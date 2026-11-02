#!/usr/bin/env python
"""Explain Creditcard frozen R-decoder squared-residual vs Gaussian-NLL ranking.

This is intentionally bound to one registered Full source and its existing
frozen-decoder diagnostic. It performs no training, encoder forward, GMM fit,
calibration, or threshold search.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_mnrf_frozen_decoder_diagnostic as frozen


SOURCE = ROOT / "result/score/MNRF-ViewProbe/Creditcard/full_gmm-full-20261024T115131Z-3977864"
OLD_DIAGNOSTIC = ROOT / "result/analysis/MNRF-FrozenDecoder/Creditcard/full_gmm-full-20261024T115131Z-3977864"
OUTPUT = ROOT / "result/analysis/MNRF-FrozenDecoder/Creditcard/R-loss-20261102"
REVISION_OUTPUT = ROOT / "result/analysis/MNRF-FrozenDecoder/Creditcard/R-loss-20261102-revision"
SEED = 2021
NORMAL_SAMPLE_COUNT = 2048
TRAIN_SIGMA_SAMPLE_COUNT = 2048
CHUNK_SIZE = 256
ATOL = 1e-7
RTOL = 1e-6
MATCH_BINS = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")


def safe_float(value):
    return None if not np.isfinite(value) else float(value)


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "std": None, "quantiles_0_1_25_50_75_99_100": [None] * 7}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "quantiles_0_1_25_50_75_99_100": [float(v) for v in np.quantile(values, [0, .01, .25, .5, .75, .99, 1])],
    }


def weighted_dominance(z: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict:
    chosen = np.flatnonzero(mask)
    if not len(chosen):
        return {"n_points": 0, "variable_weight": {}}
    selected = z[chosen]
    maxima = np.max(selected, axis=1, keepdims=True)
    ties = np.isclose(selected, maxima, rtol=RTOL, atol=ATOL)
    shares = ties / np.maximum(ties.sum(axis=1, keepdims=True), 1)
    totals = np.sum(shares, axis=0)
    return {"n_points": int(len(chosen)), "variable_weight": {str(i): float(v) for i, v in enumerate(totals) if v > 0}}


def top_variables(dominance: dict, names: list[str], limit: int = 8) -> list[dict]:
    weights = dominance["variable_weight"]
    total = sum(weights.values())
    order = sorted(((int(i), value) for i, value in weights.items()), key=lambda item: (-item[1], item[0]))
    return [{"index": i, "variable": names[i], "weighted_points": value,
             "share": value / total if total else None} for i, value in order[:limit]]


def sample_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    abnormal = np.flatnonzero(labels == 1)
    normal = np.flatnonzero(labels == 0)
    if len(normal) < NORMAL_SAMPLE_COUNT:
        raise ValueError(f"need {NORMAL_SAMPLE_COUNT} normal points; found {len(normal)}")
    rng = np.random.default_rng(SEED)
    sampled_normal = np.sort(rng.choice(normal, size=NORMAL_SAMPLE_COUNT, replace=False))
    selected = np.sort(np.concatenate((abnormal, sampled_normal)))
    return selected, abnormal, sampled_normal


def decode_training_sigma(source: dict, trainer, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    endpoints = np.asarray(source["endpoints"]["train"], dtype=np.int64)
    if len(endpoints) < TRAIN_SIGMA_SAMPLE_COUNT:
        raise ValueError("training split has fewer rows than the fixed sigma reference sample")
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(len(endpoints), size=TRAIN_SIGMA_SAMPLE_COUNT, replace=False))
    chosen_endpoints = endpoints[rows]
    features = np.load(SOURCE / "cache/train_features.npy", mmap_mode="r", allow_pickle=False)
    start, end = source["layout"].slices["R"]
    sigma_parts = []
    for left in range(0, len(rows), CHUNK_SIZE):
        row_ids = rows[left:left + CHUNK_SIZE]
        latent = np.asarray(features[row_ids, :, start:end])
        target = np.asarray(source["normalized"][chosen_endpoints[left:left + CHUNK_SIZE]])
        _, sigma, _ = frozen.decoder_nll(trainer, latent, target, "R")
        sigma_parts.append(sigma)
    sigmas = np.concatenate(sigma_parts, axis=0)
    finite("training sampled sigma", sigmas)
    return chosen_endpoints, np.median(sigmas, axis=0).astype(np.float32)


def matched_residual_sigma(sample_residual_sq: np.ndarray, sample_sigma: np.ndarray,
                           sample_labels: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for variable, name in enumerate(names):
        error = sample_residual_sq[:, variable]
        labels = sample_labels
        if not np.any(labels == 1) or not np.any(labels == 0):
            rows.append({"index": variable, "variable": name, "matched_bins": 0, "matched_anomalies": 0,
                         "matched_normals": 0, "anomaly_sigma_median": None, "normal_sigma_median": None,
                         "sigma_ratio_anomaly_over_normal": None})
            continue
        edges = np.unique(np.quantile(error, np.linspace(0, 1, MATCH_BINS + 1)))
        if len(edges) < 2:
            bin_ids = np.zeros(len(error), dtype=np.int64)
        else:
            bin_ids = np.clip(np.digitize(error, edges[1:-1], right=True), 0, len(edges) - 2)
        anomaly_sigma, normal_sigma, matched_a, matched_n = [], [], 0, 0
        matched_bins = 0
        for bin_id in np.unique(bin_ids):
            in_bin = bin_ids == bin_id
            a = in_bin & (labels == 1)
            n = in_bin & (labels == 0)
            if a.any() and n.any():
                matched_bins += 1
                matched_a += int(a.sum()); matched_n += int(n.sum())
                anomaly_sigma.extend(sample_sigma[a, variable].tolist())
                normal_sigma.extend(sample_sigma[n, variable].tolist())
        a_med = float(np.median(anomaly_sigma)) if anomaly_sigma else None
        n_med = float(np.median(normal_sigma)) if normal_sigma else None
        rows.append({"index": variable, "variable": name, "matched_bins": matched_bins,
                     "matched_anomalies": matched_a, "matched_normals": matched_n,
                     "anomaly_sigma_median": a_med, "normal_sigma_median": n_med,
                     "sigma_ratio_anomaly_over_normal": a_med / n_med if a_med is not None and n_med else None})
    return rows


def residual_bin_statistics(rows: list[dict], bins: int = MATCH_BINS) -> list[dict]:
    """Report class composition and sigma separately inside the original pooled-error bins."""
    if bins < 1:
        raise ValueError("bins must be positive")
    by_variable: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        key = (int(row["variable_index"]), row["variable"])
        by_variable.setdefault(key, []).append(row)
    output = []
    for (variable_index, name), values in sorted(by_variable.items()):
        error = np.asarray([float(row["squared_residual"]) for row in values], dtype=np.float64)
        signed_residual = np.asarray([float(row["residual"]) for row in values], dtype=np.float64)
        sigma = np.asarray([float(row["sigma"]) for row in values], dtype=np.float64)
        labels = np.asarray([int(row["label"]) for row in values], dtype=np.int8)
        if not len(error) or not np.isin(labels, (0, 1)).all() or np.any(sigma <= 0):
            raise ValueError(f"malformed residual-bin inputs for {name}")
        # Preserve the existing pooled squared-residual quantile rule, including
        # duplicate-edge collapse. Every resulting interval is emitted even if
        # a class is absent in it.
        edges = np.unique(np.quantile(error, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            edges = np.asarray([float(error.min()), float(error.max()) + np.finfo(float).eps])
        assignments = np.clip(np.digitize(error, edges[1:-1], right=True), 0, len(edges) - 2)
        for bin_id in range(len(edges) - 1):
            in_bin = assignments == bin_id
            class_stats = {}
            for label, class_name in ((0, "normal"), (1, "anomaly")):
                mask = in_bin & (labels == label)
                if mask.any():
                    class_stats[class_name] = {
                        "n": int(mask.sum()),
                        "residual_min": float(np.min(signed_residual[mask])),
                        "residual_max": float(np.max(signed_residual[mask])),
                        "residual_median": float(np.median(signed_residual[mask])),
                        "squared_residual_min": float(np.min(error[mask])),
                        "squared_residual_max": float(np.max(error[mask])),
                        "squared_residual_median": float(np.median(error[mask])),
                        "sigma_median": float(np.median(sigma[mask])),
                    }
                else:
                    class_stats[class_name] = {
                        "n": 0, "residual_min": None, "residual_max": None,
                        "residual_median": None, "squared_residual_min": None,
                        "squared_residual_max": None, "squared_residual_median": None,
                        "sigma_median": None,
                    }
            normal_sigma = class_stats["normal"]["sigma_median"]
            anomaly_sigma = class_stats["anomaly"]["sigma_median"]
            if class_stats["normal"]["n"] and class_stats["anomaly"]["n"]:
                status = "coarse_bin_overlap_only"
                ratio = anomaly_sigma / normal_sigma
            elif not class_stats["normal"]["n"] and not class_stats["anomaly"]["n"]:
                status, ratio = "empty_bin", None
            elif not class_stats["normal"]["n"]:
                status, ratio = "no_normal_class", None
            else:
                status, ratio = "no_anomaly_class", None
            output.append({
                "variable_index": variable_index, "variable": name,
                "bin_index": bin_id, "bin_count_after_duplicate_edge_collapse": len(edges) - 1,
                "squared_residual_lower": float(edges[bin_id]),
                "squared_residual_upper": float(edges[bin_id + 1]),
                "lower_inclusive": bin_id == 0, "upper_inclusive": True,
                "normal_n": class_stats["normal"]["n"], "anomaly_n": class_stats["anomaly"]["n"],
                "normal_residual_min": class_stats["normal"]["residual_min"],
                "normal_residual_max": class_stats["normal"]["residual_max"],
                "normal_residual_median": class_stats["normal"]["residual_median"],
                "anomaly_residual_min": class_stats["anomaly"]["residual_min"],
                "anomaly_residual_max": class_stats["anomaly"]["residual_max"],
                "anomaly_residual_median": class_stats["anomaly"]["residual_median"],
                "normal_squared_residual_median": class_stats["normal"]["squared_residual_median"],
                "anomaly_squared_residual_median": class_stats["anomaly"]["squared_residual_median"],
                "normal_sigma_median": normal_sigma, "anomaly_sigma_median": anomaly_sigma,
                "anomaly_over_normal_sigma_median_ratio": ratio,
                "comparison_status": status,
                "interpretation": "coarse pooled-residual bin only; not exact residual matching",
            })
    return output


def recompute_max_tie_shares(rows: list[dict], variables: int = 29) -> list[dict]:
    """Recompute each score's max membership jointly across all variables per endpoint."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["global_endpoint"], []).append(row)
    for endpoint, point_rows in grouped.items():
        point_rows.sort(key=lambda row: int(row["variable_index"]))
        if len(point_rows) != variables or [int(row["variable_index"]) for row in point_rows] != list(range(variables)):
            raise ValueError(f"endpoint {endpoint} does not contain variables 0..{variables - 1} exactly once")
        for value_key, flag_key, share_key in (
                ("nll_standardized_z", "nll_is_max_variable", "nll_max_tie_share"),
                ("residual_standardized_z", None, "residual_max_tie_share")):
            scores = np.asarray([float(row[value_key]) for row in point_rows], dtype=np.float64)
            maximum = np.max(scores)
            tied = np.isclose(scores, maximum, rtol=RTOL, atol=ATOL)
            count = int(np.count_nonzero(tied))
            if count < 1:
                raise AssertionError("a finite point must have at least one maximum variable")
            shares = tied.astype(np.float64) / count
            if not np.isclose(shares.sum(), 1.0, rtol=0, atol=1e-15):
                raise AssertionError(f"max shares for endpoint {endpoint} do not sum to one")
            for index, row in enumerate(point_rows):
                row[share_key] = repr(float(shares[index]))
                if flag_key is not None:
                    row[flag_key] = "True" if tied[index] else "False"
    return rows


def _synthetic_revision_checks() -> dict:
    """Exercise the exact revision functions on class-composition and tie counterexamples."""
    synthetic = []
    anomaly_counts = [1] * 9 + [20]
    for residual_bin, anomaly_count in enumerate(anomaly_counts):
        sigma_value = float(residual_bin + 1)
        for index in range(10):
            signed = float(residual_bin + 1)
            synthetic.append({"variable_index": "0", "variable": "V0", "squared_residual": repr(signed ** 2),
                              "residual": repr(signed), "sigma": repr(sigma_value), "label": "0"})
        for index in range(anomaly_count):
            signed = float(residual_bin + 1)
            synthetic.append({"variable_index": "0", "variable": "V0", "squared_residual": repr(signed ** 2),
                              "residual": repr(signed), "sigma": repr(sigma_value), "label": "1"})
    bins = residual_bin_statistics(synthetic)
    overlapping = [row for row in bins if row["normal_n"] and row["anomaly_n"]]
    if not overlapping or any(not math.isclose(row["anomaly_over_normal_sigma_median_ratio"], 1.0)
                              for row in overlapping):
        raise AssertionError("per-bin equal sigma must yield ratio one despite different class composition")
    if len({row["normal_sigma_median"] for row in overlapping}) < 2:
        raise AssertionError("synthetic bins must vary sigma between bins to expose pooling confounding")

    def rows_with_ties(tie_count: int) -> list[dict]:
        scores = [5.0] * tie_count + [0.0] * (29 - tie_count)
        return [{"global_endpoint": "100", "variable_index": str(i),
                 "nll_standardized_z": repr(scores[i]), "residual_standardized_z": repr(scores[i]),
                 "nll_is_max_variable": "False", "nll_max_tie_share": "0", "residual_max_tie_share": "0"}
                for i in range(29)]
    tie_results = {}
    for count in (2, 3):
        tied_rows = recompute_max_tie_shares(rows_with_ties(count))
        shares = [float(row["nll_max_tie_share"]) for row in tied_rows]
        if not np.allclose(shares[:count], 1 / count) or any(shares[count:]) or not math.isclose(sum(shares), 1.0):
            raise AssertionError(f"{count}-way max tie shares are incorrect")
        if not np.allclose([float(row["residual_max_tie_share"]) for row in tied_rows], shares):
            raise AssertionError("residual max tie shares did not follow the same rule")
        tie_results[str(count)] = {"tied_share": shares[0], "sum": sum(shares)}
    return {"residual_bins_equal_within_bin_sigma": True, "overlap_bins": len(overlapping),
            "between_bin_sigma_varies": True, "max_ties": tie_results}


def revise_existing() -> Path:
    """Create a corrected copy using only the already exported sample CSV/summary."""
    source_csv = OUTPUT / "sampled_test_point_variables.csv"
    source_summary = OUTPUT / "summary.json"
    source_readme = OUTPUT / "README.md"
    if not all(path.is_file() for path in (source_csv, source_summary, source_readme)):
        raise FileNotFoundError("existing Creditcard R-loss export is incomplete")
    original_csv_hash = sha256(source_csv)
    original_summary_hash = sha256(source_summary)
    original_readme_hash = sha256(source_readme)
    original_summary = json.loads(source_summary.read_text(encoding="utf-8"))
    if REVISION_OUTPUT.exists():
        prior_revision = REVISION_OUTPUT / "summary.json"
        if not prior_revision.is_file():
            raise FileExistsError(f"revision directory exists without its ownership record: {REVISION_OUTPUT}")
        prior_record = json.loads(prior_revision.read_text(encoding="utf-8")).get("revision", {})
        if (prior_record.get("operation") != "read_only_revise_residual_bins_and_max_tie_shares"
                or prior_record.get("input_sample_csv_sha256") != sha256(source_csv)):
            raise FileExistsError(f"revision directory is not the prior output for these exact inputs: {REVISION_OUTPUT}")
    rows = list(csv.DictReader(source_csv.open(encoding="utf-8", newline="")))
    sample = original_summary["sample"]
    if sample != {"all_valid_anomalies": 221, "normal_points": 2048, "rows_exported": 65801,
                  "seed": 2021, "selected_points": 2269, "selected_variables": 29}:
        raise ValueError(f"existing sample binding differs from expected Creditcard sample: {sample}")
    endpoints: dict[str, list[dict]] = {}
    for row in rows:
        endpoints.setdefault(row["global_endpoint"], []).append(row)
    labels_per_endpoint = {key: {int(row["label"]) for row in group} for key, group in endpoints.items()}
    if (len(rows) != sample["rows_exported"] or len(endpoints) != sample["selected_points"]
            or sum(next(iter(v)) for v in labels_per_endpoint.values()) != sample["all_valid_anomalies"]
            or any(len(v) != 1 for v in labels_per_endpoint.values())):
        raise ValueError("existing sample row count, labels, or endpoint count failed validation")
    normal_points = sum(next(iter(v)) == 0 for v in labels_per_endpoint.values())
    if normal_points != sample["normal_points"]:
        raise ValueError(f"existing normal sample count mismatch: {normal_points}")
    synthetic_checks = _synthetic_revision_checks()
    revised_rows = recompute_max_tie_shares(rows)
    bin_rows = residual_bin_statistics(rows)

    # The full-test ranking/threshold/confusion/fixed-sigma figures are copied
    # unchanged; only the bounded sample summaries below are superseded.
    revision = {
        "operation": "read_only_revise_residual_bins_and_max_tie_shares",
        "input_sample_csv_sha256": original_csv_hash,
        "input_summary_sha256": original_summary_hash,
        "input_readme_sha256": original_readme_hash,
        "source_binding": {
            "source_run_id": original_summary["source_run_id"],
            "source_path": original_summary["source_path"],
            "old_diagnostic_path": original_summary["old_diagnostic_path"],
            "source_file_sha256": original_summary["source_file_sha256"],
            "prior_diagnostic_file_sha256": original_summary["prior_diagnostic_file_sha256"],
        },
        "sample_validation": {"all_valid_anomalies": sample["all_valid_anomalies"],
                              "normal_points": normal_points, "variables": 29,
                              "points": len(endpoints), "rows": len(rows)},
        "residual_bins": {"rule": "within-variable deciles of pooled sampled squared_residual; duplicate quantile edges collapsed as in the prior match helper",
                          "bin_rows": len(bin_rows),
                          "coarse_bins_are_not_exact_residual_matches": True,
                          "no_cross_bin_pooled_sigma_ratio_reported": True},
        "max_tie_shares": {"rule": f"np.isclose(rtol={RTOL}, atol={ATOL}) jointly over all 29 variables; each tied maximum receives 1/k",
                           "endpoint_sums_verified": len(endpoints),
                           "all_endpoint_sums_equal_one": True},
        "synthetic_checks": synthetic_checks,
        "unchanged_fields": ["full_test_metrics_recomputed_from_existing_saved_scores",
                             "forward_reproduction", "training_sigma_counterfactual",
                             "source_file_sha256", "prior_diagnostic_file_sha256"],
    }
    revised_summary = dict(original_summary)
    superseded_match = revised_summary.pop("fixed_residual_strata_sigma_comparison", None)
    focus_bins = {}
    for name in ("V14", "V17"):
        matching = [row for row in bin_rows if row["variable"] == name]
        overlap = [row for row in matching if row["comparison_status"] == "coarse_bin_overlap_only"]
        ratios = [row["anomaly_over_normal_sigma_median_ratio"] for row in overlap]
        focus_bins[name] = {
            "overlap_bins": len(overlap),
            "bins_without_anomaly": sum(row["comparison_status"] == "no_anomaly_class" for row in matching),
            "coarse_bin_sigma_ratio_min": min(ratios) if ratios else None,
            "coarse_bin_sigma_ratio_max": max(ratios) if ratios else None,
            "these_are_not_exact_residual_matches": True,
        }
    revised_summary["revision"] = revision
    revised_summary["revision"]["superseded_pooled_sigma_ratio"] = {
        "old_field_removed": "fixed_residual_strata_sigma_comparison",
        "prior_values_not_reused_as_residual_controlled_evidence": superseded_match is not None,
        "reason": "pooling observations from different residual bins does not control residual magnitude",
        "replacement_focus_bin_summary": focus_bins,
    }

    REVISION_OUTPUT.mkdir(parents=True, exist_ok=True)
    revised_csv = REVISION_OUTPUT / "sampled_test_point_variables.csv"
    with revised_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(revised_rows[0]))
        writer.writeheader(); writer.writerows(revised_rows)
    bins_csv = REVISION_OUTPUT / "residual_bin_sigma_by_variable.csv"
    with bins_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bin_rows[0]))
        writer.writeheader(); writer.writerows(bin_rows)
    (REVISION_OUTPUT / "summary.json").write_text(
        json.dumps(revised_summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    comparable = sum(row["comparison_status"] == "coarse_bin_overlap_only" for row in bin_rows)
    missing_class = len(bin_rows) - comparable
    focus_text = []
    for name, value in focus_bins.items():
        if value["coarse_bin_sigma_ratio_min"] is None:
            focus_text.append(f"{name}: no overlapping class bins")
        else:
            focus_text.append(
                f"{name}: {value['overlap_bins']} overlapping bins, "
                f"{value['bins_without_anomaly']} bins without anomalies, coarse-bin sigma ratios "
                f"{value['coarse_bin_sigma_ratio_min']:.2f}–{value['coarse_bin_sigma_ratio_max']:.2f}"
            )
    readme = (
        "# Creditcard R-loss statistics revision\n\n"
        f"This is a read-only correction of `{OUTPUT}`. It consumes only its existing sample CSV and summary; it does not load a checkpoint or source cache.\n\n"
        "## Residual-bin correction\n\n"
        f"For each variable, bins reuse the previous pooled sampled squared-residual quantile rule ({MATCH_BINS} requested quantiles, duplicate edges collapsed). "
        f"The table contains {len(bin_rows)} variable/bin rows: {comparable} bins contain both classes and {missing_class} preserve a missing class or empty-bin state. "
        "Each row reports both class counts, signed-residual and squared-residual ranges/medians, sigma medians, and an anomaly/normal sigma ratio only when both classes are present. "
        "These are coarse bins, not exact residual matches. No cross-bin pooled ratio is reported; the earlier 63.4x/76.2x wording is withdrawn as a residual-controlled claim.\n\n"
        + "For the prespecified variables, " + "; ".join(focus_text) + ". These within-bin ratios are still descriptive only: each coarse interval can contain materially different residuals between classes, so they do not isolate a sigma effect at fixed residual. The corrected evidence is therefore insufficient to quantify an independent residual-controlled sigma difference.\n\n"
        "## Max tie shares\n\n"
        f"For each endpoint, NLL and residual standardized values are compared jointly across all 29 variables using the original `np.isclose(rtol={RTOL}, atol={ATOL})`. "
        "If k variables tie at the maximum each now receives 1/k; other variables receive zero. Endpoint shares sum to one. The full-test `weighted_dominance` summary was not recalculated from this sample and remains byte-for-byte copied from the original summary.\n\n"
        "## Preserved evidence\n\n"
        "The original residual/NLL full-test AUC/AP, confusion matrices, thresholds, fixed-sigma counterfactual, and source bindings are copied unchanged into this revision summary. "
        "The revised per-point CSV retains the original sample selection: all 221 anomalies, 2,048 seed-2021 sampled normals, and 29 variables (65,801 rows).\n\n"
        "Synthetic checks exercised the actual binning and tie-share functions: within-bin equal sigmas yielded ratios of one despite differing class proportions across bins; two- and three-way ties yielded 1/2 and 1/3 shares summing to one.\n"
    )
    (REVISION_OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    if (sha256(source_csv) != original_csv_hash or sha256(source_summary) != original_summary_hash
            or sha256(source_readme) != original_readme_hash):
        raise RuntimeError("original R-loss export changed while creating read-only revision")
    print(json.dumps({"revision_output": str(REVISION_OUTPUT), "sample_points": len(endpoints),
                      "rows": len(revised_rows), "residual_bin_rows": len(bin_rows),
                      "bins_with_both_classes": comparable, "synthetic_checks": synthetic_checks,
                      "original_full_metrics_unchanged": True}, indent=2))
    return REVISION_OUTPUT


def main() -> Path:
    import torch
    from threadpoolctl import threadpool_limits
    from sklearn.metrics import average_precision_score, roc_auc_score

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        if OUTPUT.exists():
            raise FileExistsError(f"refusing to overwrite diagnostic output: {OUTPUT}")
        if not SOURCE.is_dir() or not OLD_DIAGNOSTIC.is_dir():
            raise FileNotFoundError("fixed Full source or its bound frozen-decoder diagnostic is missing")

        source = frozen.validate_source_contract(SOURCE)
        trainer, checkpoint_hash = frozen.load_decoder(source)
        old_manifest = json.loads((OLD_DIAGNOSTIC / "diagnostic_manifest.json").read_text(encoding="utf-8"))
        if old_manifest.get("source_run") != str(SOURCE) or old_manifest.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("existing R diagnostic is not bound to the fixed Full source/checkpoint")
        if old_manifest.get("encoder_forward_called") is not False or old_manifest.get("gmm_fit_called") is not False:
            raise ValueError("existing decoder diagnostic has unexpected encoder/GMM execution record")

        evidence_path = OLD_DIAGNOSTIC / "R_test_prediction_evidence.npz"
        nll_path = OLD_DIAGNOSTIC / "R_nll_test_scores.npz"
        residual_score_path = OLD_DIAGNOSTIC / "R_squared_residual_test_scores.npz"
        with np.load(evidence_path, allow_pickle=False) as z:
            old_residual = np.asarray(z["residual"])
            old_per_variable_nll = np.asarray(z["per_variable_nll"])
            old_endpoints = np.asarray(z["endpoints"])
            names = [str(value) for value in np.asarray(z["variable_names"]).tolist()]
        with np.load(nll_path, allow_pickle=False) as z:
            old_nll_scores = np.asarray(z["scores"])
            old_nll_labels = np.asarray(z["labels"])
            old_nll_indices = np.asarray(z["global_indices"])
            old_nll_threshold = float(np.asarray(z["threshold"]).reshape(()))
            nll_center = np.asarray(z["center"])
            nll_scale = np.asarray(z["scale"])
        with np.load(residual_score_path, allow_pickle=False) as z:
            old_residual_scores = np.asarray(z["scores"])
            old_residual_labels = np.asarray(z["labels"])
            old_residual_indices = np.asarray(z["global_indices"])
            old_residual_threshold = float(np.asarray(z["threshold"]).reshape(()))
            residual_center = np.asarray(z["center"])
            residual_scale = np.asarray(z["scale"])

        endpoints = np.asarray(source["endpoints"]["test"], dtype=np.int64)
        axes = source["axes"]
        test_axis = np.asarray(axes["test_global"], dtype=np.int64)
        if len(names) != source["variables"] or names != list(source["manifest"]["variable_names"]):
            raise ValueError("existing evidence variable order differs from frozen Full source")
        if not (np.array_equal(old_endpoints, endpoints)
                and np.array_equal(old_nll_indices, endpoints)
                and np.array_equal(old_residual_indices, endpoints)
                and np.array_equal(old_nll_labels, old_residual_labels)):
            raise ValueError("existing R diagnostic coordinates/labels do not match source test endpoints")
        test_positions = np.searchsorted(test_axis, endpoints)
        if np.any(test_positions >= len(test_axis)) or not np.array_equal(test_axis[test_positions], endpoints):
            raise ValueError("source test endpoints do not map to complete test axis")
        labels = np.asarray(axes["test_labels"], dtype=np.int8)[test_positions]
        times = np.asarray(axes["test_times"], dtype=np.int64)[test_positions]
        if not np.array_equal(labels, old_nll_labels.astype(np.int8)):
            raise ValueError("existing diagnostic labels differ from source labels")
        if nll_center.shape != (len(names),) or nll_scale.shape != nll_center.shape or np.any(nll_scale <= 0):
            raise ValueError("saved NLL normalization parameters are malformed")
        if residual_center.shape != (len(names),) or residual_scale.shape != residual_center.shape or np.any(residual_scale <= 0):
            raise ValueError("saved residual normalization parameters are malformed")

        # Run exactly the existing frozen R decoder over cached raw R latent rows.
        fresh = frozen._split_predictions(source, trainer, "test", "R", CHUNK_SIZE)
        fresh_residual = fresh["target"] - fresh["mean"]
        fresh_residual_sq = fresh_residual ** 2
        fresh_nll = fresh["nll"]
        for name, actual, expected in (
                ("R residual", fresh_residual, old_residual),
                ("R NLL", fresh_nll, old_per_variable_nll)):
            if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=RTOL, atol=ATOL):
                delta = float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape else None
                raise ValueError(f"{name} does not reproduce existing per-point evidence (max_abs={delta})")
        finite("fresh test sigma", fresh["sigma"])
        finite("fresh test residual", fresh_residual)
        finite("fresh test NLL", fresh_nll)

        # Reproduce the existing full-test scores using saved (not refit) center/scale.
        nll_z = (fresh_nll - nll_center) / nll_scale
        residual_z = (fresh_residual_sq - residual_center) / residual_scale
        rebuilt_nll_scores = np.max(nll_z, axis=1)
        rebuilt_residual_scores = np.max(residual_z, axis=1)
        if not np.allclose(rebuilt_nll_scores, old_nll_scores, rtol=RTOL, atol=ATOL):
            raise ValueError("saved R max-NLL scores do not reproduce from saved per-variable NLL/center/scale")
        if not np.allclose(rebuilt_residual_scores, old_residual_scores, rtol=RTOL, atol=ATOL):
            raise ValueError("saved R max-residual scores do not reproduce from saved residual/center/scale")

        auc_residual = float(roc_auc_score(labels, old_residual_scores))
        ap_residual = float(average_precision_score(labels, old_residual_scores))
        auc_nll = float(roc_auc_score(labels, old_nll_scores))
        ap_nll = float(average_precision_score(labels, old_nll_scores))
        selected, abnormal_rows, sampled_normal_rows = sample_indices(labels)
        train_sample_endpoints, train_sigma_median = decode_training_sigma(source, trainer, names)

        # Compose fixed-sigma counterfactual for selected test points only.
        counterfactual_nll = (fresh_residual[selected] ** 2 / (2.0 * train_sigma_median[None, :] ** 2)
                              + np.log(train_sigma_median[None, :]) + 0.5 * math.log(2 * math.pi))
        sample_z = nll_z[selected]
        residual_sample_z = residual_z[selected]
        sample_labels = labels[selected]
        sample_sq = fresh_residual_sq[selected]
        sample_sigma = fresh["sigma"][selected]
        matched = matched_residual_sigma(sample_sq, sample_sigma, sample_labels, names)

        from sklearn.metrics import average_precision_score as ap_score
        per_variable = []
        for variable, name in enumerate(names):
            per_variable.append({
                "index": variable, "variable": name,
                "squared_residual_ap_full_test": float(ap_score(labels, fresh_residual_sq[:, variable])),
                "auxiliary_nll_ap_full_test": float(ap_score(labels, fresh_nll[:, variable])),
                "squared_residual_auc_full_test": float(roc_auc_score(labels, fresh_residual_sq[:, variable])),
                "auxiliary_nll_auc_full_test": float(roc_auc_score(labels, fresh_nll[:, variable])),
                "sigma_median_normal_sample": float(np.median(sample_sigma[sample_labels == 0, variable])),
                "sigma_median_anomaly_all": float(np.median(fresh["sigma"][labels == 1, variable])),
                "sigma_distribution_normal_sample": distribution(sample_sigma[sample_labels == 0, variable]),
                "sigma_distribution_anomaly_all": distribution(fresh["sigma"][labels == 1, variable]),
                "residual_sq_distribution_normal_sample": distribution(sample_sq[sample_labels == 0, variable]),
                "residual_sq_distribution_anomaly_all": distribution(fresh_residual_sq[labels == 1, variable]),
                "weighted_error_distribution_normal_sample": distribution(
                    (sample_sq[sample_labels == 0, variable] / (2 * sample_sigma[sample_labels == 0, variable] ** 2))),
                "weighted_error_distribution_anomaly_all": distribution(
                    (fresh_residual_sq[labels == 1, variable] / (2 * fresh["sigma"][labels == 1, variable] ** 2))),
                "log_sigma_distribution_normal_sample": distribution(np.log(sample_sigma[sample_labels == 0, variable])),
                "log_sigma_distribution_anomaly_all": distribution(np.log(fresh["sigma"][labels == 1, variable])),
                "nll_distribution_normal_sample": distribution(fresh_nll[selected[sample_labels == 0], variable]),
                "nll_distribution_anomaly_all": distribution(fresh_nll[labels == 1, variable]),
                "saved_nll_center": float(nll_center[variable]), "saved_nll_scale": float(nll_scale[variable]),
                "saved_residual_center": float(residual_center[variable]), "saved_residual_scale": float(residual_scale[variable]),
                "nll_mad_floor_active": bool(nll_scale[variable] <= source["config"].score_mad_floor),
                "residual_mad_floor_active": bool(residual_scale[variable] <= source["config"].score_mad_floor),
            })

        dominance = {}
        for score_name, z, scores, threshold in (
                ("max_nll", nll_z, old_nll_scores, old_nll_threshold),
                ("max_squared_residual", residual_z, old_residual_scores, old_residual_threshold)):
            class_groups = {
                "all_anomalies": labels == 1,
                "all_normals": labels == 0,
                "tp": (labels == 1) & (scores > threshold),
                "fp": (labels == 0) & (scores > threshold),
                "fn": (labels == 1) & (scores <= threshold),
            }
            dominance[score_name] = {}
            for group_name, mask in class_groups.items():
                result = weighted_dominance(z, labels, mask)
                result["top_variables"] = top_variables(result, names)
                dominance[score_name][group_name] = result

        # Keep component arithmetic in torch float32, matching decoder_nll's
        # operation order and dtype, so this check measures implementation
        # consistency rather than a float64 re-expression's rounding delta.
        with torch.no_grad():
            torch_target = torch.as_tensor(fresh["target"][selected])
            torch_mean = torch.as_tensor(fresh["mean"][selected])
            torch_sigma = torch.as_tensor(fresh["sigma"][selected])
            torch_residual = torch_target - torch_mean
            torch_weighted = torch_residual.square() / (2.0 * torch_sigma.square())
            torch_log_sigma = torch.log(torch_sigma)
            torch_constant = torch.full_like(torch_sigma, 0.5 * math.log(2.0 * math.pi))
            torch_reconstructed = torch_weighted + torch_log_sigma + 0.5 * math.log(2.0 * math.pi)
        sample_components = {
            "squared_error_weighted_by_predicted_sigma": torch_weighted.numpy(),
            "log_sigma": torch_log_sigma.numpy(),
            "gaussian_constant": torch_constant.numpy(),
            "reconstructed_nll": torch_reconstructed.numpy(),
            "counterfactual_fixed_train_sigma_nll": counterfactual_nll,
        }
        formula_reconstructed = (sample_components["squared_error_weighted_by_predicted_sigma"]
                                 + sample_components["log_sigma"] + sample_components["gaussian_constant"])
        if not np.allclose(formula_reconstructed, fresh_nll[selected], rtol=RTOL, atol=ATOL):
            raise ValueError("component formula does not reproduce decoder_nll output")

        OUTPUT.mkdir(parents=True)
        old_resid_row = next(row for row in csv.DictReader((OLD_DIAGNOSTIC / "metrics.csv").open())
                             if row["view"] == "R" and row["variant"] == "decoder" and row["score"] == "squared_residual")
        old_nll_row = next(row for row in csv.DictReader((OLD_DIAGNOSTIC / "metrics.csv").open())
                           if row["view"] == "R" and row["variant"] == "decoder" and row["score"] == "nll")
        output_rows = []
        original_prediction = old_nll_scores > old_nll_threshold
        for local_row in selected:
            endpoint = int(endpoints[local_row])
            source_axis_pos = np.searchsorted(np.asarray(axes["all_global"], dtype=np.int64), endpoint)
            if source_axis_pos >= len(axes["all_global"]) or int(axes["all_global"][source_axis_pos]) != endpoint:
                raise ValueError(f"sample endpoint {endpoint} is absent from full source axis")
            for variable, name in enumerate(names):
                output_rows.append({
                    "global_endpoint": endpoint, "time_ns": int(times[local_row]), "label": int(labels[local_row]),
                    "variable_index": variable, "variable": name,
                    "target_y": float(fresh["target"][local_row, variable]),
                    "mu": float(fresh["mean"][local_row, variable]),
                    "sigma": float(fresh["sigma"][local_row, variable]),
                    "residual": float(fresh_residual[local_row, variable]),
                    "squared_residual": float(fresh_residual_sq[local_row, variable]),
                    "weighted_error_term": float(sample_components["squared_error_weighted_by_predicted_sigma"][np.searchsorted(selected, local_row), variable]),
                    "log_sigma_term": float(np.log(fresh["sigma"][local_row, variable])),
                    "gaussian_constant_term": float(0.5 * math.log(2 * math.pi)),
                    "reconstructed_nll": float(fresh_nll[local_row, variable]),
                    "saved_per_variable_nll": float(old_per_variable_nll[local_row, variable]),
                    "nll_abs_error_vs_saved": float(abs(fresh_nll[local_row, variable] - old_per_variable_nll[local_row, variable])),
                    "counterfactual_fixed_train_sigma_nll": float(counterfactual_nll[np.searchsorted(selected, local_row), variable]),
                    "nll_center": float(nll_center[variable]), "nll_scale": float(nll_scale[variable]),
                    "nll_standardized_z": float(nll_z[local_row, variable]),
                    "nll_is_max_variable": bool(np.isclose(nll_z[local_row, variable], np.max(nll_z[local_row]), rtol=RTOL, atol=ATOL)),
                    "nll_max_tie_share": (float(np.isclose(nll_z[local_row], np.max(nll_z[local_row]), rtol=RTOL, atol=ATOL).sum())
                                           and (1.0 / np.isclose(nll_z[local_row, variable], np.max(nll_z[local_row]), rtol=RTOL, atol=ATOL).sum())
                                           if np.isclose(nll_z[local_row, variable], np.max(nll_z[local_row]), rtol=RTOL, atol=ATOL) else 0.0),
                    "residual_center": float(residual_center[variable]), "residual_scale": float(residual_scale[variable]),
                    "residual_standardized_z": float(residual_z[local_row, variable]),
                    "residual_max_tie_share": (1.0 / np.isclose(residual_z[local_row], np.max(residual_z[local_row]), rtol=RTOL, atol=ATOL).sum()
                                                if np.isclose(residual_z[local_row, variable], np.max(residual_z[local_row]), rtol=RTOL, atol=ATOL) else 0.0),
                    "full_nll_prediction": bool(original_prediction[local_row]),
                    "sampling_group": "anomaly_all" if labels[local_row] else "normal_seed2021_2048",
                })
        points_path = OUTPUT / "sampled_test_point_variables.csv"
        with points_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader(); writer.writerows(output_rows)

        summary = {
            "task": "Creditcard", "source_run_id": SOURCE.name, "source_path": str(SOURCE),
            "old_diagnostic_path": str(OLD_DIAGNOSTIC), "variables": names,
            "test_valid_points": int(len(endpoints)), "test_anomalies": int(np.sum(labels == 1)),
            "sample": {"seed": SEED, "normal_points": int(len(sampled_normal_rows)),
                       "all_valid_anomalies": int(len(abnormal_rows)), "selected_points": int(len(selected)),
                       "selected_variables": int(len(names)), "rows_exported": int(len(output_rows))},
            "full_test_metrics_recomputed_from_existing_saved_scores": {
                "squared_residual": {"roc_auc": auc_residual, "average_precision": ap_residual,
                                     "saved_threshold": old_residual_threshold,
                                     "saved_metrics_row": old_resid_row},
                "auxiliary_decoder_nll": {"roc_auc": auc_nll, "average_precision": ap_nll,
                                           "saved_threshold": old_nll_threshold,
                                           "saved_metrics_row": old_nll_row},
                "ap_difference_nll_minus_residual": ap_nll - ap_residual,
            },
            "forward_reproduction": {"residual_max_abs_diff": float(np.max(np.abs(fresh_residual - old_residual))),
                                      "nll_max_abs_diff": float(np.max(np.abs(fresh_nll - old_per_variable_nll))),
                                      "rtol": RTOL, "atol": ATOL,
                                      "aggregate_residual_max_abs_diff": float(np.max(np.abs(rebuilt_residual_scores - old_residual_scores))),
                                      "aggregate_nll_max_abs_diff": float(np.max(np.abs(rebuilt_nll_scores - old_nll_scores)))},
            "component_distributions_sampled_points": {
                "residual_squared": {"normal": distribution(sample_sq[sample_labels == 0]),
                                     "anomaly": distribution(sample_sq[sample_labels == 1])},
                "sigma": {"normal": distribution(sample_sigma[sample_labels == 0]),
                          "anomaly": distribution(sample_sigma[sample_labels == 1])},
                "weighted_error_term": {"normal": distribution(sample_components["squared_error_weighted_by_predicted_sigma"][sample_labels == 0]),
                                        "anomaly": distribution(sample_components["squared_error_weighted_by_predicted_sigma"][sample_labels == 1])},
                "log_sigma_term": {"normal": distribution(sample_components["log_sigma"][sample_labels == 0]),
                                   "anomaly": distribution(sample_components["log_sigma"][sample_labels == 1])},
                "nll": {"normal": distribution(fresh_nll[selected[sample_labels == 0]]),
                        "anomaly": distribution(fresh_nll[selected[sample_labels == 1]])},
            },
            "fixed_residual_strata_sigma_comparison": {"definition": f"Within-variable decile bins of pooled sampled squared residual; compare anomaly/normal sigma only in bins containing both; {MATCH_BINS} bins requested.",
                                                       "variables": matched},
            "training_sigma_counterfactual": {"seed": SEED, "sampled_training_endpoints": int(len(train_sample_endpoints)),
                                               "endpoint_sha256": hashlib.sha256(np.asarray(train_sample_endpoints, dtype=np.int64).tobytes()).hexdigest(),
                                               "sigma_median_by_variable": {name: float(value) for name, value in zip(names, train_sigma_median)},
                                               "purpose": "fixed-sigma mechanism diagnostic only; no recalibration or formal metrics"},
            "full_test_per_variable_ranking": per_variable,
            "max_aggregation_variable_dominance_tie_split": dominance,
            "limitations": [
                "This is the frozen Stage-1 R auxiliary decoder, not the Full GMM scoring head.",
                "The 2,048-normal sample is for bounded distributional inspection; its AP is not a full-test estimate.",
                "Sigma/error associations are observational and do not alone prove variance masking is causal.",
                "The fixed training-sigma counterfactual is diagnostic only and does not produce calibrated alarms.",
            ],
            "source_file_sha256": {rel: sha256(SOURCE / rel) for rel in (
                "probe_manifest.json", "analysis_inputs.npz", "input_statistics.npz",
                "checkpoints/best_stage1.pt", "checkpoints/best_encoder.pt", "cache/normalized_values.npy",
                "cache/test_features.npy", "cache/test_window_end_indices.npy")},
            "prior_diagnostic_file_sha256": {name: sha256(OLD_DIAGNOSTIC / name) for name in (
                "diagnostic_manifest.json", "R_test_prediction_evidence.npz", "R_nll_test_scores.npz",
                "R_squared_residual_test_scores.npz", "metrics.csv")},
            "execution": {"decoder": "existing decoder_nll via frozen._split_predictions", "task": "R",
                          "encoder_forward": False, "training": False, "gmm_fit": False,
                          "recalibration": False, "threshold_search": False,
                          "cpu_threads": 1, "chunk_size": CHUNK_SIZE,
                          "elapsed_seconds": time.perf_counter() - started},
        }
        summary_path = OUTPUT / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

        focused = {row["variable"]: row for row in per_variable if row["variable"] in {"V14", "V17"}}
        credit_answer = (
            "Full-test residual ranking exceeds the R auxiliary-NLL ranking, and per-variable AP/AUC tables identify where the loss first appears. "
            "The Gaussian weighted-error and log-sigma term distributions, plus residual-matched sigma strata, show whether predicted scale covaries with errors on this sample. "
            "Max aggregation dominance tables separately show which variables control the final score. These are diagnostic associations, not proof that sigma alone causes the ranking loss."
        )
        readme = (
            "# Creditcard R auxiliary-decoder NLL loss diagnosis\n\n"
            f"Source: `{SOURCE}`  \nBound prior frozen-decoder diagnostic: `{OLD_DIAGNOSTIC}`  \n"
            f"New output: `{OUTPUT}`\n\n"
            "## Verified full-test result\n\n"
            f"From the existing full-test saved scores, squared-residual AP={ap_residual:.12f}, ROC-AUC={auc_residual:.12f}; "
            f"auxiliary Gaussian-NLL AP={ap_nll:.12f}, ROC-AUC={auc_nll:.12f}. Fresh frozen R decoder reproduction passed "
            f"with max absolute residual error {summary['forward_reproduction']['residual_max_abs_diff']:.3g} and NLL error "
            f"{summary['forward_reproduction']['nll_max_abs_diff']:.3g} (rtol={RTOL:g}, atol={ATOL:g}).\n\n"
            "## What the evidence says\n\n"
            f"{credit_answer}\n\n"
            "- `sampled_test_point_variables.csv` contains every valid test anomaly plus 2,048 valid normal endpoints sampled without replacement with seed 2021, across all 29 variables. It includes y, mu, sigma, residual, squared residual, all NLL terms, saved NLL, the fixed training-sigma counterfactual, saved normalization parameters, and max-variable/tie contribution.\n"
            "- `summary.json` separates full-test per-variable ranking from bounded sample distributions and records source/prior-evidence hashes. The sample is not used to claim full-test AP.\n"
            "- The fixed-sigma counterfactual uses one sigma median per variable from 2,048 seed-2021 training endpoints; it does not recalibrate or create a formal alarm result.\n\n"
            "## Focus variables\n\n"
            + ("; ".join(f"{name}: residual AP {row['squared_residual_ap_full_test']:.6g}, NLL AP {row['auxiliary_nll_ap_full_test']:.6g}, normal sampled sigma median {row['sigma_median_normal_sample']:.6g}, anomaly sigma median {row['sigma_median_anomaly_all']:.6g}" for name, row in focused.items()) or "V14/V17 rows were not available by source variable name.")
            + "\n\n## Scope\n\nThis concerns only the frozen Stage-1 R auxiliary decoder. It does not establish the cause of the separate Full GMM score. No training, encoder forward, GMM fitting, recalibration, or threshold search was run.\n"
        )
        (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
        print(json.dumps({"output_dir": str(OUTPUT), "sampled_points": len(selected),
                          "exported_rows": len(output_rows), "residual_ap": ap_residual,
                          "nll_ap": ap_nll,
                          "max_abs_residual_diff": summary["forward_reproduction"]["residual_max_abs_diff"],
                          "max_abs_nll_diff": summary["forward_reproduction"]["nll_max_abs_diff"]}, indent=2))
    return OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revise-existing", action="store_true",
                        help="read the existing R-loss CSV/summary only; do not load source data or model weights")
    args = parser.parse_args()
    if args.revise_existing:
        _synthetic_revision_checks()
        revise_existing()
    else:
        main()
