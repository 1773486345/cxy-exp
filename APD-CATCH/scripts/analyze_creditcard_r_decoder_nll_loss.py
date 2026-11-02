#!/usr/bin/env python
"""Explain Creditcard frozen R-decoder squared-residual vs Gaussian-NLL ranking.

This is intentionally bound to one registered Full source and its existing
frozen-decoder diagnostic. It performs no training, encoder forward, GMM fit,
calibration, or threshold search.
"""

from __future__ import annotations

import csv
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
    main()
