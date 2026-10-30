#!/usr/bin/env python
"""Compare Full covariance/aggregation choices on the eight pinned Full sources."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mnrf_branch_ablation import TASKS, _sha256_file
from scripts.run_mnrf_branch_probe_diagnostic import (
    _aligned_bundle, _load_source_axes, _metrics, _nll_artifacts, _verify_head,
    _config_for, _load_or_fit_head, _json, validate_pinned_source_binding,
)
from scripts.summarize_mnrf_branch_ablation import _evaluation
from scripts.run_mnrf_view_probe import _read_task
from scripts.validate_mnrf_view_probe import validate
from ts_benchmark.baselines.mnrf_view_probe import MNRFViewProbe
from ts_benchmark.baselines.mnrf_view_probe.evaluation import calibrated_quantile

BATCH = Path("result/analysis/MNRF-ViewProbe-Ablation/branch-ablation-20261024T210034Z-2771438")
DIAG_EXISTING = BATCH / "branch-probe"
SOURCE_ROOT = ROOT / "result" / "score" / "MNRF-ViewProbe"
AGGREGATION_INPUT = ROOT / BATCH / "aggregation_comparison.csv"
COMMON_BATCH = ROOT / "result" / "analysis" / "MNRF-ViewProbe-Common" / "common-baseline-20261029T-newrun"
SEED = 2021
Q = 0.995
LAYOUT = {"S": (0, 1), "E": (1, 9), "P": (9, 17), "R": (17, 25)}


def _save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _aggregate(calibration_nll, test_nll, center, scale, aggregation, quantile=Q):
    cal = np.asarray(calibration_nll)
    test = np.asarray(test_nll)
    center = np.asarray(center)
    scale = np.asarray(scale)
    if cal.ndim != 2 or test.ndim != 2 or cal.shape[1] != test.shape[1]:
        raise ValueError("calibration/test NLL must be 2D with the same variable count")
    if center.shape != (cal.shape[1],) or scale.shape != center.shape:
        raise ValueError("center/scale do not match NLL variables")
    if not all(np.isfinite(x).all() for x in (cal, test, center, scale)) or np.any(scale <= 0):
        raise ValueError("NLL and calibration statistics must be finite with positive scales")
    if aggregation not in ("max", "mean"):
        raise ValueError("aggregation must be max or mean")
    cal_z = (cal - center) / scale
    test_z = (test - center) / scale
    axis = 1
    if aggregation == "max":
        cal_score = np.max(cal_z, axis=axis)
        test_score = np.max(test_z, axis=axis)
    else:
        cal_score = np.mean(cal_z, axis=axis)
        test_score = np.mean(test_z, axis=axis)
    threshold = calibrated_quantile(cal_score, float(quantile))
    prediction = (test_score > threshold).astype(np.int8)
    return test_score, float(threshold), prediction


def _check_equal(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("{} differs: saved={} replayed={}".format(label, actual, expected))


def _effect_deltas(by_config, task):
    full_max = by_config.get((task, "full", "max"))
    full_mean = by_config.get((task, "full", "mean"))
    diag_max = by_config.get((task, "diag", "max"))
    diag_mean = by_config.get((task, "diag", "mean"))

    def delta(left, right, metric):
        if left is None or right is None or left.get(metric) is None or right.get(metric) is None:
            return "n/a"
        return "{:+.5f}".format(float(left[metric]) - float(right[metric]))

    values = {
        "covariance_ap": "{}/{}".format(delta(diag_max, full_max, "average_precision"),
                                          delta(diag_mean, full_mean, "average_precision")),
        "covariance_auc": "{}/{}".format(delta(diag_max, full_max, "roc_auc"),
                                           delta(diag_mean, full_mean, "roc_auc")),
        "aggregation_ap": "{}/{}".format(delta(full_mean, full_max, "average_precision"),
                                            delta(diag_mean, diag_max, "average_precision")),
        "aggregation_auc": "{}/{}".format(delta(full_mean, full_max, "roc_auc"),
                                             delta(diag_mean, diag_max, "roc_auc")),
    }
    values["interaction_ap"] = "n/a"
    values["interaction_auc"] = "n/a"
    if all(row is not None for row in (full_max, full_mean, diag_max, diag_mean)):
        if all(row.get("average_precision") is not None for row in (full_max, full_mean, diag_max, diag_mean)):
            values["interaction_ap"] = "{:+.5f}".format(
                (diag_mean["average_precision"] - diag_max["average_precision"])
                - (full_mean["average_precision"] - full_max["average_precision"]))
        if all(row.get("roc_auc") is not None for row in (full_max, full_mean, diag_max, diag_mean)):
            values["interaction_auc"] = "{:+.5f}".format(
                (diag_mean["roc_auc"] - diag_max["roc_auc"])
                - (full_mean["roc_auc"] - full_max["roc_auc"]))
    return values


def _diagnostic_output_paths(reuse_batch, timestamp=None, pid=None):
    timestamp = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    pid = os.getpid() if pid is None else pid
    score_parent = ROOT / "result" / "score" / "MNRF-ViewProbe-FullDiagnostic"
    analysis_parent = ROOT / "result" / "analysis" / "MNRF-ViewProbe-FullDiagnostic"
    if reuse_batch:
        name = Path(reuse_batch).name
        if name != reuse_batch or not name.startswith("full-head-diagnostic-"):
            raise ValueError("--reuse-batch must be an existing diagnostic batch directory name")
        score_root = score_parent / name
        if not score_root.is_dir():
            raise FileNotFoundError("reusable diagnostic score directory is missing: {}".format(score_root))
        analysis_root = analysis_parent / "full-head-diagnostic-reuse-{}-{}-{}".format(
            name.rsplit("-", 1)[-1], timestamp, time.time_ns())
        return analysis_root, score_root, True
    name = "full-head-diagnostic-{}-{}".format(timestamp, pid)
    return analysis_parent / name, score_parent / name, False


def _result_row(task, config_name, covariance, aggregation, run_path, data, aligned,
                source, source_manifest, axes, fit_seconds=0.0, status="reused"):
    if aggregation == "max":
        scores = np.max((data["test_nll"] - data["center"]) / data["scale"], axis=1)
        threshold = data["threshold"]
        predictions = (scores > threshold).astype(np.int8)
        if not np.allclose(scores, aligned["scores"], rtol=1e-6, atol=1e-7):
            raise ValueError("{} max scores do not replay from NLL".format(config_name))
        if not np.array_equal(predictions, aligned["predictions"]):
            raise ValueError("{} max predictions do not replay from threshold".format(config_name))
        saved_threshold = float(aligned["threshold"])
        _check_equal(threshold, saved_threshold, config_name + " max threshold")
    else:
        scores, threshold, predictions = _aggregate(data["calibration_nll"], data["test_nll"],
                                                    data["center"], data["scale"], "mean", data["quantile"])
    candidate = dict(aligned)
    candidate.update(scores=scores, threshold=threshold, predictions=predictions)
    metrics = _metrics(candidate["labels"], scores, predictions, candidate, axes)
    manifest = data["manifest"]
    status_path = Path(run_path) / "heads" / "gmm_status.json"
    gmm_status = _json(status_path) if status_path.is_file() else {}
    metrics_path = Path(run_path) / "results" / "metrics.json"
    run_metrics = _json(metrics_path) if metrics_path.is_file() else {}
    historical = status == "reused"
    return {
        "dataset": task, "configuration": config_name, "covariance_type": covariance,
        "aggregation": aggregation, "head_seed": SEED, "source_run_id": source.name,
        "source_identity_sha256": source_manifest.get("source_identity_sha256"),
        "feature_fingerprint": source_manifest.get("feature_fingerprint"),
        "run_path": str(Path(run_path).resolve()), "status": status,
        "selected_dim": 25, "variable_count": int(data["test_nll"].shape[1]),
        "gmm_components": manifest.get("config", {}).get("gmm_components"),
        "gmm_covariance_type": covariance,
        "head_parameter_count": gmm_status.get("head_parameter_count",
            _json(Path(run_path) / "results" / "metrics.json").get("head_parameter_count")
            if (Path(run_path) / "results" / "metrics.json").is_file() else None),
        "converged": all(item.get("converged") for item in gmm_status.get("variables", []))
            if gmm_status.get("variables") else None,
        "threshold": float(threshold), "roc_auc": metrics["auc_roc"], "average_precision": metrics["ap"],
        "precision": metrics["precision"], "recall": metrics["recall"], "point_f1": metrics["f1"],
        "normal_fpr": metrics["normal_fpr"], "tp": metrics["tp"], "fp": metrics["fp"],
        "fn": metrics["fn"], "tn": metrics["tn"],
        "affiliation_precision": metrics["affiliation_precision"],
        "affiliation_recall": metrics["affiliation_recall"], "affiliation_f": metrics["affiliation_f"],
        "affiliation_scope": metrics["affiliation_scope"],
        "anomaly_segments_hit": metrics["anomaly_segments_hit"],
        "anomaly_segments_scored": metrics["anomaly_segments_scored"],
        "anomaly_segments_total": metrics["anomaly_segments_total"],
        "fit_time_seconds_historical": run_metrics.get("fit_time_seconds") if historical else None,
        "inference_time_seconds_historical": run_metrics.get("inference_time_seconds") if historical else None,
        "fit_time_seconds_this_run": run_metrics.get("fit_time_seconds") if not historical else None,
        "inference_time_seconds_this_run": run_metrics.get("inference_time_seconds") if not historical else None,
        "end_to_end_seconds_this_run": float(fit_seconds) if not historical else None,
        "fit_endpoints_sha256": manifest.get("gmm_fit_endpoints_sha256"),
        "failure_reason": None,
    }, candidate


def _contributors(data, aligned, source_manifest, config_name, aggregation):
    z = (data["test_nll"] - data["center"]) / data["scale"]
    if aggregation == "max":
        variable_scores = z
    else:
        variable_scores = z
    maxima = np.max(variable_scores, axis=1, keepdims=True)
    ties = np.isclose(variable_scores, maxima, rtol=0, atol=1e-7)
    labels = np.asarray(aligned["labels"], dtype=np.int8)
    pred = np.asarray(aligned["predictions"], dtype=np.int8)
    names = source_manifest["variable_names"]
    rows = []
    for group, mask in (("TP", (labels == 1) & (pred == 1)), ("FP", (labels == 0) & (pred == 1))):
        counts = ties[mask].sum(axis=0, dtype=np.float64)
        denom = counts.sum()
        order = np.argsort(-counts, kind="stable")
        top = [{"variable": str(names[i]), "index": int(i), "tie_split_count": float(counts[i]),
                "share": None if denom == 0 else float(counts[i] / denom)}
               for i in order[:10] if counts[i] > 0]
        rows.append({"dataset": source_manifest.get("task_name", "").split(".")[0],
                     "configuration": config_name, "aggregation": aggregation,
                     "population": group, "point_count": int(mask.sum()),
                     "max_contribution_units": float(denom),
                     "top_variables_json": json.dumps(top, ensure_ascii=False, allow_nan=False)})
    return rows


def _feature_scale_summary(source, source_manifest):
    names = source_manifest["variable_names"]
    input_path = source / "input_statistics.npz"
    with np.load(input_path, allow_pickle=False) as stats:
        input_scale = np.asarray(stats["scale"], dtype=np.float64)
    cache = np.load(source / "cache" / "train_features.npy", mmap_mode="r", allow_pickle=False)
    # Use a deterministic bounded sample from the already frozen training feature cache.
    sample_n = min(int(cache.shape[0]), 8192)
    sample_index = np.linspace(0, cache.shape[0] - 1, sample_n, dtype=np.int64)
    sample = np.asarray(cache[sample_index], dtype=np.float32)
    sampled_std = np.std(sample, axis=0, dtype=np.float64)
    slices = source_manifest.get("layout", {}).get("slices", {})
    view_stats = {}
    for view in ("S", "E", "P", "R"):
        start, end = slices[view]
        sub = sampled_std[:, start:end]
        view_stats[view] = {"feature_dimensions": int(end - start),
                            "sampled_low_std_variable_pairs_lt_0.05": int(np.sum(sub < 0.05)),
                            "sample_count": sample_n,
                            "min_feature_std": float(np.min(sub)),
                            "median_feature_std": float(np.median(sub)),
                            "max_feature_std": float(np.max(sub))}
    order = np.argsort(-input_scale, kind="stable")[:10]
    return {"input_scale_min": float(np.min(input_scale)), "input_scale_median": float(np.median(input_scale)),
            "input_scale_max": float(np.max(input_scale)),
            "largest_input_scales": [{"index": int(i), "variable": str(names[i]), "scale": float(input_scale[i])}
                                     for i in order],
            "feature_std_sampled": view_stats,
            "feature_summary_sampling": {"split": "train", "rows": sample_n, "seed_rule": "linspace endpoints"}}


def run_diagnostic(reuse_batch=None):
    plan_path = ROOT / BATCH / "plan.json"
    plan = _json(plan_path)
    if int(plan.get("head_seed", -1)) != SEED or set(plan.get("tasks", ())) != set(TASKS):
        raise ValueError("pinned Full diagnostic requires the original eight-task seed=2021 plan")
    if not AGGREGATION_INPUT.is_file():
        raise FileNotFoundError(AGGREGATION_INPUT)
    with AGGREGATION_INPUT.open(newline="", encoding="utf-8") as handle:
        aggregation_rows = list(csv.DictReader(handle))
    analysis_root, score_root, reuse_only = _diagnostic_output_paths(reuse_batch)
    analysis_root.mkdir(parents=True, exist_ok=False)
    if not reuse_only:
        score_root.mkdir(parents=True, exist_ok=False)
    rows, contribution_rows, scale_rows, failures = [], [], [], []
    baseline_metrics_path = COMMON_BATCH / "common_protocol_metrics.csv"
    baseline_by_task = {}
    with baseline_metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            baseline_by_task.setdefault(row["dataset"], {})[row["method"]] = row
    diag_manifest_path = BATCH / "branch-probe" / "results.csv"
    with diag_manifest_path.open(newline="", encoding="utf-8") as handle:
        prior_diag_rows = list(csv.DictReader(handle))
    prior_full_concat = {r["dataset"]: r for r in prior_diag_rows
                         if r.get("configuration") == "FULL_CONCAT" and r.get("result_type") == "diagonal_gmm_head"}
    agg_full_rows = {r["dataset"]: r for r in aggregation_rows if r.get("combination") == "FULL"}

    for task in TASKS:
        source_validated = False
        source = None
        source_manifest = None
        source_data = None
        source_aligned = None
        axes = None
        try:
            source_record = plan["source_records"][task]
            source = Path(source_record["path"]).resolve()
            expected_source = (SOURCE_ROOT / task / source.name).resolve()
            if source != expected_source:
                raise ValueError("plan source differs from fixed source root")
            source_manifest = _json(source / "probe_manifest.json")
            validate_pinned_source_binding(source_record, source_manifest)
            source_manifest["_path"] = str(source)
            if (source_manifest.get("artifact_policy") != "full"
                    or source_manifest.get("head_type") != "gmm"
                    or source_manifest.get("stage_status") != "complete"
                    or source_manifest.get("head_status") != "complete"
                    or source_manifest.get("config", {}).get("gmm_covariance_type") != "full"):
                raise ValueError("pinned source is not a completed full-covariance Full run")
            axes = _load_source_axes(source)
            full_eval, full_bundle = _evaluation(source, source, task, "FULL", SEED, checked_runs={})
            source_aligned = _aligned_bundle(source)
            source_data = _nll_artifacts(source)
            stored_aggregation = agg_full_rows.get(task)
            if not stored_aggregation or stored_aggregation.get("status") != "passed":
                raise ValueError("existing max/mean aggregation row is not a passed FULL row")
            if stored_aggregation.get("source_run_id") != source.name:
                raise ValueError("aggregation row points to a different Full source")
            if int(float(stored_aggregation.get("head_seed", -1))) != SEED:
                raise ValueError("aggregation row head seed differs")
            full_baseline, _ = _result_row(task, "FULL_FULL_MAX", "full", "max", source,
                                           source_data, source_aligned, source, source_manifest, axes)
            _check_equal(full_baseline["roc_auc"], full_eval["auc_roc"], task + " native Full ROC")
            _check_equal(full_baseline["average_precision"], full_eval["average_precision"], task + " native Full AP")
            _check_equal(full_baseline["threshold"], stored_aggregation["replayed_max_threshold"], task + " prior max replay")
            _check_equal(full_baseline["average_precision"], stored_aggregation["max_ap"], task + " prior max AP")

            # Keep the verified original Full rows even if the diagonal fit fails.
            for aggregation in ("max", "mean"):
                row, candidate = _result_row(task, "FULL_" + aggregation.upper(), "full", aggregation,
                                             source, source_data, source_aligned, source,
                                             source_manifest, axes, 0.0, "reused")
                _check_equal(row["average_precision"], stored_aggregation[aggregation + "_ap"],
                             task + " prior Full " + aggregation + " AP")
                _check_equal(row["roc_auc"], stored_aggregation[aggregation + "_auc_roc"],
                             task + " prior Full " + aggregation + " AUC")
                _check_equal(row["threshold"], stored_aggregation[aggregation + "_threshold"],
                             task + " prior Full " + aggregation + " threshold")
                rows.append(row)
                if aggregation == "max":
                    contribution_rows.extend(_contributors(source_data, candidate, source_manifest,
                                                           row["configuration"], aggregation))
            source_validated = True

            old_diag = prior_full_concat.get(task)
            old_diag_path = None
            if old_diag:
                if old_diag.get("status") not in {"complete", "reused"}:
                    raise ValueError("existing FULL_CONCAT diagonal head is incomplete")
                old_diag_path = Path(old_diag["run_path"]).resolve()
                old_data, old_aligned, _ = _verify_head(old_diag_path, task, source, source_manifest,
                                                         ("S", "E", "P", "R"), axes)
                if old_diag.get("source_run_id") != source.name:
                    raise ValueError("existing diagonal Full head references another source")

            task_output = score_root / task
            task_output.mkdir(parents=True, exist_ok=True)
            frame, train_length, labels, _, _, _ = _read_task(task, "large_detect")
            output_path = task_output
            diag_path = (output_path / "heads" / "FULL_CONCAT") if reuse_only else (
                old_diag_path or (output_path / "heads" / "FULL_CONCAT"))
            if old_diag_path is not None and not reuse_only:
                diag_data, diag_aligned = old_data, old_aligned
                diag_status, diag_elapsed, diag_reused_path = "reused", 0.0, old_diag_path
            elif diag_path.exists():
                diag_data, diag_aligned, _ = _verify_head(diag_path, task, source, source_manifest,
                                                           ("S", "E", "P", "R"), axes)
                diag_status, diag_elapsed, diag_reused_path = "reused", 0.0, diag_path
            else:
                if reuse_only:
                    raise FileNotFoundError("reuse-only diagnostic head is missing: {}".format(diag_path))
                config = _config_for(task, source, source_manifest, ("S", "E", "P", "R"), diag_path)
                started = time.perf_counter()
                probe = MNRFViewProbe(**config)
                probe.fit(frame, train_length, labels=labels)
                diag_elapsed = time.perf_counter() - started
                diag_data, diag_aligned, _ = _verify_head(diag_path, task, source, source_manifest,
                                                           ("S", "E", "P", "R"), axes)
                diag_reused_path = diag_path
                diag_status = "complete"
            # Existing three diagonal heads are reused; only missing heads fit.
            for aggregation in ("max", "mean"):
                row, candidate = _result_row(task, "DIAG_" + aggregation.upper(), "diag", aggregation,
                                             diag_reused_path, diag_data, diag_aligned, source,
                                             source_manifest, axes, diag_elapsed, diag_status)
                rows.append(row)
                if aggregation == "max":
                    contribution_rows.extend(_contributors(diag_data, candidate, source_manifest,
                                                           row["configuration"], aggregation))

            # Input scale and frozen feature scale diagnostics are read-only and bounded.
            scales = _feature_scale_summary(source, source_manifest)
            scale_rows.append({"dataset": task, "source_run_id": source.name,
                               "source_identity_sha256": source_manifest.get("source_identity_sha256"),
                               "feature_fingerprint": source_manifest.get("feature_fingerprint"),
                               "summary_json": json.dumps(scales, ensure_ascii=False, allow_nan=False)})
        except Exception as error:
            failures.append({"dataset": task, "status": "failed",
                             "reason": "{}: {}".format(type(error).__name__, error),
                             "traceback": traceback.format_exc()})
            if source_validated:
                diag_path = score_root / task / "heads" / "FULL_CONCAT"
                for aggregation in ("max", "mean"):
                    rows.append({
                        "dataset": task, "configuration": "DIAG_" + aggregation.upper(),
                        "covariance_type": "diag", "aggregation": aggregation,
                        "head_seed": SEED, "source_run_id": source.name,
                        "source_identity_sha256": source_manifest.get("source_identity_sha256"),
                        "feature_fingerprint": source_manifest.get("feature_fingerprint"),
                        "run_path": str(diag_path.resolve()), "status": "failed",
                        "selected_dim": 25, "variable_count": int(source_data["test_nll"].shape[1]),
                        "gmm_components": 4, "gmm_covariance_type": "diag",
                        "threshold": None, "roc_auc": None, "average_precision": None,
                        "precision": None, "recall": None, "point_f1": None, "normal_fpr": None,
                        "tp": None, "fp": None, "fn": None, "tn": None,
                        "affiliation_precision": None, "affiliation_recall": None, "affiliation_f": None,
                        "affiliation_scope": None, "anomaly_segments_hit": None,
                        "anomaly_segments_scored": None, "anomaly_segments_total": None,
                        "failure_reason": str(error),
                    })

    # Attach already verified common-protocol baseline references, and corrected
    # PCA residual results only where the old model retained all input dimensions.
    pca_rows = []
    from scripts.run_mnrf_common_baselines import (
        _existing_light_run, _fit_light_method, _read_contract, _full_axis_metrics,
        DATASET_FILES, _verify_frame_for_contract, SOURCE_RUNS,
    )
    for task in TASKS:
        try:
            task_common = COMMON_BATCH / task
            contract_path = task_common / "common_contract" / "common_contract.json"
            contract = _read_contract(contract_path)
            contract["common_contract_path"] = str(contract_path.resolve())
            source = SOURCE_ROOT / task / SOURCE_RUNS[task]
            old_output = ROOT / "result" / "score" / "MNRF-ViewProbe-Common" / "common-baseline-20261029T-newrun" / task / "PCA"
            old_score, old_manifest = _existing_light_run(old_output, "PCA", contract)
            old_model = __import__("joblib").load(old_output / "model.joblib")
            dimension = len(contract["variable_names"])
            selected = int(old_model["estimator"].n_components_)
            frame, values, labels, times = _verify_frame_for_contract(task, DATASET_FILES[task], contract)
            corrected_output = score_root / task / "PCA-residual"
            if selected < dimension:
                with np.load(old_score, allow_pickle=False) as old_payload:
                    old_scores = np.asarray(old_payload["scores"])
                    old_indices = np.asarray(old_payload["time_indices"])
                    old_threshold = float(old_payload["threshold"])
                source_manifest = _json(source / "probe_manifest.json")
                old_metrics = _full_axis_metrics(labels, old_indices, old_scores, old_threshold,
                    source_manifest.get("source_identity", {}).get("test_segments", []),
                    None if not contract.get("boundary_ids") else np.load(contract["boundary_ids"]["path"], allow_pickle=False))
                pca_rows.append({"dataset": task, "status": "reused", "run_path": str(old_output.resolve()),
                                 "selected_components": selected, "input_dimensions": dimension,
                                 "residual_dimensions": dimension - selected,
                                 "roc_auc": old_metrics["roc_auc"],
                                 "average_precision": old_metrics["average_precision"]})
            else:
                if reuse_only:
                    raise FileNotFoundError("reuse-only PCA residual result is missing: {}".format(corrected_output))
                score_path, evidence = _fit_light_method("PCA", corrected_output, values, labels, times, contract)
                with np.load(score_path, allow_pickle=False) as scored:
                    scores = np.asarray(scored["scores"])
                    indices = np.asarray(scored["time_indices"])
                    threshold = float(scored["threshold"])
                labels_full = labels
                boundary_record = contract.get("boundary_ids")
                boundaries = None if not boundary_record else np.load(boundary_record["path"], allow_pickle=False)
                source_manifest = _json(source / "probe_manifest.json")
                metrics = _full_axis_metrics(labels_full, indices, scores, threshold,
                                             source_manifest.get("source_identity", {}).get("test_segments", []), boundaries)
                params = evidence["parameters"]
                pca_rows.append({"dataset": task, "status": "recomputed", "run_path": str(corrected_output.resolve()),
                                 "selected_components": params["selected_components"],
                                 "input_dimensions": dimension, "residual_dimensions": params["residual_dimensions"],
                                 "roc_auc": metrics["roc_auc"], "average_precision": metrics["average_precision"],
                                 "old_full_dimension_result_retained": True})
        except Exception as error:
            failures.append({"dataset": task, "method": "PCA", "status": "failed",
                             "reason": "{}: {}".format(type(error).__name__, error),
                             "traceback": traceback.format_exc()})

    # Add common protocol baselines to each of the four configuration rows.
    for row in rows:
        methods = baseline_by_task.get(row["dataset"], {})
        for method in ("CATCH-common", "IsolationForest", "PCA"):
            baseline = methods.get(method, {})
            row[method.lower().replace("-", "_") + "_roc_auc"] = baseline.get("roc_auc")
            row[method.lower().replace("-", "_") + "_average_precision"] = baseline.get("average_precision")
        corrected = next((item for item in pca_rows if item["dataset"] == row["dataset"]), None)
        if corrected:
            row["pca_rule_status"] = corrected["status"]
            row["pca_rule_roc_auc"] = corrected.get("roc_auc")
            row["pca_rule_average_precision"] = corrected.get("average_precision")
            row["pca_selected_components"] = corrected.get("selected_components")

    _save_csv(analysis_root / "full_four_configurations.csv", rows)
    _save_csv(analysis_root / "max_variable_contributions.csv", contribution_rows)
    _save_csv(analysis_root / "scale_diagnostics.csv", scale_rows)
    _save_csv(analysis_root / "pca_residual_results.csv", pca_rows)
    (analysis_root / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    readme = ["# Full covariance / aggregation diagnosis", "",
              "Only the eight pinned Full representations are compared. No encoder training, encoder forward, Flow, CATCH run, or MNRF branch ablation was performed. All GMM heads use the fixed source fit endpoints, full-layout fixed noise and frozen feature caches; only covariance type differs for new diagonal heads.", "",
              "`full_four_configurations.csv` has full-covariance/max, full-covariance/mean, diagonal/max, diagonal/mean. Mean is over standardized per-variable NLL; max/mean thresholds are each calibrated from the same calibration endpoints at q=0.995. Predictions are strict score > threshold. Aff-F is the existing valid-continuous-segment macro diagnostic.", "",
              "## Four configurations", "",
              "| Dataset | Covariance | Aggregation | ROC-AUC | AP | P/R/F1 | FPR | Aff-F | Segments hit/scored/total | Status |", "|---|---|---|---:|---:|---|---:|---:|---:|---|"]
    def fmt(value, digits=6):
        return "n/a" if value is None else ("{:." + str(digits) + "f}").format(float(value))
    for row in rows:
        readme.append("| {} | {} | {} | {} | {} | {}/{}/{} | {} | {} | {}/{}/{} | {} |".format(
            row["dataset"], row["covariance_type"], row["aggregation"], fmt(row.get("roc_auc")),
            fmt(row.get("average_precision")), fmt(row.get("precision"), 4), fmt(row.get("recall"), 4),
            fmt(row.get("point_f1"), 4), fmt(row.get("normal_fpr")), fmt(row.get("affiliation_f")),
            row.get("anomaly_segments_hit", "n/a"), row.get("anomaly_segments_scored", "n/a"),
            row.get("anomaly_segments_total", "n/a"), row.get("status", "unknown")))
    by_config = {(row["dataset"], row["covariance_type"], row["aggregation"]): row for row in rows}
    readme += ["", "## Covariance 与 aggregation 效应", "",
               "差值定义：diag−full 固定聚合；mean−max 固定协方差；交互=(diag mean−diag max)−(full mean−full max)。正 AP/AUC 差代表该改动提升排序指标。", "",
               "| 数据集 | 协方差效应 AP max/mean | 协方差效应 ROC max/mean | 聚合效应 AP full/diag | 聚合效应 ROC full/diag | 交互 AP/ROC |", "|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        effects = _effect_deltas(by_config, task)
        readme.append("| {} | {} | {} | {} | {} | {}/{} |".format(
            task, effects["covariance_ap"], effects["covariance_auc"],
            effects["aggregation_ap"], effects["aggregation_auc"],
            effects["interaction_ap"], effects["interaction_auc"]))
    readme += ["", "## 共同协议排名参照", "",
               "这些基线来自已验证的共同协议批次；PCA 的 NYC/CalIt2 行采用本次统一残差维修正规则，其旧全维重建结果保留在旧批次。", "",
               "| 数据集 | Full full/max AP/ROC | Full diag/max AP/ROC | CATCH AP/ROC | IF AP/ROC | PCA residual AP/ROC |", "|---|---:|---:|---:|---:|---:|"]
    pca_lookup = {item["dataset"]: item for item in pca_rows}
    for task in TASKS:
        fm, dm = by_config.get((task, "full", "max")), by_config.get((task, "diag", "max"))
        base = baseline_by_task.get(task, {})
        pca = pca_lookup.get(task, {})
        def pair(auc, ap):
            return "n/a" if auc is None or ap is None else "{:.4f}/{:.4f}".format(float(ap), float(auc))
        readme.append("| {} | {} | {} | {} | {} | {} |".format(
            task, pair(None if fm is None else fm["roc_auc"], None if fm is None else fm["average_precision"]),
            pair(None if dm is None else dm["roc_auc"], None if dm is None else dm["average_precision"]),
            pair(base.get("CATCH-common", {}).get("roc_auc"), base.get("CATCH-common", {}).get("average_precision")),
            pair(base.get("IsolationForest", {}).get("roc_auc"), base.get("IsolationForest", {}).get("average_precision")),
            pair(pca.get("roc_auc"), pca.get("average_precision"))))
    readme += ["", "## Common-protocol references", "",
               "CATCH-common, IF and PCA values are read from the separately validated common-protocol batch. The PCA residual rule is `min(components_for_95_percent_variance, D-1)` on training rows; NYC and CalIt2 are recomputed because the old PCA retained all dimensions. Old PCA results remain in their original batch and are not overwritten.", "",
               "## Effect decomposition", "",
               "For each dataset, covariance effect is diagonal minus full at fixed aggregation; aggregation effect is mean minus max at fixed covariance. The interaction is (diag-mean minus diag-max) minus (full-mean minus full-max). These are post-hoc fixed-protocol contrasts, not a per-dataset selection of a new method.", "",
               "## Scale and maximum-score contribution", "",
               "Input scales and bounded sampled training-feature standard deviations are in `scale_diagnostics.csv`; TP/FP variable tie-split attribution is in `max_variable_contributions.csv`. They are localization evidence only, not a causal decomposition. Large MSL/CICIDS scale/NLL contributions are not clipped or removed.", "",
               "## Interpretation boundary", "",
               "An improvement from diagonal covariance supports covariance complexity/fitting as a possible source of loss; improvement from mean supports sensitivity to cross-variable max aggregation. If neither improves, this comparison does not explain the loss and is not proof that a representation branch is useless. Calibration pools for Genesis and especially CalIt2 are small; q=0.995 remains fixed.", "",
               "Failures, if any, are preserved in `failures.json`; no failed configuration is treated as complete.", ""]
    (analysis_root / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("analysis_dir={}".format(analysis_root.resolve()))
    print("score_dir={}".format(score_root.resolve()))
    print("completed_rows={}/32".format(sum(row.get("status") in {"complete", "reused"} for row in rows)))
    print("failed_items={}".format(len(failures)))
    return analysis_root, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-batch", help="validate and reuse heads from an existing diagnostic batch; never fit a missing head")
    args = parser.parse_args()
    out, errors = run_diagnostic(reuse_batch=args.reuse_batch)
    if errors:
        raise SystemExit(1)
