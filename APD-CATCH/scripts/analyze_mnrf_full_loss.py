#!/usr/bin/env python
"""Read-only score, calibration and variable-attribution diagnostics for pinned Full runs."""

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mnrf_branch_ablation import TASKS
from scripts.run_mnrf_branch_probe_diagnostic import (
    _aligned_bundle, _load_source_axes, _metrics, _nll_artifacts,
)
from scripts.run_mnrf_branch_ablation import _sha256_file

BATCH = ROOT / "result/analysis/MNRF-ViewProbe-Ablation/branch-ablation-20261024T210034Z-2771438"
DIAG = ROOT / "result/analysis/MNRF-ViewProbe-FullDiagnostic/full-head-diagnostic-20261030T062555Z-3226097"
DIAG_SCORES = ROOT / "result/score/MNRF-ViewProbe-FullDiagnostic/full-head-diagnostic-20261030T062555Z-3226097"
COMMON = ROOT / "result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun"
COMMON_SCORES = ROOT / "result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun"
OUT = DIAG
FPR_LIMITS = (0.001, 0.005, 0.01, 0.05)
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _verify_gmm_evidence(run, manifest, source_manifest=None):
    status = _json(run / "run_status.json")
    if status.get("status") != "complete":
        raise ValueError("{} run status is not complete".format(run))
    if manifest.get("head_type") != "gmm" or manifest.get("stage_status") != "complete" or manifest.get("head_status") != "complete":
        raise ValueError("{} is not a completed GMM result".format(run))
    for field, relative in (
        ("test_nll_sha256", "results/test_nll.npy"),
        ("aligned_scores_sha256", "results/aligned_scores.npz"),
        ("metrics_sha256", "results/metrics.json"),
        ("calibration_sha256", "heads/calibration.npz"),
        ("feature_statistics_sha256", "heads/feature_statistics.npz"),
        ("gmm_status_sha256", "heads/gmm_status.json"),
    ):
        artifact = run / relative
        if not artifact.is_file() or not manifest.get(field) or _sha256_file(artifact) != manifest[field]:
            raise ValueError("{} artifact binding failed: {}".format(run, field))
    if source_manifest is None:
        identity = manifest.get("source_identity")
        cache_hashes = manifest.get("cache_sha256")
        endpoint_hashes = manifest.get("endpoint_sha256")
    else:
        identity = manifest.get("source_identity")
        if (identity != source_manifest.get("source_identity")
                or manifest.get("source_identity_sha256") != source_manifest.get("source_identity_sha256")
                or manifest.get("source_feature_fingerprint") != source_manifest.get("feature_fingerprint")
                or manifest.get("source_cache_sha256") != source_manifest.get("cache_sha256")
                or manifest.get("source_endpoint_sha256") != source_manifest.get("endpoint_sha256")
                or manifest.get("fixed_noise_sha256") != source_manifest.get("fixed_noise_sha256")
                or manifest.get("source_gmm_fit_endpoints_sha256") != source_manifest.get("gmm_fit_endpoints_sha256")):
            raise ValueError("{} immutable Full source binding differs".format(run))
        cache_hashes = manifest.get("source_cache_sha256")
        endpoint_hashes = manifest.get("source_endpoint_sha256")
    if not identity or not cache_hashes or not endpoint_hashes:
        raise ValueError("{} is missing frozen source/cache/endpoint identities".format(run))
    run_noise = run / "heads/fixed_noise.npy"
    if not run_noise.is_file() or not manifest.get("fixed_noise_sha256"):
        raise ValueError("{} fixed noise binding is missing".format(run))
    if _sha256_file(run_noise) != manifest["fixed_noise_sha256"]:
        raise ValueError("{} fixed noise hash is invalid".format(run))
    if source_manifest is None:
        source_path = run
        expected_cache_hashes = manifest.get("cache_sha256", {})
        expected_endpoint_hashes = manifest.get("endpoint_sha256", {})
    else:
        source_path = Path(source_manifest["_path"])
        source_noise = source_path / "heads/fixed_noise.npy"
        if (not source_noise.is_file()
                or _sha256_file(source_noise) != source_manifest.get("fixed_noise_sha256")
                or _sha256_file(run_noise) != _sha256_file(source_noise)):
            raise ValueError("{} fixed noise differs from immutable source".format(run))
        expected_cache_hashes = source_manifest.get("cache_sha256", {})
        expected_endpoint_hashes = source_manifest.get("endpoint_sha256", {})
    for split in ("train", "validation", "calibration", "test"):
        cache = source_path / "cache" / (split + "_features.npy")
        endpoints = source_path / "cache" / (split + "_window_end_indices.npy")
        if (not cache.is_file() or _sha256_file(cache) != expected_cache_hashes.get(split)
                or not endpoints.is_file() or _sha256_file(endpoints) != expected_endpoint_hashes.get(split)):
            raise ValueError("{} frozen {} cache/endpoint binding failed".format(source_path, split))
    input_stats = source_path / "input_statistics.npz"
    if (not input_stats.is_file()
            or _sha256_file(input_stats) != (source_manifest or manifest).get("input_statistics_sha256")):
        raise ValueError("{} input statistics binding failed".format(source_path))
    return source_path


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        return {"n": int(len(values)), "undefined_reason": "empty_or_nonfinite"}
    quantiles = np.quantile(values, QUANTILES, method="linear")
    return {"n": int(len(values)), **{"q{:g}".format(q): float(v) for q, v in zip(QUANTILES, quantiles)}}


def _recall_at_fpr(labels, scores, limit):
    labels = np.asarray(labels, dtype=np.int8)
    if np.unique(labels).size != 2:
        return None
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    eligible = tpr[fpr <= limit + 1e-15]
    return None if not len(eligible) else float(np.max(eligible))


def _strict_axes(score_pack, full):
    scores = np.asarray(score_pack["scores"])
    indices = np.asarray(score_pack["time_indices"])
    labels = np.asarray(score_pack["actual"])
    times = np.asarray(score_pack["time_values_ns"])
    pred = np.asarray(score_pack["predictions"])
    if any(a.ndim != 1 for a in (scores, indices, labels, times, pred)):
        raise ValueError("baseline score arrays must be one-dimensional")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("baseline test coordinates must be integer typed")
    if (len(np.unique(indices)) != len(indices) or np.any(np.diff(indices) <= 0)
            or not np.array_equal(indices, full["global_indices"])):
        raise ValueError("baseline test global coordinates differ from pinned Full")
    if not np.array_equal(labels.astype(np.int8), full["labels"]):
        raise ValueError("baseline test labels differ from pinned Full")
    if not np.array_equal(times, full["time_values_ns"]):
        raise ValueError("baseline test timestamps differ from pinned Full")
    if not np.array_equal(pred.astype(np.int8), (scores > float(np.asarray(score_pack["threshold"])) ).astype(np.int8)):
        raise ValueError("baseline predictions do not match strict score > threshold")
    return scores.astype(np.float64), float(np.asarray(score_pack["threshold"])), pred.astype(np.int8)


def _load_baseline(run, method, contract, full_aligned):
    from scripts.run_mnrf_common_baselines import _existing_catch_run, _existing_light_run

    if method == "CATCH-common":
        task = Path(contract["data_csv_path"]).stem
        _existing_catch_run(run, task)
        manifest = _json(run / "results/common_manifest.json")
        if manifest.get("status") != "complete" or Path(manifest.get("task_name", "")).stem != task:
            raise ValueError("CATCH-common score manifest does not identify a complete task")
    else:
        _existing_light_run(run, method, contract)
    with np.load(run / "results/aligned_scores.npz", allow_pickle=False) as pack:
        test_scores, threshold, predictions = _strict_axes(pack, full_aligned)
        labels = np.asarray(pack["actual"], dtype=np.int8)
        times = np.asarray(pack["time_values_ns"])
        global_indices = np.asarray(pack["time_indices"])
    # _strict_axes has already required exact global coordinate equality, so
    # the Full's local coordinates/mask are the canonical alignment here.
    local = np.asarray(full_aligned["local_indices"])
    valid = np.asarray(full_aligned["valid_mask"])
    with np.load(run / "results/calibration_scores.npz", allow_pickle=False) as pack:
        calibration_scores = np.asarray(pack["scores"], dtype=np.float64)
        calibration_indices = np.asarray(pack["time_indices"])
        calibration_threshold = float(np.asarray(pack["threshold"]))
    if (calibration_indices.dtype.kind not in "iu" or calibration_indices.ndim != 1
            or np.any(np.diff(calibration_indices) <= 0)
            or not np.isfinite(calibration_scores).all()
            or not math.isclose(threshold, calibration_threshold, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("baseline calibration scores/threshold are invalid")
    from ts_benchmark.baselines.mnrf_view_probe.evaluation import calibrated_quantile
    if not math.isclose(threshold, calibrated_quantile(calibration_scores, 0.995),
                        rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("baseline calibration threshold does not replay at q=0.995")
    return {
        "scores": test_scores, "threshold": threshold, "predictions": predictions,
        "labels": labels, "local_indices": local, "valid_mask": valid,
        "time_values_ns": times, "global_indices": global_indices,
        "calibration_scores": calibration_scores,
        "calibration_indices": calibration_indices.astype(np.int64),
        "run_path": str(run), "status": "reused",
    }


def _full_axis_context(labels, global_indices, segments, endpoints, seq_len):
    labels = np.asarray(labels, dtype=np.int8)
    global_indices = np.asarray(global_indices, dtype=np.int64)
    lookup = {int(g): int(y) for g, y in zip(global_indices, labels)}
    output = []
    for raw_end in endpoints:
        end = int(raw_end)
        segment = next(((int(a), int(b)) for a, b in segments if int(a) <= end < int(b)), None)
        if segment is None:
            raise ValueError("calibration endpoint {} is outside frozen segments".format(end))
        start = max(segment[0], end - int(seq_len) + 1)
        points = range(start, end + 1)
        ys = [lookup[p] for p in points if p in lookup]
        if len(ys) != end - start + 1:
            raise ValueError("calibration context crosses missing source coordinates")
        output.append((int(lookup[end]), int(any(ys)), int(end - start + 1)))
    return np.asarray(output, dtype=np.int64)


def _score_row(task, method, scores, threshold, predictions, labels, aligned, axes,
               calibration_scores, calibration_endpoints, all_labels, all_global,
               source_segments, seq_len):
    from scripts.run_mnrf_full_head_diagnostic import _metrics

    metrics = _metrics(labels, scores, predictions, aligned, axes)
    normal, anomaly = scores[labels == 0], scores[labels == 1]
    sorted_cal = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    anomaly_ranks = (np.searchsorted(sorted_cal, anomaly, side="right") / max(len(sorted_cal), 1)
                     if len(anomaly) else np.asarray([]))
    tail_threshold = float(np.quantile(sorted_cal, 0.995, method="linear"))
    tail = np.asarray(calibration_scores, dtype=np.float64) >= tail_threshold
    contexts = _full_axis_context(all_labels, all_global, source_segments, calibration_endpoints, seq_len)
    cal_labels = contexts[:, 0]
    cal_context_anomaly = contexts[:, 1]
    positions = np.linspace(0, max(len(tail) - 1, 0), len(tail), dtype=np.int64) if len(tail) else np.asarray([], dtype=np.int64)
    quartile_id = np.minimum((positions * 4 // max(len(tail), 1)), 3) if len(tail) else positions
    tail_by_quartile = []
    for q in range(4):
        subset = tail & (quartile_id == q)
        tail_by_quartile.append({
            "quartile": q + 1, "tail_points": int(subset.sum()),
            "label_anomaly": int(cal_labels[subset].sum()),
            "context_contains_anomaly": int(cal_context_anomaly[subset].sum()),
            "all_quartile_points": int((quartile_id == q).sum()),
            "all_quartile_label_anomalies": int(cal_labels[quartile_id == q].sum()),
            "all_quartile_context_anomalies": int(cal_context_anomaly[quartile_id == q].sum()),
        })
    return {
        "dataset": task, "method": method, "status": "complete", "n_calibration": len(calibration_scores),
        "threshold": float(threshold), "roc_auc": metrics["auc_roc"], "average_precision": metrics["ap"],
        "precision": metrics["precision"], "recall": metrics["recall"], "point_f1": metrics["f1"],
        "normal_fpr": metrics["normal_fpr"], "tp": metrics["tp"], "fp": metrics["fp"],
        "fn": metrics["fn"], "tn": metrics["tn"],
        "recall_at_fpr_0_1pct": _recall_at_fpr(labels, scores, 0.001),
        "recall_at_fpr_0_5pct": _recall_at_fpr(labels, scores, 0.005),
        "recall_at_fpr_1pct": _recall_at_fpr(labels, scores, 0.01),
        "recall_at_fpr_5pct": _recall_at_fpr(labels, scores, 0.05),
        "affiliation_precision": metrics["affiliation_precision"],
        "affiliation_recall": metrics["affiliation_recall"], "affiliation_f": metrics["affiliation_f"],
        "segments_hit": metrics["anomaly_segments_hit"],
        "segments_scored": metrics["anomaly_segments_scored"], "segments_total": metrics["anomaly_segments_total"],
        "calibration_distribution": json.dumps(_distribution(calibration_scores), allow_nan=False),
        "test_normal_distribution": json.dumps(_distribution(normal), allow_nan=False),
        "test_anomaly_distribution": json.dumps(_distribution(anomaly), allow_nan=False),
        "test_anomaly_percentile_in_calibration": json.dumps(_distribution(anomaly_ranks), allow_nan=False),
        "calibration_q995_threshold": tail_threshold, "calibration_q995_tail_n": int(tail.sum()),
        "calibration_q995_label_anomalies": int(cal_labels[tail].sum()),
        "calibration_q995_context_anomalies": int(cal_context_anomaly[tail].sum()),
        "calibration_q995_context_normal": int((~cal_context_anomaly[tail].astype(bool)).sum()),
        "calibration_label_anomaly_points": int(cal_labels.sum()),
        "calibration_windows_with_anomaly_context": int(cal_context_anomaly.sum()),
        "calibration_context_normal_points": int((~cal_context_anomaly.astype(bool)).sum()),
        "calibration_q995_temporal_quartiles": json.dumps(tail_by_quartile, allow_nan=False),
        "diagnostic_threshold_not_used_for_selection": True,
    }


def _weighted_median(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(values) or weights.sum() <= 0:
        return None
    order = np.argsort(values, kind="stable")
    cutoff = weights.sum() / 2.0
    return float(values[order[np.searchsorted(np.cumsum(weights[order]), cutoff, side="left")]])


def _gmm_variable_rows(task, method, head, source, source_manifest, aligned, scores, pred):
    data = _nll_artifacts(head)
    z = (np.asarray(data["test_nll"], dtype=np.float32) - np.asarray(data["center"], dtype=np.float32)) / np.asarray(data["scale"], dtype=np.float32)
    maxima = np.max(z, axis=1, keepdims=True)
    tied = np.isclose(z, maxima, rtol=0, atol=1e-7)
    tie_weights = tied / np.maximum(tied.sum(axis=1, keepdims=True), 1)
    labels = np.asarray(aligned["labels"], dtype=np.int8)
    normal = labels == 0
    high_cut = float(np.quantile(np.asarray(scores)[normal], 0.99)) if normal.any() else None
    groups = {
        "TP": (labels == 1) & (pred == 1),
        "FP": normal & (pred == 1),
        "FN": (labels == 1) & (pred == 0),
        "HIGH_SCORE_NORMAL_Q99": normal & (np.asarray(scores) >= high_cut) if high_cut is not None else np.zeros_like(normal),
    }
    manifest = _json(source / "probe_manifest.json")
    names = manifest["variable_names"]
    stats = source / "input_statistics.npz"
    with np.load(stats, allow_pickle=False) as pack:
        input_scale = np.asarray(pack["scale"], dtype=np.float64)
        input_mean = np.asarray(pack["mean"], dtype=np.float64)
    normalized = np.load(source / "cache/normalized_values.npy", mmap_mode="r", allow_pickle=False)
    train_features = np.load(source / "cache/train_features.npy", mmap_mode="r", allow_pickle=False)
    sample_i = np.linspace(0, max(len(train_features) - 1, 0), min(len(train_features), 8192), dtype=np.int64)
    feature_std = np.std(np.asarray(train_features[sample_i], dtype=np.float32), axis=0, dtype=np.float64)
    feature_stats_path = head / "heads/feature_statistics.npz"
    with np.load(feature_stats_path, allow_pickle=False) as pack:
        nll_scale = np.asarray(pack["scale"], dtype=np.float32)
    with np.load(head / "heads/calibration.npz", allow_pickle=False) as pack:
        score_scale = np.asarray(pack["scale"], dtype=np.float32)
    config = manifest.get("config", {})
    input_floor = float(input_mean.dtype.type(config.get("input_std_floor", 1e-6)))
    feature_floor = float(config.get("feature_std_floor", 0.05))
    mad_floor = float(config.get("score_mad_floor", 0.001))
    all_ids = np.asarray(aligned["global_indices"], dtype=np.int64)
    rows = []
    segment_ranges = source_manifest.get("source_identity", {}).get("segments", [])
    starts = np.asarray([int(item[0]) for item in segment_ranges], dtype=np.int64)
    ends = np.asarray([int(item[1]) for item in segment_ranges], dtype=np.int64)
    for group, mask in groups.items():
        var_weights = tie_weights[mask].sum(axis=0, dtype=np.float64) if mask.any() else np.zeros(z.shape[1])
        endpoints = all_ids[mask]
        sample_endpoints = endpoints[np.linspace(0, len(endpoints) - 1, min(len(endpoints), 10000), dtype=np.int64)] if len(endpoints) else endpoints
        row_idx = np.asarray(sample_endpoints, dtype=np.int64)
        valid_rows = row_idx[(row_idx >= 0) & (row_idx < normalized.shape[0])]
        for var, name in enumerate(names):
            assigned = (mask & tied[:, var])
            assigned_endpoints = all_ids[assigned]
            assigned_endpoints = assigned_endpoints[np.linspace(0, len(assigned_endpoints) - 1,
                min(len(assigned_endpoints), 10000), dtype=np.int64)] if len(assigned_endpoints) else assigned_endpoints
            values_abs = np.abs(np.asarray(normalized[assigned_endpoints, var], dtype=np.float64)) if len(assigned_endpoints) else np.asarray([])
            previous = assigned_endpoints - 1
            segment_slot = np.searchsorted(starts, assigned_endpoints, side="right") - 1
            valid_segment = ((segment_slot >= 0) & (assigned_endpoints < normalized.shape[0]))
            if len(starts):
                valid_segment &= assigned_endpoints < ends[np.maximum(segment_slot, 0)]
                valid_segment &= previous >= starts[np.maximum(segment_slot, 0)]
            else:
                valid_segment[:] = False
            ok = valid_segment
            delta_abs = (np.abs(np.asarray(normalized[assigned_endpoints[ok], var], dtype=np.float64)
                                - np.asarray(normalized[previous[ok], var], dtype=np.float64)) if ok.any() else np.asarray([]))
            sampled_idx = np.linspace(0, max(normalized.shape[0] - 1, 0), min(normalized.shape[0], 10000), dtype=np.int64)
            sampled = np.asarray(normalized[sampled_idx, var], dtype=np.float32)
            unique_count = int(np.unique(sampled).size)
            rows.append({
                "dataset": task, "configuration": method, "population": group,
                "variable_index": var, "variable": str(name), "point_count": int(mask.sum()),
                "dominant_tie_split_weight": float(var_weights[var]),
                "dominant_share": None if not mask.any() else float(var_weights[var] / mask.sum()),
                "median_abs_normalized_input_at_dominance": None if not len(values_abs) else float(np.median(values_abs)),
                "median_abs_one_step_change_at_dominance": None if not len(delta_abs) else float(np.median(delta_abs)),
                "training_input_scale": float(input_scale[var]),
                "input_scale_floor_active": bool(input_scale[var] <= input_floor * 1.01),
                "sampled_unique_normalized_values": unique_count,
                "sampled_value_count": int(len(sampled)),
                "low_cardinality_sample_heuristic": bool(unique_count <= 16),
                "sampled_train_feature_dims_below_floor": int(np.sum(feature_std[var] < feature_floor)),
                "sampled_train_feature_dim_count": int(feature_std.shape[1]),
                "feature_scale_floor_active_dims": int(np.sum(nll_scale[var] <= feature_floor * 1.001)),
                "feature_std_floor": feature_floor,
                "score_mad_floor_active": bool(score_scale[var] <= mad_floor * 1.001),
                "score_mad_floor": mad_floor,
                "high_score_normal_cutoff": high_cut,
                "raw_event_sample_points": int(len(valid_rows)),
            })
    return rows


def _msl_initialization_diagnostic(source, head):
    from sklearn.mixture import GaussianMixture
    from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
    from sklearn.utils import check_random_state

    manifest = _json(source / "probe_manifest.json")
    failure_dir = head / "heads"
    progress = _json(failure_dir / "gmm_progress.json")
    complete = sorted(int(i) for i in progress.get("completed", {}))
    variable = max(complete) + 1 if complete else 0
    endpoints = np.load(failure_dir / "gmm_fit_endpoints.npy", mmap_mode="r", allow_pickle=False)
    train_ends = np.load(source / "cache/train_window_end_indices.npy", mmap_mode="r", allow_pickle=False)
    positions = np.searchsorted(train_ends, endpoints)
    if np.any(positions >= len(train_ends)) or not np.array_equal(train_ends[positions], endpoints):
        raise ValueError("MSL failed-head fit endpoints are not source train endpoints")
    features = np.load(source / "cache/train_features.npy", mmap_mode="r", allow_pickle=False)
    with np.load(failure_dir / "feature_statistics.npz", allow_pickle=False) as pack:
        mean = np.asarray(pack["mean"][variable], dtype=np.float32)
        scale = np.asarray(pack["scale"][variable], dtype=np.float32)
    x32 = (np.asarray(features[positions, variable, :], dtype=np.float32) - mean) / scale
    result = {
        "dataset": "MSL", "failed_variable_index": variable,
        "variable_name": manifest["variable_names"][variable], "sample_count": int(len(x32)),
        "feature_count": int(x32.shape[1]), "source_feature_dtype": str(features.dtype),
        "standardized_fit_input_dtype": str(x32.dtype), "fit_input_finite": bool(np.isfinite(x32).all()),
        "fit_input_min": float(np.min(x32)), "fit_input_max": float(np.max(x32)),
        "fit_input_abs_max": float(np.max(np.abs(x32))), "parameters": {
            "n_components": 4, "covariance_type": "diag", "reg_covar": 0.001,
            "n_init": 1, "max_iter": 100, "random_state": 2021, "init_params": "kmeans",
        },
        "prior_completed_variables": complete,
        "note": "Only sklearn initialization was replayed for this variable; no alternate fitted head is accepted as the original result.",
    }
    for dtype in (np.float32, np.float64):
        model = GaussianMixture(n_components=4, covariance_type="diag", reg_covar=0.001,
                                n_init=1, max_iter=100, random_state=2021)
        observations = np.asarray(x32, dtype=dtype)
        captured = {}
        original_initialize = model._initialize

        def capture_init(x, resp):
            weights, means, covariances = _estimate_gaussian_parameters(x, resp, model.reg_covar, model.covariance_type)
            captured["component_sample_mass"] = (resp.sum(axis=0)).tolist()
            captured["component_covariance_min"] = np.min(covariances, axis=1).tolist()
            captured["component_covariance_max"] = np.max(covariances, axis=1).tolist()
            captured["component_covariance_finite"] = bool(np.isfinite(covariances).all())
            return original_initialize(x, resp)

        model._initialize = capture_init
        try:
            model._initialize_parameters(observations, check_random_state(2021))
            state = "initialized"
            error = None
        except Exception as exc:
            state = "initialization_failed"
            error = "{}: {}".format(type(exc).__name__, exc)
        result["dtype_" + np.dtype(dtype).name] = {
            "state": state, "error": error,
            "min_component_samples": None if not captured else float(min(captured["component_sample_mass"])),
            "min_initial_diagonal_covariance": None if not captured else float(min(captured["component_covariance_min"])),
            "max_initial_diagonal_covariance": None if not captured else float(max(captured["component_covariance_max"])),
            "components": captured,
        }
    return result


def run():
    plan = _json(BATCH / "plan.json")
    table = {row["dataset"]: row for row in _csv(DIAG / "full_four_configurations.csv")}
    diag_rows = _csv(DIAG / "full_four_configurations.csv")
    diag_lookup = {(r["dataset"], r["configuration"]): r for r in diag_rows}
    common_rows = {(r["dataset"], r["method"]): r for r in _csv(COMMON / "common_protocol_metrics.csv")}
    main_rows, attribution_rows, tail_rows = [], [], []
    summary = []
    msl_diag = None
    for task in TASKS:
        source_record = plan["source_records"][task]
        source = Path(source_record["path"])
        manifest = _json(source / "probe_manifest.json")
        if (Path(source_record["path"]).resolve() != source.resolve()
                or manifest.get("source_identity_sha256") != source_record.get("source_identity_sha256")
                or manifest.get("feature_fingerprint") != source_record.get("feature_fingerprint")):
            raise ValueError("{} pinned source identity changed".format(task))
        manifest["_path"] = str(source)
        _verify_gmm_evidence(source, manifest)
        axes = _load_source_axes(source)
        full = _aligned_bundle(source)
        full_data = _nll_artifacts(source)
        if len(full_data["test_nll"]) != len(full["scores"]):
            raise ValueError("{} Full NLL/test axes mismatch".format(task))
        calibration_endpoints = np.load(source / "cache/calibration_window_end_indices.npy", mmap_mode="r", allow_pickle=False)
        test_endpoints = np.load(source / "cache/test_window_end_indices.npy", mmap_mode="r", allow_pickle=False)
        if not np.array_equal(test_endpoints, full["global_indices"]):
            raise ValueError("{} pinned Full test endpoint axis mismatch".format(task))
        seq_len = int(manifest["config"]["seq_len"])
        all_labels, all_times, all_global, all_segments, _ = axes
        source_segments = manifest.get("source_identity", {}).get("segments", [])
        head_specs = []
        for aggregation in ("max", "mean"):
            nll = np.asarray(full_data["test_nll"], dtype=np.float32)
            cal_nll = np.asarray(full_data["calibration_nll"], dtype=np.float32)
            center = np.asarray(full_data["center"], dtype=np.float32)
            scale = np.asarray(full_data["scale"], dtype=np.float32)
            score_cal = (cal_nll - center) / scale
            score_test = (nll - center) / scale
            reducer = np.max if aggregation == "max" else np.mean
            cal_scores = reducer(score_cal, axis=1)
            scores = reducer(score_test, axis=1)
            from ts_benchmark.baselines.mnrf_view_probe.evaluation import calibrated_quantile
            threshold = calibrated_quantile(cal_scores, 0.995)
            if aggregation == "max":
                if (not np.allclose(scores, full["scores"], rtol=1e-6, atol=1e-7)
                        or not np.array_equal((scores > threshold).astype(np.int8), full["predictions"])):
                    raise ValueError("{} original Full max score/threshold replay failed".format(task))
                if not math.isclose(threshold, full["threshold"], rel_tol=1e-6, abs_tol=1e-7):
                    raise ValueError("{} original Full max threshold replay failed".format(task))
            candidate = dict(full)
            candidate.update(scores=scores, threshold=threshold, predictions=(scores > threshold).astype(np.int8))
            row = _score_row(task, "Full-{}-{}".format(manifest["config"]["gmm_covariance_type"], aggregation),
                scores, threshold, candidate["predictions"], full["labels"], candidate, axes,
                cal_scores, calibration_endpoints, all_labels, all_global, source_segments, seq_len)
            row.update({"source_run_id": source.name, "score_origin": "saved per-variable NLL",
                        "head_fit_repeated": False, "formal_threshold_rule": "calibration q=0.995"})
            main_rows.append(row)
            head_specs.append(("Full-full-" + aggregation, source, full_data, candidate,
                               scores, candidate["predictions"], cal_scores, calibration_endpoints))
            if aggregation == "max":
                attribution_rows.extend(_gmm_variable_rows(task, "Full-full-max", source, source,
                    manifest, candidate, scores, candidate["predictions"]))

        diag_record = diag_lookup[(task, "DIAG_MAX")]
        if diag_record["status"] in ("complete", "reused"):
            diag_head = Path(diag_record["run_path"])
            diag_manifest = _json(diag_head / "probe_manifest.json")
            _verify_gmm_evidence(diag_head, diag_manifest, manifest)
            diag_data = _nll_artifacts(diag_head)
            diag_aligned = _aligned_bundle(diag_head)
            for key in ("global_indices", "local_indices", "time_values_ns"):
                if not np.array_equal(diag_aligned[key], full[key]):
                    raise ValueError("{} diagonal {} axis differs from Full".format(task, key))
            if not np.array_equal(diag_aligned["valid_mask"], full["valid_mask"]) or not np.array_equal(diag_aligned["labels"], full["labels"]):
                raise ValueError("{} diagonal mask/labels differ from Full".format(task))
            for aggregation in ("max", "mean"):
                cal_z = (np.asarray(diag_data["calibration_nll"], dtype=np.float32)
                         - np.asarray(diag_data["center"], dtype=np.float32)) / np.asarray(diag_data["scale"], dtype=np.float32)
                test_z = (np.asarray(diag_data["test_nll"], dtype=np.float32)
                          - np.asarray(diag_data["center"], dtype=np.float32)) / np.asarray(diag_data["scale"], dtype=np.float32)
                reducer = np.max if aggregation == "max" else np.mean
                cal_scores = reducer(cal_z, axis=1)
                scores = reducer(test_z, axis=1)
                threshold = calibrated_quantile(cal_scores, 0.995)
                if aggregation == "max" and (not np.allclose(scores, diag_aligned["scores"], rtol=1e-6, atol=1e-7)
                        or not np.array_equal((scores > threshold).astype(np.int8), diag_aligned["predictions"])):
                    raise ValueError("{} diagonal max replay failed".format(task))
                candidate = dict(diag_aligned)
                candidate.update(scores=scores, threshold=threshold, predictions=(scores > threshold).astype(np.int8))
                row = _score_row(task, "Full-diag-{}".format(aggregation), scores, threshold,
                    candidate["predictions"], full["labels"], candidate, axes, cal_scores,
                    calibration_endpoints, all_labels, all_global, source_segments, seq_len)
                row.update({"source_run_id": source.name, "score_origin": "saved diagonal-head NLL",
                            "head_fit_repeated": False, "formal_threshold_rule": "calibration q=0.995"})
                main_rows.append(row)
                head_specs.append(("Full-diag-" + aggregation, diag_head, diag_data, candidate,
                                   scores, candidate["predictions"], cal_scores, calibration_endpoints))
                if aggregation == "max":
                    attribution_rows.extend(_gmm_variable_rows(task, "Full-diag-max", diag_head,
                        source, manifest, candidate, scores, candidate["predictions"]))
        else:
            for aggregation in ("max", "mean"):
                main_rows.append({"dataset": task, "method": "Full-diag-{}".format(aggregation),
                                  "status": "failed", "source_run_id": source.name,
                                  "failure_reason": diag_record.get("failure_reason", "diagonal head not complete"),
                                  "score_origin": "not available", "head_fit_repeated": False})
            if task == "MSL":
                msl_diag = _msl_initialization_diagnostic(source, Path(diag_record["run_path"]))

        baseline_contract = _json(COMMON / task / "common_contract/common_contract.json")
        baseline_contract["common_contract_path"] = str((COMMON / task / "common_contract/common_contract.json").resolve())
        for method in ("CATCH-common", "IsolationForest", "PCA"):
            base_row = common_rows[(task, method)]
            if (base_row.get("source_full_run_id") != source.name
                    or int(float(base_row.get("seed", -1))) != 2021):
                raise ValueError("{} {} reference row is not bound to pinned Full/seed".format(task, method))
            base_run = Path(base_row.get("run_path") or (COMMON_SCORES / task / method))
            if method == "PCA" and task in ("NYC", "CalIt2"):
                base_run = DIAG_SCORES / task / "PCA-residual"
            baseline = _load_baseline(base_run, method, baseline_contract, full)
            if not np.array_equal(baseline["calibration_indices"], calibration_endpoints):
                raise ValueError("{} {} calibration coordinates differ from Full".format(task, method))
            score_pack = {"scores": baseline["scores"], "time_indices": baseline["global_indices"],
                          "actual": baseline["labels"], "time_values_ns": baseline["time_values_ns"],
                          "predictions": baseline["predictions"], "threshold": baseline["threshold"]}
            aligned = dict(full)
            aligned.update(scores=baseline["scores"], threshold=baseline["threshold"],
                           predictions=baseline["predictions"])
            row = _score_row(task, method, baseline["scores"], baseline["threshold"],
                baseline["predictions"], full["labels"], aligned, axes,
                baseline["calibration_scores"], calibration_endpoints, all_labels, all_global,
                source_segments, seq_len)
            row.update({"source_run_id": source.name, "score_origin": "saved baseline scores",
                        "head_fit_repeated": False, "formal_threshold_rule": "saved independent calibration"})
            main_rows.append(row)
            if method == "CATCH-common":
                scores_dir = baseline["scores"]
        summary.append({"dataset": task, "source_run_id": source.name,
                        "test_points": len(full["labels"]), "test_anomaly_points": int(full["labels"].sum()),
                        "calibration_points": len(calibration_endpoints),
                        "diag_status": diag_record["status"]})
    _write_csv(OUT / "loss_decomposition.csv", main_rows)
    _write_csv(OUT / "variable_error_attribution.csv", attribution_rows)
    _write_csv(OUT / "calibration_tail_context.csv", [
        {k: row.get(k) for k in ("dataset", "method", "n_calibration", "threshold", "calibration_q995_threshold",
         "calibration_q995_tail_n", "calibration_q995_label_anomalies", "calibration_q995_context_anomalies",
         "calibration_q995_context_normal", "calibration_q995_temporal_quartiles", "status")}
        for row in main_rows])
    (OUT / "msl_diag_initialization.json").write_text(
        json.dumps(msl_diag, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "loss_decomposition_sources.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return main_rows, attribution_rows, msl_diag


if __name__ == "__main__":
    run()
