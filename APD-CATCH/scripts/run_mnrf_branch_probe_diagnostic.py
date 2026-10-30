#!/usr/bin/env python
"""CPU-only pure-view heads and fixed equal-branch fusion diagnostics.

This diagnostic is intentionally separate from the formal full-covariance
nine-combination ablation matrix. It consumes one pinned Full source per task,
fits only diagonal-covariance GMM heads over its frozen caches, then derives
the fixed branch-axis fusion rows from the four pure-head NLL artifacts.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mnrf_branch_ablation import ROOT as PROJECT_ROOT
from scripts.run_mnrf_branch_ablation import _source_record
from scripts.run_mnrf_view_probe import _read_task
from scripts.summarize_mnrf_branch_ablation import (
    _affiliation_for_bundle, _evaluation, segment_counts_from_axis,
)
from scripts.validate_mnrf_view_probe import validate
from ts_benchmark.baselines.mnrf_view_probe import MNRFViewProbe


BATCH_REL = Path("result/analysis/MNRF-ViewProbe-Ablation/branch-ablation-20261024T210034Z-2771438")
DIAGNOSTIC_REL = BATCH_REL / "branch-probe"
TASKS = ("PSM", "GECCO", "Creditcard")
PURE = ("S", "E", "P", "R")
FUSIONS = {
    "EQ_SE": ("S", "E"), "EQ_SP": ("S", "P"), "EQ_SR": ("S", "R"),
    "EQ_FULL": PURE, "EQ_WITHOUT_S": ("E", "P", "R"),
    "EQ_WITHOUT_E": ("S", "P", "R"), "EQ_WITHOUT_P": ("S", "E", "R"),
    "EQ_WITHOUT_R": ("S", "E", "P"),
}
HEADS = ("S", "E", "P", "R", "FULL_CONCAT")
BASELINES = (
    "Modern", "iTrans", "DualTF", "ATrans", "DC", "TsNet", "Patch", "DLin", "NLin",
    "AE", "OCSVM", "IF", "PCA", "HBOS", "TFAD",
)

# CATCH ICLR 2025 Table 2, reported at the paper's three-decimal precision.
# Values are references under the paper's protocol, not local reruns.
PAPER = {
    "PSM": {
        "AUC-ROC": [0.593, 0.592, 0.600, 0.514, 0.501, 0.592, 0.586, 0.580, 0.585, 0.650, 0.619, 0.542, 0.648, 0.620, 0.500],
        "Aff-F": [0.825, 0.854, 0.725, 0.710, 0.682, 0.842, 0.831, 0.831, 0.843, 0.707, 0.531, 0.620, 0.702, 0.658, 0.628],
    },
    "GECCO": {
        "AUC-ROC": [0.952, 0.795, 0.714, 0.516, 0.555, 0.954, 0.949, 0.947, 0.936, 0.769, 0.804, 0.619, 0.711, 0.557, 0.499],
        "Aff-F": [0.893, 0.839, 0.701, 0.782, 0.687, 0.894, 0.906, 0.893, 0.882, 0.823, 0.666, 0.424, 0.785, 0.708, 0.627],
    },
    "Creditcard": {
        "AUC-ROC": [0.957, 0.934, 0.703, 0.552, 0.504, 0.957, 0.957, 0.954, 0.948, 0.909, 0.953, 0.860, 0.871, 0.951, 0.500],
        "Aff-F": [0.744, 0.713, 0.663, 0.650, 0.632, 0.744, 0.746, 0.738, 0.742, 0.561, 0.714, 0.634, 0.710, 0.695, 0.600],
    },
}


def validate_coordinates(indices, name):
    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in "iu" or values.dtype.kind == "b":
        raise ValueError("{} must be a one-dimensional integer coordinate array".format(name))
    if len(values) and (np.any(np.diff(values.astype(np.int64, copy=False)) <= 0)):
        raise ValueError("{} must be strictly increasing and unique".format(name))
    return values.astype(np.int64, copy=False)


def fuse_branch_nll(nll_by_view, statistics_by_view, views):
    """Normalize per variable per branch, average branch axis, then max variables."""
    if not views or any(view not in nll_by_view or view not in statistics_by_view for view in views):
        raise ValueError("fusion views must have NLL and validation statistics")
    normalized = []
    expected_shape = None
    for view in views:
        nll = np.asarray(nll_by_view[view])
        center, scale = (np.asarray(item) for item in statistics_by_view[view])
        if nll.ndim != 2 or center.shape != (nll.shape[1],) or scale.shape != center.shape:
            raise ValueError("{} NLL and center/scale dimensions differ".format(view))
        if not np.isfinite(nll).all() or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("{} NLL/statistics must be finite with positive scales".format(view))
        if expected_shape is None:
            expected_shape = nll.shape
        elif nll.shape != expected_shape:
            raise ValueError("branch NLL arrays must have exactly matching rows and variables")
        # Keep source artifacts' dtype arithmetic (normally float32).
        normalized.append((nll - center) / scale)
    branch_axis = np.stack(normalized, axis=-1)
    variable_scores = np.mean(branch_axis, axis=-1)
    return np.max(variable_scores, axis=1), variable_scores


def calibrate_fusion(calibration_nll_by_view, test_nll_by_view, stats_by_view, views, quantile=0.995):
    calibration, _ = fuse_branch_nll(calibration_nll_by_view, stats_by_view, views)
    test, _ = fuse_branch_nll(test_nll_by_view, stats_by_view, views)
    if not 0 < float(quantile) < 1:
        raise ValueError("calibration quantile must be inside (0,1)")
    threshold = float(np.quantile(calibration, float(quantile), method="linear"))
    return test, threshold, (test > threshold).astype(np.int8)


def same_axis(reference, candidate, name):
    left = validate_coordinates(reference, "reference {}".format(name))
    right = validate_coordinates(candidate, "candidate {}".format(name))
    if not np.array_equal(left, right):
        raise ValueError("{} differs from pinned Full source coordinates".format(name))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _npz(path):
    return np.load(path, allow_pickle=False)


def validate_pinned_source_binding(source_record, manifest):
    """Require all available source-record identity fields to match the file."""
    fields = (("source_identity", "source_identity"),
              ("source_identity_sha256", "source_identity_sha256"),
              ("feature_fingerprint", "feature_fingerprint"),
              ("cache_sha256", "cache_sha256"),
              ("endpoint_sha256", "endpoint_sha256"),
              ("gmm_fit_endpoints_sha256", "gmm_fit_endpoints_sha256"),
              ("fixed_noise_sha256", "fixed_noise_sha256"),
              ("input_statistics_sha256", "input_statistics_sha256"),
              ("normalized_cache_sha256", "normalized_cache_sha256"))
    for record_key, manifest_key in fields:
        expected = source_record.get(record_key)
        if expected is not None and expected != manifest.get(manifest_key):
            raise ValueError("pinned source {} differs from its manifest".format(record_key))
    return True


def _load_source_axes(source):
    with _npz(source / "analysis_inputs.npz") as pack:
        all_labels = np.asarray(pack["all_labels"], dtype=np.int8)
        all_times = np.asarray(pack["all_time_values_ns"], dtype=np.int64)
        all_global = validate_coordinates(pack["all_global_indices"], "full global axis")
        all_segments = np.asarray(pack["all_segment_ids"], dtype=np.int64)
        test_global = validate_coordinates(pack["test_global_indices"], "test global axis")
    if not (all_labels.shape == all_times.shape == all_global.shape == all_segments.shape):
        raise ValueError("pinned Full complete-axis analysis inputs have mismatched lengths")
    if not np.isin(all_labels, (0, 1)).all() or len(np.unique(all_global)) != len(all_global):
        raise ValueError("pinned Full labels/global axis are invalid")
    return all_labels, all_times, all_global, all_segments, test_global


def _aligned_bundle(run):
    with _npz(run / "results/aligned_scores.npz") as pack:
        score = np.asarray(pack["scores"])
        pred = np.asarray(pack["predictions"])
        actual = np.asarray(pack["actual"])
        local = np.asarray(pack["local_indices"])
        valid = np.asarray(pack["valid_mask"])
        global_indices = np.asarray(pack["time_indices"])
        times = np.asarray(pack["time_values_ns"])
        threshold = float(np.asarray(pack["threshold"]).reshape(()))
    global_indices = validate_coordinates(global_indices, "scored global coordinates")
    local = validate_coordinates(local, "scored local coordinates")
    if not (score.ndim == pred.ndim == actual.ndim == 1 and len(score) == len(pred) == len(actual) == len(local) == len(global_indices) == len(times)):
        raise ValueError("aligned scoring arrays have incompatible shape")
    if (valid.dtype.kind != "b" or valid.ndim != 1
            or not np.array_equal(np.flatnonzero(valid), local)):
        raise ValueError("aligned valid mask is malformed")
    if not np.isin(actual, (0, 1)).all() or not np.isin(pred, (0, 1)).all() or not np.isfinite(score).all():
        raise ValueError("aligned scores, predictions, or labels are invalid")
    if not np.array_equal(pred.astype(np.int8), (score > threshold).astype(np.int8)):
        raise ValueError("saved predictions do not match strict score > threshold")
    aligned_actual = np.full(len(valid), -1, dtype=np.int8)
    aligned_actual[local] = actual.astype(np.int8)
    return {"scores": score, "predictions": pred.astype(np.int8), "labels": actual.astype(np.int8),
            "local_indices": local, "valid_mask": valid, "global_indices": global_indices,
            "time_values_ns": times, "threshold": threshold, "aligned_actual": aligned_actual}


def _metrics(labels, scores, pred, run, source_axes):
    from scripts.summarize_mnrf_branch_ablation import _binary_metrics
    from scripts.summarize_mnrf_branch_ablation import _affiliation_for_bundle

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores)
    pred = np.asarray(pred, dtype=np.int8)
    metrics = _binary_metrics(labels, scores, pred.astype(bool))
    all_labels, all_times, all_global, all_segments, test_global = source_axes
    if len(all_labels) != len(all_global) or len(test_global) == 0:
        raise ValueError("source full axis lacks a test interval")
    # The score bundle is dense only over valid endpoints; lift it onto the
    # full test axis so cold-start prefixes and coordinate gaps remain intact.
    local_axis = np.asarray(test_global, dtype=np.int64)
    axis_labels = np.empty(len(local_axis), dtype=np.int8)
    label_map = {int(idx): int(label) for idx, label in zip(all_global, all_labels)}
    segment_map = {int(idx): int(seg) for idx, seg in zip(all_global, all_segments)}
    time_map = {int(idx): int(value) for idx, value in zip(all_global, all_times)}
    axis_labels[:] = [label_map[int(idx)] for idx in local_axis]
    axis_segments = np.asarray([segment_map[int(idx)] for idx in local_axis], dtype=np.int64)
    axis_times = np.asarray([time_map[int(idx)] for idx in local_axis], dtype=np.int64)
    scored_global = np.asarray(run["global_indices"], dtype=np.int64)
    scored_local = np.searchsorted(local_axis, scored_global)
    if (np.any(scored_local >= len(local_axis))
            or not np.array_equal(local_axis[scored_local], scored_global)):
        raise ValueError("scored global coordinate is outside source test axis")
    mask = np.zeros(len(local_axis), dtype=bool)
    mask[scored_local] = True
    if not np.array_equal(axis_labels[scored_local], labels):
        raise ValueError("candidate endpoint labels differ from pinned Full labels")
    bundle = {"valid_mask": mask, "local_indices": scored_local,
              "predictions": pred}
    aff = _affiliation_for_bundle(bundle, axis_labels, local_axis, axis_segments)
    segment = segment_counts_from_axis(
        all_labels, all_global, all_segments, test_global,
        scored_global, pred,
        all_times=all_times,
    )
    return {**metrics, "affiliation_precision": aff.get("affiliation_precision"),
            "affiliation_recall": aff.get("affiliation_recall"),
            "affiliation_f": aff.get("affiliation_f"),
            "affiliation_scope": aff.get("calculation_scope"),
            "affiliation_undefined_reason": ";".join(aff.get("undefined_reasons", [])) or None,
            "anomaly_segments_hit": segment["hit"], "anomaly_segments_scored": segment["scored"],
            "anomaly_segments_total": segment["total"]}


def _nll_artifacts(run):
    manifest = _json(run / "probe_manifest.json")
    stats_path = run / "heads/feature_statistics.npz"
    calibration_path = run / "heads/calibration.npz"
    with _npz(stats_path) as pack:
        mean, scale = np.asarray(pack["mean"]), np.asarray(pack["scale"])
    with _npz(calibration_path) as pack:
        center, score_scale = np.asarray(pack["center"]), np.asarray(pack["scale"])
        threshold = float(np.asarray(pack["threshold"]).reshape(()))
        quantile = float(np.asarray(pack["quantile"]).reshape(()))
    # Feature statistics are the source of per-variable NLL calibration center
    # and scale in this artifact schema.
    if mean.ndim != 2 or scale.shape != mean.shape:
        raise ValueError("feature statistics shape is invalid")
    validation_nll = np.load(run / "heads/validation_nll.npy", mmap_mode="r")
    calibration_nll = np.load(run / "heads/calibration_nll.npy", mmap_mode="r")
    test_nll = np.load(run / "results/test_nll.npy", mmap_mode="r")
    if center.shape != (test_nll.shape[1],) or score_scale.shape != center.shape:
        raise ValueError("calibration center/scale do not match NLL variables")
    if not np.isfinite(center).all() or not np.isfinite(score_scale).all() or np.any(score_scale <= 0):
        raise ValueError("calibration NLL statistics are invalid")
    if not (validation_nll.ndim == calibration_nll.ndim == test_nll.ndim == 2):
        raise ValueError("saved split NLL arrays must be two-dimensional")
    return {"manifest": manifest, "center": center, "scale": score_scale,
            "threshold": threshold, "quantile": quantile,
            "validation_nll": validation_nll, "calibration_nll": calibration_nll,
            "test_nll": test_nll}


def _verify_head(run, task, source, source_manifest, views, source_axes,
                 expected_compute_dtype="float32"):
    checked = validate(run)
    manifest = _json(run / "probe_manifest.json")
    config = manifest.get("config", {})
    source_dir = Path(manifest.get("source_dir", "")).resolve()
    if source_dir != source.resolve():
        raise ValueError("head does not reference the pinned Full source")
    if (manifest.get("source_identity") != source_manifest.get("source_identity")
            or manifest.get("source_identity_sha256") != source_manifest.get("source_identity_sha256")
            or manifest.get("source_feature_fingerprint") != source_manifest.get("feature_fingerprint")
            or manifest.get("source_cache_sha256") != source_manifest.get("cache_sha256")
            or manifest.get("source_endpoint_sha256") != source_manifest.get("endpoint_sha256")
            or manifest.get("source_gmm_fit_endpoints_sha256") != source_manifest.get("gmm_fit_endpoints_sha256")
            or manifest.get("raw_input_statistics_sha256") != source_manifest.get("input_statistics_sha256")):
        raise ValueError("head immutable source/cache/endpoint/noise binding differs")
    source_config = source_manifest.get("config", {})
    expected_numeric = {
        "seed": 2021, "gmm_components": 4, "gmm_covariance_type": "diag",
        "gmm_compute_dtype": expected_compute_dtype,
        "gmm_reg_covar": 0.001, "gmm_n_init": 1, "gmm_max_iter": 100,
        "gmm_fit_max_windows": 20000, "train_noise_std": source_config.get("train_noise_std"),
        "noise_seed": source_config.get("noise_seed"),
        "feature_std_floor": source_config.get("feature_std_floor"),
        "score_mad_floor": source_config.get("score_mad_floor"),
        "calibration_quantile": source_config.get("calibration_quantile"),
        "seq_len": source_config.get("seq_len"), "pattern_len": source_config.get("pattern_len"),
        "hidden_dim": source_config.get("hidden_dim"), "view_dim": source_config.get("view_dim"),
        "evolution_dim": source_config.get("evolution_dim"),
        "pattern_dim": source_config.get("pattern_dim"),
        "relation_dim": source_config.get("relation_dim"),
        "task_name": source_config.get("task_name"), "data_source": source_config.get("data_source"),
    }
    if (tuple(manifest.get("selected_views", ())) != tuple(views)
            or list(manifest.get("selected_indices", [])) != _indices(views)
            or any(config.get(key, "float32" if key == "gmm_compute_dtype" else None) != value
                   for key, value in expected_numeric.items())):
        raise ValueError("head configuration does not match frozen diagnostic protocol")
    data = _nll_artifacts(run)
    source_sample = source / "heads/gmm_fit_endpoints.npy"
    head_sample = run / "heads/gmm_fit_endpoints.npy"
    source_noise = source / "heads/fixed_noise.npy"
    head_noise = run / "heads/fixed_noise.npy"
    for path, declared, label in (
            (source_sample, source_manifest.get("gmm_fit_endpoints_sha256"), "source fit endpoints"),
            (head_sample, manifest.get("gmm_fit_endpoints_sha256"), "head fit endpoints"),
            (source_noise, source_manifest.get("fixed_noise_sha256"), "source fixed noise"),
            (head_noise, manifest.get("fixed_noise_sha256"), "head fixed noise")):
        if not path.is_file() or not declared:
            raise ValueError("{} binding is missing".format(label))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != declared:
            raise ValueError("{} digest mismatch".format(label))
    if not np.array_equal(np.load(source_sample, mmap_mode="r"), np.load(head_sample, mmap_mode="r")):
        raise ValueError("head GMM fit endpoints differ from pinned source sample")
    if not np.array_equal(np.load(source_noise, mmap_mode="r"), np.load(head_noise, mmap_mode="r")):
        raise ValueError("head fixed noise differs from pinned source noise")
    aligned = _aligned_bundle(run)
    source_aligned = _aligned_bundle(source)
    for key in ("global_indices", "local_indices", "time_values_ns"):
        same_axis(source_aligned[key], aligned[key], key)
    if not np.array_equal(source_aligned["valid_mask"], aligned["valid_mask"]):
        raise ValueError("head valid scoring mask differs from pinned Full")
    if not np.array_equal(source_aligned["labels"], aligned["labels"]):
        raise ValueError("head labels differ from pinned Full")
    nll = data["test_nll"]
    source_endpoints = {
        split: np.load(source / "cache" / (split + "_window_end_indices.npy"), mmap_mode="r")
        for split in ("train", "validation", "calibration", "test")
    }
    expected_rows = {"validation_nll": len(source_endpoints["validation"]),
                     "calibration_nll": len(source_endpoints["calibration"]),
                     "test_nll": len(source_endpoints["test"])}
    for name, count in expected_rows.items():
        if len(data[name]) != count:
            raise ValueError("{} row count differs from pinned source endpoints".format(name))
    if nll.shape[0] != len(aligned["scores"]) or nll.shape[1] != len(source_manifest["variable_names"]):
        raise ValueError("head test NLL does not align to source scored coordinates/variables")
    score = np.max((nll - data["center"]) / data["scale"], axis=1)
    pred = (score > data["threshold"]).astype(np.int8)
    if not np.allclose(score, aligned["scores"], rtol=1e-6, atol=1e-7):
        raise ValueError("head scores do not replay from saved per-variable NLL and calibration")
    if not np.array_equal(pred, aligned["predictions"]):
        raise ValueError("head predictions do not replay from saved threshold")
    if checked.get("status") != "complete":
        raise ValueError("native validator did not report a complete head")
    return data, aligned, checked


def _indices(views):
    widths = {"S": (0, 1), "E": (1, 9), "P": (9, 17), "R": (17, 25)}
    return [index for view in views for index in range(*widths[view])]


def _config_for(task, source, source_manifest, views, output, gmm_compute_dtype="float32"):
    cfg = dict(source_manifest["config"])
    for key in ("resolved_layout", "selected_indices", "selected_dim", "resume_from", "output_dir"):
        cfg.pop(key, None)
    cfg.update({"mode": "select_gmm", "selected_views": tuple(views),
                "source_dir": str(source), "output_dir": str(output), "device": "cpu",
                "seed": 2021, "gmm_components": 4, "gmm_covariance_type": "diag",
                "gmm_compute_dtype": gmm_compute_dtype,
                "gmm_reg_covar": 0.001, "gmm_n_init": 1, "gmm_max_iter": 100,
                "gmm_fit_max_windows": 20000, "cpu_threads": 1,
                "artifact_policy": "full", "task_name": task + ".csv"})
    return cfg


def _load_or_fit_head(task, source, source_manifest, views, frame, train_length, labels,
                      source_axes, run_dir):
    output = run_dir / "heads" / ("".join(views) if views != ("S", "E", "P", "R") else "FULL_CONCAT")
    if output.exists():
        data, aligned, checked = _verify_head(output, task, source, source_manifest, views, source_axes)
        return output, data, aligned, checked, "reused", 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    config = _config_for(task, source, source_manifest, views, output)
    started = time.perf_counter()
    probe = MNRFViewProbe(**config)
    probe.fit(frame, train_length, labels=labels)
    elapsed = time.perf_counter() - started
    data, aligned, checked = _verify_head(output, task, source, source_manifest, views, source_axes)
    return output, data, aligned, checked, "complete", elapsed


def _row(task, name, kind, views, run_id, run, data, aligned, source_axes, source_manifest,
         status, elapsed, error=None):
    if error:
        return {"dataset": task, "configuration": name, "result_type": kind,
                "views": ",".join(views), "source_run_id": Path(source_manifest["_path"]).name,
                "run_id": run_id, "status": "failed", "failure_reason": error,
                "elapsed_seconds_this_run": elapsed}
    metrics = _metrics(aligned["labels"], aligned["scores"], aligned["predictions"], aligned, source_axes)
    manifest = data["manifest"]
    gmm_status_path = run / "heads/gmm_status.json" if run else None
    gmm_status = _json(gmm_status_path) if gmm_status_path and gmm_status_path.is_file() else {}
    return {"dataset": task, "configuration": name, "result_type": kind,
            "views": ",".join(views), "source_run_id": Path(source_manifest["_path"]).name,
            "source_identity": source_manifest.get("source_identity_sha256"),
            "feature_fingerprint": source_manifest.get("feature_fingerprint"),
            "run_id": run_id, "run_path": str(run) if run else "derived_from_pure_head_nll",
            "status": status, "selected_dim": sum(len(range(*{"S": (0,1), "E": (1,9), "P": (9,17), "R": (17,25)}[v])) for v in views),
            "variable_count": len(source_manifest.get("variable_names", [])),
            "gmm_components": manifest.get("config", {}).get("gmm_components"),
            "gmm_covariance_type": manifest.get("config", {}).get("gmm_covariance_type"),
            "head_parameter_count": gmm_status.get("head_parameter_count"),
            "converged": all(item.get("converged") for item in gmm_status.get("variables", [])) if gmm_status.get("variables") else None,
            "threshold": float(aligned["threshold"]), "auc_roc": metrics["auc_roc"],
            "average_precision": metrics["ap"], "affiliation_precision": metrics["affiliation_precision"],
            "affiliation_recall": metrics["affiliation_recall"], "affiliation_f": metrics["affiliation_f"],
            "affiliation_scope": metrics["affiliation_scope"],
            "affiliation_undefined_reason": metrics["affiliation_undefined_reason"],
            "precision": metrics["precision"], "recall": metrics["recall"], "point_f1": metrics["f1"],
            "normal_fpr": metrics["normal_fpr"], "tp": metrics["tp"], "fp": metrics["fp"],
            "fn": metrics["fn"], "tn": metrics["tn"],
            "anomaly_segments_hit": metrics["anomaly_segments_hit"],
            "anomaly_segments_scored": metrics["anomaly_segments_scored"],
            "anomaly_segments_total": metrics["anomaly_segments_total"],
            "fit_time_seconds_historical": _json(run / "results/metrics.json").get("fit_time_seconds") if run else None,
            "inference_time_seconds_historical": _json(run / "results/metrics.json").get("inference_time_seconds") if run else None,
            "elapsed_seconds_this_run": elapsed,
            "fit_endpoint_hash": manifest.get("gmm_fit_endpoints_sha256"),
            "failure_reason": None}


def _fusion_row(task, name, views, pure, source, source_manifest, source_axes):
    test_nll = {v: pure[v]["data"]["test_nll"] for v in views}
    cal_nll = {v: pure[v]["data"]["calibration_nll"] for v in views}
    stats = {v: (pure[v]["data"]["center"], pure[v]["data"]["scale"]) for v in views}
    q = float(pure[views[0]]["data"]["quantile"])
    score, threshold, pred = calibrate_fusion(cal_nll, test_nll, stats, views, q)
    reference = pure[views[0]]["aligned"]
    for view in views[1:]:
        for key in ("global_indices", "local_indices", "time_values_ns"):
            same_axis(reference[key], pure[view]["aligned"][key], key)
        if not np.array_equal(reference["valid_mask"], pure[view]["aligned"]["valid_mask"]):
            raise ValueError("pure branch masks differ before fusion")
    aligned = dict(reference)
    aligned.update({"scores": score, "threshold": threshold, "predictions": pred})
    metrics = _metrics(aligned["labels"], score, pred, aligned, source_axes)
    return {"dataset": task, "configuration": name, "result_type": "equal_branch_fusion",
            "views": ",".join(views), "source_run_id": source.name,
            "source_identity": source_manifest.get("source_identity_sha256"),
            "feature_fingerprint": source_manifest.get("feature_fingerprint"),
            "run_id": "derived-" + task + "-" + name,
            "run_path": "derived from validated pure-head per-variable NLL",
            "status": "complete", "selected_dim": None,
            "variable_count": len(source_manifest.get("variable_names", [])),
            "gmm_components": None, "gmm_covariance_type": None,
            "head_parameter_count": None, "converged": None, "threshold": threshold,
            "auc_roc": metrics["auc_roc"], "average_precision": metrics["ap"],
            "affiliation_precision": metrics["affiliation_precision"],
            "affiliation_recall": metrics["affiliation_recall"],
            "affiliation_f": metrics["affiliation_f"], "affiliation_scope": metrics["affiliation_scope"],
            "affiliation_undefined_reason": metrics["affiliation_undefined_reason"],
            "precision": metrics["precision"], "recall": metrics["recall"], "point_f1": metrics["f1"],
            "normal_fpr": metrics["normal_fpr"], "tp": metrics["tp"], "fp": metrics["fp"],
            "fn": metrics["fn"], "tn": metrics["tn"],
            "anomaly_segments_hit": metrics["anomaly_segments_hit"],
            "anomaly_segments_scored": metrics["anomaly_segments_scored"],
            "anomaly_segments_total": metrics["anomaly_segments_total"],
            "fit_time_seconds_historical": None, "inference_time_seconds_historical": None,
            "elapsed_seconds_this_run": 0.0, "fit_endpoint_hash": pure[views[0]]["data"]["manifest"].get("gmm_fit_endpoints_sha256"),
            "failure_reason": None}


def _prediction_bundle_for(task, row, root):
    """Load saved pure/head points or reconstruct a fixed fusion from pure NLL."""
    run_path = row.get("run_path")
    if row.get("result_type") == "diagonal_gmm_head":
        run = Path(run_path)
        data = _nll_artifacts(run)
        aligned = _aligned_bundle(run)
        return aligned, data
    views = tuple(row.get("views", "").split(","))
    pure = {}
    for view in views:
        run = root / task / "heads" / view
        pure[view] = {"data": _nll_artifacts(run), "aligned": _aligned_bundle(run)}
    score, threshold, prediction = calibrate_fusion(
        {v: pure[v]["data"]["calibration_nll"] for v in views},
        {v: pure[v]["data"]["test_nll"] for v in views},
        {v: (pure[v]["data"]["center"], pure[v]["data"]["scale"]) for v in views},
        views, pure[views[0]]["data"]["quantile"])
    aligned = dict(pure[views[0]]["aligned"])
    aligned.update({"scores": score, "threshold": threshold, "predictions": prediction})
    return aligned, None


def _paired_point_and_segment_changes(task, base, changed, root):
    if (not np.array_equal(base["global_indices"], changed["global_indices"])
            or not np.array_equal(base["local_indices"], changed["local_indices"])
            or not np.array_equal(base["time_values_ns"], changed["time_values_ns"])
            or not np.array_equal(base["valid_mask"], changed["valid_mask"])
            or not np.array_equal(base["labels"], changed["labels"])):
        raise ValueError("paired result axes/labels differ")
    labels = np.asarray(base["labels"], dtype=np.int8)
    left, right = np.asarray(base["predictions"], dtype=bool), np.asarray(changed["predictions"], dtype=bool)
    added_tp = int(np.sum((labels == 1) & ~left & right))
    lost_tp = int(np.sum((labels == 1) & left & ~right))
    added_fp = int(np.sum((labels == 0) & ~left & right))
    removed_fp = int(np.sum((labels == 0) & left & ~right))
    plan = _json(PROJECT_ROOT / BATCH_REL / "plan.json")
    source = Path(plan["source_records"][task]["path"])
    all_labels, all_times, all_global, all_segments, test_global = _load_source_axes(source)
    base_seg = segment_counts_from_axis(all_labels, all_global, all_segments, test_global,
                                        base["global_indices"], base["predictions"], all_times)
    changed_seg = segment_counts_from_axis(all_labels, all_global, all_segments, test_global,
                                          changed["global_indices"], changed["predictions"], all_times)
    def key(item):
        return (source.name, int(item["segment_id"]), int(item["start_global"]), int(item["end_global"]))
    b_hits = {key(item) for item in base_seg["details"] if item["hit"]}
    c_hits = {key(item) for item in changed_seg["details"] if item["hit"]}
    return {"added_tp": added_tp, "lost_tp": lost_tp, "added_fp": added_fp,
            "removed_fp": removed_fp, "added_segment_hits": len(c_hits - b_hits),
            "lost_segment_hits": len(b_hits - c_hits),
            "total_segment_hits_base": len(b_hits), "total_segment_hits_changed": len(c_hits),
            "segment_hit_keys_added": sorted(c_hits - b_hits),
            "segment_hit_keys_lost": sorted(b_hits - c_hits)}


def _format_metric(value):
    return "n/a" if value is None else "{:.4f}".format(float(value))


def metric_delta(changed, base):
    """Subtract serialized or numeric metric values without silent null coercion."""
    if changed in (None, "") or base in (None, ""):
        return None
    return float(changed) - float(base)


def _diagnostic_comparisons(rows, root):
    by_key = {(row.get("dataset"), row.get("configuration")): row for row in rows
              if row.get("status") in {"complete", "reused", "historical_reference"}}
    report = []
    paired_specs = [("EQ_SE", "S", "SE vs S"), ("EQ_SP", "S", "SP vs S"),
                    ("EQ_SR", "S", "SR vs S"), ("EQ_FULL", "S", "4-way equal fusion vs S"),
                    ("EQ_WITHOUT_S", "EQ_FULL", "remove S from equal fusion"),
                    ("EQ_WITHOUT_E", "EQ_FULL", "remove E from equal fusion"),
                    ("EQ_WITHOUT_P", "EQ_FULL", "remove P from equal fusion"),
                    ("EQ_WITHOUT_R", "EQ_FULL", "remove R from equal fusion"),
                    ("FULL_CONCAT", "S", "diag concatenated Full vs diag S")]
    for task in TASKS:
        report.extend(["", "### {}: paired changes".format(task), "",
                       "| Comparison | AP delta | ROC delta | TP +/− | FP +/− | segment hits +/− |",
                       "|---|---:|---:|---:|---:|---:|"])
        task_rows = [row for row in rows if row.get("dataset") == task]
        bundles = {}
        for row in task_rows:
            if row.get("status") not in {"complete", "reused", "historical_reference"}:
                continue
            if row.get("result_type") == "historical_reference":
                continue
            try:
                bundles[row["configuration"]] = _prediction_bundle_for(task, row, root)[0]
            except Exception:
                continue
        for changed_name, base_name, title in paired_specs:
            base_row = by_key.get((task, base_name))
            changed_row = by_key.get((task, changed_name))
            if not base_row or not changed_row or base_name not in bundles or changed_name not in bundles:
                report.append("| {} | unavailable | unavailable | | | |".format(title))
                continue
            try:
                changes = _paired_point_and_segment_changes(task, bundles[base_name], bundles[changed_name], root)
                ap_delta = metric_delta(changed_row.get("average_precision"), base_row.get("average_precision"))
                auc_delta = metric_delta(changed_row.get("auc_roc"), base_row.get("auc_roc"))
                report.append("| {} | {:+.4f} | {:+.4f} | +{}/−{} | +{}/−{} | +{}/−{} |".format(
                    title, ap_delta, auc_delta, changes["added_tp"], changes["lost_tp"],
                    changes["added_fp"], changes["removed_fp"], changes["added_segment_hits"],
                    changes["lost_segment_hits"]))
            except Exception as error:
                report.append("| {} | invalid pair: {} | | | | |".format(title, error))
        report.extend(["", "Pure-branch detections unique relative to S (same frozen test points):"])
        s_bundle = bundles.get("S")
        for view in ("E", "P", "R"):
            branch = bundles.get(view)
            if s_bundle is None or branch is None:
                report.append("- {}: unavailable".format(view))
                continue
            changes = _paired_point_and_segment_changes(task, s_bundle, branch, root)
            branch_row = by_key.get((task, view), {})
            report.append("- {}: +{} / −{} TP, +{} / −{} FP, +{} / −{} anomaly-segment hits; AP {}, Aff-F {}".format(
                view, changes["added_tp"], changes["lost_tp"], changes["added_fp"], changes["removed_fp"],
                changes["added_segment_hits"], changes["lost_segment_hits"],
                _format_metric(branch_row.get("average_precision")),
                _format_metric(branch_row.get("affiliation_f"))))

    report.extend(["", "## Literature-reference comparison", "",
                   "For each new complete row, the following summarizes its position against the 15 CATCH Table 2 non-CATCH references at the paper's three-decimal precision. Counts are `above / tied / below`; ties are rounded-value ties, not evidence of protocol equivalence. Selected time-series baselines are shown explicitly.", "",
                   "| Dataset | Configuration | AUC vs 15 (above/tied/below) | Modern | iTrans | TsNet | Patch | DLin | Aff-F vs 15 (above/tied/below) | Modern | iTrans | TsNet | Patch | DLin |",
                   "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        if row.get("status") not in {"complete", "reused"} or row.get("result_type") == "historical_reference":
            continue
        task, config = row["dataset"], row["configuration"]
        values = []
        for metric in ("auc_roc", "affiliation_f"):
            own = row.get(metric)
            baseline_values = PAPER[task]["AUC-ROC" if metric == "auc_roc" else "Aff-F"]
            if own is None:
                values.extend(["n/a", "n/a", "n/a"] + ["n/a"] * 5)
                continue
            rounded = round(float(own), 3)
            above = sum(rounded > value for value in baseline_values)
            tied = sum(rounded == value for value in baseline_values)
            below = len(baseline_values) - above - tied
            selected = [baseline_values[BASELINES.index(name)] for name in ("Modern", "iTrans", "TsNet", "Patch", "DLin")]
            values.extend(["{}/{}/{}".format(above, tied, below)] + ["{:.3f}".format(x) for x in selected])
        report.append("| {} | {} | {} |".format(task, config, " | ".join(values)))
    return report


def run_diagnostic(plan_path=None):
    plan_path = Path(plan_path) if plan_path else PROJECT_ROOT / BATCH_REL / "plan.json"
    plan_path = plan_path.resolve()
    plan = _json(plan_path)
    if int(plan.get("head_seed", -1)) != 2021:
        raise ValueError("diagnostic requires the original plan's head_seed=2021")
    batch_dir = plan_path.parent
    diag_root = batch_dir / "branch-probe"
    diag_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in TASKS:
        source_record = plan.get("source_records", {}).get(task)
        if not source_record:
            raise ValueError("pinned plan has no source record for {}".format(task))
        source = Path(source_record["path"]).resolve()
        expected_path = Path(source_record["path"]).resolve()
        if source != expected_path:
            raise ValueError("source path changed from pinned plan")
        source_checked = validate(source)
        if source_checked.get("status") != "complete":
            raise ValueError("pinned source is not complete: {}".format(task))
        source_manifest = _json(source / "probe_manifest.json")
        # Cross-check actual frozen identities against the original plan record.
        validate_pinned_source_binding(source_record, source_manifest)
        source_manifest["_path"] = str(source)
        source_axes = _load_source_axes(source)
        frame, train_length, labels, _, _, _ = _read_task(task, "large_detect")
        task_dir = diag_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        pure = {}
        views_for_head = {"S": ("S",), "E": ("E",), "P": ("P",), "R": ("R",),
                          "FULL_CONCAT": PURE}
        for name in HEADS:
            views = views_for_head[name]
            started = time.perf_counter()
            try:
                run, data, aligned, checked, status, elapsed = _load_or_fit_head(
                    task, source, source_manifest, views, frame, train_length, labels,
                    source_axes, task_dir)
                row = _row(task, name, "diagonal_gmm_head", views, run.name, run,
                           data, aligned, source_axes, source_manifest, status, elapsed)
                rows.append(row)
                if name in PURE:
                    pure[name] = {"run": run, "data": data, "aligned": aligned}
            except Exception as error:
                rows.append({"dataset": task, "configuration": name,
                             "result_type": "diagonal_gmm_head", "views": ",".join(views),
                             "source_run_id": source.name, "run_id": None, "status": "failed",
                             "failure_reason": "{}: {}".format(type(error).__name__, error),
                             "elapsed_seconds_this_run": time.perf_counter() - started})
        if len(pure) == 4:
            for name, views in FUSIONS.items():
                started = time.perf_counter()
                try:
                    rows.append(_fusion_row(task, name, views, pure, source, source_manifest, source_axes))
                except Exception as error:
                    rows.append({"dataset": task, "configuration": name,
                                 "result_type": "equal_branch_fusion", "views": ",".join(views),
                                 "source_run_id": source.name, "run_id": "derived-" + task + "-" + name,
                                 "status": "failed", "failure_reason": "{}: {}".format(type(error).__name__, error),
                                 "elapsed_seconds_this_run": time.perf_counter() - started})
        else:
            for name, views in FUSIONS.items():
                rows.append({"dataset": task, "configuration": name,
                             "result_type": "equal_branch_fusion", "views": ",".join(views),
                             "source_run_id": source.name, "run_id": "derived-" + task + "-" + name,
                             "status": "blocked", "failure_reason": "all four pure branch heads must validate",
                             "elapsed_seconds_this_run": 0.0})
        # Native strict comparison-path evaluation is reserved for the pinned
        # historical full-covariance S and Full references.
        for combo in ("S", "FULL"):
            plan_row = next((r for r in plan["rows"] if r.get("dataset") == task and r.get("combination") == combo and r.get("head_seed") == 2021), None)
            if not plan_row:
                raise ValueError("pinned plan is missing historical {} reference for {}".format(combo, task))
            run = Path(plan_row["candidate_or_output_path"]).resolve()
            try:
                result, bundle = _evaluation(run, source, task, combo, 2021, checked_runs={})
                rows.append({"dataset": task, "configuration": "HISTORICAL_{}_FULL_COV".format(combo),
                             "result_type": "historical_reference", "views": result.get("selected_views"),
                             "source_run_id": source.name, "source_identity": result.get("source_identity"),
                             "feature_fingerprint": result.get("feature_fingerprint"),
                             "run_id": run.name, "run_path": str(run), "status": "historical_reference",
                             "selected_dim": result.get("selected_dim"), "variable_count": len(source_manifest["variable_names"]),
                             "gmm_components": result.get("gmm_components"), "gmm_covariance_type": result.get("gmm_covariance_type"),
                             "head_parameter_count": result.get("head_parameter_count"), "converged": True,
                             "threshold": float(bundle["threshold"]), "auc_roc": result.get("auc_roc"),
                             "average_precision": result.get("average_precision"),
                             "affiliation_precision": result.get("affiliation_precision"),
                             "affiliation_recall": result.get("affiliation_recall"),
                             "affiliation_f": result.get("affiliation_f"),
                             "affiliation_scope": result.get("affiliation_scope"),
                             "affiliation_undefined_reason": result.get("affiliation_undefined_reason"),
                             "precision": result.get("precision"), "recall": result.get("recall"),
                             "point_f1": result.get("point_f1"), "normal_fpr": result.get("normal_fpr"),
                             "tp": result.get("tp"), "fp": result.get("fp"), "fn": result.get("fn"), "tn": result.get("tn"),
                             "anomaly_segments_hit": result.get("anomaly_segments_hit"),
                             "anomaly_segments_scored": result.get("anomaly_segments_scored"),
                             "anomaly_segments_total": result.get("anomaly_segments_total"),
                             "fit_time_seconds_historical": result.get("fit_time_seconds"),
                             "inference_time_seconds_historical": result.get("inference_time_seconds"),
                             "elapsed_seconds_this_run": 0.0,
                             "fit_endpoint_hash": result.get("fit_endpoint_hash"), "failure_reason": None})
            except Exception as error:
                rows.append({"dataset": task, "configuration": "HISTORICAL_{}_FULL_COV".format(combo),
                             "result_type": "historical_reference", "views": combo,
                             "source_run_id": source.name, "run_id": str(run), "status": "failed",
                             "failure_reason": "{}: {}".format(type(error).__name__, error),
                             "elapsed_seconds_this_run": 0.0})
    fieldnames = sorted({key for row in rows for key in row})
    csv_path = diag_root / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    readme_path = diag_root / "README.md"
    _write_readme(readme_path, rows, plan_path)
    complete_heads = sum(r.get("result_type") == "diagonal_gmm_head" and r.get("status") in {"complete", "reused"} for r in rows)
    complete_fusions = sum(r.get("result_type") == "equal_branch_fusion" and r.get("status") == "complete" for r in rows)
    failed = sum(r.get("status") == "failed" for r in rows)
    print(json.dumps({"diagnostic_dir": str(diag_root), "results_csv": str(csv_path),
                      "head_configurations_complete_or_reused": complete_heads,
                      "equal_branch_fusions_complete": complete_fusions, "failed": failed,
                      "new_scoring_rows_expected": 39, "historical_references_expected": 6}, sort_keys=True))
    return 1 if failed else 0


def refresh_historical_references(plan_path=None):
    """Revalidate and refresh only the six pinned full-covariance references."""
    plan_path = Path(plan_path) if plan_path else PROJECT_ROOT / BATCH_REL / "plan.json"
    plan_path = plan_path.resolve()
    plan = _json(plan_path)
    csv_path = plan_path.parent / "branch-probe/results.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    affinity_path = plan_path.parent / "affiliation_results.csv"
    affinity_rows = {}
    if affinity_path.is_file():
        with affinity_path.open("r", encoding="utf-8", newline="") as handle:
            for item in csv.DictReader(handle):
                if item.get("metric_status") == "passed":
                    affinity_rows[(item.get("dataset"), item.get("combination"), item.get("source_run_id"))] = item
    checked_runs = {}
    failures = 0
    for task in TASKS:
        source = Path(plan["source_records"][task]["path"]).resolve()
        source_manifest = _json(source / "probe_manifest.json")
        for combo in ("S", "FULL"):
            row = next(r for r in rows if r.get("dataset") == task
                       and r.get("configuration") == "HISTORICAL_{}_FULL_COV".format(combo))
            plan_row = next(r for r in plan["rows"] if r.get("dataset") == task
                            and r.get("combination") == combo and r.get("head_seed") == 2021)
            run = Path(plan_row["candidate_or_output_path"]).resolve()
            try:
                result, bundle = _evaluation(run, source, task, combo, 2021, checked_runs=checked_runs)
                row.update({
                    "views": ",".join(result.get("selected_views", [])), "source_run_id": source.name,
                    "source_identity": result.get("source_identity"),
                    "feature_fingerprint": result.get("feature_fingerprint"), "run_id": run.name,
                    "run_path": str(run), "status": "historical_reference",
                    "selected_dim": result.get("selected_dim"),
                    "variable_count": len(source_manifest.get("variable_names", [])),
                    "gmm_components": result.get("gmm_components"),
                    "gmm_covariance_type": result.get("gmm_covariance_type"),
                    "head_parameter_count": result.get("head_parameter_count"), "converged": True,
                    "threshold": float(bundle["threshold"]), "auc_roc": result.get("auc_roc"),
                    "average_precision": result.get("average_precision"),
                    "affiliation_precision": result.get("affiliation_precision"),
                    "affiliation_recall": result.get("affiliation_recall"),
                    "affiliation_f": result.get("affiliation_f"),
                    "affiliation_scope": result.get("affiliation_scope"),
                    "affiliation_undefined_reason": result.get("affiliation_undefined_reason"),
                    "precision": result.get("precision"), "recall": result.get("recall"),
                    "point_f1": result.get("point_f1"), "normal_fpr": result.get("normal_fpr"),
                    "tp": result.get("tp"), "fp": result.get("fp"), "fn": result.get("fn"),
                    "tn": result.get("tn"), "anomaly_segments_hit": result.get("anomaly_segments_hit"),
                    "anomaly_segments_scored": result.get("anomaly_segments_scored"),
                    "anomaly_segments_total": result.get("anomaly_segments_total"),
                    "fit_time_seconds_historical": result.get("fit_time_seconds"),
                    "inference_time_seconds_historical": result.get("inference_time_seconds"),
                    "elapsed_seconds_this_run": 0.0, "fit_endpoint_hash": result.get("fit_endpoint_hash"),
                    "failure_reason": ""})
                # The historical batch's Aff-F supplement is tied to this
                # same pinned source/candidate and preserves its validated
                # effective-interval calculation; native _evaluation does
                # not include the supplementary affiliation fields.
                affinity = affinity_rows.get((task, combo, source.name))
                if affinity is not None:
                    row.update({"affiliation_precision": affinity.get("affiliation_precision"),
                                "affiliation_recall": affinity.get("affiliation_recall"),
                                "affiliation_f": affinity.get("affiliation_f"),
                                "affiliation_scope": affinity.get("calculation_scope"),
                                "affiliation_undefined_reason": affinity.get("undefined_or_failure_reason")})
            except Exception as error:
                failures += 1
                row.update({"status": "failed", "failure_reason": "{}: {}".format(type(error).__name__, error)})
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_readme(plan_path.parent / "branch-probe/README.md", rows, plan_path)
    print(json.dumps({"historical_references_validated": 6 - failures, "failed": failures,
                      "results_csv": str(csv_path)}, sort_keys=True))
    return 1 if failures else 0


def _write_readme(path, rows, plan_path):
    diagnostic_root = path.parent
    lines = ["# Frozen-branch validity and simple-fusion diagnostic", "",
             "Pinned input plan: `{}`. CPU only; head seed 2021; no Stage1, encoder forward/backprop, Flow, cache rebuild, or refitting of the historical references.".format(plan_path), "",
             "The five new heads per task are S/E/P/R-only and a 25D concatenated Full, each with an independent K=4 diagonal GMM. The 24 derived rows reuse only the four pure-head per-variable NLL outputs. For each branch, validation median/MAD normalization is performed per original variable; selected branch z-scores are averaged over the branch axis, then max is taken over the original-variable axis. Each fusion uses its own calibration scores and the fixed 0.995 quantile; prediction is strictly score > threshold. This is branch-axis equal weighting, not the separate variable-axis max/mean sensitivity analysis.", "",
             "## New and historical rows", "",
             "`results.csv` includes the 39 new score rows and six separately marked full-covariance historical references (S and Full per task). Failed configurations retain a status and reason; they are not assigned metrics.", "",
             "## CATCH Table 2 references", "",
             "The table below reproduces the CATCH paper's [Table 2](https://proceedings.iclr.cc/paper_files/paper/2025/file/2b25c39788e5cf11d3541de433ebf4c0-Paper-Conference.pdf) three-decimal AUC-ROC and Aff-F references for the 15 non-CATCH methods. These are literature values under that paper's protocol, not reruns under this frozen-source endpoint protocol. AP is not substituted for Aff-F.", "",
             "| Dataset | Metric | " + " | ".join(BASELINES) + " |",
             "|---|---|" + "---:|" * len(BASELINES)]
    for task in TASKS:
        for metric in ("AUC-ROC", "Aff-F"):
            lines.append("| {} | {} | {} |".format(task, metric, " | ".join("{:.3f}".format(v) for v in PAPER[task][metric])))
    lines.extend(["", "## Reading the comparison", "",
                  "The paper values are context only: data splits, windowing, calibration and affiliation evaluation may differ. They do not establish a common-protocol win/loss. In particular, compare Modern, iTrans, TsNet, Patch and DLin explicitly rather than inferring competitiveness only from classical baselines.", "",
                  "The branch heads, concatenated Full head, and branch-axis fusion are distinct methods. A weaker pure branch can still contain complementary anomaly information. This diagnostic is not evidence that an encoder branch should be deleted; branch removal from the architecture requires a separately trained compact model.", "",
                  "## Diagnostic results", "",
                  "| Dataset | Configuration | Status | AUC-ROC | AP | Aff-F | Precision | Recall | F1 | Normal FPR | TP/FP/FN/TN | Segments hit/scored/total |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        def fmt(key):
            value = row.get(key)
            return "" if value is None else ("{:.6f}".format(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value))
        if row.get("status") not in {"failed", "blocked"}:
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{}/{}/{} | {}/{}/{} |".format(
                row.get("dataset"), row.get("configuration"), row.get("status"), fmt("auc_roc"), fmt("average_precision"), fmt("affiliation_f"), fmt("precision"), fmt("recall"), fmt("point_f1"), fmt("normal_fpr"), fmt("tp"), fmt("fp"), fmt("fn"), fmt("tn"), fmt("anomaly_segments_hit"), fmt("anomaly_segments_scored"), fmt("anomaly_segments_total")))
        else:
            lines.append("| {} | {} | {}: {} | | | | | | | | | |".format(row.get("dataset"), row.get("configuration"), row.get("status"), row.get("failure_reason", "")))
    lines.extend(_diagnostic_comparisons(rows, diagnostic_root))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(PROJECT_ROOT / BATCH_REL / "plan.json"))
    parser.add_argument("--refresh-historical-references", action="store_true")
    args = parser.parse_args(argv)
    if args.refresh_historical_references:
        return refresh_historical_references(args.plan)
    return run_diagnostic(args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
