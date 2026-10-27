#!/usr/bin/env python
"""Read-only metric recheck and paired summary for frozen MNRF ablations."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score
from ts_benchmark.evaluation.metrics.classification_metrics_label import (
    affiliation_f as original_affiliation_f,
    affiliation_precision as original_affiliation_precision,
    affiliation_recall as original_affiliation_recall,
)
from ts_benchmark.evaluation.metrics.affiliation.generics import convert_vector_to_events
from ts_benchmark.evaluation.metrics.affiliation.metrics import pr_from_events

from scripts.run_mnrf_branch_ablation import (
    COMBINATIONS, TASKS, _json, _same_conditions, _sha256_file, _validate_run, _views, selected_indices,
)
from scripts.mnrf_branch_source_registry import selected_source
from ts_benchmark.baselines.mnrf_view_probe.evaluation import affiliation_f_segment_macro


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp-{}".format(__import__("os").getpid()))
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_csv(path, rows, lineterminator="\r\n"):
    path = Path(path)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(path.name + ".tmp-{}".format(__import__("os").getpid()))
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore",
                                lineterminator=lineterminator)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _atomic_text(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def _npz(path):
    return np.load(path, allow_pickle=False)


def _evaluation(run_dir, pinned_source, dataset, combination, seed, checked_runs=None):
    run_dir = Path(run_dir)
    pinned_source = Path(pinned_source)
    checked_runs = checked_runs if checked_runs is not None else {}
    source_key, run_key = pinned_source.resolve(), run_dir.resolve()
    if source_key not in checked_runs:
        checked_runs[source_key] = _validate_run(pinned_source)
    if run_key not in checked_runs:
        checked_runs[run_key] = checked_runs[source_key] if run_key == source_key else _validate_run(run_dir)
    source_checked, run_checked = checked_runs[source_key], checked_runs[run_key]
    manifest = run_checked["manifest"]
    aligned = _npz(run_dir / "results/aligned_scores.npz")
    source_manifest = source_checked["manifest"]
    source_config = source_manifest.get("config", {})
    if dataset in TASKS:
        expected_run_id, expected_seq_len, _ = TASKS[dataset]
        expected_source, registration = selected_source(dataset, expected_run_id)
        if expected_source is None:
            raise ValueError("no verified replacement Full source is registered for {}".format(dataset))
        expected_source = expected_source.resolve()
        if pinned_source.resolve() != expected_source:
            raise ValueError("summary source path is not the pinned Full run for {}".format(dataset))
        if (pinned_source.name != expected_source.name
                or source_config.get("seq_len") != expected_seq_len):
            raise ValueError("pinned source Run ID or window length differs from the ablation contract")
        if registration is not None:
            if (source_manifest.get("source_identity_sha256") != registration.get("source_identity_sha256")
                    or source_manifest.get("feature_fingerprint") != registration.get("feature_fingerprint")
                    or _sha256_file(pinned_source / "probe_manifest.json") != registration.get("manifest_sha256")):
                raise ValueError("registered source identity/manifest changed before summary")
    if (source_manifest.get("artifact_policy") != "full" or source_manifest.get("stage_status") != "complete"
            or source_manifest.get("head_type") != "gmm" or source_manifest.get("head_status") != "complete"
            or source_manifest.get("layout") != {
                "F": 25, "order": ["S", "E", "P", "R"],
                "dimensions": {"S": 1, "E": 8, "P": 8, "R": 8},
                "slices": {"S": [0, 1], "E": [1, 9], "P": [9, 17], "R": [17, 25]}}
            or source_manifest.get("artifact_type") != "probe_head"
            or _views(source_manifest) != ("S", "E", "P", "R")):
        raise ValueError("pinned source is not the required full-layout completed GMM run")
    if dataset in TASKS:
        expected_source_numeric = {
            "seed": 2021, "gmm_components": 4, "gmm_covariance_type": "full",
            "gmm_reg_covar": 0.001, "gmm_n_init": 1, "gmm_max_iter": 100,
            "gmm_fit_max_windows": 20000, "train_noise_std": 0.01, "noise_seed": 20261013,
            "feature_std_floor": 0.05, "score_mad_floor": 0.001, "calibration_quantile": 0.995,
        }
        for name, expected_value in expected_source_numeric.items():
            actual_value = source_config.get(name)
            if actual_value != expected_value:
                raise ValueError("pinned source config {}={} expected {}".format(name, actual_value, expected_value))
    from scripts.export_mnrf_view_probe_analysis import _load_analysis_inputs
    source_analysis = _load_analysis_inputs(pinned_source, source_manifest)
    canonical_test_labels = source_analysis[0]
    canonical_test_times = source_analysis[1]
    canonical_test_global = source_analysis[2]
    canonical_test_segments = source_analysis[4]
    canonical_full_labels = source_analysis[5]
    canonical_full_times = source_analysis[6]
    canonical_full_global = source_analysis[7]
    canonical_full_segments = source_analysis[8]
    if manifest.get("artifact_type") != "probe_head" or manifest.get("head_type") != "gmm":
        raise ValueError("candidate is not a completed GMM probe head")
    if run_key == source_key:
        if combination != "FULL" or int(seed) != 2021:
            raise ValueError("pinned Full source can only fill FULL/head_seed=2021")
    else:
        source_identity = source_manifest.get("source_identity") or {}
        source_record = {
            "path": str(pinned_source.resolve()), "run_id": pinned_source.name,
            "source_identity": source_identity,
            "source_identity_sha256": source_manifest.get("source_identity_sha256"),
            "feature_fingerprint": source_manifest.get("feature_fingerprint"),
            "cache_sha256": source_manifest.get("cache_sha256"),
            "endpoint_sha256": source_manifest.get("endpoint_sha256"),
            "gmm_fit_endpoints_sha256": source_manifest.get("gmm_fit_endpoints_sha256"),
            "gmm_fit_endpoints_path": str((pinned_source / "heads/gmm_fit_endpoints.npy").resolve()),
            "input_statistics_sha256": source_manifest.get("raw_input_statistics_sha256",
                                                              source_manifest.get("input_statistics_sha256")),
            "fixed_noise_sha256": source_manifest.get("fixed_noise_sha256"),
            "fixed_noise_path": str((pinned_source / "heads/fixed_noise.npy").resolve()),
            "config": source_manifest.get("config", {}),
        }
        compatible, reason = _same_conditions(run_checked, source_record, combination, int(seed))
        if not compatible:
            raise ValueError("candidate does not match pinned source contract: " + reason)
    views = tuple(manifest.get("selected_views") or manifest.get("config", {}).get("selected_views") or ())
    expected_views = dict(COMBINATIONS)[combination]
    indices = manifest.get("selected_indices") or manifest.get("config", {}).get("selected_indices")
    if indices is None:
        indices = selected_indices(views)
    if (not isinstance(indices, (list, tuple))
            or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in indices)
            or views != expected_views or list(indices) != selected_indices(expected_views)):
        raise ValueError("run combination or selected feature coordinates differ")
    config = manifest.get("config", {})
    if (isinstance(config.get("seed"), bool) or not isinstance(config.get("seed"), int)
            or config.get("seed") != int(seed)):
        raise ValueError("run head seed differs from summary cell")
    scores = np.asarray(aligned["scores"])
    predictions = np.asarray(aligned["predictions"])
    actual = np.asarray(aligned["actual"])
    global_indices = np.asarray(aligned["time_indices"])
    local_indices = np.asarray(aligned["local_indices"])
    valid_raw = np.asarray(aligned["valid_mask"])
    if valid_raw.dtype.kind != "b":
        raise ValueError("saved valid_mask must be boolean")
    valid_mask = valid_raw
    threshold = float(aligned["threshold"])
    if scores.ndim != 1 or actual.shape != scores.shape or predictions.shape != scores.shape:
        raise ValueError("saved endpoint score arrays do not align")
    if (global_indices.shape != scores.shape or local_indices.shape != scores.shape
            or global_indices.ndim != 1 or local_indices.ndim != 1):
        raise ValueError("saved endpoint coordinate arrays do not align")
    for name, coordinate in (("global", global_indices), ("local", local_indices)):
        if coordinate.dtype.kind not in "iu":
            raise ValueError("{} coordinates must have integer dtype".format(name))
        if np.any(np.diff(coordinate.astype(np.int64, copy=False)) <= 0):
            raise ValueError("{} coordinates must be strictly increasing and unique".format(name))
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("saved scores or threshold are non-finite")
    if actual.dtype.kind not in "iu" or not np.isin(actual, (0, 1)).all():
        raise ValueError("saved endpoint labels are not binary integers")
    if predictions.dtype.kind not in "iu" or not np.isin(predictions, (0, 1)).all():
        raise ValueError("saved predictions are not binary integers")
    if not np.array_equal(predictions, scores > threshold):
        raise ValueError("saved predictions disagree with calibrated threshold")
    for key in ("aligned_scores", "aligned_predictions", "aligned_actual"):
        if key not in aligned.files:
            raise ValueError("saved aligned score bundle is missing {}".format(key))
    aligned_scores = np.asarray(aligned["aligned_scores"])
    aligned_predictions = np.asarray(aligned["aligned_predictions"])
    aligned_actual = np.asarray(aligned["aligned_actual"])
    if (aligned_scores.shape != valid_mask.shape or aligned_predictions.shape != valid_mask.shape
            or aligned_actual.shape != valid_mask.shape):
        raise ValueError("full aligned score arrays do not match valid_mask")
    if (not np.allclose(aligned_scores[valid_mask], scores, rtol=0, atol=0)
            or not np.array_equal(aligned_predictions[valid_mask], predictions)
            or not np.array_equal(aligned_actual[valid_mask].astype(np.int8), actual.astype(np.int8))):
        raise ValueError("aligned score/label arrays disagree with valid endpoint arrays")

    with _npz(pinned_source / "analysis_inputs.npz") as inputs:
        all_labels = np.asarray(inputs["all_labels"])
        all_times = np.asarray(inputs["all_time_values_ns"])
        all_global = np.asarray(inputs["all_global_indices"])
        all_segments = np.asarray(inputs["all_segment_ids"])
        test_labels = np.asarray(inputs["test_labels"])
        test_times = np.asarray(inputs["test_time_values_ns"])
        test_global = np.asarray(inputs["test_global_indices"])
        test_segments = np.asarray(inputs["test_segment_ids"])
    if (all_labels.ndim != 1 or all_times.shape != all_labels.shape
            or all_global.shape != all_labels.shape or all_segments.shape != all_labels.shape
            or test_labels.ndim != 1 or test_times.shape != test_labels.shape
            or test_global.shape != test_labels.shape or test_segments.shape != test_labels.shape):
        raise ValueError("pinned source full-axis label/coordinate inputs are invalid")
    if not np.array_equal(test_global, all_global[len(all_labels) - len(test_labels):]):
        raise ValueError("source test coordinate axis is not the suffix of its full axis")
    for actual_axis, expected_axis, name in (
            (test_labels, canonical_test_labels, "test labels"),
            (all_labels, canonical_full_labels, "full labels"),
            (test_global, canonical_test_global, "test global indices"),
            (test_segments, canonical_test_segments, "test segment IDs"),
            (all_global, canonical_full_global, "full global indices"),
            (all_segments, canonical_full_segments, "full segment IDs"),
            (all_times, canonical_full_times, "full time axis"),
            (test_times, canonical_test_times, "test time axis")):
        if not np.array_equal(actual_axis, expected_axis):
            raise ValueError("validated source analysis input {} differs".format(name))
    if all_global.dtype.kind not in "iu" or test_global.dtype.kind not in "iu":
        raise ValueError("pinned full-axis coordinates must have integer dtype")
    if not np.array_equal(test_labels, all_labels[len(all_labels) - len(test_labels):]):
        raise ValueError("pinned test labels differ from full label axis suffix")
    if not np.array_equal(test_segments, all_segments[len(all_segments) - len(test_segments):]):
        raise ValueError("pinned test segment IDs differ from full-axis canonical IDs")
    local = local_indices.astype(np.int64, copy=False)
    if np.any(local < 0) or np.any(local >= len(test_labels)):
        raise ValueError("scored local coordinate is outside pinned source test axis")
    if not np.array_equal(global_indices, test_global[local]):
        raise ValueError("scored global coordinates differ from pinned source")
    if not np.array_equal(actual.astype(np.int8), test_labels[local].astype(np.int8)):
        raise ValueError("saved scored labels differ from pinned full label axis")
    expected_mask = np.zeros(len(test_labels), dtype=bool)
    expected_mask[local] = True
    if not np.array_equal(expected_mask, valid_mask):
        raise ValueError("saved valid mask differs from scored local coordinates")
    if len(test_labels) != len(valid_mask):
        raise ValueError("candidate test axis length differs from pinned full source")
    if not np.array_equal(expected_mask, valid_mask):
        raise ValueError("candidate scored point set differs from its saved coordinates")

    with _npz(pinned_source / "results/aligned_scores.npz") as source_aligned:
        for key in ("time_indices", "time_values_ns", "local_indices", "valid_mask", "aligned_actual"):
            if key not in aligned.files or key not in source_aligned.files:
                raise ValueError("candidate/source aligned evidence missing {}".format(key))
            if not np.array_equal(np.asarray(aligned[key]), np.asarray(source_aligned[key])):
                raise ValueError("candidate {} differs from pinned Full scoring axis".format(key))
        if "time_values_ns" not in aligned.files or not np.array_equal(
                np.asarray(aligned["time_values_ns"]), np.asarray(source_aligned["time_values_ns"])):
            raise ValueError("candidate time_values_ns differ from pinned Full scoring axis")

    nll = np.load(run_dir / "results/test_nll.npy", mmap_mode="r", allow_pickle=False)
    calibration_path = run_dir / "heads/calibration.npz"
    if nll.ndim != 2 or nll.shape[0] != int(valid_mask.sum()) or not np.isfinite(nll).all():
        raise ValueError("saved test_nll shape/values do not match the exact scored points")
    with _npz(calibration_path) as calibration:
        center = np.asarray(calibration["center"])
        scale = np.asarray(calibration["scale"])
        saved_threshold = float(np.asarray(calibration["threshold"]))
    if center.shape != (nll.shape[1],) or scale.shape != center.shape:
        raise ValueError("calibration center/scale shape differs from test_nll")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("calibration statistics are invalid")
    if not np.isclose(threshold, saved_threshold, rtol=0, atol=1e-7):
        raise ValueError("aligned threshold differs from saved calibration threshold")
    recomputed = np.max((np.asarray(nll) - center) / scale, axis=1)
    if not np.allclose(scores, recomputed, rtol=1e-6, atol=1e-7):
        raise ValueError("saved scores do not reproduce from test_nll and calibration statistics")
    run_metrics = run_checked["metrics"]
    gmm_status_path = run_dir / "heads/gmm_status.json"
    gmm_status = _json(gmm_status_path) if gmm_status_path.is_file() else {}
    gmm_config = manifest.get("config", {})
    head_parameter_count = run_metrics.get("head_parameter_count", gmm_status.get("head_parameter_count"))
    if head_parameter_count is None:
        raise ValueError("validated GMM head parameter count is missing")

    tp = int(np.sum((actual == 1) & (predictions == 1)))
    fp = int(np.sum((actual == 0) & (predictions == 1)))
    fn = int(np.sum((actual == 1) & (predictions == 0)))
    tn = int(np.sum((actual == 0) & (predictions == 0)))
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = float(roc_auc_score(actual, scores)) if np.unique(actual).size == 2 else None
    ap = float(average_precision_score(actual, scores)) if len(actual) else None
    metrics = run_checked["metrics"]
    for metric_key, computed in (("auc_roc", auc), ("auc_pr", ap), ("threshold", threshold)):
        recorded = metrics.get(metric_key)
        if computed is None:
            if recorded is not None:
                raise ValueError("metrics {} should be undefined for this label set".format(metric_key))
        elif recorded is None or not np.isclose(float(recorded), float(computed), rtol=1e-6, atol=1e-7):
            raise ValueError("metrics {} differs from aligned score recomputation".format(metric_key))
    if int(metrics.get("valid_points", -1)) != int(valid_mask.sum()):
        raise ValueError("metrics valid_points differs from exact scoring mask")
    input_metadata = _json(pinned_source / "analysis_inputs.json")
    segment_identity = {int(item["segment_id"]): item.get("boundary_value")
                        for item in input_metadata.get("test_segment_identity", [])}
    segment = segment_counts_from_axis(all_labels, all_global, all_segments, test_global,
                                       global_indices, predictions, all_times,
                                       segment_identity)
    result = {
        "dataset": dataset, "combination": combination, "head_seed": int(seed),
        "candidate_or_output_path": str(run_dir.resolve()), "status": "complete",
        "selected_views": ",".join(expected_views), "selected_indices": json.dumps(selected_indices(expected_views)),
        "selected_dim": len(selected_indices(expected_views)), "threshold": threshold,
        "auc_roc": auc, "average_precision": ap, "precision": precision, "recall": recall,
        "point_f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "normal_fpr": fp / float(fp + tn) if fp + tn else None,
        "coverage": float(valid_mask.mean()) if len(valid_mask) else 0.0,
        "anomaly_segments_total": segment["total"], "anomaly_segments_scored": segment["scored"],
        "anomaly_segments_hit": segment["hit"], "segment_details": segment["details"],
        "source_run_id": pinned_source.name, "source_encoder_seed": 2021,
        "source_identity": source_manifest.get("source_identity_sha256"),
        "feature_fingerprint": source_manifest.get("feature_fingerprint"),
        "gmm_components": gmm_config.get("gmm_components"),
        "gmm_covariance_type": gmm_config.get("gmm_covariance_type"),
        "gmm_reg_covar": gmm_config.get("gmm_reg_covar"),
        "gmm_n_init": gmm_config.get("gmm_n_init"),
        "gmm_max_iter": gmm_config.get("gmm_max_iter"),
        "gmm_fit_max_windows": gmm_config.get("gmm_fit_max_windows"),
        "head_parameter_count": head_parameter_count,
        "fit_time_seconds": run_metrics.get("fit_time_seconds"),
        "inference_time_seconds": run_metrics.get("inference_time_seconds"),
        "elapsed_seconds_kind": "historical_or_reused",
        "independent_recheck": True,
    }
    bundle = {
        "scores": scores.astype(np.float32, copy=False), "predictions": predictions.astype(np.int8, copy=False),
        "labels": actual.astype(np.int8, copy=False), "global_indices": global_indices.astype(np.int64, copy=False),
        "local_indices": local.astype(np.int64, copy=False), "valid_mask": valid_mask,
        "threshold": np.asarray(threshold, dtype=np.float64),
    }
    return result, bundle


def segment_counts_from_axis(all_labels, all_global, all_segments, test_global, scored_global, predictions,
                             all_times=None, segment_identity=None):
    """Count complete test-axis anomaly intervals without joining gaps/entities."""
    test_start = int(test_global[0]) if len(test_global) else int(all_global[-1]) + 1
    test_end = int(test_global[-1]) + 1 if len(test_global) else test_start
    label_by_global = {int(g): int(label) for g, label in zip(all_global, all_labels)}
    segment_by_global = {int(g): int(s) for g, s in zip(all_global, all_segments)}
    prediction_by_global = {int(g): int(p) for g, p in zip(scored_global, predictions)}
    test_positions = [int(g) for g in test_global]
    time_by_global = {} if all_times is None else {int(g): int(t) for g, t in zip(all_global, all_times)}
    details = []
    i = 0
    while i < len(test_positions):
        g = test_positions[i]
        if label_by_global.get(g) != 1:
            i += 1
            continue
        segid = segment_by_global[g]
        start_i = i
        i += 1
        while i < len(test_positions):
            prev_g, cur_g = test_positions[i - 1], test_positions[i]
            if (cur_g != prev_g + 1 or segment_by_global.get(cur_g) != segid
                    or label_by_global.get(cur_g) != 1):
                break
            i += 1
        points = test_positions[start_i:i]
        predicted = [(point, prediction_by_global[point]) for point in points if point in prediction_by_global]
        hit_points = sum(pred for _, pred in predicted)
        details.append({"segment_id": segid, "boundary_value": (segment_identity or {}).get(segid),
                        "start_global": points[0], "end_global": points[-1],
                        "start_time_ns": time_by_global.get(points[0]),
                        "end_time_ns": time_by_global.get(points[-1]),
                        "points": len(points), "valid_points": len(predicted), "hit_points": int(hit_points),
                        "hit": bool(hit_points), "no_valid_scores": len(predicted) == 0})
    return {"total": len(details), "scored": sum(d["valid_points"] > 0 for d in details),
            "hit": sum(d["hit"] for d in details), "details": details}


def segment_counts(source_path, scored_global, predictions):
    """Small adapter shared by execution-time result recording and summaries."""
    source_path = Path(source_path)
    with _npz(source_path / "analysis_inputs.npz") as inputs:
        all_labels = np.asarray(inputs["all_labels"])
        all_times = np.asarray(inputs["all_time_values_ns"])
        all_global = np.asarray(inputs["all_global_indices"])
        all_segments = np.asarray(inputs["all_segment_ids"])
        test_global = np.asarray(inputs["test_global_indices"])
    result = segment_counts_from_axis(all_labels, all_global, all_segments, test_global,
                                      np.asarray(scored_global), np.asarray(predictions), all_times)
    return result


def contribution_deltas(metrics_by_combo):
    """Return prescribed added/deleted AP and ROC deltas; positive deletion means removal hurts."""
    result = {}
    full = metrics_by_combo.get("FULL")
    s = metrics_by_combo.get("S")
    epr = metrics_by_combo.get("WITHOUT_S")
    for view in ("S", "E", "P", "R"):
        if view == "S":
            added = {"average_precision": None, "auc_roc": None}
            removed_combo = "WITHOUT_S"
        else:
            added_combo = "S" + view
            added = {}
            for metric in ("average_precision", "auc_roc"):
                left, right = metrics_by_combo.get(added_combo), s
                added[metric] = (left.get(metric) - right.get(metric)
                                if left and right and left.get(metric) is not None and right.get(metric) is not None else None)
            removed_combo = "WITHOUT_" + view
        removed = {}
        for metric in ("average_precision", "auc_roc"):
            left, right = full, metrics_by_combo.get(removed_combo)
            removed[metric] = (left.get(metric) - right.get(metric)
                               if left and right and left.get(metric) is not None and right.get(metric) is not None else None)
        result[view] = {"added": added, "deleted": removed, "removal_combination": removed_combo}
    return result


def cross_dataset_summary(tasks, evals, tolerance=1e-12):
    """Summarize fixed-seed contribution signs across requested datasets.

    Only cells validated in this invocation are included. Missing, blocked and
    invalid cells remain represented by ``observed=0`` rather than being
    treated as ties or zeros.
    """
    rows = []
    for branch in ("S", "E", "P", "R"):
        for direction in ("added", "deleted"):
            if direction == "added" and branch == "S":
                combinations = (None,)
            else:
                combinations = ("average_precision", "auc_roc")
            for metric in combinations:
                values = []
                participating = []
                for task in tasks:
                    table = {combo: evals.get((task, combo, 2021)) for combo, _ in COMBINATIONS}
                    delta = contribution_deltas(table)[branch][direction]
                    value = None if metric is None else delta.get(metric)
                    if value is not None and np.isfinite(value):
                        values.append(float(value))
                        participating.append(task)
                rows.append({
                    "branch": branch, "direction": direction,
                    "metric": "not_applicable" if metric is None else metric,
                    "requested": len(tasks), "observed": len(values),
                    "wins": sum(value > tolerance for value in values),
                    "ties": sum(abs(value) <= tolerance for value in values),
                    "losses": sum(value < -tolerance for value in values),
                    "mean": float(np.mean(values)) if values else None,
                    "participating_datasets": json.dumps(participating),
                    "missing_datasets": json.dumps([task for task in tasks if task not in participating]),
                    "tie_tolerance": tolerance,
                    "not_applicable": metric is None,
                })
    return rows


PAIRED_COMPARISONS = (("S", "SE"), ("S", "SP"), ("S", "SR"),
                      ("FULL", "WITHOUT_S"), ("FULL", "WITHOUT_E"),
                      ("FULL", "WITHOUT_P"), ("FULL", "WITHOUT_R"))


def _point_and_segment_changes(base, changed, source_run_id, limit=8):
    base_global = np.asarray(base["global_indices"])
    changed_global = np.asarray(changed["global_indices"])
    if not np.array_equal(base_global, changed_global):
        raise ValueError("paired point comparisons require identical global indices")
    labels = np.asarray(base["labels"])
    if not np.array_equal(labels, changed["labels"]):
        raise ValueError("paired point comparisons require identical labels")
    bp = np.asarray(base["predictions"], dtype=bool)
    cp = np.asarray(changed["predictions"], dtype=bool)
    categories = {
        "added_tp": (labels == 1) & ~bp & cp,
        "lost_tp": (labels == 1) & bp & ~cp,
        "added_fp": (labels == 0) & ~bp & cp,
        "removed_fp": (labels == 0) & bp & ~cp,
    }
    point_indices = {name: np.flatnonzero(mask) for name, mask in categories.items()}
    base_segments = {(_segment_key(source_run_id, item)): item
                     for item in base.get("segment_details", [])}
    changed_segments = {(_segment_key(source_run_id, item)): item
                        for item in changed.get("segment_details", [])}
    added_hits, lost_hits = [], []
    for key in sorted(set(base_segments) | set(changed_segments)):
        left, right = base_segments.get(key), changed_segments.get(key)
        left_hit = bool(left and left.get("hit"))
        right_hit = bool(right and right.get("hit"))
        item = dict(right or left)
        item.update({"base_hit": left_hit, "changed_hit": right_hit,
                     "base_threshold": float(base["threshold"]),
                     "changed_threshold": float(changed["threshold"])})
        if not left_hit and right_hit:
            added_hits.append(item)
        elif left_hit and not right_hit:
            lost_hits.append(item)
    return point_indices, added_hits, lost_hits


def _segment_key(source_run_id, item):
    return (source_run_id, int(item["segment_id"]), int(item["start_global"]), int(item["end_global"]))


def _change_cases(base, changed, indices, added_segments, lost_segments, limit=8):
    cases = {}
    for kind, positions in indices.items():
        cases[kind] = [{
            "global_index": int(base["global_indices"][pos]),
            "local_index": int(base["local_indices"][pos]),
            "label": int(base["labels"][pos]),
            "base_score": float(base["scores"][pos]),
            "changed_score": float(changed["scores"][pos]),
            "base_threshold": float(base["threshold"]),
            "changed_threshold": float(changed["threshold"]),
            "base_prediction": int(base["predictions"][pos]),
            "changed_prediction": int(changed["predictions"][pos]),
        } for pos in positions[:limit]]
    for name, items in (("added_segment_hit", added_segments), ("lost_segment_hit", lost_segments)):
        cases[name] = [{
            "segment_id": int(item["segment_id"]), "boundary_value": item.get("boundary_value"),
            "start_global": int(item["start_global"]), "end_global": int(item["end_global"]),
            "points": int(item["points"]), "valid_points": int(item["valid_points"]),
            "hit_points": int(item["hit_points"]),
            "base_hit": bool(item["base_hit"]), "changed_hit": bool(item["changed_hit"]),
            "base_threshold": float(item["base_threshold"]),
            "changed_threshold": float(item["changed_threshold"]),
        } for item in items[:limit]]
    return cases


def completion_counts(tasks, seeds, rows):
    expected = {(task, combo, int(seed)) for task in tasks
                for combo, _ in COMBINATIONS for seed in seeds}
    seen, duplicate = {}, set()
    for row in rows:
        try:
            key = (row["dataset"], row["combination"], int(row["head_seed"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key in seen:
            duplicate.add(key)
        else:
            seen[key] = row
    statuses = ("complete", "reused", "planned/pending", "running", "failed", "blocked", "conflict", "invalid")
    known = set(statuses)
    if any(row.get("status") not in known for row in rows):
        rows = [dict(row, status="invalid") if row.get("status") not in known else row for row in rows]
    counts = {status: sum(row.get("status") == status for row in rows) for status in statuses}
    missing, unexpected = expected - set(seen), set(seen) - expected
    complete_datasets = datasets_with_results = 0
    for task in tasks:
        keys = {(task, combo, int(seed)) for combo, _ in COMBINATIONS for seed in seeds}
        success = {key for key in keys if key in seen and seen[key].get("status") in {"complete", "reused"}}
        datasets_with_results += int(bool(success))
        complete_datasets += int(keys == success and not (keys & duplicate))
    illegal = bool(duplicate or missing or unexpected or counts["invalid"])
    hard_error = illegal or any(counts[key] for key in ("failed", "blocked", "conflict", "invalid"))
    successful = counts["complete"] + counts["reused"]
    status = "complete" if successful == len(expected) and not hard_error else ("failed" if hard_error else "partial")
    return {"positions_expected": len(expected), "positions_present": len(rows),
            "complete": counts["complete"], "reused": counts["reused"],
            "complete_or_reused": successful, "pending": counts["planned/pending"],
            "running": counts["running"], "failed": counts["failed"], "blocked": counts["blocked"],
            "conflict": counts["conflict"],
            "invalid": counts["invalid"] + len(missing) + len(unexpected) + len(duplicate),
            "datasets_requested": len(tasks), "datasets_complete": complete_datasets,
            "datasets_with_results": datasets_with_results, "batch_status": status,
            "plan_errors": {"missing": len(missing), "unexpected": len(unexpected),
                            "duplicate": len(duplicate)}}


def summarize(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = _json(plan_path)
    source_records = plan["source_records"]
    requested_tasks = list(plan.get("tasks", ()))
    requested_seeds = [int(seed) for seed in plan.get("seeds", ())]
    if not requested_tasks or requested_seeds != [2021]:
        raise ValueError("this summarizer accepts exactly the fixed head seed [2021]; repeat plans are obsolete")
    expected_keys = {(task, combo, seed) for task in requested_tasks
                     for combo, _ in COMBINATIONS for seed in requested_seeds}
    raw_rows = plan.get("rows", [])
    by_key = {}
    duplicates = set()
    malformed = []
    for row in raw_rows:
        try:
            key = (row["dataset"], row["combination"], int(row["head_seed"]))
        except (KeyError, TypeError, ValueError):
            malformed.append(row)
            continue
        if key in by_key:
            duplicates.add(key)
        else:
            by_key[key] = row
    rows = []
    bundles = {}
    evals = {}
    checked_runs = {}
    for expected in sorted(expected_keys, key=lambda k: (requested_tasks.index(k[0]), requested_seeds.index(k[2]),
                                                         [c for c, _ in COMBINATIONS].index(k[1]))):
        planned = by_key.get(expected)
        if planned is None:
            rows.append({"dataset": expected[0], "combination": expected[1], "head_seed": expected[2],
                         "status": "invalid", "reason": "expected matrix key is missing"})
            continue
        row = dict(planned)
        task, combo, seed = expected
        if expected in duplicates:
            row.update(status="invalid", reason="duplicate matrix key")
        elif row.get("status") in {"complete", "reused"}:
            source = source_records[task]
            path = row.get("candidate_or_output_path")
            try:
                if source.get("status") != "complete":
                    raise ValueError("pinned source is not verified complete: {}".format(source.get("reason")))
                result, bundle = _evaluation(path, source["path"], task, combo, seed, checked_runs)
                elapsed_kind = row.get("elapsed_seconds_kind")
                for key, value in result.items():
                    if key != "segment_details":
                        row[key] = value
                row["segment_details"] = json.dumps(result["segment_details"], separators=(",", ":"))
                row["status"] = "reused" if planned.get("status") == "reused" else "complete"
                row["elapsed_seconds_kind"] = elapsed_kind if elapsed_kind in {
                    "this_run", "historical_or_reused"} else "historical_or_reused"
                key = "{}_{}_{}".format(task, combo, seed)
                bundles[key] = bundle
                result["_bundle"] = bundle
                evals[(task, combo, seed)] = result
            except Exception as error:
                row["status"] = "invalid"
                row["reason"] = "read-only summary verification failed: {}: {}".format(type(error).__name__, error)
                for key in ("auc_roc", "average_precision", "precision", "recall", "point_f1",
                            "fp", "fn", "tp", "tn", "normal_fpr", "coverage"):
                    row[key] = None
        elif row.get("status") not in {"blocked", "conflict", "failed", "planned/pending", "running"}:
            row.update(status="invalid", reason="illegal plan status")
        if row.get("status") == "blocked" and not row.get("reason"):
            row["reason"] = "blocked by plan/source validation"
        if row.get("status") not in {"complete", "reused"}:
            for metric_key in ("auc_roc", "average_precision", "precision", "recall", "point_f1",
                               "fp", "fn", "tp", "tn", "normal_fpr", "coverage",
                               "anomaly_segments_total", "anomaly_segments_scored", "anomaly_segments_hit"):
                row[metric_key] = None
        rows.append(row)

    extra_keys = set(by_key) - expected_keys
    if malformed:
        rows.append({"dataset": "", "combination": "", "head_seed": "", "status": "invalid",
                     "reason": "plan contains {} malformed rows".format(len(malformed))})
    for key in sorted(extra_keys):
        rows.append({"dataset": key[0], "combination": key[1], "head_seed": key[2], "status": "invalid",
                     "reason": "unexpected matrix key outside requested task/seed set"})

    contributions = []
    paired = []
    for task in plan["tasks"]:
        for seed in plan["seeds"]:
            table = {combo: evals.get((task, combo, int(seed))) for combo, _ in COMBINATIONS}
            delta = contribution_deltas(table)
            for view, values in delta.items():
                row = {"dataset": task, "head_seed": int(seed), "branch": view,
                       "added_ap": values["added"]["average_precision"],
                       "deleted_ap": values["deleted"]["average_precision"],
                       "added_auc_roc": values["added"]["auc_roc"],
                       "deleted_auc_roc": values["deleted"]["auc_roc"],
                       "removal_combination": values["removal_combination"]}
                contributions.append(row)
            for base, changed in PAIRED_COMPARISONS:
                left, right = table.get(base), table.get(changed)
                if not left or not right:
                    continue
                left_bundle, right_bundle = left["_bundle"], right["_bundle"]
                left_points = dict(left_bundle, segment_details=left["segment_details"])
                right_points = dict(right_bundle, segment_details=right["segment_details"])
                changes, added_segments, lost_segments = _point_and_segment_changes(
                    left_points, right_points, left["source_run_id"])
                cases = _change_cases(left_points, right_points, changes, added_segments, lost_segments)
                paired.append({"dataset": task, "head_seed": int(seed), "base": base, "changed": changed,
                               "delta_ap_changed_minus_base": (right["average_precision"] - left["average_precision"]
                                   if right["average_precision"] is not None and left["average_precision"] is not None else None),
                               "delta_auc_changed_minus_base": (right["auc_roc"] - left["auc_roc"]
                                   if right["auc_roc"] is not None and left["auc_roc"] is not None else None),
                               "delta_tp": right["tp"] - left["tp"], "delta_fp": right["fp"] - left["fp"],
                               "delta_fn": right["fn"] - left["fn"],
                               "delta_segments_hit": right["anomaly_segments_hit"] - left["anomaly_segments_hit"],
                               "added_tp": len(changes["added_tp"]), "lost_tp": len(changes["lost_tp"]),
                               "added_fp": len(changes["added_fp"]), "removed_fp": len(changes["removed_fp"]),
                               "added_segment_hits": len(added_segments), "lost_segment_hits": len(lost_segments),
                               "change_cases": json.dumps(cases, separators=(",", ":"))})

    return rows, contributions, paired, bundles


def _binary_metrics(labels, scores, predictions):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(np.sum((labels == 1) & predictions))
    fp = int(np.sum((labels == 0) & predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    tn = int(np.sum((labels == 0) & ~predictions))
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None
    ap = float(average_precision_score(labels, scores)) if len(labels) else None
    return {"ap": ap, "auc_roc": auc, "precision": precision, "recall": recall,
            "f1": f1, "normal_fpr": fp / float(fp + tn) if fp + tn else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _max_variable_summary(z, labels, max_predictions):
    """Return tie-split variable shares for max-score FP and TP points."""
    z = np.asarray(z, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    max_predictions = np.asarray(max_predictions, dtype=bool)
    maxima = np.max(z, axis=1)
    ties = np.isclose(z, maxima[:, None], rtol=0.0, atol=1e-7)
    shares = ties / np.maximum(ties.sum(axis=1, keepdims=True), 1)
    summaries = []
    for name, selection in (
            ("fp", (labels == 0) & max_predictions),
            ("tp", (labels == 1) & max_predictions)):
        counts = shares[selection].sum(axis=0)
        total = float(counts.sum())
        if not total:
            summaries.append("none (n=0)")
            continue
        order = np.argsort(-counts, kind="stable")
        summaries.append("{} (n={})".format(
            name, int(selection.sum())) + "; " + ", ".join(
                "v{}={:.3f}".format(int(index), float(counts[index] / total))
                for index in order if counts[index] > 0))
    return " | ".join(summaries)


def _replay_calibration_aggregations(calibration_nll, center, scale, saved_max_threshold,
                                    manifest_quantile, saved_quantile):
    """Replay frozen max and mean calibration thresholds without fitting."""
    calibration_nll = np.asarray(calibration_nll)
    center = np.asarray(center)
    scale = np.asarray(scale)
    if calibration_nll.ndim != 2 or calibration_nll.shape[0] == 0:
        raise ValueError("calibration NLL must be a non-empty 2D array")
    if center.shape != (calibration_nll.shape[1],) or scale.shape != center.shape:
        raise ValueError("calibration NLL columns differ from center/scale")
    if (not np.isfinite(calibration_nll).all() or not np.isfinite(center).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.isfinite(saved_max_threshold)):
        raise ValueError("calibration NLL/statistics/threshold must be finite and scales positive")
    if (not np.isfinite(manifest_quantile) or manifest_quantile != saved_quantile
            or saved_quantile != 0.995):
        raise ValueError("calibration quantile differs from manifest or fixed 0.995")
    # Preserve the persisted dtype arithmetic used by the original max score.
    calibration_z = (calibration_nll - center) / scale
    replayed_max = float(np.quantile(np.max(calibration_z, axis=1), saved_quantile,
                                     method="linear"))
    if not np.isclose(replayed_max, saved_max_threshold, rtol=1e-6, atol=1e-7):
        raise ValueError("replayed max calibration threshold differs from saved threshold")
    replayed_mean = float(np.quantile(np.mean(calibration_z, axis=1), saved_quantile,
                                      method="linear"))
    return calibration_z, replayed_max, replayed_mean


def _single_variable_consistent(max_scores, mean_scores, max_threshold, mean_threshold,
                                max_predictions, mean_predictions):
    """A one-variable reduction must make max and mean exactly identical."""
    return (np.array_equal(max_scores, mean_scores) and max_threshold == mean_threshold
            and np.array_equal(max_predictions, mean_predictions))


def aggregation_comparison(plan_path):
    """Recompute max and an additive mean-variable aggregation read-only."""
    plan_path = Path(plan_path).resolve()
    plan = _json(plan_path)
    rows = []
    successful = 0
    failed = 0
    quantile_default = 0.995
    checked_runs = {}
    for planned in plan.get("rows", []):
        dataset = planned.get("dataset", "")
        combination = planned.get("combination", "")
        seed = int(planned.get("head_seed", 2021))
        row = {"dataset": dataset, "combination": combination, "head_seed": seed,
               "source_run_id": planned.get("source_run_id"),
               "candidate_run_id": Path(planned.get("candidate_or_output_path", "")).name,
               "status": planned.get("status"), "original_status": planned.get("status"),
               "max_reproduced": False,
               "mean_status": "not_attempted", "reason": planned.get("reason", "")}
        if planned.get("status") not in {"complete", "reused"}:
            failed += 1
            row.update({"status": planned.get("status", "invalid"), "mean_status": "not_attempted"})
            rows.append(row)
            continue
        try:
            run_dir = Path(planned["candidate_or_output_path"]).resolve()
            source_dir = Path(plan["source_records"][dataset]["path"]).resolve()
            manifest = _json(run_dir / "probe_manifest.json")
            # Re-enter the same native source/candidate contract used by the
            # normal summary.  The cache is shared across cells so each source
            # and candidate is fully checked at most once in this invocation.
            evaluation, bundle = _evaluation(
                run_dir, source_dir, dataset, combination, seed, checked_runs)
            source_manifest = checked_runs[source_dir.resolve()]["manifest"]
            candidate_names = tuple(manifest.get("variable_names") or
                                    manifest.get("config", {}).get("variable_names", ()))
            source_names = tuple(source_manifest.get("variable_names") or
                                 source_manifest.get("config", {}).get("variable_names", ()))
            if candidate_names != source_names:
                raise ValueError("candidate/source variable order differs")
            for relative in ("results/test_nll.npy", "heads/calibration_nll.npy",
                             "heads/calibration.npz"):
                if not (run_dir / relative).is_file():
                    raise ValueError("missing {}".format(relative))
            nll = np.load(run_dir / "results/test_nll.npy", mmap_mode="r", allow_pickle=False)
            calibration_nll = np.load(run_dir / "heads/calibration_nll.npy", mmap_mode="r", allow_pickle=False)
            if nll.ndim != 2 or calibration_nll.ndim != 2 or nll.shape[1] != calibration_nll.shape[1]:
                raise ValueError("test/calibration NLL dimensions differ")
            selected_dim = len(selected_indices(dict(COMBINATIONS)[combination]))
            variable_count = len(source_names) if source_names else int(nll.shape[1])
            if nll.shape[1] != variable_count:
                raise ValueError("NLL columns do not match frozen variable order")
            # select_gmm heads intentionally reference the frozen source cache;
            # self-contained Full runs carry their own endpoint file.  The
            # endpoint identity is the same contract in either case.
            endpoint_owner = run_dir if (run_dir / "cache/calibration_window_end_indices.npy").is_file() else source_dir
            calibration_endpoint_path = endpoint_owner / "cache/calibration_window_end_indices.npy"
            if not calibration_endpoint_path.is_file():
                raise ValueError("missing cache/calibration_window_end_indices.npy in candidate or source")
            calibration_endpoints = np.load(
                calibration_endpoint_path, mmap_mode="r", allow_pickle=False)
            source_calibration_endpoints = np.load(
                source_dir / "cache/calibration_window_end_indices.npy",
                mmap_mode="r", allow_pickle=False)
            if (calibration_endpoints.ndim != 1 or
                    calibration_endpoints.dtype.kind not in "iu" or
                    calibration_nll.shape[0] != calibration_endpoints.shape[0]):
                raise ValueError("calibration NLL rows do not match calibration endpoints")
            if not np.array_equal(calibration_endpoints, source_calibration_endpoints):
                raise ValueError("candidate calibration endpoints differ from pinned Full source")
            with _npz(run_dir / "heads/calibration.npz") as calibration:
                # Keep the saved float32 arithmetic used by the production
                # max path; widening here changes threshold-boundary
                # predictions for otherwise valid historical heads.
                center = np.asarray(calibration["center"])
                scale = np.asarray(calibration["scale"])
                max_threshold = float(np.asarray(calibration["threshold"]))
                if "quantile" not in calibration.files:
                    raise ValueError("calibration quantile is missing")
                quantile = float(np.asarray(calibration["quantile"]))
            config = manifest.get("config", {})
            manifest_quantile = float(config.get("calibration_quantile", float("nan")))
            # Replay the original max calibration threshold from the saved
            # calibration NLL. This is deliberately separate from the saved
            # threshold comparison performed by _evaluation().
            if not np.isfinite(nll).all():
                raise ValueError("test NLL contains non-finite values")
            calibration_z, replayed_max_threshold, mean_threshold = _replay_calibration_aggregations(
                calibration_nll, center, scale, max_threshold, manifest_quantile, quantile)
            bundle_scores = np.asarray(bundle["scores"])
            bundle_predictions = np.asarray(bundle["predictions"], dtype=bool)
            labels = np.asarray(bundle["labels"], dtype=np.int8)
            global_indices = np.asarray(bundle["global_indices"])
            max_scores = np.max((np.asarray(nll) - center) / scale, axis=1)
            if len(bundle_scores) != len(max_scores) or not np.allclose(
                    bundle_scores, max_scores, rtol=1e-6, atol=1e-7):
                raise ValueError("native max scores do not reproduce from test_nll")
            max_predictions = max_scores > replayed_max_threshold
            if not np.array_equal(bundle_predictions, max_predictions):
                raise ValueError("native max predictions do not reproduce")
            # Calibration and test NLL use the same selected variable order.
            z = (np.asarray(nll) - center) / scale
            mean_scores = np.mean(z, axis=1)
            mean_predictions = mean_scores > mean_threshold
            max_metrics = _binary_metrics(labels, max_scores, max_predictions)
            mean_metrics = _binary_metrics(labels, mean_scores, mean_predictions)
            native_ap = evaluation["average_precision"]
            ap_matches = ((max_metrics["ap"] is None and native_ap is None) or
                          (max_metrics["ap"] is not None and native_ap is not None and
                           np.isclose(max_metrics["ap"], native_ap, rtol=1e-6, atol=1e-7)))
            native_auc = evaluation["auc_roc"]
            auc_matches = ((max_metrics["auc_roc"] is None and native_auc is None) or
                           (max_metrics["auc_roc"] is not None and native_auc is not None and
                            np.isclose(max_metrics["auc_roc"], native_auc, rtol=1e-6, atol=1e-7)))
            if not ap_matches or not auc_matches:
                raise ValueError("replayed max metrics differ from native evaluation")
            max_segments = segment_counts(source_dir, global_indices, max_predictions.astype(np.int8))
            mean_segments = segment_counts(source_dir, global_indices, mean_predictions.astype(np.int8))
            max_var_summary = _max_variable_summary(z, labels, max_predictions)
            row.update({"status": "passed", "mean_status": "passed", "max_reproduced": True,
                        "max_threshold": replayed_max_threshold, "mean_threshold": mean_threshold,
                        "replayed_max_threshold": replayed_max_threshold,
                        "quantile": quantile, "selected_dim": selected_dim,
                        "variable_count": variable_count,
                        "single_variable_check": ("passed" if variable_count == 1 and
                                                   _single_variable_consistent(
                                                       max_scores, mean_scores,
                                                       replayed_max_threshold, mean_threshold,
                                                       max_predictions, mean_predictions)
                                                   else ("not_applicable_actual_variable_count={}"
                                                         .format(variable_count) if variable_count != 1
                                                         else "failed")),
                        "max_ap": max_metrics["ap"], "mean_ap": mean_metrics["ap"],
                        "delta_ap_mean_minus_max": mean_metrics["ap"] - max_metrics["ap"] if mean_metrics["ap"] is not None else None,
                        "max_auc_roc": max_metrics["auc_roc"], "mean_auc_roc": mean_metrics["auc_roc"],
                        "delta_auc_roc_mean_minus_max": (mean_metrics["auc_roc"] - max_metrics["auc_roc"]
                                                          if mean_metrics["auc_roc"] is not None and max_metrics["auc_roc"] is not None else None),
                        "max_precision": max_metrics["precision"], "mean_precision": mean_metrics["precision"],
                        "max_recall": max_metrics["recall"], "mean_recall": mean_metrics["recall"],
                        "max_f1": max_metrics["f1"], "mean_f1": mean_metrics["f1"],
                        "max_normal_fpr": max_metrics["normal_fpr"], "mean_normal_fpr": mean_metrics["normal_fpr"],
                        "max_tp": max_metrics["tp"], "mean_tp": mean_metrics["tp"],
                        "max_fp": max_metrics["fp"], "mean_fp": mean_metrics["fp"],
                        "max_fn": max_metrics["fn"], "mean_fn": mean_metrics["fn"],
                        "max_tn": max_metrics["tn"], "mean_tn": mean_metrics["tn"],
                        "max_segments_hit": max_segments["hit"], "max_segments_scored": max_segments["scored"],
                        "max_segments_total": max_segments["total"], "mean_segments_hit": mean_segments["hit"],
                        "mean_segments_scored": mean_segments["scored"], "mean_segments_total": mean_segments["total"],
                        "segment_delta_hit_mean_minus_max": mean_segments["hit"] - max_segments["hit"],
                        "max_contributing_variables_fp_tp": max_var_summary})
            if variable_count == 1 and row["single_variable_check"] != "passed":
                raise ValueError("single-variable max/mean consistency check failed")
            successful += 1
        except Exception as error:
            failed += 1
            row.update({"status": "invalid", "mean_status": "not_attempted",
                        "reason": "aggregation comparison failed: {}: {}".format(type(error).__name__, error)})
        rows.append(row)
    return rows, successful, failed


def _affiliation_for_bundle(bundle, test_labels, test_global, test_segments):
    """Calculate affiliation scores on valid, continuous source-axis intervals."""
    labels = np.asarray(test_labels)
    global_axis = np.asarray(test_global)
    segment_axis = np.asarray(test_segments)
    if (labels.ndim != 1 or global_axis.shape != labels.shape or segment_axis.shape != labels.shape
            or labels.dtype.kind not in "iu" or not np.isin(labels, (0, 1)).all()
            or global_axis.dtype.kind not in "iu"):
        raise ValueError("validated test label/coordinate/segment axes are invalid")

    valid = np.asarray(bundle["valid_mask"])
    local = np.asarray(bundle["local_indices"])
    pred = np.asarray(bundle["predictions"])
    if valid.dtype.kind != "b" or valid.shape != labels.shape:
        raise ValueError("validated scoring mask does not align to full test axis")
    if (local.ndim != 1 or local.dtype.kind not in "iu" or pred.shape != local.shape
            or pred.dtype.kind not in "iu" or not np.isin(pred, (0, 1)).all()
            or np.any(local < 0) or np.any(local >= len(labels))):
        raise ValueError("validated prediction coordinates are invalid")
    if not np.array_equal(np.flatnonzero(valid), local.astype(np.int64, copy=False)):
        raise ValueError("prediction coordinates differ from the valid scoring mask")

    predicted_axis = np.zeros(len(labels), dtype=np.int8)
    predicted_axis[local] = pred.astype(np.int8, copy=False)
    blocks = []
    start = None
    previous = None
    for idx in range(len(labels)):
        if not valid[idx]:
            if start is not None:
                blocks.append((start, idx))
                start = None
            previous = None
            continue
        if (start is None or previous is None or idx != previous + 1
                or int(global_axis[idx]) != int(global_axis[previous]) + 1
                or segment_axis[idx] != segment_axis[previous]):
            if start is not None:
                blocks.append((start, idx))
            start = idx
        previous = idx
    if start is not None:
        blocks.append((start, len(labels)))

    records = []
    p_values, r_values, f_values = [], [], []
    for left, right in blocks:
        truth = labels[left:right].astype(np.int8, copy=False)
        guess = predicted_axis[left:right]
        gt_events = convert_vector_to_events(truth)
        pred_events = convert_vector_to_events(guess)
        record = {"local_start": left, "local_end_exclusive": right,
                  "global_start": int(global_axis[left]),
                  "global_end_exclusive": int(global_axis[right - 1]) + 1,
                  "segment_id": int(segment_axis[left]),
                  "ground_truth_events": len(gt_events), "predicted_events": len(pred_events),
                  "precision": None, "recall": None, "f": None, "undefined_reasons": []}
        if not gt_events:
            record["undefined_reasons"].append("no_ground_truth_event")
            records.append(record)
            continue
        metrics = pr_from_events(pred_events, gt_events, (0, len(truth)))
        segment_affiliation = affiliation_f_segment_macro(truth, guess)
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        if np.isfinite(precision):
            record["precision"] = precision
            p_values.append(precision)
        else:
            record["undefined_reasons"].append("precision_undefined_no_predicted_event")
        if np.isfinite(recall):
            record["recall"] = recall
            r_values.append(recall)
        else:
            record["undefined_reasons"].append("recall_non_finite")
        f_value = segment_affiliation["affiliation_f_segment_macro"]
        if f_value is not None:
            record["f"] = float(f_value)
            f_values.append(float(f_value))
        else:
            record["undefined_reasons"].append("f_undefined_from_affiliation_segment_core")
        records.append(record)

    full_contiguous = (len(blocks) == 1 and blocks[0] == (0, len(labels))
                       and len(np.unique(segment_axis)) == 1)
    if full_contiguous and np.any(labels == 1):
        truth = labels.astype(np.int8, copy=False)
        guess = predicted_axis
        if not records[0]["predicted_events"]:
            # The official wrapper has undefined precision/F in this known case.
            return {"affiliation_precision": None,
                    "affiliation_recall": float(np.mean(r_values)) if r_values else None,
                    "affiliation_f": None,
                    "calculation_scope": "full_single_continuous_valid_interval",
                    "aggregation_method": "affiliation_f_valid_segment_macro",
                    "valid_points": int(valid.sum()), "interval_count": len(blocks),
                    "defined_interval_count": 0,
                    "undefined_reasons": ["precision_undefined_no_predicted_event",
                                          "f_undefined_no_predicted_event"],
                    "intervals": records,
                    "original_algorithm_crosscheck": "undefined",
                    "crosscheck_reason": "no_predicted_event"}
        raw = pr_from_events(convert_vector_to_events(guess),
                             convert_vector_to_events(truth), (0, len(truth)))
        raw_p, raw_r = float(raw["precision"]), float(raw["recall"])
        if np.isfinite(raw_p) and np.isfinite(raw_r) and raw_p + raw_r == 0:
            return {"affiliation_precision": raw_p, "affiliation_recall": raw_r,
                    "affiliation_f": float(np.mean(f_values)) if f_values else None,
                    "calculation_scope": "full_single_continuous_valid_interval",
                    "aggregation_method": "affiliation_f_valid_segment_macro",
                    "valid_points": int(valid.sum()), "interval_count": len(blocks),
                    "defined_interval_count": len(f_values),
                    "undefined_reasons": ["original_f_undefined_zero_precision_recall"],
                    "intervals": records,
                    "original_algorithm_crosscheck": "undefined",
                    "crosscheck_reason": "original_f_zero_precision_recall"}
        p = float(original_affiliation_precision(truth, guess))
        r = float(original_affiliation_recall(truth, guess))
        f = float(original_affiliation_f(truth, guess))
        if (not np.isclose(p, raw["precision"], rtol=1e-12, atol=1e-12)
                or not np.isclose(r, raw["recall"], rtol=1e-12, atol=1e-12)
                or not np.isclose(f, 2 * p * r / (p + r), rtol=1e-12, atol=1e-12)):
            raise ValueError("single-interval affiliation differs from original implementation")
        return {"affiliation_precision": p, "affiliation_recall": r,
                "affiliation_f": f, "calculation_scope": "full_single_continuous_valid_interval",
                "aggregation_method": "original_affiliation_f", "valid_points": int(valid.sum()),
                "interval_count": len(blocks), "defined_interval_count": 1,
                "undefined_reasons": [], "intervals": records,
                "original_algorithm_crosscheck": "passed"}

    reasons = sorted({reason for record in records for reason in record["undefined_reasons"]})
    if not blocks:
        reasons.append("no_valid_scoring_points")
    return {"affiliation_precision": float(np.mean(p_values)) if p_values else None,
            "affiliation_recall": float(np.mean(r_values)) if r_values else None,
            "affiliation_f": float(np.mean(f_values)) if f_values else None,
            "calculation_scope": "valid_continuous_intervals",
            "aggregation_method": "affiliation_f_valid_segment_macro",
            "valid_points": int(valid.sum()), "interval_count": len(blocks),
            "defined_interval_count": len(f_values), "undefined_reasons": reasons,
            "intervals": records,
            "original_algorithm_crosscheck": "not_applicable_multiple_or_partial_intervals"}


_AFFILIATION_PLAN_STATUSES = {
    "complete", "reused", "failed", "pending", "planned/pending",
    "blocked", "running", "conflict", "invalid",
}


def _validate_affiliation_plan(plan):
    """Validate the declared task scope as a complete nine-combination matrix."""
    if not isinstance(plan, dict) or plan.get("mode") != "first_round":
        raise ValueError("affiliation plan must be a first_round plan object")
    if not isinstance(plan.get("batch_id"), str) or not plan["batch_id"]:
        raise ValueError("affiliation plan must declare a non-empty batch_id")
    if type(plan.get("head_seed")) is not int or plan["head_seed"] != 2021:
        raise ValueError("affiliation plan must declare fixed head_seed=2021")
    tasks = plan.get("tasks")
    if (not isinstance(tasks, list) or not tasks
            or any(not isinstance(task, str) or task not in TASKS for task in tasks)
            or len(set(tasks)) != len(tasks)):
        raise ValueError("affiliation plan tasks must explicitly declare unique known datasets")
    if plan.get("seeds") != [2021]:
        raise ValueError("affiliation plan must declare seeds=[2021]")
    expected = {(task, combo, 2021) for task in tasks for combo, _ in COMBINATIONS}
    declared_positions = plan.get("positions")
    if type(declared_positions) is not int or declared_positions != len(expected):
        raise ValueError("affiliation plan positions must equal its declared task scope ({})".format(
            len(expected)))
    planned_rows = plan.get("rows")
    if not isinstance(planned_rows, list):
        raise ValueError("affiliation plan rows must be a list")
    observed = []
    unknown_statuses = []
    incomplete = []
    for index, row in enumerate(planned_rows):
        if not isinstance(row, dict):
            raise ValueError("affiliation plan row {} must be an object".format(index))
        dataset, combination, seed = row.get("dataset"), row.get("combination"), row.get("head_seed")
        if type(seed) is not int:
            raise ValueError("affiliation plan row {} has invalid head_seed".format(index))
        if seed != 2021:
            raise ValueError("affiliation plan row {} must use head_seed=2021".format(index))
        if not isinstance(dataset, str) or not isinstance(combination, str):
            raise ValueError("affiliation plan row {} has invalid dataset/combination".format(index))
        key = (dataset, combination, seed)
        observed.append(key)
        status = row.get("status")
        if not isinstance(status, str) or status not in _AFFILIATION_PLAN_STATUSES:
            unknown_statuses.append("{}: {}".format(key, status))
        elif status in {"pending", "planned/pending", "blocked", "running", "conflict", "invalid"}:
            incomplete.append("{}/{}/{}={}".format(dataset, combination, seed, status))
    if len(observed) != len(set(observed)):
        duplicates = sorted(key for key in set(observed) if observed.count(key) > 1)
        raise ValueError("affiliation plan contains duplicate positions: {}".format(duplicates))
    extra = sorted(set(observed) - expected)
    missing = sorted(expected - set(observed))
    if extra or missing or len(planned_rows) != len(expected):
        raise ValueError("affiliation plan scope mismatch: missing={}, extra={}".format(missing, extra))
    if unknown_statuses:
        raise ValueError("affiliation plan has unknown statuses: {}".format(unknown_statuses))
    if incomplete:
        raise ValueError("affiliation plan is incomplete; states retained: {}".format(
            ", ".join(incomplete)))
    source_records = plan.get("source_records")
    if not isinstance(source_records, dict) or set(source_records) != set(tasks):
        raise ValueError("affiliation plan source_records must match its declared task scope")
    failed = [row for row in planned_rows if row["status"] == "failed"]
    return {"tasks": tasks, "expected_positions": len(expected), "failed_rows": failed}


def affiliation_only(plan_path):
    """Revalidate each plan cell and add original max-threshold affiliation metrics."""
    plan_path = Path(plan_path).resolve()
    plan = _json(plan_path)
    _validate_affiliation_plan(plan)
    checked_runs = {}
    rows = []
    for planned in plan.get("rows", []):
        dataset = planned.get("dataset", "")
        combination = planned.get("combination", "")
        seed = planned["head_seed"]
        status = planned["status"]
        row = {"dataset": dataset, "combination": combination, "head_seed": seed,
               "source_run_id": planned.get("source_run_id"), "original_status": status,
               "metric_status": "not_attempted", "threshold": None,
               "calculation_scope": None, "aggregation_method": None, "valid_points": None,
               "affiliation_precision": None, "affiliation_recall": None,
               "affiliation_f": None, "undefined_or_failure_reason": planned.get("reason", "")}
        if status == "failed":
            row["metric_status"] = "original_failed"
            rows.append(row)
            continue
        if status not in {"complete", "reused"}:
            raise AssertionError("nonterminal plan status escaped preflight: {}".format(status))
        try:
            run_dir = Path(planned["candidate_or_output_path"]).resolve()
            source_dir = Path(plan["source_records"][dataset]["path"]).resolve()
            evaluation, bundle = _evaluation(run_dir, source_dir, dataset, combination, seed, checked_runs)
            with _npz(source_dir / "analysis_inputs.npz") as inputs:
                test_labels = np.asarray(inputs["test_labels"])
                test_global = np.asarray(inputs["test_global_indices"])
                test_segments = np.asarray(inputs["test_segment_ids"])
            result = _affiliation_for_bundle(bundle, test_labels, test_global, test_segments)
            row.update({"metric_status": "passed", "threshold": float(evaluation["threshold"]),
                        "undefined_or_failure_reason": ";".join(result["undefined_reasons"]),
                        **result})
        except Exception as error:
            row.update({"metric_status": "invalid",
                        "undefined_or_failure_reason": "{}: {}".format(type(error).__name__, error)})
        rows.append(row)
    return rows


def _replace_readme_section(existing, marker, replacement):
    """Replace one level-two section while preserving all surrounding content."""
    lines = existing.splitlines()
    matches = [index for index, line in enumerate(lines) if line.strip() == marker]
    replacement_lines = replacement.rstrip().splitlines()
    if len(matches) > 1:
        raise ValueError("README contains duplicate section marker: {}".format(marker))
    if matches:
        start = matches[0]
        end = next((index for index in range(start + 1, len(lines))
                    if lines[index].startswith("## ")), len(lines))
        lines = lines[:start] + replacement_lines + lines[end:]
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(replacement_lines)
    return "\n".join(lines).rstrip() + "\n"


def _update_affiliation_readme(batch_dir, rows):
    batch_dir = Path(batch_dir)
    readme = batch_dir / "README.md"
    existing = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    marker = "## 正式 max 的 Aff-F 补算"
    by_key = {(row["dataset"], row["combination"], int(row["head_seed"])): row for row in rows}
    datasets = list(TASKS)
    combinations = [name for name, _ in COMBINATIONS]
    lines = [marker, "",
             "本节使用原正式 max 分数、保存阈值与预测；逐格调用原生 `_evaluation()` 验证源/头、分数、标签、坐标及有效点绑定。没有重训、拟合、重新校准或修改预测。历史审查 ZIP 不包含本节或 `affiliation_results.csv`。", "",
             "Affiliation Precision/Recall/F 使用仓库原始 event-affiliation 核心算法。评价仅覆盖有效预测点；连续有效区间依据有效掩码、全局坐标连续性及冻结段 ID 切分。单一完整连续区间报告原始 `affiliation_precision`、`affiliation_recall`、`affiliation_f` 并与原实现交叉核对；多个/部分区间的 F 指标命名为 `affiliation_f_valid_segment_macro`（仅对该指标有定义的区间取算术宏平均），Precision/Recall 同样按有定义区间分别作宏平均。没有真实异常事件时相应指标未定义；没有预测事件时 Precision/F 可未定义而 Recall 可合法为 0。未定义值留空并保留原因，不替换为零。", "",
             "阈值取每个组合原 `calibration.npz` 的 max 评分校准阈值，预测保持严格 `score > threshold`。无效点不补值、不压缩跨缺口，不作 point adjustment。", "",
             "公平比较需要对齐阈值选择协议、评分区间、边界处理和汇总方式；不同方法各自使用的数值阈值不必相同。本批次是冻结端点有效区间协议；原始 CATCH 文献/历史结果的窗口端点与区间汇总口径未由本次补算证明一致，因此仅并列作为协议参照，不据此宣称胜出。", "",
             "### 八个 Full 的 Affiliation 指标", "",
             "| Dataset | Precision | Recall | F | 状态 |", "|---|---:|---:|---:|---|"]
    for dataset in datasets:
        row = by_key.get((dataset, "FULL", 2021), {})
        def fmt(value):
            return "" if value is None else "{:.6f}".format(float(value))
        lines.append("| {} | {} | {} | {} | {} |".format(
            dataset, fmt(row.get("affiliation_precision")), fmt(row.get("affiliation_recall")),
            fmt(row.get("affiliation_f")), row.get("metric_status", "missing")))
    lines += ["", "### 八数据集九组合 Affiliation F", "",
              "| Dataset | " + " | ".join(combinations) + " |",
              "|---|" + "---:|" * len(combinations)]
    for dataset in datasets:
        values = []
        for combination in combinations:
            row = by_key.get((dataset, combination, 2021), {})
            value = row.get("affiliation_f")
            values.append("" if value is None else "{:.6f}".format(float(value)))
        lines.append("| {} | {} |".format(dataset, " | ".join(values)))
    lines += ["", "### CATCH 论文 Aff-F 参照（非同协议复现）", "",
              "| Dataset | CATCH Table 2 Aff-F | 本批 FULL Aff-F |", "|---|---:|---:|"]
    catch_reference = {"PSM": 0.859, "MSL": 0.740, "GECCO": 0.908,
                       "Genesis": 0.896, "NYC": 0.994, "CalIt2": 0.835,
                       "Creditcard": 0.750, "CICIDS": 0.787}
    for dataset in datasets:
        row = by_key.get((dataset, "FULL", 2021), {})
        value = row.get("affiliation_f")
        rendered = "" if value is None else "{:.6f}".format(float(value))
        lines.append("| {} | {:.3f} | {} |".format(dataset, catch_reference[dataset], rendered))
    lines += ["", "CATCH 数值引自 Wu et al., *CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching*, ICLR 2025, Table 2。论文值按其原始评估协议报告；本批次使用本地 Full 源冻结的测试端点、有效掩码和分段边界。窗口端点、无效区间与分段汇总未被证明相同，因此仅作并列参照，不据此宣称统一口径下胜出。"]
    _atomic_text(readme, _replace_readme_section(existing, marker, "\n".join(lines)))


def _update_aggregation_readme(batch_dir, rows, successful, failed):
    readme = Path(batch_dir) / "README.md"
    existing = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    marker = "## 变量评分聚合对照"
    passed = [row for row in rows if row.get("status") == "passed"]
    lines = [marker, "", "本节由 `--aggregation-comparison` 只读生成；未改写原 max 分数、校准、manifest 或模型。",
             "通过对照格：{}/{}；失败/未验证格：{}。mean 只使用各格保存的 `calibration_nll.npy`、原 center/scale 和原 quantile。".format(
                 successful, len(rows), failed), "",
             "max 复现要求保存 test NLL 聚合、原阈值、预测和指定 Full 的有效坐标轴全部一致。变量贡献按原 max 预测的 FP/TP 点统计，并列最大值按份额平分；该统计只描述变量评分聚合，不证明编码器分支因果贡献。", ""]
    nyc_dims = sorted({int(row["variable_count"]) for row in passed
                       if row.get("dataset") == "NYC" and row.get("variable_count")})
    if nyc_dims:
        lines.append("NYC 实际保存的逐变量 NLL 宽度为 {}；因此本批次不是单变量特例，max 与 mean 不要求相等。若变量数为 1，CSV 的 `single_variable_check` 要求分数、阈值和预测逐项一致。".format(
            ",".join(str(value) for value in nyc_dims)))
        lines.append("")
    def value(row, key):
        value = row.get(key)
        return None if value in (None, "") else float(value)

    def fmt(value):
        return "NA" if value is None else "{:+.4f}".format(value)

    def delta(table, left, right, metric):
        a, b = table.get(left), table.get(right)
        if not a or not b:
            return None
        av, bv = value(a, metric), value(b, metric)
        return None if av is None or bv is None else bv - av

    for dataset in TASKS:
        subset = [row for row in passed if row.get("dataset") == dataset]
        if not subset:
            continue
        table = {row.get("combination"): row for row in subset}
        full, only_s = table.get("FULL"), table.get("S")
        if full and only_s:
            lines.append("- {}（参与 {}/9 格）：Full AP max/mean {}/{}，S-only AP max/mean {}/{}；Full mean-max AP {}、ROC-AUC {}。".format(
                dataset, len(subset), fmt(value(full, "max_ap")), fmt(value(full, "mean_ap")),
                fmt(value(only_s, "max_ap")), fmt(value(only_s, "mean_ap")),
                fmt(value(full, "delta_ap_mean_minus_max")), fmt(value(full, "delta_auc_roc_mean_minus_max"))))
            lines.append("  添加 E AP max/mean {}/{}、AUC max/mean {}/{}；添加 P AP max/mean {}/{}、AUC max/mean {}/{}；删除 E AP max/mean {}/{}、删除 P {}/{}、删除 R {}/{}。".format(
                fmt(delta(table, "S", "SE", "max_ap")), fmt(delta(table, "S", "SE", "mean_ap")),
                fmt(delta(table, "S", "SE", "max_auc_roc")), fmt(delta(table, "S", "SE", "mean_auc_roc")),
                fmt(delta(table, "S", "SP", "max_ap")), fmt(delta(table, "S", "SP", "mean_ap")),
                fmt(delta(table, "S", "SP", "max_auc_roc")), fmt(delta(table, "S", "SP", "mean_auc_roc")),
                fmt(delta(table, "WITHOUT_E", "FULL", "max_ap")), fmt(delta(table, "WITHOUT_E", "FULL", "mean_ap")),
                fmt(delta(table, "WITHOUT_P", "FULL", "max_ap")), fmt(delta(table, "WITHOUT_P", "FULL", "mean_ap")),
                fmt(delta(table, "WITHOUT_R", "FULL", "max_ap")), fmt(delta(table, "WITHOUT_R", "FULL", "mean_ap"))))
    _atomic_text(readme, _replace_readme_section(existing, marker, "\n".join(lines)))


def _write_bundle(path, bundles):
    temp = Path(path).with_name(Path(path).name + ".tmp")
    np.savez_compressed(temp, **{key + "_" + name: value for key, bundle in bundles.items()
                                 for name, value in bundle.items()})
    # numpy appends .npz when the path lacks that suffix.
    generated = Path(str(temp) + ".npz")
    generated.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="plan.json from run_mnrf_branch_ablation.py")
    parser.add_argument("--aggregation-comparison", action="store_true",
                        help="read-only max versus mean variable-score aggregation")
    parser.add_argument("--affiliation-only", action="store_true",
                        help="revalidate fixed max outputs and calculate affiliation metrics only")
    args = parser.parse_args()
    if args.aggregation_comparison and args.affiliation_only:
        parser.error("--aggregation-comparison and --affiliation-only are mutually exclusive")
    plan_path = Path(args.plan).resolve()
    if args.aggregation_comparison:
        rows, successful, failed = aggregation_comparison(plan_path)
        _atomic_csv(plan_path.parent / "aggregation_comparison.csv", rows,
                    lineterminator="\n")
        _update_aggregation_readme(plan_path.parent, rows, successful, failed)
        print("aggregation_csv={}".format(plan_path.parent / "aggregation_comparison.csv"))
        original_skipped = sum(row.get("original_status") not in {"complete", "reused"}
                               for row in rows)
        invalid = sum(row.get("status") == "invalid" for row in rows)
        print("aggregation_passed={} aggregation_original_skipped={} aggregation_invalid={} aggregation_failed={} max_reproduction=checked".format(
            successful, original_skipped, invalid, failed))
        # The input batch intentionally retains the pre-existing failed cell
        # (CICIDS WITHOUT_E).  It is reported as an original skip; only a new
        # invalid cell means this read-only comparison failed.
        return 1 if invalid else 0
    if args.affiliation_only:
        try:
            rows = affiliation_only(plan_path)
        except Exception as error:
            print("affiliation plan rejected: {}: {}".format(type(error).__name__, error),
                  file=sys.stderr)
            return 2
        output = plan_path.parent / "affiliation_results.csv"
        _atomic_csv(output, rows, lineterminator="\n")
        _update_affiliation_readme(plan_path.parent, rows)
        passed = sum(row["metric_status"] == "passed" for row in rows)
        original_failed = sum(row["metric_status"] == "original_failed" for row in rows)
        invalid = sum(row["metric_status"] == "invalid" for row in rows)
        defined = sum(row["metric_status"] == "passed" and row["affiliation_f"] is not None
                      for row in rows)
        undefined = sum(row["metric_status"] == "passed" and row["affiliation_f"] is None
                        for row in rows)
        print("affiliation_csv={}".format(output))
        print("affiliation_validated={} affiliation_f_defined={} affiliation_f_undefined={} original_failed={} invalid={}".format(
            passed, defined, undefined, original_failed, invalid))
        print("affiliation_matrix_positions={}/{} matrix_complete={}".format(
            len(rows) - original_failed, len(rows), original_failed == 0 and invalid == 0))
        return 1 if invalid else 0
    rows, contributions, paired, bundles = summarize(plan_path)
    batch_dir = plan_path.parent
    plan = _json(plan_path)
    _atomic_csv(batch_dir / "long_table.csv", rows)
    _atomic_csv(batch_dir / "branch_contributions.csv", contributions)
    _atomic_csv(batch_dir / "paired_changes.csv", paired)
    # Always regenerate this table from this invocation's validated cells; an
    # older table must never be carried into a new partial/failed summary.
    evals = {}
    for row in rows:
        if row.get("status") in {"complete", "reused"}:
            evals[(row.get("dataset"), row.get("combination"), int(row.get("head_seed")))] = row
    _atomic_csv(batch_dir / "cross_dataset_summary.csv",
                cross_dataset_summary(plan["tasks"], evals))
    for obsolete in ("repeat_summary.csv", "contribution_repeat_summary.csv"):
        # Do not carry stale repeat tables into a regenerated single-seed batch.
        (batch_dir / obsolete).unlink(missing_ok=True)
    _atomic_json(batch_dir / "change_cases.json", {
        "case_limit_per_category": 8,
        "ordering": "increasing global coordinate / source segment key",
        "pairs": [{"dataset": row["dataset"], "head_seed": row["head_seed"],
                   "base": row["base"], "changed": row["changed"],
                   "counts": {key: row[key] for key in ("added_tp", "lost_tp", "added_fp", "removed_fp",
                               "added_segment_hits", "lost_segment_hits")},
                   "cases": json.loads(row["change_cases"])} for row in paired]})
    point_path = batch_dir / "paired_point_scores.npz"
    if bundles:
        _write_bundle(point_path, bundles)
    else:
        point_path.unlink(missing_ok=True)
    report = completion_counts(plan["tasks"], plan["seeds"], rows)
    report.update({"batch_id": plan.get("batch_id"),
              "positions_present": len(plan.get("rows", [])),
              "interpretation": "Frozen encoder representations with independently refit GMM heads; not encoder-seed variance or causal proof.",
              "contribution_sign": "added = AP(S+branch)-AP(S); deleted = AP(FULL)-AP(FULL-minus-branch); positive deletion means removal hurts."})
    _atomic_json(batch_dir / "summary.json", report)
    _atomic_json(batch_dir / "status.json", report)
    _atomic_text(batch_dir / "README.md", """# MNRF Frozen-View Ablation Summary

Batch: `{batch}`

Batch status: `{status}`. Coverage: {datasets_complete}/{datasets_requested} complete datasets, {complete_or_reused}/{positions_expected} validated cells. Missing combinations are not filled with zero. `long_table.csv` is independently recalculated from each saved NLL and matched source labels. `paired_changes.csv` contains same-task, same-seed point and anomaly-segment additions/losses; bounded examples are in `change_cases.json`. `paired_point_scores.npz` contains only cells validated in this summary invocation.

For E/P/R, addition is `metric(S+branch)-metric(S)` and deletion is `metric(FULL)-metric(FULL-minus-branch)`. Positive deletion means removal hurt. S has deletion only. AP is average precision, not Aff-F. Segment hits mean at least one predicted point and are diagnostic only.

The head seed is fixed to 2021 for this matrix; it is a GMM-head seed on frozen features, not an encoder seed. These results do not prove causal representation contribution, paper-protocol equivalence, or superiority to CATCH. Full caches, encoders and GMM files stay in their server source runs.
""".format(batch=report["batch_id"], status=report["batch_status"],
           datasets_complete=report["datasets_complete"], datasets_requested=report["datasets_requested"],
           complete_or_reused=report["complete_or_reused"], positions_expected=report["positions_expected"]))
    review_root = ROOT / "result/review/MNRF-ViewProbe"
    review_root.mkdir(parents=True, exist_ok=True)
    archive = review_root / ("MNRF_BRANCH_ABLATION_{}.zip".format(report["batch_id"]))
    temporary_archive = archive.with_name(archive.name + ".tmp-{}".format(os.getpid()))
    with zipfile.ZipFile(str(temporary_archive), "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name in ("plan.json", "matrix.csv", "long_table.csv", "branch_contributions.csv",
                     "paired_changes.csv", "cross_dataset_summary.csv", "change_cases.json",
                     "summary.json", "README.md", "paired_point_scores.npz"):
            candidate = batch_dir / name
            if candidate.is_file():
                output.write(str(candidate), arcname=name)
    if archive.exists():
        try:
            with zipfile.ZipFile(str(archive), "r") as old:
                prior = json.loads(old.read("plan.json").decode("utf-8"))
            if prior.get("batch_id") != report["batch_id"]:
                raise ValueError("archive belongs to another batch")
        except Exception as error:
            temporary_archive.unlink(missing_ok=True)
            raise ValueError("refusing to replace unrecognized review archive: {}".format(error))
    os.replace(str(temporary_archive), str(archive))
    print("summary_dir={}".format(batch_dir))
    print("review_package={}".format(archive))
    print("batch_status={} datasets_complete={}/{} datasets_with_results={} positions={}/{} complete={} reused={} pending={} running={} failed={} blocked={} conflict={} invalid={}".format(
        report["batch_status"], report["datasets_complete"], report["datasets_requested"],
        report["datasets_with_results"], report["positions_present"], report["positions_expected"],
        report["complete"], report["reused"], report["pending"], report["running"],
        report["failed"], report["blocked"], report["conflict"], report["invalid"]))
    # A plan with pending cells is an intentional, usable partial inventory;
    # blocked/conflicting cells remain non-success so callers cannot mistake
    # an invalid source for a merely unrun combination.
    return 1 if report["batch_status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
