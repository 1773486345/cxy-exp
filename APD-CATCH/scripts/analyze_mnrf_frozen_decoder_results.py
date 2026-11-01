#!/usr/bin/env python
"""Analyze saved frozen E/R decoder diagnostics without model execution.

The script consumes only the diagnostic NPZ/CSV files and the bound source
axis. It does not load a model, run an encoder, fit a head, or recalibrate a
source run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("PSM", "MSL", "GECCO", "Genesis", "NYC", "CalIt2", "Creditcard", "CICIDS")
DEFAULT_ROOT = ROOT / "result/analysis/MNRF-FrozenDecoder"
DEFAULT_OUTPUT = ROOT / "result/analysis/MNRF-FrozenDecoder-Analysis-20261101"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _finite_quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"q01": None, "q50": None, "q99": None}
    q01, q50, q99 = np.quantile(values, [0.01, 0.5, 0.99])
    return {"q01": float(q01), "q50": float(q50), "q99": float(q99)}


def _metric_row(rows, view, variant, score):
    for row in rows:
        if row.get("view") == view and row.get("variant") == variant and row.get("score") == score:
            return row
    raise ValueError(f"missing metric row: {view}/{variant}/{score}")


def _float_or_none(value):
    if value in (None, "", "null", "None"):
        return None
    return float(value)


def _safe_metric(values, labels, metric):
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int8)
    values = np.asarray(values, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return None
    if metric == "auc_roc":
        return float(roc_auc_score(labels, values))
    return float(average_precision_score(labels, values))


def _ranking_pair(values, labels):
    """Compute ROC/AP with one stable score sort for each variable."""
    from sklearn.metrics._ranking import _binary_clf_curve

    labels = np.asarray(labels, dtype=np.int8)
    values = np.asarray(values, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return None, None
    fps, tps, _ = _binary_clf_curve(labels, values, pos_label=1)
    positives = float(tps[-1])
    negatives = float(np.sum(labels == 0))
    if positives <= 0 or negatives <= 0:
        return None, None
    recall = tps / positives
    precision = tps / np.maximum(tps + fps, 1.0)
    # ROC integration must include the classifier's (0, 0) origin.  Without
    # it, a tied score ordering can report AUC=0 for a genuinely random
    # ranking (the correct value is 0.5).
    auc = float(np.trapz(np.r_[0.0, tps / positives],
                         np.r_[0.0, fps / negatives]))
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return auc, ap


def _ranking_matrix(values, labels):
    """Compute per-column ROC/AP after one vectorized stable sort."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if len(np.unique(labels)) < 2:
        return [(None, None)] * values.shape[1]
    order = np.argsort(-values, axis=0, kind="mergesort")
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_labels = labels[order]
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    result = []
    for column in range(values.shape[1]):
        scores = sorted_values[:, column]
        y = sorted_labels[:, column].astype(np.float64)
        distinct = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
        tps = np.cumsum(y)[distinct]
        fps = np.cumsum(1.0 - y)[distinct]
        recall = tps / positives
        precision = tps / np.maximum(tps + fps, 1.0)
        result.append((float(np.trapz(np.r_[0.0, tps / positives],
                                      np.r_[0.0, fps / negatives])),
                       float(np.sum(np.diff(np.r_[0.0, recall]) * precision))))
    return result


def _source_axis(source_run):
    with np.load(source_run / "analysis_inputs.npz", allow_pickle=False) as axis:
        required = ("all_global_indices", "all_labels", "all_time_values_ns", "all_segment_ids",
                    "test_global_indices", "test_labels", "test_time_values_ns", "test_segment_ids")
        if any(name not in axis.files for name in required):
            raise ValueError(f"source analysis axis is incomplete: {source_run}")
        values = {name: np.asarray(axis[name]) for name in required}
    if any(values[name].ndim != 1 for name in required):
        raise ValueError(f"source analysis axis is not one-dimensional: {source_run}")
    full = values["all_global_indices"]
    test = values["test_global_indices"]
    positions = np.searchsorted(full, test)
    if (np.any(positions >= len(full)) or not np.array_equal(full[positions], test)
            or not np.array_equal(values["test_labels"], values["all_labels"][positions])
            or not np.array_equal(values["test_time_values_ns"], values["all_time_values_ns"][positions])
            or not np.array_equal(values["test_segment_ids"], values["all_segment_ids"][positions])):
        raise ValueError(f"source test axis does not map to its complete axis: {source_run}")
    return values


def _source_run_for(manifest):
    source = Path(manifest["source_run"]).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"registered source is absent: {source}")
    source_manifest = _read_json(source / "probe_manifest.json")
    if source_manifest.get("stage_status") != "complete" or source_manifest.get("head_status") != "complete":
        raise ValueError(f"source is not complete: {source}")
    if source.name != Path(manifest["source_run"]).name:
        raise ValueError("diagnostic source identity is malformed")
    return source, source_manifest


def _finite_array_stats(path, expected_shape, label):
    """Inspect a saved float32 array in bounded chunks without rewriting it."""
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != tuple(expected_shape):
        raise ValueError(f"{label} shape {array.shape} != {expected_shape}")
    if array.dtype != np.dtype(np.float32):
        raise ValueError(f"{label} dtype {array.dtype} != float32")
    finite = 0
    minimum = math.inf
    maximum = -math.inf
    for start in range(0, int(array.shape[0]), 4096):
        block = np.asarray(array[start:start + 4096])
        mask = np.isfinite(block)
        finite += int(mask.sum())
        if mask.any():
            values = block[mask]
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
    total = int(array.size)
    if finite != total:
        raise ValueError(f"{label} contains non-finite values")
    return {"path": str(Path(path)), "shape": list(array.shape), "dtype": str(array.dtype),
            "finite_values": finite, "nbytes": int(array.nbytes),
            "min": minimum, "max": maximum}


def _source_reference(source_run, source_manifest, axis, metrics, names):
    """Validate and record the Full GMM artifacts used by this diagnostic."""
    test_nll_path = source_run / "results/test_nll.npy"
    calibration_nll_path = source_run / "heads/calibration_nll.npy"
    calibration_path = source_run / "heads/calibration.npz"
    aligned_path = source_run / "results/aligned_scores.npz"
    for path in (test_nll_path, calibration_nll_path, calibration_path, aligned_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing Full reference artifact: {path}")
    expected_test_points = int(source_manifest.get("test_endpoint_count", len(axis["test_global_indices"])))
    test_nll = _finite_array_stats(test_nll_path,
                                   (expected_test_points, len(names)),
                                   "Full test_nll")
    calibration_nll = np.load(calibration_nll_path, mmap_mode="r", allow_pickle=False)
    if calibration_nll.ndim != 2 or calibration_nll.shape[1] != len(names):
        raise ValueError(f"Full calibration_nll shape is not [N,C]: {calibration_nll.shape}")
    calibration_info = _finite_array_stats(calibration_nll_path, calibration_nll.shape,
                                            "Full calibration_nll")
    with np.load(calibration_path, allow_pickle=False) as calibration:
        required = ("center", "scale", "threshold", "quantile")
        if any(key not in calibration.files for key in required):
            raise ValueError(f"Full calibration statistics are incomplete: {calibration_path}")
        center = np.asarray(calibration["center"])
        scale = np.asarray(calibration["scale"])
        threshold = float(np.asarray(calibration["threshold"]).reshape(()))
        quantile = float(np.asarray(calibration["quantile"]).reshape(()))
    if center.shape != (len(names),) or scale.shape != center.shape or not np.isfinite(center).all():
        raise ValueError(f"Full calibration center is malformed: {calibration_path}")
    if not np.isfinite(scale).all() or np.any(scale <= 0) or not np.isfinite(threshold):
        raise ValueError(f"Full calibration scale/threshold is malformed: {calibration_path}")
    with np.load(aligned_path, allow_pickle=False) as aligned:
        required = ("scores", "predictions", "time_indices", "time_values_ns", "local_indices", "valid_mask", "actual")
        if any(key not in aligned.files for key in required):
            raise ValueError(f"Full aligned scores are incomplete: {aligned_path}")
        aligned_indices = np.asarray(aligned["time_indices"], dtype=np.int64)
        if len(aligned_indices) == 0 or np.any(np.diff(aligned_indices) <= 0):
            raise ValueError(f"Full aligned score coordinates are not ordered and unique: {aligned_path}")
        source_positions = {int(g): i for i, g in enumerate(axis["test_global_indices"])}
        if any(int(g) not in source_positions for g in aligned_indices):
            raise ValueError(f"Full aligned score coordinates differ from source axis: {aligned_path}")
        positions = np.asarray([source_positions[int(g)] for g in aligned_indices], dtype=np.int64)
        if not np.array_equal(np.asarray(aligned["actual"], dtype=np.int8), axis["test_labels"][positions].astype(np.int8)):
            raise ValueError(f"Full aligned labels differ from source axis: {aligned_path}")
        if not np.array_equal(np.asarray(aligned["time_values_ns"]), axis["test_time_values_ns"][positions]):
            raise ValueError(f"Full aligned times differ from source axis: {aligned_path}")
    full_row = _metric_row(metrics, "FULL", "saved_full_gmm_reference", "saved_original_aggregate")
    config = source_manifest.get("config", {})
    return {
        "source_run_id": source_run.name,
        "source_run": str(source_run),
        "variable_count": len(names),
        "test_points": len(axis["test_global_indices"]),
        "calibration_points": int(calibration_nll.shape[0]),
        "test_nll": test_nll,
        "calibration_nll": calibration_info,
        "calibration_threshold": threshold,
        "calibration_quantile": quantile,
        "score_mad_floor": config.get("score_mad_floor"),
        "gmm_covariance_type": config.get("gmm_covariance_type"),
        "gmm_components": config.get("gmm_components"),
        "feature_fingerprint": source_manifest.get("feature_fingerprint"),
        "source_identity": source_manifest.get("source_identity"),
        "fit_endpoint_count": source_manifest.get("gmm_fit_endpoint_count"),
        "fit_endpoint_sha256": source_manifest.get("gmm_fit_endpoints_sha256"),
        "fixed_noise_sha256": source_manifest.get("fixed_noise_sha256"),
        "input_statistics_sha256": source_manifest.get("input_statistics_sha256"),
        "cache_sha256": source_manifest.get("cache_sha256"),
        "full_metric": full_row,
    }


def _event_rows(axis):
    labels = axis["all_labels"]
    global_indices = axis["all_global_indices"]
    segments = axis["all_segment_ids"]
    test_set = set(map(int, axis["test_global_indices"]))
    events = []
    index = 0
    while index < len(labels):
        if int(labels[index]) != 1:
            index += 1
            continue
        start = index
        event_segment = int(segments[index])
        index += 1
        while index < len(labels) and int(labels[index]) == 1:
            if (int(global_indices[index]) != int(global_indices[index - 1]) + 1
                    or int(segments[index]) != event_segment):
                break
            index += 1
        points = global_indices[start:index]
        scored = [int(point) for point in points if int(point) in test_set]
        events.append({"event_index": len(events), "segment_id": event_segment,
                       "start_global": int(points[0]), "end_global": int(points[-1]),
                       "points": int(len(points)), "scored_points": int(len(scored)),
                       "scored_globals": scored})
    return events


def _max_attribution(values, center, scale, labels, score, threshold, names, task, view, score_name):
    z = (np.asarray(values, dtype=np.float64) - center[None, :]) / scale[None, :]
    rows = []
    pred = np.asarray(score > threshold, dtype=bool)
    groups = {
        "normal": labels == 0,
        "anomaly": labels == 1,
        "TP": (labels == 1) & pred,
        "FP": (labels == 0) & pred,
        "FN": (labels == 1) & ~pred,
    }
    finite = np.isfinite(z).all(axis=1)
    for population, population_mask in groups.items():
        selected = np.flatnonzero(population_mask & finite)
        if not len(selected):
            continue
        block = z[selected]
        maxima = np.max(block, axis=1)
        ties = np.isclose(block, maxima[:, None], rtol=0.0, atol=1e-10)
        shares = ties / ties.sum(axis=1, keepdims=True)
        for variable, name in enumerate(names):
            contribution = float(shares[:, variable].sum())
            rows.append({"dataset": task, "view": view, "score": score_name,
                         "population": population, "variable_index": variable,
                         "variable": name, "point_count": int(len(selected)),
                         "max_attribution_share": contribution / len(selected),
                         "max_attribution_points": contribution})
    return rows


def _distribution_rows(values, labels, names, task, view, score_name):
    rows = []
    for variable, name in enumerate(names):
        for population, mask in (("normal", labels == 0), ("anomaly", labels == 1)):
            selected = np.asarray(values)[mask, variable]
            q = _finite_quantiles(selected)
            rows.append({"dataset": task, "view": view, "score": score_name,
                         "variable_index": variable, "variable": name,
                         "population": population, "n": int(np.isfinite(selected).sum()),
                         "mean": float(np.nanmean(selected)) if np.isfinite(selected).any() else None,
                         "std": float(np.nanstd(selected)) if np.isfinite(selected).any() else None,
                         **q})
    return rows


def _variable_metric_rows(values, labels, names, task, view, score_name, aggregate):
    rows = []
    values = np.asarray(values)
    ranking = _ranking_matrix(values, labels)
    for variable, name in enumerate(names):
        column = values[:, variable]
        auc, ap = ranking[variable]
        rows.append({"dataset": task, "view": view, "score": score_name,
                     "variable_index": variable, "variable": name,
                     "n": int(len(column)), "auc_roc": auc,
                     "average_precision": ap,
                     "aggregate_auc_roc": _float_or_none(aggregate.get("auc_roc")),
                     "aggregate_average_precision": _float_or_none(aggregate.get("average_precision"))})
    return rows


def _nyc_event_rows(task, source_axis, evidence_by_view, names):
    if task != "NYC":
        return []
    test_positions = {int(g): i for i, g in enumerate(source_axis["test_global_indices"])}
    rows = []
    for event in _event_rows(source_axis):
        scored = [g for g in event["scored_globals"] if g in test_positions]
        for view, evidence in evidence_by_view.items():
            residual = evidence["residual"]
            nll = evidence["nll"]
            for position, global_index in enumerate(scored):
                local = test_positions[global_index]
                if position == 0:
                    phase = "start"
                elif position == len(scored) - 1:
                    phase = "end"
                else:
                    phase = "interior"
                rows.append({"dataset": task, "event_index": event["event_index"],
                             "segment_id": event["segment_id"], "event_start_global": event["start_global"],
                             "event_end_global": event["end_global"], "phase": phase,
                             "global_index": global_index, "view": view,
                             "target_variable": names[0], "abs_residual": float(abs(residual[local, 0])),
                             "nll": float(nll[local, 0])})
    return rows


def _nyc_event_neighborhood_rows(task, source_axis, evidence_by_view, names, k=8):
    """Summarize a fixed, boundary-safe k-point event neighborhood.

    The phase labels are based only on scored event endpoints.  Normal
    neighbors are selected from the complete axis only when they remain in
    the same source segment, have consecutive global coordinates, and have a
    saved score for the corresponding diagnostic view.
    """
    if task != "NYC":
        return []
    full_globals = np.asarray(source_axis["all_global_indices"], dtype=np.int64)
    full_labels = np.asarray(source_axis["all_labels"], dtype=np.int8)
    full_segments = np.asarray(source_axis["all_segment_ids"], dtype=np.int64)
    full_times = np.asarray(source_axis["all_time_values_ns"])
    full_pos = {int(g): i for i, g in enumerate(full_globals)}
    rows = []
    for event in _event_rows(source_axis):
        event_globals = list(event["scored_globals"])
        if not event_globals:
            continue
        before = []
        after = []
        start_pos = full_pos[event["start_global"]]
        end_pos = full_pos[event["end_global"]]
        for pos in range(start_pos - 1, -1, -1):
            if (full_labels[pos] != 0 or full_segments[pos] != event["segment_id"]
                    or pos + 1 >= len(full_globals)
                    or full_globals[pos + 1] - full_globals[pos] != 1):
                break
            before.append(int(full_globals[pos]))
            if len(before) == k:
                break
        before.reverse()
        for pos in range(end_pos + 1, len(full_globals)):
            if (full_labels[pos] != 0 or full_segments[pos] != event["segment_id"]
                    or pos == 0 or full_globals[pos] - full_globals[pos - 1] != 1):
                break
            after.append(int(full_globals[pos]))
            if len(after) == k:
                break
        for view, evidence in evidence_by_view.items():
            endpoint_globals = np.asarray(evidence["endpoints"], dtype=np.int64)
            endpoint_pos = {int(g): i for i, g in enumerate(endpoint_globals)}
            values_by_score = {
                "squared_residual": np.asarray(evidence["residual"], dtype=np.float64) ** 2,
                "nll": np.asarray(evidence["nll"], dtype=np.float64),
            }
            phase_globals = {
                "event_first8": event_globals[:k],
                "event_last8": event_globals[-k:],
                "event_interior": event_globals[k:-k] if len(event_globals) > 2 * k else [],
                "normal_before8": [g for g in before if g in endpoint_pos],
                "normal_after8": [g for g in after if g in endpoint_pos],
            }
            for score_name, values in values_by_score.items():
                for phase, globals_for_phase in phase_globals.items():
                    indices = [endpoint_pos[g] for g in globals_for_phase if g in endpoint_pos]
                    if not indices:
                        continue
                    selected = values[np.asarray(indices, dtype=np.int64), 0]
                    finite = selected[np.isfinite(selected)]
                    if not len(finite):
                        continue
                    quantiles = _finite_quantiles(finite)
                    rows.append({
                        "dataset": task, "event_index": event["event_index"],
                        "segment_id": event["segment_id"], "view": view,
                        "score": score_name, "phase": phase, "k": k,
                        "n": int(len(finite)), "mean": float(np.mean(finite)),
                        "median": float(np.median(finite)), **quantiles,
                        "global_start": int(min(globals_for_phase)),
                        "global_end": int(max(globals_for_phase)),
                        "time_start_ns": int(full_times[full_pos[min(globals_for_phase)]]),
                        "time_end_ns": int(full_times[full_pos[max(globals_for_phase)]]),
                    })
    return rows


def analyze_task(task, diagnostic_root):
    dirs = sorted((diagnostic_root / task).glob("*/"))
    if len(dirs) != 1:
        raise ValueError(f"expected one diagnostic output for {task}, found {len(dirs)}")
    out = dirs[0]
    manifest = _read_json(out / "diagnostic_manifest.json")
    source_run, source_manifest = _source_run_for(manifest)
    axis = _source_axis(source_run)
    metrics = _read_csv(out / "metrics.csv")
    if len(metrics) != 13:
        raise ValueError(f"{task} expected 13 metrics rows, found {len(metrics)}")
    names = list(manifest["variable_names"])
    evidence_by_view = {}
    distribution_rows = []
    metric_rows = []
    attribution_rows = []
    scale_rows = []
    summary_rows = []
    for view in ("E", "R"):
        with np.load(out / f"{view}_test_prediction_evidence.npz", allow_pickle=False) as evidence:
            residual = np.asarray(evidence["residual"], dtype=np.float32)
            nll = np.asarray(evidence["per_variable_nll"], dtype=np.float32)
            endpoints = np.asarray(evidence["endpoints"], dtype=np.int64)
            evidence_names = [str(x) for x in evidence["variable_names"]]
        if evidence_names != names:
            raise ValueError(f"{task}/{view} variable names differ from source")
        if residual.shape != nll.shape or residual.ndim != 2 or residual.shape[1] != len(names):
            raise ValueError(f"{task}/{view} prediction evidence has inconsistent shape")
        for score_name, values, score_file, variant in (
                ("squared_residual", residual ** 2, f"{view}_squared_residual_test_scores.npz", "decoder"),
                ("nll", nll, f"{view}_nll_test_scores.npz", "decoder")):
            with np.load(out / score_file, allow_pickle=False) as saved:
                score = np.asarray(saved["scores"], dtype=np.float64)
                labels = np.asarray(saved["labels"], dtype=np.int8)
                saved_endpoints = np.asarray(saved["global_indices"], dtype=np.int64)
                center = np.asarray(saved["center"], dtype=np.float64)
                scale = np.asarray(saved["scale"], dtype=np.float64)
                threshold = float(np.asarray(saved["threshold"]).reshape(()))
            if (not np.array_equal(endpoints, saved_endpoints) or len(score) != len(labels)
                    or len(endpoints) == 0 or np.any(np.diff(endpoints) <= 0)):
                raise ValueError(f"{task}/{view}/{score_name} coordinate binding differs")
            source_test = axis["test_global_indices"]
            source_label_map = {int(g): int(label) for g, label in zip(source_test, axis["test_labels"])}
            if any(int(g) not in source_label_map for g in saved_endpoints):
                raise ValueError(f"{task}/{view}/{score_name} contains coordinates outside source test axis")
            source_labels = np.asarray([source_label_map[int(g)] for g in saved_endpoints], dtype=np.int8)
            if not np.array_equal(labels, source_labels):
                raise ValueError(f"{task}/{view}/{score_name} labels differ from source test axis")
            if center.shape != (len(names),) or scale.shape != center.shape or np.any(scale <= 0):
                raise ValueError(f"{task}/{view}/{score_name} calibration vectors are malformed")
            row = _metric_row(metrics, view, variant, score_name)
            distribution_rows.extend(_distribution_rows(values, labels, names, task, view, score_name))
            metric_rows.extend(_variable_metric_rows(values, labels, names, task, view, score_name, row))
            attribution_rows.extend(_max_attribution(values, center, scale, labels, score, threshold,
                                                      names, task, view, score_name))
            floor = float(source_manifest.get("config", {}).get("score_mad_floor", 0.001))
            # Saved float32 scales can round the configured floor upward by a few ulps.
            floor_count = int(np.sum(scale <= floor * (1.0 + 1e-6) + 1e-9))
            scale_rows.append({"dataset": task, "view": view, "score": score_name,
                               "variable_count": len(names), "mad_floor": floor,
                               "scale_floor_active_count": floor_count,
                               "scale_floor_active_fraction": floor_count / len(names),
                               "scale_min": float(scale.min()), "scale_median": float(np.median(scale)),
                               "scale_max": float(scale.max()), "threshold": threshold,
                               "normal_points": int(np.sum(labels == 0)),
                               "anomaly_points": int(np.sum(labels == 1)),
                               "anomaly_fraction": float(np.mean(labels == 1))})
            evidence_by_view.setdefault(view, {"residual": residual, "nll": nll,
                                               "endpoints": endpoints})
            summary_rows.append({"dataset": task, "view": view, "score": score_name,
                                 "source_run_id": source_run.name, "diagnostic_run_id": out.name,
                                 "variable_count": len(names), "test_points": len(labels),
                                 "normal_points": int(np.sum(labels == 0)),
                                 "anomaly_points": int(np.sum(labels == 1)),
                                 "anomaly_fraction": float(np.mean(labels == 1)),
                                 "aggregate_auc_roc": row.get("auc_roc"),
                                 "aggregate_average_precision": row.get("average_precision"),
                                 "aggregate_precision": row.get("precision"),
                                 "aggregate_recall": row.get("recall"),
                                 "aggregate_f1": row.get("f1"),
                                 "aggregate_fpr": row.get("fpr"),
                                 "tp": row.get("tp"), "fp": row.get("fp"),
                                 "fn": row.get("fn"), "tn": row.get("tn"),
                                 "affiliation_precision": row.get("affiliation_precision_valid_segment_macro"),
                                 "affiliation_recall": row.get("affiliation_recall_valid_segment_macro"),
                                 "affiliation_f": row.get("affiliation_f_valid_segment_macro"),
                                 "affiliation_defined_segments": row.get("affiliation_defined_segments"),
                                 "affiliation_total_segments": row.get("affiliation_total_segments"),
                                 "threshold": threshold})
    nyc_rows = _nyc_event_rows(task, axis, evidence_by_view, names)
    nyc_neighborhood_rows = _nyc_event_neighborhood_rows(task, axis, evidence_by_view, names, k=8)
    full_reference = _source_reference(source_run, source_manifest, axis, metrics, names)
    reference_metrics = []
    for metric in metrics:
        reference_metrics.append({"dataset": task, "source_run_id": source_run.name,
                                  "diagnostic_run_id": out.name, **metric})
    return {"summary": summary_rows, "reference_metrics": reference_metrics,
            "distribution": distribution_rows, "variable_metrics": metric_rows,
            "attribution": attribution_rows, "scale": scale_rows, "nyc_events": nyc_rows,
            "nyc_neighborhood": nyc_neighborhood_rows,
            "full_reference": full_reference,
            "source": {"task": task, "source_run_id": source_run.name,
                        "source_run": str(source_run), "diagnostic_run_id": out.name,
                        "variable_count": len(names), "variable_names": names,
                        "source_feature_fingerprint": source_manifest.get("feature_fingerprint"),
                        "diagnostic_manifest": str(out / "diagnostic_manifest.json")}}


def _top_attribution(rows, task, view, score, population, limit=5):
    selected = [r for r in rows if r["dataset"] == task and r["view"] == view
                and r["score"] == score and r["population"] == population]
    selected.sort(key=lambda r: (-float(r["max_attribution_share"]), int(r["variable_index"])))
    return selected[:limit]


def _creditcard_note(summary, variable_metrics, attribution):
    rows = [r for r in summary if r["dataset"] == "Creditcard" and r["view"] == "R"]
    residual = next(r for r in rows if r["score"] == "squared_residual")
    nll = next(r for r in rows if r["score"] == "nll")
    tops = {}
    for score in ("squared_residual", "nll"):
        top = _top_attribution(attribution, "Creditcard", "R", score, "anomaly", 5)
        tops[score] = [{"variable": r["variable"], "share": r["max_attribution_share"]} for r in top]
    return {"residual_ap": _float_or_none(residual["aggregate_average_precision"]),
            "nll_ap": _float_or_none(nll["aggregate_average_precision"]),
            "residual_auc": _float_or_none(residual["aggregate_auc_roc"]),
            "nll_auc": _float_or_none(nll["aggregate_auc_roc"]),
            "top_anomaly_max_variables": tops,
            "pointwise_sigma_available": False,
            "interpretation": "Residual and NLL use the same decoder mean but NLL additionally ranks learned uncertainty; the saved evidence has no pointwise sigma, so variance masking cannot be assigned to specific samples."}


def _read_rawcount_status():
    """Search only the known RawCount roots for a verified S-only result."""
    report_root = ROOT / "result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031"
    source = ROOT / "result/score/MNRF-ViewProbe-RawCount/CalIt2/rawcount-full-20261031T20261031T092320Z-1093827"
    # These are the bounded result roots used by the RawCount repair work;
    # do not turn this status check into a repository-wide result scan.
    roots = [ROOT / "result/score/MNRF-ViewProbe-RawCount",
             ROOT / "result/score/MNRF-ViewProbe-SOnly",
             ROOT / "result/score/MNRF-ViewProbe-Components",
             report_root]
    source_name = source.name
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for manifest_path in sorted(root.rglob("probe_manifest.json")):
            if manifest_path.parent == source:
                continue
            candidate = manifest_path.parent
            record = {"path": str(candidate), "status": "rejected", "reasons": []}
            try:
                manifest = _read_json(manifest_path)
                config = manifest.get("config", {})
                selected = config.get("selected_views", manifest.get("selected_views", []))
                mode = config.get("mode", manifest.get("mode"))
                status = manifest.get("stage_status")
                head_status = manifest.get("head_status")
                bound = config.get("source_dir", manifest.get("source_dir", ""))
                if list(selected) != ["S"]:
                    record["reasons"].append(f"selected_views={selected!r}")
                if mode not in ("select_gmm", "s_only"):
                    record["reasons"].append(f"mode={mode!r}")
                if status != "complete" or head_status != "complete":
                    record["reasons"].append(f"incomplete stage/head status: {status!r}/{head_status!r}")
                if source_name not in str(bound):
                    record["reasons"].append("source binding does not identify the fixed RawCount Full source")
                if not (candidate / "results/test_nll.npy").is_file():
                    record["reasons"].append("missing test_nll.npy")
                if not record["reasons"]:
                    record["status"] = "verified"
            except Exception as exc:
                record["reasons"].append(f"manifest unreadable: {type(exc).__name__}: {exc}")
            candidates.append(record)
    metrics = report_root / "baselines-v2/metrics.json"
    command = (
        "python scripts/run_mnrf_view_probe.py --task CalIt2 --data-source large_detect "
        "--mode select_gmm --views S "
        f"--source-dir result/score/MNRF-ViewProbe-RawCount/CalIt2/{source_name} "
        "--data-root result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031/data_root "
        "--output-dir result/score/MNRF-ViewProbe-RawCount/CalIt2/rawcount-s-only-$(date -u +%Y%m%dT%H%M%SZ)-$$ "
        "--seed 2021 --seq-len 192 --pattern-len 32 --hidden-dim 32 --batch-size 128 "
        "--encoding-batch-size 128 --stage1-epochs 50 --stage2-epochs 30 --patience 5 "
        "--learning-rate 0.001 --weight-decay 0.0001 --feature-std-floor 0.05 "
        "--score-mad-floor 0.001 --calibration-quantile 0.995 --device cpu --cpu-threads 1 "
        "--cache-chunk-size 256 --view-dim 8 --evolution-dim 8 --pattern-dim 8 --relation-dim 8 "
        "--gmm-components 4 --gmm-covariance-type full --gmm-reg-covar 0.001 --gmm-n-init 1 "
        "--gmm-max-iter 100 --gmm-fit-max-windows 20000 --train-noise-std 0.01 "
        "--noise-seed 20261013 --log-every-steps 100 --log-every-seconds 30 "
        "--gmm-log-every-iters 10 --artifact-policy full "
        "--startup-script scripts/multivariate_detection/detect_score/CalIt2_script/MNRFRawCountSOnly.sh"
    )
    verified = [r for r in candidates if r["status"] == "verified"]
    return {
        "directory": str(report_root),
        "searched_roots": [str(root) for root in roots],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "verified_candidates": [r["path"] for r in verified],
        "s_only_present": bool(verified),
        "s_only_note": ("A same-source completed S-only result was verified."
                         if verified else
                         "No same-source completed RawCount S-only output was verified; the available report is RawCount Full plus CATCH/IF/PCA."),
        "full_and_baselines_complete": metrics.is_file(),
        "metrics": json.loads(metrics.read_text()) if metrics.is_file() else None,
        "command": command,
        "completion_criterion": "run_status=complete, selected_views=['S'], fixed RawCount Full source binding, native validation passed, and S-only artifacts/metrics present",
    }


def build_readme(results, output, rawcount):
    summary = results["summary"]
    reference_metrics = results["reference_metrics"]
    attribution = results["attribution"]
    variable_metrics = results["variable_metrics"]
    lines = ["# Frozen E/R decoder result analysis", "",
             "This report is a read-only analysis of the eight already completed frozen E/R diagnostics. It does not train, run an encoder, fit a GMM, or recalibrate a source.", "",
             "## Evidence and limits", "",
             "Each task has 13 diagnostic rows: saved Full GMM reference; E/R decoder, latent-train-mean intervention, persistence, and train-target-mean references, with squared-residual and/or NLL scores as saved by the diagnostic. The analysis reuses saved residuals, per-variable NLL, centers/scales, thresholds and coordinates. It does not treat these scores as one common detector.", "",
             "The diagnostic manifests bind these outputs to the following source runs:", "",
             "| Dataset | Source Run | Variables | Test points | Anomaly points |", "|---|---|---:|---:|---:|"]
    for task in TASKS:
        item = results["source"][task]
        row = next(r for r in summary if r["dataset"] == task and r["view"] == "E" and r["score"] == "nll")
        lines.append(f"| {task} | `{item['source_run_id']}` | {item['variable_count']} | {row['test_points']} | {row['anomaly_points']} |")
    lines += ["", "## Main findings", ""]
    for task in TASKS:
        rows = [r for r in summary if r["dataset"] == task]
        def get(view, variant, score):
            return next((r for r in rows if r["view"] == view and r["score"] == score
                         and (variant == "decoder" or variant in ("decoder",))), None)
        rres = next(r for r in rows if r["view"] == "R" and r["score"] == "squared_residual")
        rnll = next(r for r in rows if r["view"] == "R" and r["score"] == "nll")
        er = next(r for r in rows if r["view"] == "E" and r["score"] == "squared_residual")
        lines.append(f"- **{task}:** R residual AP/AUC={rres['aggregate_average_precision']}/{rres['aggregate_auc_roc']}; R NLL AP/AUC={rnll['aggregate_average_precision']}/{rnll['aggregate_auc_roc']}; E residual AP/AUC={er['aggregate_average_precision']}/{er['aggregate_auc_roc']}. These are diagnostic rankings, not claims of a new detector.")
    cc = _creditcard_note(summary, variable_metrics, attribution)
    lines += ["", "## Creditcard residual/NLL reversal", "",
               f"For R decoder, squared-residual AP is **{cc['residual_ap']:.6f}** while NLL AP is **{cc['nll_ap']:.6f}**. The ranking deterioration is already present at the single-variable auxiliary-NLL layer; a positive median/MAD affine standardization cannot change a variable's ordering. Cross-variable scaling and the final max aggregation can further change the overall result. The top anomaly-point max contributors are recorded in `max_attribution.csv`. Pointwise sigma is not saved, so this report does not claim that particular anomalies were hidden by high predicted variance; a fixed-sample export of per-point mu/sigma would be required.", ""]
    lines += ["## E-branch reference comparison", "",
              "E residual scores are compared separately with persistence and the training-target-mean reference in `reference_metrics.csv`. GECCO, the historical modulo-10 CalIt2 source, and Creditcard have E residual AP above persistence but not above the target-mean reference; this is not evidence that historical conditions add detection information. Small differences are reported as such. A stronger conditional increment requires consistent improvement over both simple references under the same coordinates and threshold protocol.", ""]
    def reference_ap(task, view, variant):
        row = next(r for r in reference_metrics if r["dataset"] == task and r["view"] == view
                   and r["variant"] == variant and r["score"] == "squared_residual")
        return row.get("average_precision")
    lines += ["| Dataset | E decoder AP | E persistence AP | E target-mean AP | E latent-mean AP | R decoder AP | R persistence AP | R target-mean AP | R latent-mean AP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            task, reference_ap(task, "E", "decoder"), reference_ap(task, "E", "persistence"),
            reference_ap(task, "E", "train_target_mean"), reference_ap(task, "E", "decoder_latent_train_mean"),
            reference_ap(task, "R", "decoder"), reference_ap(task, "R", "persistence"),
            reference_ap(task, "R", "train_target_mean"), reference_ap(task, "R", "decoder_latent_train_mean")))
    lines.append("")
    lines += ["## MSL target and condition fields", "",
              "MSL is analyzed as 55 input fields and 27 source sequences are not reconstructed by this diagnostic into separate retrained models. `col_0` is reported separately in `variable_metrics.csv`; `col_37`, `col_51` and `col_25` are retained as ordinary fields in the full attribution table. A large residual contribution does not by itself mean a field controls the max score; use `max_attribution.csv` for the latter.", ""]
    lines += ["## NYC event profile", "",
              "NYC retains the source event boundaries, global coordinates and unscored gaps. `nyc_event_profile.csv` reports the original target-variable rows. `nyc_event_neighborhood.csv` adds a pre-fixed k=8 summary for event_first8, event_last8, remaining event_interior, and same-segment consecutive normal_before8/normal_after8 points when available. This source has one scored anomaly event, so the neighborhoods describe that event only; they do not establish a general persistence mechanism. Passenger/calendar role interpretation remains a diagnostic annotation, not evidence that calendar fields are causal or that lower prediction error proves anomaly information.", ""]
    lines += ["## Task-specific interpretation", "",
              "- PSM and GECCO show R squared-residual ranking above their simple static/persistence references in the saved metrics, but fixed-threshold alarms and event coverage remain separate outcomes.",
              "- Genesis R residual improves ranking over persistence and target-mean references while its fixed threshold still has substantial false positives; ranking gain is not alarm calibration gain.",
              "- CalIt2 is the historical modulo-10 representation. The separate RawCount report exists and shows a RawCount Full rerun plus CATCH/IF/PCA, but no RawCount S-only result; it must not be substituted into the old source comparison.",
              "- CICIDS has no verified entity/session identity in the frozen input; row-adjacent event interpretations remain limited.", ""]
    roots = rawcount["searched_roots"]
    if len(roots) <= 1:
        roots_text = "" if not roots else "`{}`".format(roots[0])
    else:
        roots_text = ", ".join("`{}`".format(root) for root in roots[:-1])
        roots_text += ", and `{}`".format(roots[-1])
    lines += ["## RawCount CalIt2 status", "",
              f"RawCount output: `{rawcount['directory']}`. Search roots were {roots_text}; {rawcount['candidate_count']} manifest candidate(s) were inspected and {len(rawcount['verified_candidates'])} same-source S-only candidate(s) passed the binding/status checks. {rawcount['s_only_note']}",
              "The report contains RawCount Full plus CATCH/IF/PCA metrics, but this read-only analysis did not execute an S-only fit. The accurate future entry is `run_mnrf_view_probe.py --mode select_gmm --views S` with the fixed RawCount `--source-dir`, explicit `--data-root`, CPU and a unique output directory; the complete command is recorded in `analysis_config.json`.", ""]
    lines += ["## Next P diagnostic", "",
              "Use the training-time 32-step local window with a continuous 8-step P mask, the saved valid flag, relative-position input and the original auxiliary decoder. Apply the identical deterministic mask rule to every split without using test labels. Persistence must read the unmasked history only, never the held-out masked truth. Map each recovered position back to its original global coordinate and segment; aggregate overlaps by a fixed rule (for example mean over valid reconstructions), and retain only common valid points. Treat 256/128/128/512 as a process-validation budget, not a formal performance sample. With small calibration sets, q=.995 is unstable; do not publish a comparative claim without enough positives and a predeclared budget. Existing axis validation and chunked I/O can be reused, but the full P cache cannot stand in for a masked representation. This round did not run it.", ""]
    lines += ["## Output files", "",
              "`summary.csv` contains aggregate E/R diagnostic rows; `reference_metrics.csv` preserves the complete 13-row distinction between Full GMM, decoder, latent-mean intervention, persistence and train-target-mean references; `full_reference.csv` records the source Full GMM NLL, calibration arrays, axes and binding hashes; `variable_distributions.csv` contains normal/anomaly per-variable residual/NLL statistics; `variable_metrics.csv` contains corrected per-variable and aggregate ranking metrics; `max_attribution.csv` contains tie-split max attribution for normal/anomaly/TP/FP/FN; `scale_diagnostics.csv` records MAD floors; `nyc_event_profile.csv` and `nyc_event_neighborhood.csv` contain bounded event rows and fixed-k summaries; `source_bindings.csv` and `analysis_config.json` record source identities, layouts, protocol fields and RawCount status.", ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {"summary": [], "reference_metrics": [], "distribution": [], "variable_metrics": [], "attribution": [],
               "scale": [], "nyc_events": [], "nyc_neighborhood": [], "full_reference": [], "source": {}}
    for task in TASKS:
        result = analyze_task(task, args.diagnostic_root)
        for key in ("summary", "reference_metrics", "distribution", "variable_metrics", "attribution", "scale", "nyc_events", "nyc_neighborhood"):
            results[key].extend(result[key])
        results["full_reference"].append(result["full_reference"])
        results["source"][task] = result["source"]
    rawcount = _read_rawcount_status()
    _write_csv(args.output_dir / "summary.csv", results["summary"])
    _write_csv(args.output_dir / "reference_metrics.csv", results["reference_metrics"])
    _write_csv(args.output_dir / "variable_distributions.csv", results["distribution"])
    _write_csv(args.output_dir / "variable_metrics.csv", results["variable_metrics"])
    _write_csv(args.output_dir / "max_attribution.csv", results["attribution"])
    _write_csv(args.output_dir / "scale_diagnostics.csv", results["scale"])
    _write_csv(args.output_dir / "nyc_event_profile.csv", results["nyc_events"])
    _write_csv(args.output_dir / "nyc_event_neighborhood.csv", results["nyc_neighborhood"])
    _write_csv(args.output_dir / "full_reference.csv", results["full_reference"])
    source_rows = []
    for task, binding in results["source"].items():
        source_rows.append({"dataset": task, "source_run_id": binding["source_run_id"],
                            "source_run": binding["source_run"], "variable_count": binding["variable_count"],
                            "variable_names": json.dumps(binding["variable_names"], ensure_ascii=False),
                            "feature_fingerprint": binding["source_feature_fingerprint"],
                            "diagnostic_run_id": binding["diagnostic_run_id"],
                            "diagnostic_manifest": binding["diagnostic_manifest"]})
    _write_csv(args.output_dir / "source_bindings.csv", source_rows)
    config = {"diagnostic_root": str(args.diagnostic_root.resolve()), "tasks": list(TASKS),
              "source_bindings": results["source"], "rawcount": rawcount,
              "model_execution": False, "encoder_forward": False, "gmm_fit": False,
              "new_calibration": False, "thread_budget": 1}
    (args.output_dir / "analysis_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(build_readme(results, args.output_dir, rawcount), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "datasets": len(TASKS),
                      "summary_rows": len(results["summary"]), "distribution_rows": len(results["distribution"]),
                      "reference_metric_rows": len(results["reference_metrics"]),
                      "variable_metric_rows": len(results["variable_metrics"]),
                      "attribution_rows": len(results["attribution"]), "nyc_event_rows": len(results["nyc_events"]),
                      "nyc_neighborhood_rows": len(results["nyc_neighborhood"]),
                      "full_reference_rows": len(results["full_reference"]),
                      "model_execution": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
