#!/usr/bin/env python
"""Export bounded, read-only diagnostics from a completed full ViewProbe run.

The source run is validated once before any output is created.  This command
never trains, refits a GMM, re-encodes windows, recalibrates a threshold, or
removes source artifacts.  Large feature arrays are read with memmap and
bounded chunks; the analysis directory contains evidence and diagnostics, not
the source cache or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_mnrf_view_probe import validate


EVIDENCE_FILES = (
    "probe_manifest.json",
    "run_status.json",
    "run_request.json",
    "input_statistics.npz",
    "results/metrics.json",
    "results/aligned_scores.npz",
    "results/test_nll.npy",
    "heads/validation_nll.npy",
    "heads/calibration_nll.npy",
    "heads/feature_statistics.npz",
    "heads/calibration.npz",
    "heads/gmm_fit_endpoints.npy",
    "heads/fixed_noise.npy",
    "heads/gmm_status.json",
    "heads/gmm_progress.json",
    "analysis_inputs.npz",
    "analysis_inputs.json",
    "training_provenance.json",
    "provenance/startup_script.sh",
    "training.jsonl",
    "run.log",
)
SPLITS = ("train", "validation", "calibration", "test")
BRANCHES = ("S", "E", "P", "R")


def _json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _copy_evidence(source: Path, output: Path):
    records = []
    for relative in EVIDENCE_FILES:
        source_path = source / relative
        record = {"path": relative, "present": source_path.is_file()}
        if source_path.is_file():
            destination = output / "evidence" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            record.update({"bytes": destination.stat().st_size, "sha256": _sha256(destination)})
        records.append(record)
    return records


def _cache_inventory(source: Path, manifest):
    declared = manifest.get("cache_sha256", {})
    endpoint_hashes = manifest.get("endpoint_sha256", {})
    records = []
    paths = ["cache/normalized_values.npy"]
    paths.extend(f"cache/{split}_features.npy" for split in SPLITS)
    paths.extend(f"cache/{split}_window_end_indices.npy" for split in SPLITS)
    for relative in paths:
        path = source / relative
        if not path.is_file():
            raise ValueError(f"validated source cache is missing {relative}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if relative.endswith("_features.npy"):
            split = relative.split("/")[-1][:-len("_features.npy")]
            expected = declared.get(split)
        elif relative.endswith("_window_end_indices.npy"):
            split = relative.split("/")[-1][:-len("_window_end_indices.npy")]
            expected = endpoint_hashes.get(split)
        else:
            expected = manifest.get("normalized_cache_sha256")
        if not expected:
            raise ValueError(f"source manifest has no validated cache hash for {relative}")
        records.append({"path": relative, "shape": list(array.shape), "dtype": str(array.dtype),
                        "bytes": int(path.stat().st_size), "nbytes": int(array.nbytes),
                        "sha256": expected, "hash_source": "validated source manifest"})
    return records


def _tree_bytes(path: Path):
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return int(total)


def _layout(manifest):
    layout = manifest.get("layout") or manifest.get("config", {}).get("resolved_layout")
    if not isinstance(layout, dict) or not isinstance(layout.get("slices"), dict):
        raise ValueError("source manifest has no canonical branch layout")
    slices = {}
    for branch in BRANCHES:
        raw = layout["slices"].get(branch)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"source layout has no slice for {branch}")
        slices[branch] = (int(raw[0]), int(raw[1]))
    total = int(layout.get("F", max(stop for _, stop in slices.values())))
    if total != max(stop for _, stop in slices.values()):
        raise ValueError("source layout total dimension is inconsistent")
    return layout, slices, total


def _feature_overview(source: Path, output: Path, manifest, rng):
    _, slices, total = _layout(manifest)
    result = {"sampling": {"max_windows": 4096, "seed": 20261013}, "splits": {}}
    for split in SPLITS:
        path = source / "cache" / f"{split}_features.npy"
        if not path.is_file():
            result["splits"][split] = {"status": "missing"}
            continue
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or array.shape[2] != total:
            raise ValueError(f"{path} has unexpected feature shape {array.shape}; expected F={total}")
        count = min(len(array), 4096)
        indices = np.arange(count, dtype=np.int64) if count == len(array) else np.sort(rng.choice(len(array), count, replace=False))
        # Keep the variable axis separate: temporal variation is measured over
        # windows (N), not over variables or feature dimensions.
        sampled = np.asarray(array[indices], dtype=np.float32)
        split_result = {"shape": list(array.shape), "dtype": str(array.dtype), "sample_count": int(count), "branches": {}}
        for branch in BRANCHES:
            start, stop = slices[branch]
            values = sampled[:, :, start:stop]  # [N, C, branch_dim]
            finite = np.isfinite(values)
            with np.errstate(invalid="ignore"):
                per_pair_mean = np.nanmean(np.where(finite, values, np.nan), axis=0)
                per_pair_std = np.nanstd(np.where(finite, values, np.nan), axis=0)
            finite_values = values[finite]
            if finite_values.size and count >= 2:
                low_pair = np.isfinite(per_pair_std) & (per_pair_std < 1e-6)
                low_by_variable = np.sum(low_pair, axis=1).astype(int).tolist()
                safe_mean = np.where(np.isfinite(per_pair_mean), per_pair_mean, np.nan).tolist()
                safe_std = np.where(np.isfinite(per_pair_std), per_pair_std, np.nan).tolist()
                safe_mean = [[_finite_float(item) for item in row] for row in safe_mean]
                safe_std = [[_finite_float(item) for item in row] for row in safe_std]
                summary = {"finite_fraction": float(np.mean(finite)),
                           "mean": float(np.mean(finite_values, dtype=np.float64)),
                           "std": float(np.std(finite_values, dtype=np.float64)),
                           "min": float(np.min(finite_values)), "max": float(np.max(finite_values)),
                           "variance_axis": "windows_N",
                           "low_variance_threshold": 1e-6,
                           "low_variance_unit": "variable x feature dimension pair",
                           "low_variance_dimensions": int(np.sum(low_pair)),
                           "low_variance_by_variable": low_by_variable,
                           "per_pair_mean_sample": safe_mean,
                           "per_pair_std_sample": safe_std,
                           "undefined_pair_count": int(np.sum(~np.isfinite(per_pair_std))),
                           "reason": None if np.isfinite(per_pair_std).all() else "one or more variable-feature pairs have no finite observations"}
            else:
                summary = {"finite_fraction": 0.0, "mean": None, "std": None, "min": None, "max": None,
                           "variance_axis": "windows_N", "low_variance_unit": "variable x feature dimension pair",
                           "low_variance_dimensions": None, "low_variance_by_variable": None,
                           "reason": "fewer than two windows" if count < 2 else "no finite sampled values"}
            split_result["branches"][branch] = summary
        result["splits"][split] = split_result
    (output / "feature_overview.json").write_text(_json_text(result))
    return result


def _metric_recheck(source: Path, output: Path, analysis_inputs):
    result_path = source / "results" / "aligned_scores.npz"
    if not result_path.is_file():
        raise ValueError("completed source is missing results/aligned_scores.npz")
    with np.load(result_path, allow_pickle=False) as payload:
        required = {"scores", "predictions", "time_indices", "local_indices", "valid_mask",
                    "aligned_scores", "aligned_predictions", "threshold"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"aligned score evidence is missing {sorted(missing)}")
        scores = np.asarray(payload["scores"])
        predictions = np.asarray(payload["predictions"])
        endpoints = np.asarray(payload["time_indices"])
        local = np.asarray(payload["local_indices"])
        valid = np.asarray(payload["valid_mask"])
        threshold = float(np.asarray(payload["threshold"]))
        actual = np.asarray(payload["actual"]) if "actual" in payload.files else None
        time_values = np.asarray(payload["time_values_ns"]) if "time_values_ns" in payload.files else None
    if scores.ndim != 1 or predictions.ndim != 1 or endpoints.ndim != 1 or local.ndim != 1 or valid.ndim != 1:
        raise ValueError("aligned score evidence must contain one-dimensional vectors")
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("aligned score evidence contains NaN or Inf")
    input_labels, _, input_indices, _, _, _, _, _, _ = analysis_inputs
    position = {int(endpoint): index for index, endpoint in enumerate(input_indices)}
    positions = np.asarray([position.get(int(endpoint), -1) for endpoint in endpoints], dtype=np.int64)
    if np.any(positions < 0):
        raise ValueError("aligned score endpoint falls outside the bound complete test label axis")
    expected = input_labels[positions]
    if actual is not None and not np.array_equal(actual, expected):
        raise ValueError("aligned result labels do not match the bound complete test label axis")
    actual = expected
    if actual is None:
        metrics = {"status": "incomplete", "reason": "source has no test labels"}
    else:
        from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
        if len(valid) == len(input_labels):
            scored = valid[positions].astype(bool, copy=False)
            coverage = float(np.mean(valid)) if len(valid) else 0.0
        elif len(valid) == len(endpoints):
            scored = valid.astype(bool, copy=False)
            coverage = float(np.sum(scored) / max(len(input_labels), 1))
        else:
            raise ValueError("valid mask is neither complete-test-axis nor score-endpoint aligned")
        if len(scores) != len(endpoints) or len(predictions) != len(endpoints):
            raise ValueError("score, prediction, and endpoint vectors are not aligned")
        actual = actual[scored].astype(np.int64, copy=False)
        scores = scores[scored]
        predicted_stored = predictions[scored]
        predicted = (scores > threshold).astype(np.int64)
        if not np.array_equal(predicted, predicted_stored):
            raise ValueError("stored predictions do not match the fixed score threshold")
        counts = {"tp": int(np.sum((actual == 1) & (predicted == 1))),
                  "fp": int(np.sum((actual == 0) & (predicted == 1))),
                  "tn": int(np.sum((actual == 0) & (predicted == 0))),
                  "fn": int(np.sum((actual == 1) & (predicted == 0)))}
        precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="binary", zero_division=0)
        metrics = {"status": "complete", "average_precision": float(average_precision_score(actual, scores)),
                   "auc_roc": float(roc_auc_score(actual, scores)) if np.unique(actual).size == 2 else None,
                   "auc_roc_undefined_reason": None if np.unique(actual).size == 2 else "single class",
                   "threshold": threshold, "alarm_points": int(np.sum(predicted)), "anomaly_points": int(np.sum(actual)),
                   "coverage": coverage, "precision": float(precision), "recall": float(recall),
                   "f1": float(f1), **counts}
        np.savez_compressed(output / "rechecked_scores.npz", scores=scores, predictions=predicted,
                            actual=actual, time_indices=endpoints[scored], local_indices=local[scored],
                            valid_mask=np.ones(len(scores), dtype=bool),
                            time_values_ns=np.asarray(time_values) if time_values is not None else np.array([], dtype=np.int64))
    (output / "metric_recheck.json").write_text(_json_text(metrics))
    return metrics


def _segment_diagnostics(source: Path, output: Path, manifest, analysis_inputs):
    result_path = source / "results" / "aligned_scores.npz"
    with np.load(result_path, allow_pickle=False) as payload:
        endpoints = np.asarray(payload["time_indices"], dtype=np.int64)
        scores = np.asarray(payload["scores"], dtype=np.float64)
        valid = np.asarray(payload["valid_mask"], dtype=bool)
        actual = np.asarray(payload["actual"], dtype=np.int64) if "actual" in payload.files else None
        aligned_actual = np.asarray(payload["aligned_actual"], dtype=np.int64) if "aligned_actual" in payload.files else None
        aligned_scores = np.asarray(payload["aligned_scores"], dtype=np.float64)
        times = np.asarray(payload["time_values_ns"], dtype=np.int64) if "time_values_ns" in payload.files else None
        threshold = float(np.asarray(payload["threshold"]))
    # Full-coordinate labels are an explicit source artifact.  The aligned
    # result remains authoritative for scoring, but cannot describe an
    # anomaly before the first valid endpoint.
    input_labels, input_times, input_indices, input_metadata, input_segment_ids = analysis_inputs[:5]
    if actual is not None:
        source_identity = manifest.get("source_identity", {})
        boundaries = source_identity.get("test_segments", [])
        train_length = int(source_identity.get("train_length", 0))
        # Prefer the aligned label axis so anomalies with no valid endpoint
        # are retained as explicit no-valid segments.  Fall back to scored
        # labels for older result bundles that lack aligned_actual.
        anomaly_axis = input_labels == 1
        endpoint_axis = input_indices
        time_by_global = {int(endpoint): int(time) for endpoint, time in zip(input_indices, input_times)}
        score_axis = np.full(len(input_labels), np.nan, dtype=np.float64)
        valid_axis = np.zeros(len(input_labels), dtype=bool)
        position = {int(endpoint): index for index, endpoint in enumerate(input_indices)}
        for score_index, (endpoint, score_value) in enumerate(zip(endpoints, scores)):
            local = position.get(int(endpoint))
            if local is not None:
                score_axis[local] = score_value
                if len(valid) == len(input_labels):
                    valid_axis[local] = bool(valid[local])
                elif len(valid) == len(endpoints):
                    valid_axis[local] = bool(valid[score_index])
                else:
                    raise ValueError("aligned valid mask has no recognized test-axis alignment")
        runs = []
        segment_identity = (input_metadata.get("boundary_identity") or {}).get("test_segment_identity", [])
        for segment_index, (left, right) in enumerate(boundaries or [(train_length, train_length + len(anomaly_axis))]):
            left = max(train_length, int(left))
            right = min(train_length + len(anomaly_axis), int(right))
            if right <= left:
                continue
            local_left, local_right = left - train_length, right - train_length
            anomaly = anomaly_axis[local_left:local_right]
            start = None
            for index, value in enumerate(np.r_[anomaly, False]):
                if value and start is None:
                    start = index
                if start is not None and not value:
                    stop = index
                    global_left = left + start
                    global_right = left + stop
                    score_slice = score_axis[global_left - train_length:global_right - train_length]
                    valid_slice = valid_axis[global_left - train_length:global_right - train_length]
                    finite = valid_slice & np.isfinite(score_slice)
                    observed_scores = score_slice[finite]
                    observed_endpoints = endpoint_axis[global_left - train_length:global_right - train_length][finite]
                    hits = finite & (score_slice > threshold)
                    hit_endpoints = endpoint_axis[global_left - train_length:global_right - train_length][hits]
                    peak = int(observed_endpoints[int(np.argmax(observed_scores))]) if observed_scores.size else None
                    runs.append({"start_original": int(global_left), "end_original": int(global_right - 1),
                                 "segment_id": segment_index, "point_count": int(stop - start),
                                 "segment_identity": segment_identity[segment_index] if segment_index < len(segment_identity) else None,
                                 "valid_points": int(np.sum(valid_slice)), "hit_points": int(np.sum(hits)),
                                 "hit": bool(np.any(hits)),
                                 "max_score": _finite_float(np.max(observed_scores)) if observed_scores.size else None,
                                 "median_score": _finite_float(np.median(observed_scores)) if observed_scores.size else None,
                                 "peak_original": peak,
                                 "first_hit_original": int(hit_endpoints[0]) if hit_endpoints.size else None,
                                 "start_time_ns": time_by_global.get(int(global_left)),
                                 "end_time_ns": time_by_global.get(int(global_right - 1)),
                                 "no_valid_scores": bool(not np.any(finite))})
                    start = None
        segments = runs
    else:
        segments = []
    payload = {"diagnostic_only": True, "segments": segments,
               "segment_summary": {
                   "raw_anomaly_segments": int(len(segments)),
                   "segments_with_valid_scores": int(sum(item["valid_points"] > 0 for item in segments)),
                   "segments_without_valid_scores": int(sum(item["valid_points"] == 0 for item in segments)),
                   "segments_with_any_alarm": int(sum(item["hit"] for item in segments)),
               },
               "note": "anomaly hit means at least one fixed-threshold alarm; this is not Aff-F or point adjustment"}
    (output / "anomaly_segments.json").write_text(_json_text(payload))
    return payload


def _variable_diagnostics(source: Path, output: Path, manifest, analysis_inputs=None):
    nll_path = source / "results" / "test_nll.npy"
    calibration_path = source / "heads" / "calibration.npz"
    if not nll_path.is_file() or not calibration_path.is_file():
        return {"status": "missing", "reason": "test_nll.npy or calibration.npz missing"}
    nll = np.load(nll_path, mmap_mode="r", allow_pickle=False)
    with np.load(calibration_path, allow_pickle=False) as calibration:
        center = np.asarray(calibration["center"], dtype=np.float32)
        scale = np.asarray(calibration["scale"], dtype=np.float32)
        threshold = float(np.asarray(calibration["threshold"]))
    if nll.ndim != 2 or center.shape != (nll.shape[1],) or scale.shape != center.shape:
        raise ValueError("test NLL and calibration shapes disagree")
    with np.load(source / "results" / "aligned_scores.npz", allow_pickle=False) as aligned:
        actual = np.asarray(aligned["actual"], dtype=np.int64) if "actual" in aligned.files else None
        score_endpoints = np.asarray(aligned["time_indices"], dtype=np.int64)
        stored_scores = np.asarray(aligned["scores"], dtype=np.float32)
        stored_predictions = np.asarray(aligned["predictions"], dtype=np.int64)
    endpoint_path = source / "cache" / "test_window_end_indices.npy"
    if not endpoint_path.is_file():
        raise ValueError("test endpoint cache is required for NLL diagnostics")
    endpoints = np.load(endpoint_path, mmap_mode="r", allow_pickle=False)
    if actual is None or len(actual) != len(nll):
        if len(score_endpoints) != len(nll):
            raise ValueError("test NLL rows do not align with result endpoints")
        input_labels = (analysis_inputs or _load_analysis_inputs(source, manifest))[0]
        positions = {int(value): index for index, value in enumerate(np.arange(
            int(manifest["source_identity"]["train_length"]),
            int(manifest["source_identity"]["input_length"]), dtype=np.int64))}
        actual = np.asarray([input_labels[positions[int(value)]] for value in score_endpoints], dtype=np.int64)
    elif len(score_endpoints) != len(actual):
        raise ValueError("test labels, NLL rows, and score endpoints are not aligned")
    normal = actual == 0 if actual is not None else np.ones(len(nll), dtype=bool)
    anomalous = actual == 1 if actual is not None else np.zeros(len(nll), dtype=bool)
    floor = float((manifest.get("config") or {}).get("score_mad_floor", 0.001))
    names = (manifest.get("source_identity") or {}).get("variable_names", [])
    variable = []
    for column in range(nll.shape[1]):
        values = np.asarray(nll[:, column], dtype=np.float64)
        standard = (values - float(center[column])) / float(scale[column])
        variable.append({"variable_index": column,
                         "variable_name": str(names[column]) if column < len(names) else None,
                         "center": float(center[column]), "scale": float(scale[column]),
                         "score_mad_floor": floor,
                         "scale_floor_active": bool(scale[column] <= floor + max(1e-7, floor * 1e-6)),
                         "normal_median": _finite_float(np.median(values[normal])) if np.any(normal) else None,
                         "anomaly_median": _finite_float(np.median(values[anomalous])) if np.any(anomalous) else None,
                         "normal_standard_q95": _finite_float(np.quantile(standard[normal], .95)) if np.any(normal) else None,
                         "anomaly_standard_q95": _finite_float(np.quantile(standard[anomalous], .95)) if np.any(anomalous) else None,
                         "max_score_frequency": None})
    if len(endpoints) != len(nll) or len(score_endpoints) != len(nll):
        raise ValueError("test NLL, feature endpoints, and scores have inconsistent lengths")
    if not np.array_equal(np.asarray(endpoints, dtype=np.int64), score_endpoints):
        raise ValueError("test NLL rows are not aligned with scored global endpoints")
    owner_counts = np.zeros(nll.shape[1], dtype=np.int64)
    aggregate = np.empty(len(nll), dtype=np.float32)
    for start in range(0, len(nll), 4096):
        stop = min(len(nll), start + 4096)
        block = (np.asarray(nll[start:stop]) - center[None, :]) / scale[None, :]
        owner_counts += np.bincount(np.argmax(block, axis=1), minlength=nll.shape[1])
        aggregate[start:stop] = np.max(block, axis=1)
    threshold_matches = bool(np.array_equal((aggregate > threshold).astype(np.int64), stored_predictions))
    score_matches = bool(np.allclose(aggregate, stored_scores, atol=1e-6, rtol=1e-6))
    if not threshold_matches or not score_matches:
        raise ValueError("test NLL aggregation does not reproduce stored scores and fixed-threshold predictions")
    for entry in variable:
        entry["max_score_frequency"] = int(owner_counts[entry["variable_index"]])
    result = {"status": "complete", "threshold": threshold, "variables": variable,
              "formula": "max_c((test_nll[:,c]-center[c])/scale[c])",
              "aggregation_matches_scores": score_matches,
              "aggregation_matches_predictions": threshold_matches,
              "scale_floor": floor}
    (output / "variable_diagnostics.json").write_text(_json_text(result))
    return result


def _load_analysis_inputs(source: Path, manifest):
    npz_path = source / "analysis_inputs.npz"
    meta_path = source / "analysis_inputs.json"
    if not npz_path.is_file() or not meta_path.is_file():
        raise ValueError("full source is missing bound analysis_inputs.npz/json; cannot report unscored labels")
    metadata = _json(meta_path)
    if metadata.get("schema") != "mnrf-view-probe-analysis-inputs-v1":
        raise ValueError("analysis input metadata schema is missing or unsupported")
    if metadata.get("source_manifest_sha256") != _sha256(source / "probe_manifest.json"):
        raise ValueError("analysis input source manifest binding does not match the source")
    identity = manifest.get("source_identity") or {}
    train_length = int(identity.get("train_length", 0))
    task_name = str(manifest.get("task_name", (manifest.get("config") or {}).get("task_name", "")))
    expected_task = task_name[:-4] if task_name.endswith(".csv") else task_name
    if metadata.get("source_identity_sha256") != manifest.get("source_identity_sha256"):
        raise ValueError("analysis input source identity binding does not match the source")
    if metadata.get("labels_sha256") != _sha256(npz_path):
        raise ValueError("analysis input array SHA-256 does not match its metadata")
    bound = manifest.get("analysis_artifacts") or {}
    if bound.get("analysis_inputs_sha256") != _sha256(npz_path):
        raise ValueError("analysis input array is not bound by the source manifest")
    provenance_path = source / "training_provenance.json"
    if not provenance_path.is_file() or bound.get("training_provenance_sha256") != _sha256(provenance_path):
        raise ValueError("training-time provenance is missing or not bound by the source manifest")
    if metadata.get("provenance_sha256") != _sha256(provenance_path):
        raise ValueError("analysis input metadata does not bind the training provenance file")
    provenance = _json(provenance_path)
    source_data = provenance.get("source_data") or {}
    if metadata.get("data_file") != source_data.get("data_file") or metadata.get("data_file_sha256") != source_data.get("data_file_sha256"):
        raise ValueError("analysis input data source differs from training provenance")
    if metadata.get("metadata_file") != source_data.get("metadata_file") or metadata.get("metadata_file_sha256") != source_data.get("metadata_file_sha256"):
        raise ValueError("analysis input metadata source differs from training provenance")
    startup = provenance.get("startup_script") or {}
    snapshot = source / str(startup.get("snapshot", ""))
    if not startup.get("sha256") or not snapshot.is_file() or _sha256(snapshot) != startup.get("snapshot_sha256") or startup.get("snapshot_sha256") != startup.get("sha256"):
        raise ValueError("training startup script snapshot is missing or failed its hash binding")
    for path_key, hash_key in (("data_file", "data_file_sha256"), ("metadata_file", "metadata_file_sha256")):
        value, expected_hash = metadata.get(path_key), metadata.get(hash_key)
        if not value or not expected_hash:
            raise ValueError(f"analysis input metadata is missing {path_key} identity/hash")
        source_file = Path(value)
        if not source_file.is_file() or _sha256(source_file) != expected_hash:
            raise ValueError(f"bound {path_key} is missing or changed")
    selected_row = metadata.get("selected_metadata_row") or {}
    if Path(str(metadata.get("data_file", ""))).name != f"{expected_task}.csv":
        raise ValueError("bound raw task CSV name differs from source identity")
    with Path(metadata["metadata_file"]).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("file_name") == f"{expected_task}.csv"]
    if len(rows) != 1 or int(rows[0].get("train_lens", -1)) != train_length:
        raise ValueError("bound source metadata does not contain the frozen unique train boundary")
    if str(selected_row.get("file_name")) != f"{expected_task}.csv" or int(selected_row.get("train_lens", -1)) != train_length:
        raise ValueError("selected metadata row differs from the source run boundary")
    with np.load(npz_path, allow_pickle=False) as payload:
        required = {"test_labels", "test_time_values_ns", "test_global_indices", "test_segment_ids",
                    "all_labels", "all_time_values_ns", "all_global_indices", "all_segment_ids"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"analysis input is missing {sorted(missing)}")
        labels = np.asarray(payload["test_labels"])
        times = np.asarray(payload["test_time_values_ns"])
        indices = np.asarray(payload["test_global_indices"])
        segment_ids = np.asarray(payload["test_segment_ids"]) if "test_segment_ids" in payload.files else None
        all_labels = np.asarray(payload["all_labels"])
        all_times = np.asarray(payload["all_time_values_ns"])
        all_indices = np.asarray(payload["all_global_indices"])
        all_segment_ids = np.asarray(payload["all_segment_ids"])
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("analysis input labels must be one-dimensional binary integers")
    if times.ndim != 1 or times.dtype.kind not in "iu" or indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("analysis input coordinates must be one-dimensional integer arrays")
    if not (len(labels) == len(times) == len(indices)) or len(indices) != int((manifest.get("source_identity") or {}).get("input_length", 0)) - int((manifest.get("source_identity") or {}).get("train_length", 0)):
        raise ValueError("analysis input length does not match source test axis")
    if not np.array_equal(indices, np.arange(int((manifest.get("source_identity") or {}).get("train_length", 0)),
                                             int((manifest.get("source_identity") or {}).get("input_length", 0)), dtype=np.int64)):
        raise ValueError("analysis input global coordinates do not match the source test interval")
    if metadata.get("train_length") != train_length or metadata.get("test_length") != len(labels):
        raise ValueError("analysis input split boundaries do not match the source identity")
    input_length = int(identity.get("input_length", 0))
    if (metadata.get("full_rows") != input_length or all_labels.shape != (input_length,)
            or all_times.shape != (input_length,) or all_indices.shape != (input_length,)
            or all_segment_ids.shape != (input_length,)):
        raise ValueError("complete source analysis axis has inconsistent lengths")
    if all_labels.dtype.kind not in "iu" or not np.all(np.isin(all_labels, [0, 1])):
        raise ValueError("complete source labels are not binary integers")
    if any(array.dtype.kind not in "iu" for array in (all_times, all_indices, all_segment_ids)):
        raise ValueError("complete source time/coordinate/segment arrays must be integer typed")
    if not np.array_equal(all_indices, np.arange(input_length, dtype=np.int64)):
        raise ValueError("complete source global row coordinates are invalid")
    if not np.array_equal(labels, all_labels[train_length:]) or not np.array_equal(times, all_times[train_length:]):
        raise ValueError("test analysis axis does not match its full source axis")
    if not np.array_equal(indices, all_indices[train_length:]) or not np.array_equal(segment_ids, all_segment_ids[train_length:]):
        raise ValueError("test coordinates/segments do not match the full source axis")
    full_hashes = bound.get("full_axis_sha256") or {}
    for key, array in (("labels", all_labels), ("time_values_ns", all_times),
                       ("global_indices", all_indices), ("segment_ids", all_segment_ids)):
        if full_hashes.get(key) != _array_sha256(array):
            raise ValueError(f"complete source {key} differs from its frozen hash")
    if metadata.get("variable_names") != identity.get("variable_names"):
        raise ValueError("analysis input variable order does not match the source identity")
    if source_data.get("rows") != input_length or source_data.get("train_length") != train_length:
        raise ValueError("training provenance row/split identity differs from the source manifest")
    task_name = str(manifest.get("task_name", (manifest.get("config") or {}).get("task_name", "")))
    expected_task = task_name[:-4] if task_name.endswith(".csv") else task_name
    if source_data.get("task") != expected_task or source_data.get("data_source") != (manifest.get("config") or {}).get("data_source"):
        raise ValueError("training provenance task/source differs from the source manifest")
    if source_data.get("variable_names") != identity.get("variable_names"):
        raise ValueError("training provenance variable order differs from the source manifest")
    if source_data.get("time_values_sha256") != _array_sha256(all_times):
        raise ValueError("training provenance time axis differs from the bound analysis axis")
    if source_data.get("time_values_sha256") != (identity.get("time_index") or {}).get("values_sha256"):
        raise ValueError("training provenance time axis differs from frozen source identity")
    if source_data.get("time_kind") != metadata.get("time_kind"):
        raise ValueError("analysis time kind differs from training provenance")
    if source_data.get("feature_data_sha256") != identity.get("feature_data_sha256"):
        raise ValueError("training provenance feature values differ from frozen source identity")
    if source_data.get("boundary_columns") != identity.get("boundary_columns", []):
        raise ValueError("training provenance entity/session columns differ from frozen source identity")
    if source_data.get("boundary_values_sha256") != identity.get("boundary_values_sha256"):
        raise ValueError("training provenance entity/session values differ from frozen source identity")
    if metadata.get("test_segments") != identity.get("test_segments"):
        raise ValueError("analysis input continuity segments do not match the source identity")
    if metadata.get("boundary_columns") != identity.get("boundary_columns", []):
        raise ValueError("analysis input entity/session boundary columns do not match the source identity")
    if (metadata.get("boundary_identity") or {}).get("columns") != identity.get("boundary_columns", []):
        raise ValueError("analysis input boundary identity columns do not match the source identity")
    if metadata.get("test_segment_identity") != (metadata.get("boundary_identity") or {}).get("test_segment_identity"):
        raise ValueError("analysis test segment identity binding is inconsistent")
    if bound.get("test_time_values_sha256") != _array_sha256(times):
        raise ValueError("analysis input test time axis differs from the frozen source identity")
    if segment_ids is None or segment_ids.shape != labels.shape or segment_ids.dtype.kind not in "iu":
        raise ValueError("analysis input continuous-segment ids have invalid shape or dtype")
    if metadata.get("boundary_identity") is None:
        raise ValueError("analysis input is missing entity/session identity metadata")
    # The result's valid-point labels must agree with the full raw test axis.
    aligned_path = source / "results" / "aligned_scores.npz"
    if aligned_path.is_file():
        with np.load(aligned_path, allow_pickle=False) as aligned:
            if "actual" in aligned.files:
                endpoints = np.asarray(aligned["time_indices"])
                actual = np.asarray(aligned["actual"])
                if endpoints.ndim != 1 or endpoints.dtype.kind not in "iu" or actual.shape != endpoints.shape:
                    raise ValueError("aligned endpoint/label vectors are malformed")
                positions = endpoints.astype(np.int64, copy=False) - train_length
                if np.any((positions < 0) | (positions >= len(labels))) or not np.array_equal(actual, labels[positions]):
                    raise ValueError("full-label analysis input disagrees with scored endpoint labels")
    return (labels.astype(np.int64, copy=False), times.astype(np.int64, copy=False),
            indices.astype(np.int64, copy=False), metadata, segment_ids.astype(np.int64, copy=False),
            all_labels.astype(np.int64, copy=False), all_times.astype(np.int64, copy=False),
            all_indices.astype(np.int64, copy=False), all_segment_ids.astype(np.int64, copy=False))


def _array_sha256(array):
    digest = hashlib.sha256()
    array = np.asarray(array)
    digest.update(json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True).encode())
    view = np.ascontiguousarray(array).view(np.uint8)
    for start in range(0, view.size, 1024 * 1024):
        digest.update(view[start:start + 1024 * 1024])
    return digest.hexdigest()


def _samples(source: Path, output: Path, manifest, rng, analysis_inputs=None):
    _, _, total = _layout(manifest)
    limits = {"train": 256, "validation": 128, "calibration": 128, "test": 512}
    records = {}
    _, _, _, input_metadata, _, input_labels, input_times, global_axis, segment_ids = analysis_inputs
    label_by_global = dict(zip(global_axis.tolist(), input_labels.tolist()))
    time_by_global = dict(zip(global_axis.tolist(), input_times.tolist()))
    segment_by_global = dict(zip(global_axis.tolist(), segment_ids.tolist()))
    train_length = int((manifest.get("source_identity") or {}).get("train_length", 0))
    test_scores = {}
    with np.load(source / "results" / "aligned_scores.npz", allow_pickle=False) as payload:
        endpoints = np.asarray(payload["time_indices"], dtype=np.int64)
        score_values = np.asarray(payload["scores"])
        prediction_values = np.asarray(payload["predictions"])
        valid_values = np.asarray(payload["valid_mask"])
        if len(endpoints) != len(score_values) or len(endpoints) != len(prediction_values):
            raise ValueError("test scores, predictions, and endpoints are not aligned")
        for index, (endpoint, score, prediction) in enumerate(zip(endpoints, score_values, prediction_values)):
            if len(valid_values) == len(global_axis) - train_length:
                local = int(endpoint) - train_length
                valid_value = 0 <= local < len(valid_values) and bool(valid_values[local])
            elif len(valid_values) == len(endpoints):
                valid_value = bool(valid_values[index])
            else:
                raise ValueError("aligned valid mask has no recognized test-axis alignment")
            if valid_value:
                test_scores[int(endpoint)] = (float(score), int(prediction))
    with np.load(source / "heads" / "calibration.npz", allow_pickle=False) as payload:
        test_threshold = float(np.asarray(payload["threshold"]))
    for split, limit in limits.items():
        path = source / "cache" / f"{split}_features.npy"
        endpoint_path = source / "cache" / f"{split}_window_end_indices.npy"
        if not path.is_file() or not endpoint_path.is_file():
            continue
        features = np.load(path, mmap_mode="r", allow_pickle=False)
        endpoints = np.load(endpoint_path, mmap_mode="r", allow_pickle=False)
        count = min(len(features), limit)
        endpoints_array = np.asarray(endpoints, dtype=np.int64)
        reasons = {}
        candidate_counts = {}
        selected = []
        if split == "test":
            score_values = np.full(len(features), np.nan, dtype=np.float64)
            labels_for_rows = np.full(len(features), -1, dtype=np.int8)
            categories = {name: [] for name in ("tp", "fp", "tn", "fn")}
            for row, endpoint in enumerate(endpoints_array):
                endpoint = int(endpoint)
                labels_for_rows[row] = label_by_global.get(endpoint, -1)
                pair = test_scores.get(endpoint)
                if pair is None or labels_for_rows[row] not in (0, 1):
                    continue
                score_values[row] = pair[0]
                pred = int(pair[1])
                if labels_for_rows[row] == 1:
                    category = "tp" if pred == 1 else "fn"
                else:
                    category = "fp" if pred == 1 else "tn"
                categories[category].append(row)
            candidate_counts = {name: len(rows) for name, rows in categories.items()}
            # Keep every rare anomaly endpoint if it fits; otherwise round-robin
            # across frozen continuous segments before filling common classes.
            anomaly_rows = np.flatnonzero(labels_for_rows == 1).tolist()
            if len(anomaly_rows) <= count:
                for row in anomaly_rows:
                    prediction = test_scores.get(int(endpoints_array[row]), (np.nan, -1))[1]
                    selected.append(row)
                    reasons[row] = "tp" if prediction == 1 else "fn"
            else:
                by_segment = {}
                for row in anomaly_rows:
                    sid = segment_by_global.get(int(endpoints_array[row]), -1)
                    by_segment.setdefault(sid, []).append(row)
                while len(selected) < min(count, max(1, count // 2)) and any(by_segment.values()):
                    for sid in sorted(by_segment):
                        if by_segment[sid] and len(selected) < min(count, max(1, count // 2)):
                            row = by_segment[sid].pop(0)
                            selected.append(row); reasons[row] = "anomaly_segment_stratified"
            # Preserve a representative from every available confusion-matrix
            # class; FP and FN are ordered toward their most diagnostic scores.
            for category in ("tp", "fp", "fn", "tn"):
                rows = categories[category]
                if not rows or len(selected) >= count:
                    continue
                ordered = sorted(rows, key=lambda row: ((-score_values[row] if category == "fp" else
                                                          score_values[row] if category == "fn" else
                                                          -score_values[row] if category == "tp" else
                                                          score_values[row]), int(endpoints_array[row])))
                for row in ordered:
                    if row not in reasons:
                        selected.append(row); reasons[row] = category
                        break
            remaining = [row for row in range(len(features)) if row not in reasons]
            need = count - len(selected)
            fill = remaining if need >= len(remaining) else np.sort(rng.choice(remaining, need, replace=False)).tolist()
            for row in fill:
                selected.append(int(row)); reasons[int(row)] = "deterministic_uniform"
            indices = np.asarray(sorted(selected), dtype=np.int64)
            selected_reasons = [reasons[int(row)] for row in indices]
            selected_scores = score_values[indices]
            selected_labels = labels_for_rows[indices]
            selected_predictions = np.asarray([test_scores.get(int(endpoints_array[row]), (np.nan, -1))[1]
                                               for row in indices], dtype=np.int8)
        else:
            indices = np.arange(count, dtype=np.int64) if count == len(features) else np.sort(rng.choice(len(features), count, replace=False))
            selected_reasons = ["deterministic_uniform"] * len(indices)
            candidate_counts = {"deterministic_uniform": int(len(features))}
            selected_scores = np.full(len(indices), np.nan, dtype=np.float64)
            selected_labels = np.asarray([label_by_global.get(int(endpoints_array[row]), -1) for row in indices], dtype=np.int8)
            selected_predictions = np.full(len(indices), -1, dtype=np.int8)
        selected_endpoints = endpoints_array[indices]
        selected_local = selected_endpoints - train_length
        selected_times = np.asarray([time_by_global.get(int(endpoint), -1) for endpoint in selected_endpoints], dtype=np.int64)
        selected_segments = np.asarray([segment_by_global.get(int(endpoint), -1) for endpoint in selected_endpoints], dtype=np.int32)
        np.savez_compressed(output / "samples" / f"{split}.npz", features=np.asarray(features[indices], dtype=np.float32),
                            window_end_indices=selected_endpoints, local_indices=selected_local,
                            time_values_ns=selected_times, segment_ids=selected_segments,
                            actual=selected_labels, scores=selected_scores, predictions=selected_predictions,
                            sample_indices=indices, sample_reason=np.asarray(selected_reasons, dtype="U32"),
                            diagnostic_only=np.asarray(True))
        records[split] = {"count": int(len(indices)), "shape": [int(len(indices)), *features.shape[1:]], "total_windows": int(len(features)),
                          "candidate_counts": candidate_counts,
                          "selected_counts": {str(k): int(v) for k, v in zip(*np.unique(selected_reasons, return_counts=True))},
                          "reason_counts": {str(k): int(v) for k, v in zip(*np.unique(selected_reasons, return_counts=True))}}
    return records


def _validate_output_target(source: Path, requested: Path, overwrite: bool):
    """Validate source/target relation before any output-side mutation."""
    source = source.resolve()
    raw = requested.expanduser()
    # Reject a link itself before normalization can hide its original identity.
    if raw.is_symlink():
        raise ValueError("analysis output must not be a symlink; choose a real new path")
    # Resolve aliases for the relationship check, but retain the raw object so
    # an existing symlink cannot be replaced through --overwrite.
    target = raw.resolve(strict=False)
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("analysis output must be outside the source tree and must not contain it")
    if raw.exists() and raw.is_file():
        raise ValueError("analysis output must be a directory, not a file")
    if raw.exists() and not raw.is_dir():
        raise ValueError("analysis output is not a directory")
    if raw.exists() and overwrite:
        marker = raw / "analysis_manifest.json"
        complete = raw / "analysis_complete.json"
        if not marker.is_file() or not complete.is_file():
            raise ValueError("--overwrite only accepts a recognized completed analysis directory")
        try:
            existing = _json(marker)
            completed = _json(complete)
            analysis_status = _json(raw / "analysis_manifest.json")
        except Exception as error:
            raise ValueError("--overwrite target has invalid analysis metadata") from error
        source_hash = _sha256(source / "probe_manifest.json")
        if (existing.get("schema") != "mnrf-view-probe-analysis-v1"
                or existing.get("status") != "complete"
                or existing.get("source_run") != str(source)
                or completed.get("status") != "complete"
                or completed.get("source_manifest_sha256") != source_hash
                or analysis_status.get("schema") != existing.get("schema")
                or analysis_status.get("source_run") != str(source)
                or analysis_status.get("source_manifest_sha256") != source_hash
                or analysis_status.get("analysis_complete") is not True):
            raise ValueError("--overwrite target belongs to another source run")
        required_package = ("README.md", "metric_recheck.json", "anomaly_segments.json",
                            "feature_overview.json", "variable_diagnostics.json",
                            "samples/sample_manifest.json")
        missing_package = [item for item in required_package if not (raw / item).is_file()]
        if missing_package:
            raise ValueError("--overwrite target is an incomplete analysis package: " + ", ".join(missing_package))
        for record in existing.get("evidence", []):
            if not record.get("present"):
                continue
            evidence = raw / "evidence" / str(record.get("path", ""))
            if (not evidence.is_file() or not record.get("sha256")
                    or _sha256(evidence) != record["sha256"]
                    or evidence.stat().st_size != int(record.get("bytes", -1))):
                raise ValueError("--overwrite target has damaged or missing evidence: {}".format(record.get("path")))
    elif raw.exists() and not overwrite:
        raise FileExistsError(f"analysis output exists; choose a new directory or pass --overwrite: {raw}")
    return raw, target


def _require_full_analysis_evidence(run_dir: Path):
    required = (
        "probe_manifest.json", "run_status.json", "input_statistics.npz",
        "checkpoints/initial_encoder.pt", "checkpoints/best_encoder.pt",
        "cache/normalized_values.npy", "heads/feature_statistics.npz",
        "cache/train_features.npy", "cache/validation_features.npy",
        "cache/calibration_features.npy", "cache/test_features.npy",
        "cache/train_window_end_indices.npy", "cache/validation_window_end_indices.npy",
        "cache/calibration_window_end_indices.npy", "cache/test_window_end_indices.npy",
        "heads/gmm_models.pkl", "heads/gmm_status.json", "heads/gmm_progress.json",
        "heads/feature_statistics.npz", "heads/calibration.npz",
        "heads/validation_nll.npy", "heads/calibration_nll.npy",
        "heads/gmm_fit_endpoints.npy", "heads/fixed_noise.npy",
        "results/test_nll.npy", "results/aligned_scores.npz", "results/metrics.json",
        "analysis_inputs.npz", "analysis_inputs.json",
    )
    missing = [item for item in required if not (run_dir / item).is_file()]
    if missing:
        raise ValueError("full source is missing required analysis evidence: " + ", ".join(missing))
    manifest = _json(run_dir / "probe_manifest.json")
    status = _json(run_dir / "run_status.json")
    if manifest.get("artifact_policy") != "full" or status.get("status") != "complete" or manifest.get("stage_status") != "complete":
        raise ValueError("analysis export requires a complete self-contained full_gmm source")
    if manifest.get("source_dir"):
        raise ValueError("analysis export does not treat source-referencing heads as self-contained full sources")


def export(run_dir: Path, output_dir: Path, overwrite=False, already_validated=False):
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise ValueError(f"source run directory does not exist: {run_dir}")
    manifest_path = run_dir / "probe_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("source run has no probe_manifest.json")
    output_dir, _ = _validate_output_target(run_dir, output_dir, overwrite)
    old_snapshot = _tree_snapshot(output_dir) if output_dir.exists() else None
    manifest = _json(manifest_path)
    if manifest.get("artifact_policy") != "full":
        raise ValueError("analysis export requires artifact_policy=full; results-only runs cannot be recovered")
    if already_validated:
        raise ValueError("already_validated is forbidden; export always performs exactly one complete source validation")
    _require_full_analysis_evidence(run_dir)
    validation = validate(run_dir)
    if validation.get("status") != "complete":
        raise ValueError(f"source run is not a complete full artifact: {validation}")
    analysis_inputs = _load_analysis_inputs(run_dir, manifest)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    started = time.time()
    try:
        temporary.mkdir()
        (temporary / "metadata").mkdir()
        (temporary / "samples").mkdir()
        (temporary / "evidence").mkdir()
        source_manifest_hash = _sha256(manifest_path)
        (temporary / "metadata" / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        status_path = run_dir / "run_status.json"
        if status_path.is_file():
            shutil.copy2(status_path, temporary / "metadata" / "source_status.json")
        records = _copy_evidence(run_dir, temporary)
        cache_records = _cache_inventory(run_dir, manifest)
        rng = np.random.default_rng(20261013)
        overview = _feature_overview(run_dir, temporary, manifest, rng)
        metrics = _metric_recheck(run_dir, temporary, analysis_inputs)
        segments = _segment_diagnostics(run_dir, temporary, manifest, analysis_inputs)
        variables = _variable_diagnostics(run_dir, temporary, manifest, analysis_inputs)
        samples = _samples(run_dir, temporary, manifest, rng, analysis_inputs)
        (temporary / "samples" / "sample_manifest.json").write_text(_json_text({
            "diagnostic_only": True, "seed": 20261013,
            "budgets": {"train": 256, "validation": 128, "calibration": 128, "test": 512},
            "source_manifest_sha256": source_manifest_hash,
            "selection": "deterministic bounded sampling; test stratifies available confusion classes and anomalies",
            "splits": samples,
        }), encoding="utf-8")
        runtime = {"python": sys.version, "platform": platform.platform()}
        try:
            import torch
            runtime.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                            "cuda_available": bool(torch.cuda.is_available())})
        except Exception as error:
            runtime["torch_error"] = str(error)
        provenance_path = run_dir / "training_provenance.json"
        training_provenance = _json(provenance_path) if provenance_path.is_file() else None
        analysis_manifest = {
            "schema": "mnrf-view-probe-analysis-v1", "status": "complete",
            "source_run": str(run_dir), "source_run_identity": manifest.get("run_identity"),
            "source_manifest_sha256": source_manifest_hash, "source_validation": validation,
            "model_complete": True, "analysis_complete": True,
            "training_environment": (training_provenance or {}).get("training_environment"),
            "export_environment": runtime, "evidence": records,
            "source_cache_inventory": cache_records,
            "feature_overview": overview,
            "metric_recheck": metrics, "segment_count": len(segments.get("segments", [])),
            "variable_diagnostics": variables, "samples": samples,
            "elapsed_seconds": float(time.time() - started),
            "source_directory_logical_bytes": _tree_bytes(run_dir),
            "analysis_directory_logical_bytes": None,
            "note": "diagnostic export; sampled feature summaries do not replace full-test metrics",
        }
        (temporary / "README.md").write_text(
            "# MNRF ViewProbe analysis\n\n"
            f"Source run: `{run_dir}`\n\n"
            "This directory is a bounded, read-only diagnostic export. The source full artifact, "
            "cache and checkpoints remain authoritative. `metric_recheck.json` uses the stored "
            "fixed threshold; it does not fit or calibrate anything. Feature summaries and samples "
            "are diagnostic only and do not establish branch causality or performance superiority.\n"
        )
        (temporary / "analysis_complete.json").write_text(_json_text({"status": "complete", "source_manifest_sha256": source_manifest_hash}))
        # Fixed point accounts for the manifest's own serialized byte count.
        size = _tree_bytes(temporary)
        while True:
            analysis_manifest["analysis_directory_logical_bytes"] = size
            encoded = _json_text(analysis_manifest)
            next_size = _tree_bytes(temporary) + len(encoded.encode("utf-8"))
            if next_size == size:
                break
            size = next_size
        (temporary / "analysis_manifest.json").write_text(encoded, encoding="utf-8")
        if output_dir.exists():
            _validate_output_target(run_dir, output_dir, overwrite=True)
            if _tree_snapshot(output_dir) != old_snapshot:
                raise RuntimeError("recognized analysis output changed during export; refusing replacement")
            backup = output_dir.parent / f".{output_dir.name}.old-{uuid.uuid4().hex}"
            os.replace(output_dir, backup)
            try:
                os.replace(temporary, output_dir)
            except Exception:
                os.replace(backup, output_dir)
                raise
            try:
                shutil.rmtree(backup)
            except Exception:
                rollback = output_dir.parent / f".{output_dir.name}.failed-{uuid.uuid4().hex}"
                os.replace(output_dir, rollback)
                os.replace(backup, output_dir)
                shutil.rmtree(rollback, ignore_errors=True)
                raise
        else:
            _validate_output_target(run_dir, output_dir, overwrite=False)
            os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"status": "complete", "run_dir": str(run_dir), "output_dir": str(output_dir),
            "source_manifest_sha256": source_manifest_hash, "analysis_manifest": str(output_dir / "analysis_manifest.json")}


def _tree_snapshot(path: Path):
    if not path.exists():
        return None
    return {str(item.relative_to(path)): (_sha256(item) if item.is_file() else "directory")
            for item in sorted(path.rglob("*"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = export(Path(args.run_dir), Path(args.output_dir), overwrite=args.overwrite)
    except Exception as error:
        print(json.dumps({"status": "invalid", "reason": str(error)}, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
