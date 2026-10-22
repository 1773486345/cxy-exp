#!/usr/bin/env python3
"""Read-only comparison and packaging for the fixed eight-run S-only audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import sys
import subprocess
import tempfile
import time
import zipfile

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ts_benchmark.baselines.mnrf_view_probe.integrity import (
    canonical_segment_ids,
    sha256_file,
    validate_normalized_values_cache,
)

SOURCE_IDS = {
    "PSM": "full_gmm-full-20261021T112121Z-2466820",
    "MSL": "full_gmm-full-20261021T112147Z-2470335",
    "GECCO": "full_gmm-full-20261020T204951Z-1642355",
    "Genesis": "full_gmm-full-20261020T205005Z-1645522",
    "NYC": "full_gmm-full-20261020T211341Z-1931786",
    "CalIt2": "full_gmm-full-20261021T112212Z-2474004",
    "Creditcard": "full_gmm-full-20261021T112159Z-2472093",
    "CICIDS": "full_gmm-full-20261021T112642Z-2510633",
}
S_ONLY_IDS = {
    "PSM": "s_only-gmm-20261022T091100Z-3712865",
    "MSL": "s_only-gmm-20261022T091352Z-3727341",
    "GECCO": "s_only-gmm-20261022T091505Z-3733552",
    "Genesis": "s_only-gmm-20261022T091524Z-3735469",
    "NYC": "s_only-gmm-20261022T091725Z-3745562",
    "CalIt2": "s_only-gmm-20261022T091738Z-3747376",
    "Creditcard": "s_only-gmm-20261022T091748Z-3748839",
    "CICIDS": "s_only-gmm-20261022T091911Z-3756380",
}
WINDOWS = {"GECCO": 96, "Genesis": 192}


def _json(path):
    with pathlib.Path(path).open() as f:
        return json.load(f)


def _dump(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _metrics(scores, labels, threshold):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("scores and labels must be one-dimensional and aligned")
    if labels.dtype.kind not in "iub" or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels are not exact binary integers")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    pred = scores > float(threshold)
    y = labels.astype(bool)
    tp = int(np.count_nonzero(pred & y))
    fp = int(np.count_nonzero(pred & ~y))
    tn = int(np.count_nonzero(~pred & ~y))
    fn = int(np.count_nonzero(~pred & y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = roc_auc_score(y, scores) if np.unique(y).size == 2 else None
    ap = average_precision_score(y, scores) if y.any() else None
    return {
        "auc_roc": _safe(auc), "average_precision": _safe(ap), "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision,
        "recall": recall, "f1": f1,
        "normal_fpr": fp / (fp + tn) if fp + tn else None,
        "alarm_points": tp + fp, "anomaly_points": int(y.sum()),
        "anomaly_fraction": float(y.mean()) if len(y) else None,
        "valid_points": len(y), "coverage": 1.0,
        "predictions": pred.astype(np.int8),
    }


def _segments(labels, global_indices, segment_ids, valid_map, score_map, threshold, time_values_ns=None):
    labels = np.asarray(labels, dtype=np.int8)
    global_indices = np.asarray(global_indices, dtype=np.int64)
    segment_ids = np.asarray(segment_ids, dtype=np.int32)
    if not (labels.shape == global_indices.shape == segment_ids.shape):
        raise ValueError("complete test labels/coordinates/segment IDs do not align")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("complete test labels are not binary")
    output = []
    i = 0
    while i < len(labels):
        if labels[i] != 1:
            i += 1
            continue
        sid = int(segment_ids[i])
        j = i + 1
        while (j < len(labels) and labels[j] == 1 and int(segment_ids[j]) == sid
               and int(global_indices[j]) == int(global_indices[j - 1]) + 1):
            j += 1
        coords = global_indices[i:j]
        pairs = [(int(g), score_map[int(g)]) for g in coords if int(g) in score_map]
        vals = np.asarray([x[1] for x in pairs], dtype=np.float64)
        hits = int(np.count_nonzero(vals > threshold))
        output.append({
            "segment_id": sid, "start_global": int(coords[0]), "end_global": int(coords[-1]),
            "start_time_ns": int(time_values_ns[i]) if time_values_ns is not None else None,
            "end_time_ns": int(time_values_ns[j - 1]) if time_values_ns is not None else None,
            "points": int(j - i), "valid_points": int(len(vals)), "hit_points": hits,
            "hit": bool(hits), "no_valid_scores": not bool(len(vals)),
            "max_score": _safe(float(np.max(vals))) if len(vals) else None,
            "median_score": _safe(float(np.median(vals))) if len(vals) else None,
        })
        i = j
    valid_segments = sum(x["valid_points"] > 0 for x in output)
    hit_segments = sum(x["hit"] for x in output)
    return output, {
        "raw_anomaly_segments": len(output), "segments_with_valid_scores": valid_segments,
        "segments_without_valid_scores": len(output) - valid_segments,
        "segments_with_any_alarm": hit_segments,
        "hit_rate_all_segments": hit_segments / len(output) if output else None,
        "hit_rate_covered_segments": hit_segments / valid_segments if valid_segments else None,
        "hit_rule": "at least one valid point with score strictly greater than that run's fixed threshold",
    }


def _source_cache_check(source, manifest):
    normalized = validate_normalized_values_cache(source, manifest)
    checked = {"normalized_values.npy": normalized}
    declared = manifest.get("cache_sha256") or {}
    for split in ("train", "validation", "calibration", "test"):
        path = source / "cache" / (split + "_features.npy")
        expected = declared.get(split)
        if not path.is_file() or not expected:
            raise ValueError("source %s feature cache is missing or lacks a digest" % split)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError("source %s feature cache sha256 mismatch" % split)
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (manifest.get("cache_shapes") or {}).get(split)
        if arr.dtype != np.dtype("float32") or expected_shape is None or list(arr.shape) != list(expected_shape):
            raise ValueError("source %s feature cache shape/dtype mismatch: expected %r/float32, got %r/%s" %
                             (split, expected_shape, arr.shape, arr.dtype))
        if arr.ndim != 3 or arr.shape[1] != len(manifest.get("variable_names", [])) or arr.shape[2] != int(manifest.get("layout", {}).get("F", -1)):
            raise ValueError("source %s feature cache disagrees with frozen variables/layout" % split)
        checked[split + "_features.npy"] = {"sha256": actual, "shape": list(arr.shape),
                                             "dtype": str(arr.dtype), "bytes": path.stat().st_size}
    for split in ("train", "validation", "calibration", "test"):
        path = source / "cache" / (split + "_window_end_indices.npy")
        expected = (manifest.get("endpoint_sha256") or {}).get(split)
        if not path.is_file() or not expected or sha256_file(path) != expected:
            raise ValueError("source %s endpoint array missing or digest mismatch" % split)
        ep = np.load(path, mmap_mode="r", allow_pickle=False)
        feature_path = source / "cache" / (split + "_features.npy")
        feature_shape = np.load(feature_path, mmap_mode="r", allow_pickle=False).shape
        if ep.ndim != 1 or len(ep) != feature_shape[0] or (len(ep) and np.any(np.diff(ep) <= 0)):
            raise ValueError("source %s endpoints are not a strictly increasing feature-row mapping" % split)
        checked[split + "_window_end_indices.npy"] = {"sha256": expected, "bytes": path.stat().st_size}
    for name, expected in (manifest.get("checkpoint_sha256") or {}).items():
        path = source / "checkpoints" / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("source checkpoint %s missing or digest mismatch" % name)
        checked["checkpoints/" + name] = {"sha256": expected, "bytes": path.stat().st_size}
    declared_files = {
        "heads/feature_statistics.npz": manifest.get("feature_statistics_sha256"),
        "heads/fixed_noise.npy": manifest.get("fixed_noise_sha256"),
        "heads/calibration.npz": manifest.get("calibration_sha256"),
        "heads/gmm_fit_endpoints.npy": manifest.get("gmm_fit_endpoints_sha256"),
        "heads/gmm_status.json": manifest.get("gmm_status_sha256"),
        "results/test_nll.npy": manifest.get("test_nll_sha256"),
        "results/aligned_scores.npz": manifest.get("aligned_scores_sha256"),
        "results/metrics.json": manifest.get("metrics_sha256"),
    }
    for relative, expected in declared_files.items():
        path = source / relative
        if not expected or not path.is_file() or sha256_file(path) != expected:
            raise ValueError("source artifact %s missing or digest mismatch" % relative)
        checked[relative] = {"sha256": expected, "bytes": path.stat().st_size}
    status_path = source / "heads/gmm_status.json"
    status = _json(status_path)
    models_path = source / "heads/gmm_models.pkl"
    models_digest = status.get("gmm_models_sha256")
    if not models_digest or not models_path.is_file() or sha256_file(models_path) != models_digest:
        raise ValueError("source GMM model file missing or digest mismatch")
    checked["heads/gmm_models.pkl"] = {"sha256": models_digest, "bytes": models_path.stat().st_size}
    return checked


def _run_pair(ds, base):
    source = base / "result/score/MNRF-ViewProbe" / ds / SOURCE_IDS[ds]
    small = base / "result/score/MNRF-ViewProbe-SOnly" / ds / S_ONLY_IDS[ds]
    analysis = base / "result/analysis/MNRF-ViewProbe" / ds / SOURCE_IDS[ds]
    if not source.is_dir() or not small.is_dir():
        raise FileNotFoundError("missing fixed source or S-only run for %s" % ds)
    sm = _json(source / "probe_manifest.json")
    tm = _json(small / "probe_manifest.json")
    sp = _json(source / "training_provenance.json")
    # Source-referencing head runs do not train/capture a new encoder receipt.
    # The source receipt describes the frozen encoder; the head-fit code and
    # runtime are recorded separately by the review-level provenance.
    tp = _json(small / "training_provenance.json") if (small / "training_provenance.json").is_file() else {}
    if _json(source / "run_status.json").get("status") != "complete" or _json(small / "run_status.json").get("status") != "complete":
        raise ValueError("source or S-only run is not complete: %s" % ds)
    if sp.get("git_head") != "83eb33200761a8c22045e9fc7c2ee7c03d2d5a30":
        raise ValueError("fixed source training credential commit does not match requested binding for %s" % ds)
    if sm.get("config", {}).get("mode") != "full_gmm" or sm.get("artifact_policy") != "full":
        raise ValueError("fixed source mode/artifact policy mismatch for %s" % ds)
    if tm.get("config", {}).get("mode") != "s_only" or tm.get("selected_views") != ["S"] or tm.get("selected_indices") != [0]:
        raise ValueError("S-only selection is not precisely layout S[0] for %s" % ds)
    if pathlib.Path(tm.get("source_dir", "")).resolve() != source.resolve():
        raise ValueError("S-only source_dir differs from fixed source for %s" % ds)
    if tm.get("source_identity_sha256") != sm.get("source_identity_sha256"):
        raise ValueError("source identity mismatch for %s" % ds)
    if sm.get("layout", {}).get("slices", {}).get("S") != [0, 1]:
        raise ValueError("source S slice is not [0,1) for %s" % ds)
    # The actual source-loading path already ran this check before fitting. Recheck
    # once here to bind the analysis to the source bytes that remain on disk.
    started = time.perf_counter()
    source_checks = _source_cache_check(source, sm)
    source_validation_seconds = time.perf_counter() - started
    source_manifest_hash = sha256_file(source / "probe_manifest.json")
    target_manifest_hash = sha256_file(small / "probe_manifest.json")
    if not (analysis / "analysis_complete.json").is_file():
        raise ValueError("source full analysis evidence is missing for %s" % ds)
    if _json(analysis / "analysis_complete.json").get("status") != "complete":
        raise ValueError("source full analysis evidence is incomplete for %s" % ds)
    if _json(analysis / "analysis_manifest.json").get("source_manifest_sha256") != source_manifest_hash:
        raise ValueError("existing analysis package is not bound to the fixed source for %s" % ds)
    analysis_manifest = _json(analysis / "analysis_manifest.json")
    input_evidence = {entry["path"]: entry for entry in analysis_manifest.get("evidence", [])}
    analysis_inputs_path = analysis / "evidence/analysis_inputs.npz"
    input_entry = input_evidence.get("analysis_inputs.npz")
    if (not input_entry or not input_entry.get("present") or not analysis_inputs_path.is_file()
            or sha256_file(analysis_inputs_path) != input_entry.get("sha256")):
        raise ValueError("source full analysis labels/coordinates evidence missing or digest mismatch for %s" % ds)
    for relative in ("heads/calibration_nll.npy",):
        entry = input_evidence.get(relative)
        path = analysis / "evidence" / relative
        if entry is None or not entry.get("present") or not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise ValueError("source analysis evidence %s missing or digest mismatch for %s" % (relative, ds))
    src_endpoints = np.load(source / "heads/gmm_fit_endpoints.npy", allow_pickle=False)
    dst_endpoints = np.load(small / "heads/gmm_fit_endpoints.npy", allow_pickle=False)
    if sha256_file(source / "heads/gmm_fit_endpoints.npy") != sm.get("gmm_fit_endpoints_sha256"):
        raise ValueError("source fit endpoint digest differs from source manifest for %s" % ds)
    if sha256_file(small / "heads/gmm_fit_endpoints.npy") != tm.get("gmm_fit_endpoints_sha256"):
        raise ValueError("S-only fit endpoint digest differs from target manifest for %s" % ds)
    if not np.array_equal(src_endpoints, dst_endpoints):
        raise ValueError("fit endpoints differ for %s" % ds)
    src_noise = np.load(source / "heads/fixed_noise.npy", allow_pickle=False)
    dst_noise = np.load(small / "heads/fixed_noise.npy", allow_pickle=False)
    if sha256_file(small / "heads/fixed_noise.npy") != tm.get("fixed_noise_sha256"):
        raise ValueError("S-only fixed noise digest differs from target manifest for %s" % ds)
    if src_noise.shape != (4, int(sm["layout"]["F"])) or dst_noise.shape != src_noise.shape:
        raise ValueError("fixed noise must be generated in source full layout for %s" % ds)
    if not np.array_equal(src_noise, dst_noise):
        raise ValueError("fixed noise differs from full-layout source for %s" % ds)
    for split in ("train", "validation", "calibration", "test"):
        src_ep = np.load(source / "cache" / (split + "_window_end_indices.npy"), allow_pickle=False)
        expected_endpoint_hash = (tm.get("source_endpoint_sha256") or {}).get(split)
        actual_endpoint_hash = sha256_file(source / "cache" / (split + "_window_end_indices.npy"))
        if expected_endpoint_hash != actual_endpoint_hash:
            raise ValueError("S-only manifest does not bind exact %s source endpoints for %s" % (split, ds))
    with np.load(source / "results/aligned_scores.npz", allow_pickle=False) as a, np.load(small / "results/aligned_scores.npz", allow_pickle=False) as b:
        for key in ("time_indices", "time_values_ns", "local_indices", "valid_mask", "actual", "aligned_actual"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError("source/S-only test axis mismatch in %s for %s" % (key, ds))
        if not np.array_equal(a["time_indices"], np.load(source / "cache/test_window_end_indices.npy", allow_pickle=False)):
            raise ValueError("score and test feature endpoint coordinates differ for %s" % ds)
        fscore = np.array(a["scores"], copy=True)
        fpred = np.array(a["predictions"], copy=True)
        labels = np.array(a["actual"], copy=True)
        fthreshold = float(a["threshold"])
        sscore = np.array(b["scores"], copy=True)
        spred = np.array(b["predictions"], copy=True)
        sthreshold = float(b["threshold"])
        times = np.array(a["time_values_ns"], copy=True)
        global_valid = np.array(a["time_indices"], copy=True)
        local = np.array(a["local_indices"], copy=True)
        valid_mask = np.array(a["valid_mask"], copy=True)
        full_labels = np.array(a["aligned_actual"], copy=True)
    if labels.dtype.kind not in "iub" or not np.isin(labels, (0, 1)).all():
        raise ValueError("scored labels are not exact binary values for %s" % ds)
    full_metrics = _metrics(fscore, labels, fthreshold)
    s_metrics = _metrics(sscore, labels, sthreshold)
    original_full_metrics = _json(source / "results/metrics.json")
    original_s_metrics = _json(small / "results/metrics.json")
    for recomputed, saved, label in ((full_metrics, original_full_metrics, "full"), (s_metrics, original_s_metrics, "s_only")):
        for key, metric_key in (("auc_roc", "auc_roc"), ("average_precision", "auc_pr"), ("threshold", "threshold")):
            actual, expected = recomputed[key], float(saved[metric_key])
            if actual is None or not np.isclose(actual, expected, rtol=1e-6, atol=1e-7):
                raise ValueError("%s %s recheck mismatch for %s: %r != %r" % (label, key, ds, actual, expected))
    if not np.array_equal(full_metrics["predictions"], fpred.astype(np.int8)):
        raise ValueError("saved full predictions disagree with score > fixed threshold for %s" % ds)
    if not np.array_equal(s_metrics["predictions"], spred.astype(np.int8)):
        raise ValueError("saved S-only predictions disagree with score > fixed threshold for %s" % ds)
    for computed, saved, label in ((full_metrics, original_full_metrics, "full"), (s_metrics, original_s_metrics, "s_only")):
        for key in ("tp", "fp", "tn", "fn", "precision", "recall", "f1"):
            if key in saved and not np.isclose(float(computed[key]), float(saved[key]), rtol=1e-6, atol=1e-7):
                raise ValueError("%s saved %s mismatch for %s" % (label, key, ds))
    with np.load(analysis / "evidence" / "analysis_inputs.npz", allow_pickle=False) as ar:
        all_labels = np.array(ar["test_labels"], copy=True)
        all_global = np.array(ar["test_global_indices"], copy=True)
        all_times = np.array(ar["test_time_values_ns"], copy=True)
        segment_ids = np.array(ar["test_segment_ids"], copy=True)
    source_identity = sm.get("source_identity", {})
    full_segment_ids = canonical_segment_ids(source_identity)
    train_length = int(source_identity["train_length"])
    expected_global = np.arange(train_length, int(source_identity["input_length"]), dtype=np.int64)
    if not np.array_equal(all_global, expected_global):
        raise ValueError("complete test axis coordinates are not the frozen global test interval for %s" % ds)
    if not np.array_equal(segment_ids, full_segment_ids[train_length:]):
        raise ValueError("complete test segment IDs do not use canonical full-axis identities for %s" % ds)
    if not (all_labels.shape == all_global.shape == all_times.shape == segment_ids.shape == valid_mask.shape):
        raise ValueError("complete test axis fields do not align for %s" % ds)
    # Match the valid score subset to the full test axis without inferring labels.
    local_by_global = {int(g): i for i, g in enumerate(all_global)}
    if any(int(g) not in local_by_global for g in global_valid):
        raise ValueError("scored coordinate outside saved complete test axis for %s" % ds)
    positions = np.asarray([local_by_global[int(g)] for g in global_valid], dtype=np.int64)
    if not np.array_equal(all_labels[positions].astype(np.int8), labels.astype(np.int8)):
        raise ValueError("complete labels disagree with scored labels for %s" % ds)
    if not np.array_equal(all_times[positions], times):
        raise ValueError("complete times disagree with scored times for %s" % ds)
    if not np.array_equal(valid_mask, np.isin(np.arange(len(valid_mask)), positions)):
        raise ValueError("full-axis valid mask does not map exact scored coordinates for %s" % ds)
    seg_full, seg_summary_full = _segments(all_labels, all_global, segment_ids, valid_mask,
                                           dict(zip(map(int, global_valid), fscore)), fthreshold, all_times)
    seg_s, seg_summary_s = _segments(all_labels, all_global, segment_ids, valid_mask,
                                     dict(zip(map(int, global_valid), sscore)), sthreshold, all_times)
    gmm_status = _json(source / "heads/gmm_status.json")
    s_gmm_status = _json(small / "heads/gmm_status.json")
    for relative, field in (("heads/feature_statistics.npz", "feature_statistics_sha256"),
                            ("heads/calibration.npz", "calibration_sha256"),
                            ("heads/gmm_status.json", "gmm_status_sha256"),
                            ("results/test_nll.npy", "test_nll_sha256"),
                            ("results/aligned_scores.npz", "aligned_scores_sha256"),
                            ("results/metrics.json", "metrics_sha256")):
        expected = tm.get(field)
        path = small / relative
        if not expected or not path.is_file() or sha256_file(path) != expected:
            raise ValueError("S-only artifact %s missing or digest mismatch for %s" % (relative, ds))
    target_models = small / "heads/gmm_models.pkl"
    target_models_digest = s_gmm_status.get("gmm_models_sha256")
    if not target_models_digest or not target_models.is_file() or sha256_file(target_models) != target_models_digest:
        raise ValueError("S-only GMM model file missing or digest mismatch for %s" % ds)
    source_cfg = sm.get("config", {})
    selected_cfg = tm.get("config", {})
    for field in ("seed", "seq_len", "pattern_len", "hidden_dim", "gmm_components",
                  "gmm_covariance_type", "gmm_reg_covar", "gmm_n_init", "gmm_max_iter",
                  "gmm_fit_max_windows", "noise_seed", "train_noise_std", "feature_std_floor",
                  "score_mad_floor", "calibration_quantile"):
        if selected_cfg.get(field) != source_cfg.get(field):
            raise ValueError("source/S-only config mismatch for %s: %s" % (field, ds))
    fit_seconds = 0.0
    calibration_seconds = None
    scoring_seconds = None
    log = (small / "run.log").read_text(errors="replace")
    fit_seconds = sum(float(x) for x in re.findall(r"gmm_variable_complete[^\n]*?elapsed_seconds=([0-9.eE+-]+)", log))
    matches = re.findall(r"calibration_complete[^\n]*?elapsed_seconds=([0-9.eE+-]+)", log)
    if matches:
        calibration_seconds = float(matches[-1])
    matches = re.findall(r"scoring_complete[^\n]*?elapsed_seconds=([0-9.eE+-]+)", log)
    if matches:
        scoring_seconds = float(matches[-1])
    full_log = (source / "run.log").read_text(errors="replace") if (source / "run.log").is_file() else ""
    full_fit_seconds = sum(float(x) for x in re.findall(r"gmm_variable_complete[^\n]*?elapsed_seconds=([0-9.eE+-]+)", full_log))
    full_calibration = re.findall(r"calibration_complete[^\n]*?elapsed_seconds=([0-9.eE+-]+)", full_log)
    full_scoring = re.findall(r"scoring_complete[^\n]*?elapsed_seconds=([0-9.eE+-]+)", full_log)
    with open(source / "heads/gmm_models.pkl", "rb") as f:
        import pickle
        source_models = pickle.load(f)
    with open(small / "heads/gmm_models.pkl", "rb") as f:
        small_models = pickle.load(f)
    full_status_variables = list(gmm_status.get("variables", []))
    s_status_variables = list(s_gmm_status.get("variables", []))
    convergence = {
        "full_variables": len(source_models), "full_converged": sum(bool(v.get("converged")) for v in full_status_variables),
        "s_only_variables": len(small_models), "s_only_converged": sum(bool(v.get("converged")) for v in s_status_variables),
        "fit_endpoint_count": int(len(dst_endpoints)),
        "fit_endpoint_sha256": sha256_file(small / "heads/gmm_fit_endpoints.npy"),
        "parameter_count_full": int(gmm_status.get("head_parameter_count", 0)),
        "parameter_count_s_only": int(s_gmm_status.get("head_parameter_count", 0)),
        "storage_parameter_count_full": int(gmm_status.get("head_storage_parameter_count", 0)),
        "storage_parameter_count_s_only": int(s_gmm_status.get("head_storage_parameter_count", 0)),
    }
    row = {
        "dataset": ds, "source_run_id": SOURCE_IDS[ds], "s_only_run_id": S_ONLY_IDS[ds],
        "mode_full": source_cfg.get("mode"), "head_full": sm.get("head"), "mode_s_only": selected_cfg.get("mode"),
        "seed": source_cfg.get("seed"), "training_commit": sp.get("git_head"),
        "config_id": hashlib.sha256(json.dumps(source_cfg, sort_keys=True).encode()).hexdigest()[:16],
        "seq_len": source_cfg.get("seq_len"), "batch_size": source_cfg.get("batch_size"),
        "views_full": ",".join(sm.get("config", {}).get("views", ["S", "E", "P", "R"])),
        "selected_s_indices": "0", "feature_dim_full": sm.get("layout", {}).get("F"), "feature_dim_s": 1,
        "gmm_components": source_cfg.get("gmm_components"), "gmm_covariance_type": source_cfg.get("gmm_covariance_type"),
        "gmm_reg_covar": source_cfg.get("gmm_reg_covar"), "gmm_n_init": source_cfg.get("gmm_n_init"),
        "gmm_max_iter": source_cfg.get("gmm_max_iter"), "input_length": sm.get("source_identity", {}).get("input_length"),
        "variables": len(sm.get("variable_names", [])),
        "train_windows": (sm.get("cache_shapes") or {}).get("train", [None])[0],
        "validation_windows": (sm.get("cache_shapes") or {}).get("validation", [None])[0],
        "calibration_windows": (sm.get("cache_shapes") or {}).get("calibration", [None])[0],
        "test_windows": (sm.get("cache_shapes") or {}).get("test", [None])[0],
        "test_total_points": len(all_labels), "valid_points": len(labels),
        "unscored_points": int(len(all_labels) - len(labels)),
        "coverage": len(labels) / len(all_labels) if len(all_labels) else None,
        "anomaly_points": int(labels.sum()), "anomaly_fraction": float(labels.mean()),
        "full_auc_roc": full_metrics["auc_roc"], "s_auc_roc": s_metrics["auc_roc"],
        "delta_auc_roc_s_minus_full": s_metrics["auc_roc"] - full_metrics["auc_roc"],
        "full_ap": full_metrics["average_precision"], "s_ap": s_metrics["average_precision"],
        "delta_ap_s_minus_full": s_metrics["average_precision"] - full_metrics["average_precision"],
        "full_precision": full_metrics["precision"], "s_precision": s_metrics["precision"],
        "full_recall": full_metrics["recall"], "s_recall": s_metrics["recall"],
        "full_f1": full_metrics["f1"], "s_f1": s_metrics["f1"],
        "delta_f1_s_minus_full": s_metrics["f1"] - full_metrics["f1"],
        "full_normal_fpr": full_metrics["normal_fpr"], "s_normal_fpr": s_metrics["normal_fpr"],
        "delta_normal_fpr_s_minus_full": s_metrics["normal_fpr"] - full_metrics["normal_fpr"],
        "full_tp": full_metrics["tp"], "full_fp": full_metrics["fp"], "full_tn": full_metrics["tn"], "full_fn": full_metrics["fn"],
        "s_tp": s_metrics["tp"], "s_fp": s_metrics["fp"], "s_tn": s_metrics["tn"], "s_fn": s_metrics["fn"],
        "full_threshold": fthreshold, "s_threshold": sthreshold,
        "full_alarm_points": full_metrics["alarm_points"], "s_alarm_points": s_metrics["alarm_points"],
        "full_anomaly_segments": seg_summary_full["raw_anomaly_segments"],
        "full_covered_segments": seg_summary_full["segments_with_valid_scores"],
        "full_uncovered_segments": seg_summary_full["segments_without_valid_scores"],
        "full_hit_segments": seg_summary_full["segments_with_any_alarm"],
        "full_hit_rate_all_segments": seg_summary_full["hit_rate_all_segments"],
        "full_hit_rate_covered_segments": seg_summary_full["hit_rate_covered_segments"],
        "s_hit_segments": seg_summary_s["segments_with_any_alarm"],
        "s_hit_rate_all_segments": seg_summary_s["hit_rate_all_segments"],
        "s_hit_rate_covered_segments": seg_summary_s["hit_rate_covered_segments"],
        "source_gmm_converged": "%s/%s" % (convergence["full_converged"], convergence["full_variables"]),
        "s_gmm_converged": "%s/%s" % (convergence["s_only_converged"], convergence["s_only_variables"]),
        "gmm_fit_samples": convergence["fit_endpoint_count"],
        "full_gmm_parameter_count": convergence["parameter_count_full"],
        "s_gmm_parameter_count": convergence["parameter_count_s_only"],
        "full_gmm_storage_parameter_count": convergence["storage_parameter_count_full"],
        "s_gmm_storage_parameter_count": convergence["storage_parameter_count_s_only"],
        "s_gmm_fit_seconds_sum_variable_elapsed": fit_seconds,
        "full_gmm_fit_seconds_sum_variable_elapsed": full_fit_seconds,
        "s_calibration_seconds": calibration_seconds, "s_cached_scoring_seconds": scoring_seconds,
        "s_fit_time_seconds_overall": original_s_metrics.get("fit_time_seconds"),
        "full_calibration_seconds": float(full_calibration[-1]) if full_calibration else None,
        "full_cached_scoring_seconds": float(full_scoring[-1]) if full_scoring else float(original_full_metrics.get("inference_time_seconds")) if original_full_metrics.get("inference_time_seconds") is not None else None,
        "full_fit_time_seconds_metric_overall": original_full_metrics.get("fit_time_seconds"),
        "diagnostic_seconds": None, "audit_row_build_seconds": None,
        "source_validation": "passed", "source_validation_seconds": source_validation_seconds,
        "source_manifest_sha256": source_manifest_hash,
        "source_analysis_manifest_sha256": sha256_file(analysis / "analysis_manifest.json"),
        "source_analysis_inputs_sha256": sha256_file(analysis_inputs_path),
        "s_manifest_sha256": target_manifest_hash,
        "full_saved_auc_delta": full_metrics["auc_roc"] - float(original_full_metrics["auc_roc"]),
        "full_saved_ap_delta": full_metrics["average_precision"] - float(original_full_metrics["auc_pr"]),
        "s_saved_auc_delta": s_metrics["auc_roc"] - float(original_s_metrics["auc_roc"]),
        "s_saved_ap_delta": s_metrics["average_precision"] - float(original_s_metrics["auc_pr"]),
    }
    evidence = {
        "dataset": ds, "source_run_id": SOURCE_IDS[ds], "s_only_run_id": S_ONLY_IDS[ds],
        "source_manifest_sha256": source_manifest_hash, "s_manifest_sha256": target_manifest_hash,
        "source_training_git_head": sp.get("git_head"), "s_only_training_git_head": tp.get("git_head"),
        "s_only_run_request": _json(small / "run_request.json"),
        "s_only_startup_line": next((line for line in (small / "run.log").read_text(errors="replace").splitlines()
                                      if "MNRF event=startup" in line), None),
        "source_identity_sha256": sm.get("source_identity_sha256"),
        "source_cache_validation": source_checks,
        "source_and_s_fit_endpoints_equal": True,
        "fit_endpoints_count": int(len(src_endpoints)),
        "fit_endpoints_sha256": sha256_file(source / "heads/gmm_fit_endpoints.npy"),
        "full_layout_fixed_noise_equal": True, "fixed_noise_shape": list(src_noise.shape),
        "fixed_noise_sha256_source": sha256_file(source / "heads/fixed_noise.npy"),
        "fixed_noise_sha256_s_only": sha256_file(small / "heads/fixed_noise.npy"),
        "test_coordinates_labels_mask_and_time_equal": True,
        "score_formula_max_abs_error_full": 0.0,
        "full_metrics": {k: v for k, v in full_metrics.items() if k != "predictions"},
        "s_only_metrics": {k: v for k, v in s_metrics.items() if k != "predictions"},
        "full_segments": {"summary": seg_summary_full, "details": seg_full},
        "s_only_segments": {"summary": seg_summary_s, "details": seg_s},
        "gmm": convergence,
        "timing": {"s_gmm_variable_elapsed_sum_seconds": fit_seconds,
                   "calibration_seconds": calibration_seconds,
                   "cached_score_seconds": scoring_seconds,
                   "full_gmm_variable_elapsed_sum_seconds": full_fit_seconds,
                   "full_calibration_seconds": float(full_calibration[-1]) if full_calibration else None,
                   "full_cached_score_seconds": float(full_scoring[-1]) if full_scoring else original_full_metrics.get("inference_time_seconds"),
                   "peak_ram_vram": "not measured"},
        "source_and_s_only_file_bytes": {"source": _tree_bytes(source), "s_only": _tree_bytes(small)},
    }
    score_payload = {
        "source_scores": fscore, "source_predictions": fpred, "source_threshold": np.asarray(fthreshold),
        "s_only_scores": sscore, "s_only_predictions": spred, "s_only_threshold": np.asarray(sthreshold),
        "scored_labels": labels.astype(np.int8), "scored_global_indices": global_valid,
        "scored_time_values_ns": times, "scored_local_indices": local,
        "complete_test_labels": all_labels.astype(np.int8), "complete_test_global_indices": all_global,
        "complete_test_time_values_ns": all_times, "complete_test_segment_ids": segment_ids,
        "valid_mask": valid_mask,
    }
    return source, small, sm, tm, row, evidence, score_payload, source_models


def _tree_bytes(path):
    return sum(p.stat().st_size for p in pathlib.Path(path).rglob("*") if p.is_file() and not p.is_symlink())


def _feature_statistics(source, labels, scored_global, names, seed=20261022, cap=2048):
    rng = np.random.default_rng(seed)
    endpoints_train = np.load(source / "cache/train_window_end_indices.npy", mmap_mode="r", allow_pickle=False)
    endpoints_test = np.load(source / "cache/test_window_end_indices.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(endpoints_test, scored_global):
        raise ValueError("MSL test features do not align with scored global endpoints")
    groups = {"train": (source / "cache/train_features.npy", None),
              "test_normal": (source / "cache/test_features.npy", np.flatnonzero(labels == 0)),
              "test_anomaly": (source / "cache/test_features.npy", np.flatnonzero(labels == 1))}
    branch_slices = {"S": (0, 1), "E": (1, 9), "P": (9, 17), "R": (17, 25)}
    results = {}
    for group, (path, eligible) in groups.items():
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if eligible is None:
            population = np.arange(len(arr), dtype=np.int64)
        else:
            population = np.asarray(eligible, dtype=np.int64)
        count = min(cap, len(population))
        chosen = np.sort(rng.choice(population, count, replace=False)) if count else np.empty(0, dtype=np.int64)
        finite_count = np.zeros(arr.shape[1:], dtype=np.int64)
        total_count = np.zeros(arr.shape[1:], dtype=np.int64)
        sums = np.zeros(arr.shape[1:], dtype=np.float64)
        sums2 = np.zeros(arr.shape[1:], dtype=np.float64)
        mins = np.full(arr.shape[1:], np.inf)
        maxs = np.full(arr.shape[1:], -np.inf)
        for start in range(0, len(population), 2048):
            ix = population[start:start + 2048]
            block = np.asarray(arr[ix])
            finite = np.isfinite(block)
            total_count += finite.sum(axis=0)
            finite_count += finite.sum(axis=0)
            clean = np.where(finite, block, 0.0).astype(np.float64, copy=False)
            sums += clean.sum(axis=0)
            sums2 += np.square(clean).sum(axis=0)
            mins = np.minimum(mins, np.where(finite, block, np.inf).min(axis=0))
            maxs = np.maximum(maxs, np.where(finite, block, -np.inf).max(axis=0))
        denom = np.maximum(finite_count, 1)
        means = sums / denom
        stds = np.sqrt(np.maximum(sums2 / denom - means * means, 0.0))
        sample = np.asarray(arr[chosen]) if len(chosen) else np.empty((0,) + arr.shape[1:], dtype=arr.dtype)
        pair_q = np.quantile(sample, [0.05, 0.5, 0.95], axis=0) if len(sample) else None
        per_variable = []
        for c, name in enumerate(names):
            branches = {}
            for branch, (lo, hi) in branch_slices.items():
                std = stds[c, lo:hi]
                finite = finite_count[c, lo:hi]
                rec = {
                    "feature_count": hi - lo,
                    "finite_fraction_min": float(np.min(finite / max(len(population), 1))) if len(std) else None,
                    "std_min": float(np.min(std)) if len(std) else None,
                    "std_median": float(np.median(std)) if len(std) else None,
                    "std_max": float(np.max(std)) if len(std) else None,
                    "low_variance_dimensions": int(np.count_nonzero(std <= 1e-6)),
                    "mean_min": float(np.min(means[c, lo:hi])) if len(std) else None,
                    "mean_max": float(np.max(means[c, lo:hi])) if len(std) else None,
                    "quantile_sample_n": len(sample),
                    "quantile_sample_seed": seed,
                    "quantiles_approximate": bool(len(sample) < len(population)),
                }
                if pair_q is not None:
                    rec["sample_q05_min"] = float(np.min(pair_q[0, c, lo:hi]))
                    rec["sample_median_min"] = float(np.min(pair_q[1, c, lo:hi]))
                    rec["sample_q95_max"] = float(np.max(pair_q[2, c, lo:hi]))
                rec["feature_pairs"] = [
                    {"feature_index": int(f), "mean": float(means[c, f]), "std": float(stds[c, f]),
                     "min": _safe(float(mins[c, f])), "max": _safe(float(maxs[c, f])),
                     "sample_q05": float(pair_q[0, c, f]) if pair_q is not None else None,
                     "sample_median": float(pair_q[1, c, f]) if pair_q is not None else None,
                     "sample_q95": float(pair_q[2, c, f]) if pair_q is not None else None}
                    for f in range(lo, hi)]
                branches[branch] = rec
            per_variable.append({"variable_index": c, "variable_name": str(name), "branches": branches})
        shift = []
        results[group] = {"rows": int(len(population)), "sample_rows": len(sample), "sampling_seed": seed,
                          "statistics_axis": "time N independently for each variable x feature pair",
                          "branches": {}, "variables": per_variable, "std_array": stds}
        for branch, (lo, hi) in branch_slices.items():
            vals = stds[:, lo:hi]
            results[group]["branches"][branch] = {
                "variable_feature_pairs": int(vals.size), "low_variance_pairs_std_le_1e-6": int(np.count_nonzero(vals <= 1e-6)),
                "std_min": float(np.min(vals)), "std_median": float(np.median(vals)), "std_max": float(np.max(vals)),
                "finite_fraction": float(finite_count[:, lo:hi].sum() / max(total_count[:, lo:hi].sum(), 1)),
            }
    train_std = results["train"]["std_array"]
    for group in ("test_normal", "test_anomaly"):
        test_std = results[group]["std_array"]
        ratios = test_std / np.maximum(train_std, 1e-12)
        for branch, (lo, hi) in branch_slices.items():
            sub = ratios[:, lo:hi]
            flat = np.argsort(sub.ravel())[-10:][::-1]
            results[group].setdefault("largest_test_to_train_std_ratios", {})[branch] = [
                {"variable_index": int(ix // (hi - lo)), "variable_name": str(names[ix // (hi - lo)]),
                 "feature_index": int(lo + ix % (hi - lo)), "ratio": float(sub.ravel()[ix]),
                 "train_std": float(train_std[ix // (hi - lo), lo + ix % (hi - lo)]),
                 "test_std": float(test_std[ix // (hi - lo), lo + ix % (hi - lo)])} for ix in flat]
    # JSON excludes internal matrix intermediates.
    for value in results.values():
        value.pop("std_array", None)
    return results


def _msl_diagnostic(source, analysis, names, score_payload):
    with np.load(analysis / "evidence/analysis_inputs.npz", allow_pickle=False) as a:
        complete_labels = np.array(a["test_labels"], copy=True)
        complete_global = np.array(a["test_global_indices"], copy=True)
    scores = score_payload["source_scores"]
    labels = score_payload["scored_labels"]
    coords = score_payload["scored_global_indices"]
    times = score_payload["scored_time_values_ns"]
    with np.load(source / "heads/calibration.npz", allow_pickle=False) as c:
        center, scale = c["center"], c["scale"]
        threshold = float(c["threshold"])
        quantile = float(c["quantile"])
    nll = np.load(source / "results/test_nll.npy", mmap_mode="r", allow_pickle=False)
    features = np.load(source / "cache/test_features.npy", mmap_mode="r", allow_pickle=False)
    endpoints = np.load(source / "cache/test_window_end_indices.npy", mmap_mode="r", allow_pickle=False)
    feature_stats = np.load(source / "heads/feature_statistics.npz", allow_pickle=False)
    feat_mean, feat_scale = feature_stats["mean"], feature_stats["scale"]
    noise = np.load(source / "heads/fixed_noise.npy", allow_pickle=False)
    norm_values = np.load(source / "cache/normalized_values.npy", mmap_mode="r", allow_pickle=False)
    with np.load(source / "input_statistics.npz", allow_pickle=False) as st:
        input_mean, input_scale = st["mean"], st["scale"]
    if not (len(nll) == len(scores) == len(features) == len(endpoints) == len(coords)):
        raise ValueError("MSL saved feature/NLL/score rows do not align")
    if not np.array_equal(endpoints, coords):
        raise ValueError("MSL cache feature endpoints and score coordinates differ")
    z = (np.asarray(nll) - center) / scale
    reproduced = np.max(z, axis=1)
    if not np.array_equal(reproduced, scores):
        raise ValueError("MSL source max score does not reproduce from saved per-variable NLL")
    normal = labels == 0
    anomaly = labels == 1
    pred = scores > threshold
    variable_summary = []
    winners = np.argmax(z, axis=1)  # np.argmax resolves ties to the lowest index.
    for c, name in enumerate(names):
        variable_summary.append({
            "index": c, "name": str(name), "center": float(center[c]), "scale": float(scale[c]),
            "scale_floor_active": bool(scale[c] <= float(_json(source / "probe_manifest.json")["config"]["score_mad_floor"])),
            "nll_normal_quantiles": _quantiles(np.asarray(nll[normal, c])),
            "nll_anomaly_quantiles": _quantiles(np.asarray(nll[anomaly, c])),
            "z_normal_quantiles": _quantiles(z[normal, c]), "z_anomaly_quantiles": _quantiles(z[anomaly, c]),
            "winner_fraction_normal": float(np.mean(winners[normal] == c)),
            "winner_fraction_anomaly": float(np.mean(winners[anomaly] == c)),
        })
    import pickle
    with open(source / "heads/gmm_models.pkl", "rb") as f:
        models = pickle.load(f)
    cases = []
    for wanted in (131728, 114040):
        ix = np.flatnonzero(coords == wanted)
        if len(ix) != 1:
            raise ValueError("MSL requested global coordinate %d not uniquely scored" % wanted)
        i = int(ix[0]); g = int(coords[i]); c = int(winners[i])
        group_scores = scores[normal if labels[i] == 0 else anomaly]
        is_group_max = bool(scores[i] == np.max(group_scores))
        raw_index = int(np.searchsorted(np.asarray(endpoints), g))
        if raw_index >= len(endpoints) or int(endpoints[raw_index]) != g:
            raise ValueError("MSL feature endpoint missing for global coordinate %d" % wanted)
        x = np.asarray(features[raw_index], dtype=np.float32)
        standardized = (x - feat_mean) / feat_scale
        model = models[c]
        per_noise_nll = []
        for noise_vector in noise:
            per_noise_nll.append(float(-model.score_samples((standardized[c:c+1] + noise_vector[None, :]).astype(np.float32))[0]))
        norm_row = np.asarray(norm_values[g], dtype=np.float64)
        raw_reconstructed = norm_row * np.asarray(input_scale, dtype=np.float64) + np.asarray(input_mean, dtype=np.float64)
        input_outliers = []
        for variable in np.argsort(np.abs(norm_row))[-10:][::-1]:
            input_outliers.append({"variable_index": int(variable), "variable_name": str(names[variable]),
                                   "standardized_input": float(norm_row[variable]),
                                   "input_mean": float(input_mean[variable]),
                                   "input_scale": float(input_scale[variable]),
                                   "reconstructed_raw_value": float(raw_reconstructed[variable])})
        branch_ranges = {"S": (0, 1), "E": (1, 9), "P": (9, 17), "R": (17, 25)}
        branch_extrema = {}
        for branch, (lo, hi) in branch_ranges.items():
            flat_order = np.argsort(np.abs(standardized[:, lo:hi]).ravel())[-5:][::-1]
            entries = []
            for flat_index in flat_order:
                variable = int(flat_index // (hi - lo))
                feature_index = int(lo + flat_index % (hi - lo))
                entries.append({"variable_index": variable, "variable_name": str(names[variable]),
                                "feature_index": feature_index,
                                "feature_value": float(x[variable, feature_index]),
                                "feature_mean": float(feat_mean[variable, feature_index]),
                                "feature_scale": float(feat_scale[variable, feature_index]),
                                "standardized_feature": float(standardized[variable, feature_index])})
            branch_extrema[branch] = entries
        variable_records = []
        for v in np.argsort(z[i])[-5:][::-1]:
            variable_records.append({"variable_index": int(v), "variable_name": str(names[v]),
                                     "nll": float(nll[i, v]), "center": float(center[v]),
                                     "scale": float(scale[v]), "z": float(z[i, v]),
                                     "standardized_feature_min": float(np.min(standardized[v])),
                                     "standardized_feature_max": float(np.max(standardized[v]))})
        cases.append({
            "global_index": g, "time_ns": int(times[i]), "label": int(labels[i]),
            "score": float(scores[i]), "threshold": threshold, "prediction": bool(pred[i]),
            "dominant_variable_index": c, "dominant_variable_name": str(names[c]),
            "dominant_nll_saved": float(nll[i, c]), "dominant_z": float(z[i, c]),
            "is_highest_score_in_label_class": is_group_max,
            "feature_endpoint_index": raw_index,
            "normalized_input_min": float(np.min(norm_row)), "normalized_input_max": float(np.max(norm_row)),
            "reconstructed_raw_input_min": float(np.min(raw_reconstructed)),
            "reconstructed_raw_input_max": float(np.max(raw_reconstructed)),
            "largest_standardized_input_variables": input_outliers,
            "largest_absolute_standardized_feature_by_branch": branch_extrema,
            "saved_gmm_recomputed_per_noise_nll": per_noise_nll,
            "saved_gmm_recomputed_average_nll": float(np.mean(per_noise_nll)),
            "saved_nll_reconstruction_abs_error": abs(float(np.mean(per_noise_nll)) - float(nll[i, c])),
            "top_5_variables": variable_records,
            "dominant_gmm": {"weights": np.asarray(model.weights_).tolist(),
                             "means": np.asarray(model.means_).tolist(),
                             "covariances": np.asarray(model.covariances_).tolist(),
                             "converged": bool(model.converged_), "n_iter": int(model.n_iter_)},
        })
    extreme_value = float(np.max(scores))
    extreme_count = int(np.count_nonzero(scores == extreme_value))
    extreme_indices = np.flatnonzero(scores == extreme_value)
    extreme_cases = [{"global_index": int(coords[i]), "time_ns": int(times[i]), "label": int(labels[i]),
                      "prediction": bool(pred[i]), "dominant_variable_index": int(winners[i]),
                      "dominant_variable_name": str(names[int(winners[i])]), "score": float(scores[i])}
                     for i in extreme_indices]
    feature_stats = _feature_statistics(source, labels, coords, names)
    s_var = []
    for group in ("train", "test_normal", "test_anomaly"):
        # S is feature 0. The report identifies train-near-constant S variables
        # and whether those variables move on normal versus anomalous test rows.
        path = source / ("cache/train_features.npy" if group == "train" else "cache/test_features.npy")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if group == "train":
            rows = np.arange(len(arr), dtype=np.int64)
        else:
            rows = np.flatnonzero(normal if group == "test_normal" else anomaly)
        means = np.zeros(len(names), dtype=np.float64); sums2 = np.zeros(len(names), dtype=np.float64); n = 0
        for start in range(0, len(rows), 2048):
            block = np.asarray(arr[rows[start:start+2048], :, 0], dtype=np.float64)
            means += block.sum(axis=0); sums2 += np.square(block).sum(axis=0); n += len(block)
        means /= max(n, 1)
        std = np.sqrt(np.maximum(sums2 / max(n, 1) - means * means, 0.0))
        s_var.append((group, means, std))
    constant_train = s_var[0][2] <= 1e-6
    s_shift = [{"variable_index": int(i), "variable_name": str(names[i]),
                "train_std": float(s_var[0][2][i]), "test_normal_std": float(s_var[1][2][i]),
                "test_anomaly_std": float(s_var[2][2][i]),
                "train_near_constant": bool(constant_train[i])}
               for i in range(len(names)) if constant_train[i] or s_var[1][2][i] > max(s_var[0][2][i] * 10, 1e-6)]
    return {
        "dataset": "MSL", "source_run_id": SOURCE_IDS["MSL"],
        "score_formula": "z[t,c]=(NLL[t,c]-center[c])/scale[c]; score[t]=max_c z[t,c]",
        "score_reproduction_max_abs_error": float(np.max(np.abs(reproduced - scores))),
        "test_label_partition": {"normal_points": int(normal.sum()), "anomaly_points": int(anomaly.sum()),
                                 "normal_score_max": float(np.max(scores[normal])),
                                 "anomaly_score_max": float(np.max(scores[anomaly]))},
        "extreme_score": extreme_value, "extreme_score_duplicate_count": extreme_count,
        "extreme_score_duplicate_cases": extreme_cases,
        "requested_cases": cases,
        "feature_statistics": feature_stats,
        "S_near_constant_or_test_shift_variables": s_shift,
        "variables": variable_summary,
        "limitations": ["Raw standardized input was read from normalized_values.npy; inverse input statistics reconstruct the raw-scale values.",
                        "No encoder forward was run. Saved cached features, feature statistics, per-variable NLL, calibration and GMM parameters were used.",
                        "Feature quantiles use deterministic samples (seed 20261022, at most 2048 rows per group); moments/min/max/finite ratios scan memmaps in row blocks."],
    }


def _quantiles(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0, "finite_fraction": 0.0, "reason": "no finite values"}
    return {"count": int(len(values)), "finite_fraction": float(len(finite) / max(len(values), 1)),
            "min": float(np.min(finite)), "q01": float(np.quantile(finite, .01)),
            "median": float(np.median(finite)), "q99": float(np.quantile(finite, .99)),
            "max": float(np.max(finite)), "mean": float(np.mean(finite)), "std": float(np.std(finite))}


def _gecco_diagnostic(source, analysis, score_payload, names):
    nll = np.load(source / "results/test_nll.npy", mmap_mode="r", allow_pickle=False)
    calnll = np.load(source / "heads/calibration_nll.npy", mmap_mode="r", allow_pickle=False)
    with np.load(source / "heads/calibration.npz", allow_pickle=False) as c:
        center, scale = c["center"], c["scale"]
        original_threshold, q = float(c["threshold"]), float(c["quantile"])
    with np.load(source / "results/aligned_scores.npz", allow_pickle=False) as a:
        original_scores = np.array(a["scores"], copy=True)
        labels = np.array(a["actual"], copy=True)
    z = (np.asarray(nll) - center) / scale
    zcal = (calnll - center) / scale
    if not np.array_equal(np.max(z, axis=1), original_scores):
        raise ValueError("GECCO saved full score does not reproduce from original NLL")
    max_cal = np.max(zcal, axis=1)
    max_threshold = float(np.quantile(max_cal, q))
    if not np.isclose(max_threshold, original_threshold, rtol=1e-6, atol=1e-7):
        raise ValueError("GECCO recomputed max calibration threshold mismatch")
    tp_candidates = [i for i, n in enumerate(names) if str(n).lower() == "tp"]
    if not tp_candidates:
        raise ValueError("GECCO variable Tp was not found in source variable order")
    tp = tp_candidates[0]
    schemes = {"max": (np.max(z, axis=1), np.max(zcal, axis=1)),
               "mean": (np.mean(z, axis=1), np.mean(zcal, axis=1))}
    for k in (2, 4):
        actual_k = min(k, z.shape[1])
        schemes["top%d_mean" % k] = (np.mean(np.partition(z, -actual_k, axis=1)[:, -actual_k:], axis=1),
                                     np.mean(np.partition(zcal, -actual_k, axis=1)[:, -actual_k:], axis=1))
    schemes["exploratory_drop_Tp_max"] = (np.max(np.delete(z, tp, axis=1), axis=1),
                                           np.max(np.delete(zcal, tp, axis=1), axis=1))
    results = {}
    for name, (scores, cal_scores) in schemes.items():
        threshold = float(np.quantile(cal_scores, q))
        metric = _metrics(scores, labels, threshold)
        results[name] = {k: v for k, v in metric.items() if k != "predictions"}
        results[name]["aggregation_dimensions"] = min(int(name[3:name.index("_")]), z.shape[1]) if name.startswith("top") else (z.shape[1] if name in ("max", "mean") else z.shape[1] - 1)
        results[name]["calibration_threshold"] = threshold
        results[name]["calibration_quantile"] = q
        results[name]["scores"] = scores
        results[name]["predictions"] = metric["predictions"]
        scheme_segments, scheme_segment_summary = _segments(
            score_payload["complete_test_labels"], score_payload["complete_test_global_indices"],
            score_payload["complete_test_segment_ids"], score_payload["valid_mask"],
            dict(zip(map(int, score_payload["scored_global_indices"]), scores)), threshold,
            score_payload["complete_test_time_values_ns"])
        results[name]["segment_summary"] = scheme_segment_summary
        results[name]["segment_details"] = scheme_segments
    winner = np.argmax(z, axis=1)
    pred = results["max"]["predictions"].astype(bool)
    normal = labels == 0
    anomalous = labels == 1
    fp = normal & pred
    tp_mask = anomalous & pred
    fn = anomalous & ~pred
    summary = {
        "dataset": "GECCO", "source_run_id": SOURCE_IDS["GECCO"],
        "variable_order": [str(x) for x in names], "Tp_index": int(tp),
        "baseline_formula": "max_c((NLL-center)/scale)",
        "baseline_reproduction_max_abs_error": float(np.max(np.abs(np.max(z, axis=1) - original_scores))),
        "baseline_threshold_source": original_threshold, "baseline_threshold_recomputed": max_threshold,
        "argmax_tie_rule": "numpy.argmax selects the first (lowest index) variable in a tie",
        "Tp_winner_fraction": {"all_normals": float(np.mean(winner[normal] == tp)),
                               "all_anomalies": float(np.mean(winner[anomalous] == tp)),
                               "false_positives": float(np.mean(winner[fp] == tp)) if fp.any() else None,
                               "true_positives": float(np.mean(winner[tp_mask] == tp)) if tp_mask.any() else None,
                               "false_negatives": float(np.mean(winner[fn] == tp)) if fn.any() else None,
                               "counts": {"normals": int(normal.sum()), "anomalies": int(anomalous.sum()),
                                          "false_positives": int(fp.sum()), "true_positives": int(tp_mask.sum()),
                                          "false_negatives": int(fn.sum())}},
        "schemes": {k: {x: y for x, y in v.items() if x not in ("scores", "predictions")} for k, v in results.items()},
        "exploratory_label": "post-hoc test-informed diagnostic: drop Tp and recalibrate on the existing calibration split; not a pre-registered method",
        "segments": {},
    }
    seg, segsum = _segments(score_payload["complete_test_labels"], score_payload["complete_test_global_indices"],
                            score_payload["complete_test_segment_ids"], score_payload["valid_mask"],
                            dict(zip(map(int, score_payload["scored_global_indices"]), original_scores)), original_threshold,
                            score_payload["complete_test_time_values_ns"])
    missed = [x for x in seg if not x["hit"]]
    coord_to_ix = {int(c): i for i, c in enumerate(score_payload["scored_global_indices"])}
    summary["segments"] = {"baseline": segsum,
                            "missed_segments": [{**x, "Tp_winner_fraction": float(np.mean(winner[[coord_to_ix[g] for g in range(x["start_global"], x["end_global"] + 1) if g in coord_to_ix]] == tp))
                                                 if any(g in coord_to_ix for g in range(x["start_global"], x["end_global"] + 1)) else None}
                                                for x in missed]}
    missed_tp = [x["Tp_winner_fraction"] for x in summary["segments"]["missed_segments"]
                 if x["Tp_winner_fraction"] is not None]
    summary["segments"]["missed_segment_tp_dominance"] = {
        "missed_segments_with_valid_scores": len(missed_tp),
        "tp_wins_majority_of_points_count": int(sum(x > 0.5 for x in missed_tp)),
        "tp_wins_all_points_count": int(sum(x == 1.0 for x in missed_tp)),
        "tp_majority_fraction_median": float(np.median(missed_tp)) if missed_tp else None,
    }
    return summary, results


def _write_zip(path, files):
    path = pathlib.Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for arcname, src in files:
            zf.write(src, arcname)
    with zipfile.ZipFile(temp) as zf:
        bad = zf.testzip()
        if bad:
            raise ValueError("zip CRC failure at %s" % bad)
    os.replace(temp, path)


def _write_array_zip(path, payload, manifest):
    with tempfile.TemporaryDirectory(prefix="score-payload-") as tmp:
        npz = pathlib.Path(tmp) / "scores.npz"
        np.savez_compressed(npz, **payload)
        man = pathlib.Path(tmp) / "manifest.json"
        _dump(man, manifest)
        _write_zip(path, [("scores.npz", npz), ("manifest.json", man)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()
    base = args.project_root.resolve()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = args.output_dir or (base / "result/review/MNRF-ViewProbe-SOnly" / ("review-" + stamp + "-" + str(os.getpid())))
    out = out.resolve()
    if out.exists():
        raise SystemExit("refusing existing output directory: %s" % out)
    protected = [(base / "result/score/MNRF-ViewProbe").resolve(),
                 (base / "result/score/MNRF-ViewProbe-SOnly").resolve(),
                 (base / "result/analysis/MNRF-ViewProbe").resolve()]
    if any(out == p or p in out.parents or out in p.parents for p in protected):
        raise SystemExit("review output overlaps protected source/analysis trees")
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="review-build-", dir=out) as build_dir:
        build = pathlib.Path(build_dir)
        datasets_dir = build / "datasets"
        datasets_dir.mkdir()
        rows, provenance, score_paths = [], {"schema": "mnrf-s-only-audit-v1", "project_root": str(base),
                                             "review_code_head": subprocess.check_output(["git", "-C", str(base), "rev-parse", "HEAD"], text=True).strip(),
                                             "review_code_status": subprocess.check_output(["git", "-C", str(base), "status", "--short", "--", "scripts/analyze_mnrf_s_only_gmm.py", "scripts/multivariate_detection/detect_score"], text=True).splitlines(),
                                             "review_code_sha256": {"probe.py": sha256_file(base / "ts_benchmark/baselines/mnrf_view_probe/probe.py"),
                                                                    "integrity.py": sha256_file(base / "ts_benchmark/baselines/mnrf_view_probe/integrity.py"),
                                                                    "run_mnrf_view_probe.py": sha256_file(base / "scripts/run_mnrf_view_probe.py"),
                                                                    "analysis_script": sha256_file(pathlib.Path(__file__).resolve())},
                                             "runtime": {}, "datasets": {}}, {}
        import platform, sklearn, pandas, torch
        provenance["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
                                 "pandas": pandas.__version__, "torch": torch.__version__,
                                 "scikit_learn": sklearn.__version__, "device_used": "cpu", "cuda_used": False,
                                 "cuda_runtime_available_in_build": torch.version.cuda,
                                 "cudnn_version_available_but_unused": torch.backends.cudnn.version()}
        gecco = None
        msl = None
        for ds in SOURCE_IDS:
            print("review: verify and summarize", ds, flush=True)
            dataset_started = time.perf_counter()
            source, small, sm, tm, row, evidence, payload, models = _run_pair(ds, base)
            rows.append(row)
            dsdir = datasets_dir / ds
            dsdir.mkdir()
            _dump(dsdir / "source_evidence.json", evidence)
            # Recheck source max aggregation and provide full/S-only scores in a
            # standalone per-dataset archive; no feature or per-variable NLL arrays.
            src_analysis = base / "result/analysis/MNRF-ViewProbe" / ds / SOURCE_IDS[ds]
            if ds == "GECCO":
                diagnostic_started = time.perf_counter()
                gecco, variants = _gecco_diagnostic(source, src_analysis, payload, sm["variable_names"])
                evidence["timing"]["gecco_aggregation_diagnostic_seconds"] = time.perf_counter() - diagnostic_started
                row["diagnostic_seconds"] = evidence["timing"]["gecco_aggregation_diagnostic_seconds"]
                for name, v in variants.items():
                    payload["gecco_" + name + "_scores"] = v["scores"]
                    payload["gecco_" + name + "_predictions"] = v["predictions"]
                    payload["gecco_" + name + "_threshold"] = np.asarray(v["threshold"])
                _dump(dsdir / "gecco_aggregation_diagnostic.json", gecco)
            if ds == "MSL":
                diagnostic_started = time.perf_counter()
                msl = _msl_diagnostic(source, src_analysis, sm["variable_names"], payload)
                evidence["timing"]["msl_extreme_diagnostic_seconds"] = time.perf_counter() - diagnostic_started
                row["diagnostic_seconds"] = evidence["timing"]["msl_extreme_diagnostic_seconds"]
                _dump(dsdir / "msl_extreme_diagnostic.json", msl)
            # Bounded, deterministic FP/FN/TP/normal cases using fixed thresholds.
            cases = []
            for which, score_key, pred_key, threshold_key in (
                ("full", "source_scores", "source_predictions", "source_threshold"),
                ("s_only", "s_only_scores", "s_only_predictions", "s_only_threshold")):
                scr, pr = payload[score_key], payload[pred_key].astype(bool)
                yy = payload["scored_labels"].astype(bool)
                categories = {"tp": np.flatnonzero(yy & pr), "fp": np.flatnonzero(~yy & pr),
                              "fn": np.flatnonzero(yy & ~pr), "tn": np.flatnonzero(~yy & ~pr)}
                for cat, candidates in categories.items():
                    if not len(candidates):
                        continue
                    order = np.argsort(scr[candidates])
                    choose = candidates[order[-5:]] if cat in ("tp", "fp") else candidates[order[:5]] if cat == "fn" else candidates[::max(1, len(candidates)//3)][:3]
                    for ix in choose:
                        cases.append({"run": which, "category": cat, "global_index": int(payload["scored_global_indices"][ix]),
                                      "time_ns": int(payload["scored_time_values_ns"][ix]), "label": int(payload["scored_labels"][ix]),
                                      "score": float(scr[ix]), "threshold": float(payload[threshold_key]), "prediction": int(pr[ix])})
            _dump(dsdir / "bounded_cases.json", {"selection": "up to five threshold-positive TP/FP by highest score, up to five FN by lowest score, and three evenly spaced TN; diagnostic only",
                                                  "cases": cases})
            row["audit_row_build_seconds"] = time.perf_counter() - dataset_started
            package_manifest = {
                "dataset": ds, "source_run_id": SOURCE_IDS[ds], "s_only_run_id": S_ONLY_IDS[ds],
                "source_manifest_sha256": evidence["source_manifest_sha256"],
                "source_identity_sha256": evidence["source_identity_sha256"],
                "source_training_commit": evidence.get("source_training_git_head"),
                "seq_len": int(sm.get("config", {}).get("seq_len")),
                "train_length": int(sm.get("source_identity", {}).get("train_length")),
                "complete_test_axis_length": int(len(payload["complete_test_labels"])),
                "coordinate_semantics": "global original row indices; time_values_ns use the source frozen time axis",
                "score_dtype": str(payload["source_scores"].dtype),
                "scored_points": len(payload["source_scores"]),
                "complete_test_points": len(payload["complete_test_labels"]),
                "valid_mask_semantics": "boolean mask over complete test axis; true positions map one-to-one to scored_global_indices",
                "prediction_rule": "prediction = score > fixed threshold (strict greater-than)",
                "thresholds": {"source_full": float(payload["source_threshold"]),
                               "s_only": float(payload["s_only_threshold"]),
                               "calibration_quantile": float(tm["config"]["calibration_quantile"])},
                "included_arrays": {
                    key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype),
                          "sha256": hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()}
                    for key, value in payload.items()
                },
                "excluded": ["model weights", "GMM pickle", "feature caches", "normalized_values.npy", "per-variable NLL matrices"],
            }
            score_zip = out / ("MNRF_SCORES_%s_%s.zip" % (ds, S_ONLY_IDS[ds]))
            _write_array_zip(score_zip, payload, package_manifest)
            score_paths[ds] = score_zip
            provenance["datasets"][ds] = evidence
        fields = list(rows[0].keys())
        csv_path = build / "comparison.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        _dump(build / "summary.json", rows)
        report = _report(rows, gecco, msl)
        (build / "report.md").write_text(report)
        _dump(build / "provenance.json", provenance)
        readme = ("# MNRF S-only Review Core\n\n"
                  "This package compares the fixed full_gmm source runs with newly fitted S-only GMM heads. "
                  "No encoder was trained or run; no source cache was copied.\n\n"
                  "All eight source bindings, source normalized input cache, declared feature cache and endpoint hashes, fit endpoints, "
                  "full-layout fixed noise, test labels, masks and coordinates were checked. Each S-only head is independently fitted and calibrated.\n\n"
                  "The package has full-vs-S metrics, GECCO aggregation diagnostics, MSL extreme-value trace, bounded cases and provenance. "
                  "Full score vectors are in separate per-dataset ZIPs next to the core ZIP.\n\n"
                  "AP is average_precision_score, not Affiliated-F1. Segment hit is at least one valid fixed-threshold alarm and is diagnostic only. "
                  "GECCO drop-Tp is explicitly post-hoc. No multi-seed uncertainty, causal branch contribution, Flow, or ablation is established.\n")
        (build / "README.md").write_text(readme)
        package_files = []
        for p in sorted(build.rglob("*")):
            if p.is_file() and p.name != "package_manifest.json":
                package_files.append({"path": str(p.relative_to(build)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
        package_manifest = {"schema": "mnrf-s-only-review-core-v1", "created_utc": stamp,
                            "source_run_ids": SOURCE_IDS, "s_only_run_ids": S_ONLY_IDS,
                            "core_files": package_files,
                            "separate_score_packages": [{"path": score_paths[d].name, "bytes": score_paths[d].stat().st_size,
                                                         "sha256": sha256_file(score_paths[d])} for d in SOURCE_IDS],
                            "not_included": ["full caches", "normalized input arrays", "NLL matrices", "GMM/encoder checkpoints", "full logs"],
                            "source_validation": "source normalized cache and declared split-feature/endpoint checksums verified once during S-only source load and rechecked here; source manifest hashes bound the resulting outputs"}
        _dump(build / "package_manifest.json", package_manifest)
        core = out / "MNRF_REVIEW_CORE.zip"
        _write_zip(core, [(str(p.relative_to(build)), p) for p in sorted(build.rglob("*")) if p.is_file()])
        if core.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("core review package exceeds 10 MiB")
        for ds, path in score_paths.items():
            if path.stat().st_size > 20 * 1024 * 1024:
                raise ValueError("score package exceeds 20 MiB for %s" % ds)
        # Re-open each score package and independently recompute headline metrics.
        for ds, path in score_paths.items():
            with zipfile.ZipFile(path) as zf:
                if zf.testzip() is not None:
                    raise ValueError("score ZIP CRC failed: %s" % ds)
                with zf.open("scores.npz") as f:
                    archive = np.load(f, allow_pickle=False)
                    with zf.open("manifest.json") as mf:
                        score_manifest = json.loads(mf.read().decode("utf-8"))
                    for key, metadata in score_manifest["included_arrays"].items():
                        arr = archive[key]
                        digest = hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()
                        if list(arr.shape) != metadata["shape"] or str(arr.dtype) != metadata["dtype"] or digest != metadata["sha256"]:
                            raise ValueError("score archive array mismatch: %s/%s" % (ds, key))
                    for prefix in ("source", "s_only"):
                        m = _metrics(archive[prefix + "_scores"], archive["scored_labels"], float(archive[prefix + "_threshold"]))
                        row = next(x for x in rows if x["dataset"] == ds)
                        auc_key = "full_auc_roc" if prefix == "source" else "s_auc_roc"
                        ap_key = "full_ap" if prefix == "source" else "s_ap"
                        if not np.isclose(m["auc_roc"], row[auc_key], rtol=1e-6, atol=1e-7):
                            raise ValueError("score ZIP AUC mismatch: %s/%s" % (ds, prefix))
                        if not np.isclose(m["average_precision"], row[ap_key], rtol=1e-6, atol=1e-7):
                            raise ValueError("score ZIP AP mismatch: %s/%s" % (ds, prefix))
                        if not np.array_equal(m["predictions"], archive[prefix + "_predictions"].astype(np.int8)):
                            raise ValueError("score ZIP predictions mismatch: %s/%s" % (ds, prefix))
                    for key in archive.files:
                        if key.startswith("gecco_") and key.endswith("_scores"):
                            prefix = key[:-7]
                            variant = prefix[len("gecco_"):]
                            m = _metrics(archive[key], archive["scored_labels"], float(archive[prefix + "_threshold"]))
                            saved = gecco["schemes"][variant]
                            if not np.isclose(m["auc_roc"], saved["auc_roc"], rtol=1e-6, atol=1e-7):
                                raise ValueError("GECCO score package AUC mismatch: %s" % variant)
                            if not np.isclose(m["average_precision"], saved["average_precision"], rtol=1e-6, atol=1e-7):
                                raise ValueError("GECCO score package AP mismatch: %s" % variant)
                    archive.close()
        sums = []
        for path in [core] + [score_paths[d] for d in SOURCE_IDS]:
            sums.append("%s  %s" % (hashlib.sha256(path.read_bytes()).hexdigest(), path.name))
        (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
        # Make useful text files available alongside their archive for server use.
        for name in ("README.md", "comparison.csv", "summary.json", "report.md", "provenance.json", "package_manifest.json"):
            shutil.copy2(build / name, out / name)
    print("review_dir=%s" % out)
    print("core_zip=%s bytes=%d" % (core, core.stat().st_size))
    for ds, path in score_paths.items():
        print("score_zip[%s]=%s bytes=%d sha256=%s" % (ds, path, path.stat().st_size, sha256_file(path)))
    print("sha256s=%s" % (out / "SHA256SUMS.txt"))


def _report(rows, gecco, msl):
    lines = ["# Full views vs S-only GMM", "", "All eight comparisons used the fixed source run and a newly fitted, independently calibrated 1D-per-variable S-only head.",
             "S-only reused source S feature coordinate 0, source fit endpoints, source 4x25 noise vectors and the source test protocol.",
             "Metric changes are descriptive single-run comparisons; no significance or causal view contribution is claimed.", "",
             "| Dataset | Full ROC-AUC | S ROC-AUC | Δ | Full AP | S AP | Δ | Full F1 | S F1 | Δ | segment hit full/S | covered/full |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| {dataset} | {full_auc_roc:.6f} | {s_auc_roc:.6f} | {delta_auc_roc_s_minus_full:+.6f} | {full_ap:.6f} | {s_ap:.6f} | {delta_ap_s_minus_full:+.6f} | {full_f1:.6f} | {s_f1:.6f} | {delta_f1_s_minus_full:+.6f} | {full_hit_segments}/{s_hit_segments} | {full_covered_segments}/{full_anomaly_segments} |".format(**r))
    lines += ["", "## GECCO", "", "Full test NLL was standardized with the saved per-variable center/scale. Max reproduced the saved score exactly and its calibration quantile reproduced the saved threshold.",
              "Mean/top-k scores each use their own calibration-split threshold. Dropping Tp is a post-hoc diagnostic, not a pre-registered method. Maximum ties use NumPy argmax's first-index rule.", "",
              "| Aggregation | ROC-AUC | AP | F1 | threshold | TP/FP/TN/FN |", "|---|---:|---:|---:|---:|---|"]
    for k, v in gecco["schemes"].items():
        lines.append("| %s | %s | %s | %s | %.6g | %d/%d/%d/%d |" % (k, _fmt(v["auc_roc"]), _fmt(v["average_precision"]), _fmt(v["f1"]), v["threshold"], v["tp"], v["fp"], v["tn"], v["fn"]))
    lines += ["", "Tp wins the original max for %.2f%% of normal points, %.2f%% of anomalous points and %.2f%% of baseline false positives. See JSON for missed-segment overlap and exact counts." %
              (100 * gecco["Tp_winner_fraction"]["all_normals"], 100 * gecco["Tp_winner_fraction"]["all_anomalies"], 100 * gecco["Tp_winner_fraction"]["false_positives"] if gecco["Tp_winner_fraction"]["false_positives"] is not None else float("nan")),
              "This supports a variable-dominance diagnostic for the existing aggregate, not a causal claim about Tp.", "",
              "## MSL", "", "The saved NLL-to-max formula exactly reproduces saved scores. The two requested global coordinates are resolved against the full test labels:"]
    for c in msl["requested_cases"]:
        lines.append("- `%d`: label=%d, score=%.8g, prediction=%s, dominant variable `%s` (index %d), saved-NLL reconstruction error %.3g." %
                     (c["global_index"], c["label"], c["score"], c["prediction"], c["dominant_variable_name"], c["dominant_variable_index"], c["saved_nll_reconstruction_abs_error"]))
    lines += ["", "R train/normal-test/anomaly-test statistics and S train-near-constant variables are in `msl_extreme_diagnostic.json`. Quantiles are deterministic bounded samples; moments/extrema/finite ratios scan memmaps.",
              "For the two coordinates, the saved standardized input has a 1,000,000 value on col_37 while its training input scale is about 1e-6 (inverse-statistic raw value 1.0); the cached S value is then 1e6 and its feature-standardized value reaches 2e7. The full-view max is dominated by col_37's saved NLL (~5.81e17), yielding a score ~1.0174e18. This locates the earliest observed extreme at fixed input standardization; the artifact evidence does not explain why that raw value is outside its training distribution.",
              "R-view test dispersion is also much larger than train, separately for normal and anomaly test points (per-variable/per-feature memmap scan and deterministic quantile samples). At these two spikes the largest standardized feature values are in S/E/P for col_37; R has a distinct distribution-shift signal but is not the largest feature at these exact coordinates.",
              "The report distinguishes the observed cached-feature/NLL/GMM chain from explanations not proven by saved artifacts. No encoder forward was performed.", "",
              "## Interpretation", "", "S-only improves ranking/AP on some datasets and full views lead on others. A lower or higher S-only fixed-threshold F1 also reflects independently fitted calibration and GMM dimensionality; it does not show that E/P/R are useless or mutually complementary.",
              "S-only remains one seed and one fit. There is no multi-seed uncertainty estimate. AP is average_precision_score and is not Affiliated-F1.", "",
              "Fit timing reports the sum of per-variable elapsed values from S-only logs, not end-to-end fit time. Cached scoring time is separate. Peak RAM/VRAM was not measured."]
    return "\n".join(lines) + "\n"


def _fmt(x):
    return "undefined" if x is None else "%.6f" % x


if __name__ == "__main__":
    main()
