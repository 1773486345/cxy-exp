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
    "training.jsonl",
    "run.log",
)
SPLITS = ("train", "validation", "calibration", "test")
BRANCHES = ("S", "E", "P", "R")


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
        sampled = np.asarray(array[indices], dtype=np.float32)
        split_result = {"shape": list(array.shape), "dtype": str(array.dtype), "sample_count": int(count), "branches": {}}
        for branch in BRANCHES:
            start, stop = slices[branch]
            values = sampled[:, :, start:stop].reshape(-1, stop - start)
            finite = np.isfinite(values)
            finite_values = values[finite]
            if finite_values.size:
                mean = float(np.mean(finite_values, dtype=np.float64))
                std = float(np.std(finite_values, dtype=np.float64))
                per_dimension_std = []
                for column in range(values.shape[1]):
                    column_values = values[finite[:, column], column]
                    per_dimension_std.append(float(np.std(column_values, dtype=np.float64))
                                               if column_values.size else np.nan)
                low_variance = int(np.sum(np.isfinite(per_dimension_std) & (np.asarray(per_dimension_std) < 1e-6)))
                summary = {"finite_fraction": float(np.mean(finite)), "mean": mean, "std": std,
                           "min": float(np.min(finite_values)), "max": float(np.max(finite_values)),
                           "low_variance_dimensions": low_variance}
            else:
                summary = {"finite_fraction": 0.0, "mean": None, "std": None, "min": None, "max": None,
                           "low_variance_dimensions": None, "reason": "no finite sampled values"}
            split_result["branches"][branch] = summary
        result["splits"][split] = split_result
    (output / "feature_overview.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _metric_recheck(source: Path, output: Path):
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
    if actual is None:
        metrics = {"status": "incomplete", "reason": "source has no test labels"}
    else:
        from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
        actual = actual.astype(np.int64, copy=False)
        predicted = (scores > threshold).astype(np.int64)
        counts = {"tp": int(np.sum((actual == 1) & (predicted == 1))),
                  "fp": int(np.sum((actual == 0) & (predicted == 1))),
                  "tn": int(np.sum((actual == 0) & (predicted == 0))),
                  "fn": int(np.sum((actual == 1) & (predicted == 0)))}
        precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="binary", zero_division=0)
        metrics = {"status": "complete", "average_precision": float(average_precision_score(actual, scores)),
                   "auc_roc": float(roc_auc_score(actual, scores)) if np.unique(actual).size == 2 else None,
                   "auc_roc_undefined_reason": None if np.unique(actual).size == 2 else "single class",
                   "threshold": threshold, "alarm_points": int(np.sum(predicted)), "anomaly_points": int(np.sum(actual)),
                   "coverage": float(np.mean(valid)), "precision": float(precision), "recall": float(recall),
                   "f1": float(f1), **counts}
        np.savez_compressed(output / "rechecked_scores.npz", scores=scores, predictions=predicted,
                            actual=actual, time_indices=endpoints, local_indices=local, valid_mask=valid,
                            time_values_ns=np.asarray(time_values) if time_values is not None else np.array([], dtype=np.int64))
    (output / "metric_recheck.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def _segment_diagnostics(source: Path, output: Path, manifest):
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
    if actual is None:
        segments = []
    else:
        source_identity = manifest.get("source_identity", {})
        boundaries = source_identity.get("test_segments", [])
        train_length = int(source_identity.get("train_length", 0))
        # Prefer the aligned label axis so anomalies with no valid endpoint
        # are retained as explicit no-valid segments.  Fall back to scored
        # labels for older result bundles that lack aligned_actual.
        if aligned_actual is not None:
            anomaly_axis = aligned_actual == 1
            score_axis = aligned_scores
            valid_axis = valid
            endpoint_axis = train_length + np.arange(len(aligned_actual), dtype=np.int64)
            time_by_global = {int(endpoint): int(time) for endpoint, time in zip(endpoints, times if times is not None else [])}
        else:
            anomaly_axis = actual == 1
            score_axis = scores
            valid_axis = valid
            endpoint_axis = endpoints
            time_by_global = {int(endpoint): int(time) for endpoint, time in zip(endpoints, times if times is not None else [])}
        runs = []
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
    payload = {"diagnostic_only": True, "segments": segments,
               "note": "anomaly hit means at least one fixed-threshold alarm; this is not Aff-F or point adjustment"}
    (output / "anomaly_segments.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _variable_diagnostics(source: Path, output: Path):
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
    aligned = np.load(source / "results" / "aligned_scores.npz", allow_pickle=False)
    actual = np.asarray(aligned["actual"], dtype=np.int64) if "actual" in aligned.files else None
    normal = actual == 0 if actual is not None else np.ones(len(nll), dtype=bool)
    anomalous = actual == 1 if actual is not None else np.zeros(len(nll), dtype=bool)
    variable = []
    for column in range(nll.shape[1]):
        values = np.asarray(nll[:, column], dtype=np.float64)
        standard = (values - float(center[column])) / float(scale[column])
        variable.append({"variable": column, "center": float(center[column]), "scale": float(scale[column]),
                         "scale_floor_active": bool(scale[column] <= 0.001 + 1e-7),
                         "normal_median": _finite_float(np.median(values[normal])) if np.any(normal) else None,
                         "anomaly_median": _finite_float(np.median(values[anomalous])) if np.any(anomalous) else None,
                         "normal_standard_q95": _finite_float(np.quantile(standard[normal], .95)) if np.any(normal) else None,
                         "anomaly_standard_q95": _finite_float(np.quantile(standard[anomalous], .95)) if np.any(anomalous) else None,
                         "max_score_frequency": None})
    owner_counts = np.zeros(nll.shape[1], dtype=np.int64)
    for start in range(0, len(nll), 4096):
        stop = min(len(nll), start + 4096)
        block = (np.asarray(nll[start:stop], dtype=np.float64) - center[None, :]) / scale[None, :]
        owner_counts += np.bincount(np.argmax(block, axis=1), minlength=nll.shape[1])
    for entry in variable:
        entry["max_score_frequency"] = int(owner_counts[entry["variable"]])
    result = {"status": "complete", "threshold": threshold, "variables": variable,
              "formula": "max_c((test_nll[:,c]-center[c])/scale[c])"}
    (output / "variable_diagnostics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _samples(source: Path, output: Path, manifest, rng):
    _, _, total = _layout(manifest)
    limits = {"train": 256, "validation": 128, "calibration": 128, "test": 512}
    records = {}
    for split, limit in limits.items():
        path = source / "cache" / f"{split}_features.npy"
        endpoint_path = source / "cache" / f"{split}_window_end_indices.npy"
        if not path.is_file() or not endpoint_path.is_file():
            continue
        features = np.load(path, mmap_mode="r", allow_pickle=False)
        endpoints = np.load(endpoint_path, mmap_mode="r", allow_pickle=False)
        count = min(len(features), limit)
        indices = np.arange(count, dtype=np.int64) if count == len(features) else np.sort(rng.choice(len(features), count, replace=False))
        np.savez_compressed(output / "samples" / f"{split}.npz", features=np.asarray(features[indices], dtype=np.float32),
                            window_end_indices=np.asarray(endpoints[indices], dtype=np.int64), sample_indices=indices,
                            diagnostic_only=np.asarray(True))
        records[split] = {"count": int(count), "shape": [int(count), *features.shape[1:]], "total_windows": int(len(features))}
    return records


def export(run_dir: Path, output_dir: Path, overwrite=False, already_validated=False):
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if not run_dir.is_dir():
        raise ValueError(f"source run directory does not exist: {run_dir}")
    manifest_path = run_dir / "probe_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("source run has no probe_manifest.json")
    manifest = _json(manifest_path)
    if manifest.get("artifact_policy") != "full":
        raise ValueError("analysis export requires artifact_policy=full; results-only runs cannot be recovered")
    if already_validated:
        status_path = run_dir / "run_status.json"
        status = _json(status_path) if status_path.is_file() else {}
        required = (run_dir / "results" / "metrics.json", run_dir / "results" / "aligned_scores.npz",
                    run_dir / "results" / "test_nll.npy", run_dir / "heads" / "calibration.npz")
        if status.get("status") != "complete" or any(not item.is_file() for item in required):
            raise ValueError("source-already-validated requires a complete source result bundle")
        validation = {"status": "complete", "validation_reused": True}
    else:
        validation = validate(run_dir)
        if validation.get("status") != "complete":
            raise ValueError(f"source run is not a complete full artifact: {validation}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"analysis output exists; choose a new directory or pass --overwrite: {output_dir}")
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError("--overwrite refuses a non-directory or symlink output")
        shutil.rmtree(output_dir)
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
        rng = np.random.default_rng(20261013)
        overview = _feature_overview(run_dir, temporary, manifest, rng)
        metrics = _metric_recheck(run_dir, temporary)
        segments = _segment_diagnostics(run_dir, temporary, manifest)
        variables = _variable_diagnostics(run_dir, temporary)
        samples = _samples(run_dir, temporary, manifest, rng)
        runtime = {"python": sys.version, "platform": platform.platform()}
        try:
            import torch
            runtime.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                            "cuda_available": bool(torch.cuda.is_available())})
        except Exception as error:
            runtime["torch_error"] = str(error)
        analysis_manifest = {
            "schema": "mnrf-view-probe-analysis-v1", "status": "complete",
            "source_run": str(run_dir), "source_run_identity": manifest.get("run_identity"),
            "source_manifest_sha256": source_manifest_hash, "source_validation": validation,
            "model_complete": True, "analysis_complete": True,
            "runtime": runtime, "evidence": records, "feature_overview": overview,
            "metric_recheck": metrics, "segment_count": len(segments.get("segments", [])),
            "variable_diagnostics": variables, "samples": samples,
            "elapsed_seconds": float(time.time() - started),
            "note": "diagnostic export; sampled feature summaries do not replace full-test metrics",
        }
        (temporary / "analysis_manifest.json").write_text(json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n")
        (temporary / "README.md").write_text(
            "# MNRF ViewProbe analysis\n\n"
            f"Source run: `{run_dir}`\n\n"
            "This directory is a bounded, read-only diagnostic export. The source full artifact, "
            "cache and checkpoints remain authoritative. `metric_recheck.json` uses the stored "
            "fixed threshold; it does not fit or calibrate anything. Feature summaries and samples "
            "are diagnostic only and do not establish branch causality or performance superiority.\n"
        )
        (temporary / "analysis_complete.json").write_text(json.dumps({"status": "complete", "source_manifest_sha256": source_manifest_hash}, indent=2) + "\n")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"status": "complete", "run_dir": str(run_dir), "output_dir": str(output_dir),
            "source_manifest_sha256": source_manifest_hash, "analysis_manifest": str(output_dir / "analysis_manifest.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source-already-validated", action="store_true",
                        help="reuse a preceding validator result and avoid a second model/test score pass")
    args = parser.parse_args()
    try:
        result = export(Path(args.run_dir), Path(args.output_dir), overwrite=args.overwrite,
                        already_validated=args.source_already_validated)
    except Exception as error:
        print(json.dumps({"status": "invalid", "reason": str(error)}, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
