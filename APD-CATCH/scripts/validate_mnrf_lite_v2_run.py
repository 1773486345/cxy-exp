#!/usr/bin/env python
"""Fail the one-task shell command when benchmark artifacts are incomplete."""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def main(run_path: str, task: str) -> None:
    root = Path(run_path)
    required = [
        root / "run_identity.json", root / "benchmark_context.json", root / "run_metadata.json",
        root / "cache_manifest.json", root / "input_statistics.npz", root / "deployment.pt",
        root / "checkpoints" / "best_encoder.pt", root / "checkpoints" / "best_fusion.pt",
        root / "checkpoints" / "best_stage1.pt",
        root / "checkpoints" / "stage1_latest.pt", root / "checkpoints" / "stage2_latest.pt",
        root / "logs" / "training.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit(f"Missing required run artifacts: {missing}")
    identity = json.loads((root / "run_identity.json").read_text(encoding="utf-8"))
    context = json.loads((root / "benchmark_context.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "cache_manifest.json").read_text(encoding="utf-8"))
    if context.get("task") != f"{task}.csv" or context.get("run_id") != root.name:
        raise SystemExit(f"Task/run identity mismatch in {root / 'benchmark_context.json'}")
    if context.get("strategy_name") != "mnrf_valid_detect" or metadata.get("config", {}).get("task_name") != f"{task}.csv":
        raise SystemExit("Strategy or model-config task does not match this run")
    if identity.get("architecture_version") != "mnrf-lite-v2-tcn6b-2l-r1-segment-v2":
        raise SystemExit("run architecture version is missing or incompatible")
    if manifest.get("status") != "complete" or manifest.get("encoder_fingerprint") != metadata.get("feature_fingerprint"):
        raise SystemExit("feature cache manifest is incomplete or fingerprint-mismatched")
    for name, path in metadata.get("feature_cache_paths", {}).items():
        index_path = Path(metadata.get("feature_cache_window_end_indices", {}).get(name, ""))
        if not Path(path).is_file() or not index_path.is_file():
            raise SystemExit(f"Missing cache or original-index file for {name}")
        expected_cache_sha = manifest.get("feature_cache_sha256", {}).get(name)
        if expected_cache_sha is None or _sha256_file(path) != expected_cache_sha:
            raise SystemExit(f"Feature cache missing or fingerprint mismatch for {name}")
        cache = np.load(path, mmap_mode="r")
        indices = np.load(index_path, mmap_mode="r")
        if len(cache) != metadata["feature_cache_window_counts"][name] or len(indices) != len(cache):
            raise SystemExit(f"Cache/index length mismatch for {name}")
        if not np.isfinite(cache).all() or np.any(np.diff(indices) <= 0):
            raise SystemExit(f"Invalid cached features or indices for {name}")
        expected_sha = manifest.get("window_end_index_sha256", {}).get(name)
        if expected_sha is None:
            raise SystemExit(f"Missing cache index fingerprint for {name}")
        digest = __import__("hashlib").sha256(Path(index_path).read_bytes()).hexdigest()
        if digest != expected_sha:
            raise SystemExit(f"Cache index fingerprint mismatch for {name}")
        split = metadata.get("split_ranges", {}).get(name)
        if split is None:
            raise SystemExit(f"Missing split bounds for cache {name}")
        from ts_benchmark.baselines.mnrf_lite_v2.windows import window_endpoints, intersect_segments
        expected_indices = window_endpoints(
            intersect_segments(manifest["continuous_segments"], split["start"], split["end_exclusive"]),
            int(metadata["config"]["seq_len"]),
        )
        if not np.array_equal(indices, expected_indices):
            raise SystemExit(f"Cache positions do not match continuous split windows for {name}")
    with np.load(root / "input_statistics.npz", allow_pickle=False) as stats:
        for key in ("mean", "scale", "feature_mean", "feature_scale", "calibration_center", "calibration_scale", "threshold", "variable_names"):
            if key not in stats.files:
                raise SystemExit(f"Missing {key} in input_statistics.npz")
        if not np.isfinite(stats["threshold"]).all() or not np.isfinite(stats["calibration_center"]).all():
            raise SystemExit("Non-finite calibration artifact")
        if not np.allclose(stats["feature_mean"], manifest["feature_mean"], rtol=0, atol=0):
            raise SystemExit("Feature mean differs between statistics and cache manifest")
        if not np.allclose(stats["feature_scale"], manifest["feature_scale"], rtol=0, atol=0):
            raise SystemExit("Feature scale differs between statistics and cache manifest")
    try:
        import torch
        deployment = torch.load(root / "deployment.pt", map_location="cpu", weights_only=False)
    except Exception as error:
        raise SystemExit(f"Deployment package cannot be loaded: {error}")
    if deployment.get("architecture_version") != identity["architecture_version"]:
        raise SystemExit("Deployment/run architecture mismatch")
    if deployment.get("training_metadata", {}).get("feature_fingerprint") != manifest.get("encoder_fingerprint"):
        raise SystemExit("Deployment encoder fingerprint differs from cache manifest")
    if tuple(deployment.get("variable_names", ())) != tuple(manifest.get("variable_names", ())):
        raise SystemExit("Deployment and cache variable order differ")
    if "auxiliary_state" in deployment or "auxiliary" in deployment:
        raise SystemExit("Auxiliary pretraining head must not be part of deployment")
    for checkpoint in (root / "checkpoints" / "best_encoder.pt", root / "checkpoints" / "best_fusion.pt",
                       root / "checkpoints" / "stage1_latest.pt", root / "checkpoints" / "stage2_latest.pt"):
        try:
            torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception as error:
            raise SystemExit(f"Checkpoint cannot be loaded ({checkpoint.name}): {error}")
    result_files = sorted((root / "results").glob("*_metrics.json"))
    if len(result_files) != 1:
        raise SystemExit(f"Expected one per-task metrics JSON, found {len(result_files)} in {root}")
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("task") != f"{task}.csv" or payload.get("run_id") != root.name:
        raise SystemExit(f"Incomplete or mismatched task metrics: {payload}")
    for key in ("precision_at_calibrated_threshold", "recall_at_calibrated_threshold",
                "f1_at_calibrated_threshold", "normal_false_alarm_rate", "threshold"):
        if payload.get(key) is None or not math.isfinite(float(payload[key])):
            raise SystemExit(f"Non-finite {key} in {result_files[0]}")
    reports = sorted(root.glob("test_report*.csv"))
    if len(reports) != 1:
        raise SystemExit(f"Expected one test_report CSV, found {len(reports)} in {root}")
    report = pd.read_csv(reports[0])
    model_columns = [column for column in report.columns if "MNRFLiteV2;" in str(column)]
    if len(model_columns) != 1 or "metric_name" not in report.columns:
        raise SystemExit("Leaderboard does not contain one MNRFLiteV2 result column")
    for metric in ("auc_roc", "auc_pr"):
        rows = report.loc[report.metric_name == metric]
        if len(rows) != 1 or not math.isfinite(float(rows.iloc[0][model_columns[0]])):
            raise SystemExit(f"Missing or non-finite {metric} in {reports[0]}")
        if payload[metric] is None or not math.isclose(float(rows.iloc[0][model_columns[0]]), float(payload[metric]), rel_tol=1e-9, abs_tol=1e-9):
            raise SystemExit(f"Leaderboard {metric} differs from aligned task metrics")
    aligned_files = sorted((root / "results").glob("*_aligned_scores.npz"))
    if len(aligned_files) != 1:
        raise SystemExit(f"Expected one aligned score artifact, found {len(aligned_files)}")
    with np.load(aligned_files[0], allow_pickle=False) as aligned:
        expected = {"scores", "predictions", "time_indices", "local_time_indices", "time_index_values", "valid_mask", "local_valid_mask", "actual", "threshold"}
        if not expected.issubset(aligned.files):
            raise SystemExit(f"Aligned artifact missing fields: {expected - set(aligned.files)}")
        scores = aligned["scores"]
        predictions = aligned["predictions"]
        local = aligned["local_time_indices"]
        indices = aligned["time_indices"]
        actual = aligned["actual"]
        if not (len(scores) == len(predictions) == len(local) == len(indices) == payload["valid_points"]):
            raise SystemExit("Aligned score/index/prediction lengths do not match metrics")
        if not np.isfinite(scores).all() or not np.all(np.diff(indices) > 0) or not np.all(np.diff(local) > 0):
            raise SystemExit("Aligned scores or indices are invalid")
        if np.any(local < 0) or np.any(local >= len(actual)) or not np.array_equal(np.flatnonzero(aligned["local_valid_mask"]), local):
            raise SystemExit("Valid mask does not match scored local indices")
        if not np.array_equal(np.flatnonzero(aligned["valid_mask"]), indices):
            raise SystemExit("Full-series valid mask does not match global scored indices")
        if not np.array_equal(predictions, (scores > float(payload["threshold"])).astype(np.int64)):
            raise SystemExit("Saved predictions do not match the calibrated threshold")
        selected = actual[local]
        if np.unique(selected).size != 2:
            raise SystemExit("Metrics are not complete: aligned labels contain only one class")
        if not math.isclose(roc_auc_score(selected, scores), float(payload["auc_roc"]), rel_tol=1e-9, abs_tol=1e-9):
            raise SystemExit("Saved aligned scores do not reproduce AUC-ROC")
        if not math.isclose(average_precision_score(selected, scores), float(payload["auc_pr"]), rel_tol=1e-9, abs_tol=1e-9):
            raise SystemExit("Saved aligned scores do not reproduce AUC-PR")
        if len(aligned["time_index_values"]) != len(local):
            raise SystemExit("Saved timestamp coordinates do not match scored indices")
    aligned_task = f"{task}.csv_aligned_scores.npz"
    if aligned_files[0].name != aligned_task:
        raise SystemExit(f"Aligned artifact belongs to another task: {aligned_files[0].name}")
    if metadata.get("config", {}).get("output_dir") != str(root.resolve()):
        raise SystemExit("Run config output_dir does not match validated run path")
    print(f"MNRF-Lite V2 run complete: {root}")


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = __import__("hashlib").sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: validate_mnrf_lite_v2_run.py RUN_DIR TASK")
    main(sys.argv[1], sys.argv[2])
