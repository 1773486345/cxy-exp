#!/usr/bin/env python
"""Validate one independent MNRF View Probe run without modifying it."""

import argparse
import json
from pathlib import Path

import numpy as np

from ts_benchmark.baselines.mnrf_view_probe import load_probe
from ts_benchmark.baselines.mnrf_view_probe.probe import PROBE_SCHEMA, _sha256_file


def validate(run_dir):
    run = Path(run_dir).resolve()
    manifest_path = run / "probe_manifest.json"
    status_path = run / "run_status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise ValueError("probe manifest or run status is missing")
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    if manifest.get("schema") != PROBE_SCHEMA:
        raise ValueError("unsupported probe schema")
    if status.get("status") not in {"complete", "stage1_complete"}:
        raise ValueError(f"run is not complete: {status.get('status')}")
    mode = manifest.get("mode")
    if mode != "flow":
        for name in ("input_statistics.npz", "cache/train_features.npy", "cache/validation_features.npy",
                     "cache/calibration_features.npy", "cache/test_features.npy"):
            if not (run / name).is_file():
                raise ValueError(f"required artifact missing: {name}")
    elif not (run / "input_statistics.npz").is_file():
        raise ValueError("Flow run is missing input_statistics.npz")
    if mode in {"stage1", "pretrained", "random"} and not (run / "heads/gmm_status.json").is_file():
        for name in ("checkpoints/initial_encoder.pt", "checkpoints/best_encoder.pt"):
            if not (run / name).is_file():
                raise ValueError(f"stage-one artifact missing: {name}")
        return {"status": "stage1_complete", "run": str(run), "mode": mode,
                "task_name": manifest.get("task_name", manifest.get("config", {}).get("task_name"))}
    if mode in {"full_gmm", "select_gmm", "s_only", "random_gmm"} or (run / "heads/gmm_status.json").is_file():
        load_probe(run, device="cpu")
        metrics_path = run / "results/metrics.json"
        result_path = run / "results/aligned_scores.npz"
        if not metrics_path.is_file() or not result_path.is_file():
            raise ValueError("completed GMM run has no result artifacts")
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("status") not in {"complete", "incomplete"}:
            raise ValueError("invalid GMM result status")
        with np.load(result_path, allow_pickle=False) as result:
            required = {"scores", "predictions", "time_indices", "local_indices", "valid_mask", "aligned_scores"}
            if not required.issubset(result.files):
                raise ValueError("aligned score artifact is incomplete")
            if len(result["scores"]) != len(result["time_indices"]) or len(result["scores"]) != len(result["local_indices"]):
                raise ValueError("aligned score lengths disagree")
            if not np.isfinite(result["scores"]).all():
                raise ValueError("aligned score contains NaN or Inf")
        return {"status": metrics.get("status"), "run": str(run), "mode": mode,
                "task_name": manifest.get("config", {}).get("task_name"),
                "auc_roc": metrics.get("auc_roc"), "auc_pr": metrics.get("auc_pr")}
    if mode == "flow":
        required = ("checkpoints/best_fusion.pt", "heads/feature_statistics.npz",
                    "heads/calibration.npz", "heads/fixed_noise.npy",
                    "results/metrics.json", "results/aligned_scores.npz")
        missing = [name for name in required if not (run / name).is_file()]
        if missing:
            raise ValueError("completed Flow run is missing: " + ", ".join(missing))
        restored = load_probe(run, device="cpu")
        restored.score_cached_test()
        with np.load(run / "results/aligned_scores.npz", allow_pickle=False) as result:
            if not np.isfinite(result["scores"]).all():
                raise ValueError("Flow score contains NaN or Inf")
        return {"status": "complete", "run": str(run), "mode": mode}
    raise ValueError(f"unsupported probe mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    try:
        result = validate(args.run_dir)
    except Exception as error:
        print(json.dumps({"status": "invalid", "reason": str(error)}, sort_keys=True))
        raise SystemExit(1)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
