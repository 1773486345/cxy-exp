#!/usr/bin/env python
"""Validate one independent MNRF View Probe run without modifying it."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.mnrf_view_probe import load_probe
from ts_benchmark.baselines.mnrf_view_probe.probe import (
    ARTIFACT_RESULTS, PROBE_SCHEMA, _sha256_file, _validate_result_bundle,
)


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
    mode = manifest.get("mode")
    artifact_type = manifest.get("artifact_type")
    if artifact_type == ARTIFACT_RESULTS or manifest.get("artifact_policy") == "results":
        checked = _validate_result_bundle(
            run, allow_in_progress=manifest.get("result_cleanup_status") != "complete"
        )
        metrics = checked["metrics"]
        return {"status": "results_valid" if checked.get("cleanup_status") == "complete" else "results_valid_cleanup_failed",
                "run": str(run), "mode": mode,
                "artifact_type": ARTIFACT_RESULTS,
                "cleanup_status": checked["cleanup_status"],
                "cleanup_error": manifest.get("cleanup_error"),
                "task_name": manifest.get("task_name", manifest.get("config", {}).get("task_name")),
                "auc_roc": metrics.get("auc_roc"), "auc_pr": metrics.get("auc_pr"),
                "valid_points": metrics.get("valid_points"),
                "capabilities": manifest.get("capabilities", {})}
    if status.get("status") not in {"complete", "stage1_complete"}:
        raise ValueError(f"run is not complete: {status.get('status')}")
    head_type = manifest.get("head_type")
    if artifact_type not in {"source_encoder", "probe_head"}:
        raise ValueError("manifest has no recognized artifact_type")
    if artifact_type == "source_encoder":
        if status.get("status") != "stage1_complete" or manifest.get("stage_status") != "stage1_complete":
            raise ValueError("source encoder is not a completed stage-one artifact")
        restored = load_probe(run, device="cpu")
        return {"status": "stage1_complete", "run": str(run), "mode": mode,
                "artifact_type": artifact_type,
                "task_name": manifest.get("task_name", manifest.get("config", {}).get("task_name")),
                "feature_dim": int(manifest.get("cache_feature_dim", 0))}
    if status.get("status") != "complete" or manifest.get("stage_status") != "complete":
        raise ValueError("head artifact is not a complete run")
    if head_type not in {"gmm", "flow"}:
        raise ValueError("head artifact has no recognized head_type")
    # A source-referencing head deliberately has no local feature cache.  Its
    # source manifest and all source cache hashes are checked by load_probe.
    if not manifest.get("source_dir"):
        for name in ("cache/train_features.npy", "cache/validation_features.npy",
                     "cache/calibration_features.npy", "cache/test_features.npy"):
            if not (run / name).is_file():
                raise ValueError(f"self-contained head is missing cache artifact: {name}")
    restored = load_probe(run, device="cpu")
    if head_type == "flow":
        restored.score_cached_test()
    metrics_path = run / "results/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    return {"status": metrics.get("status"), "run": str(run), "mode": mode,
            "artifact_type": artifact_type, "head_type": head_type,
            "task_name": manifest.get("config", {}).get("task_name"),
            "auc_roc": metrics.get("auc_roc"), "auc_pr": metrics.get("auc_pr")}


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
