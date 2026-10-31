#!/usr/bin/env python
"""Post-hoc MSL col_0 aggregation diagnostic; no fitting or recalibration."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_mnrf_common_baselines import _full_axis_metrics
from ts_benchmark.baselines.mnrf_view_probe.evaluation import calibrated_quantile


def run(run_dir, output):
    run_dir = Path(run_dir).resolve(); output = Path(output).resolve()
    validation = np.load(run_dir / "heads/validation_nll.npy", mmap_mode="r", allow_pickle=False)
    calibration = np.load(run_dir / "heads/calibration_nll.npy", mmap_mode="r", allow_pickle=False)
    test = np.load(run_dir / "results/test_nll.npy", mmap_mode="r", allow_pickle=False)
    with np.load(run_dir / "heads/calibration.npz", allow_pickle=False) as saved:
        quantile = float(np.asarray(saved["quantile"])); saved_threshold = float(np.asarray(saved["threshold"]))
    if validation.ndim != 2 or calibration.ndim != 2 or test.ndim != 2 or validation.shape[1] != 55:
        raise ValueError("MSL NLL arrays are not the expected 55-variable layout")
    center = np.median(np.asarray(validation[:, 0], dtype=np.float64))
    scale = float(np.median(np.abs(np.asarray(validation[:, 0], dtype=np.float64) - center)))
    scale = max(scale, 0.001)
    calibration_score = (np.asarray(calibration[:, 0], dtype=np.float64) - center) / scale
    test_score = (np.asarray(test[:, 0], dtype=np.float64) - center) / scale
    threshold = calibrated_quantile(calibration_score, quantile)
    endpoints = np.asarray(np.load(run_dir / "cache/test_window_end_indices.npy", allow_pickle=False), dtype=np.int64)
    with np.load(run_dir / "analysis_inputs.npz", allow_pickle=False) as axis:
        labels = np.asarray(axis["all_labels"], dtype=np.int64)
        test_segments = json.loads((run_dir / "probe_manifest.json").read_text()).get("source_identity", {}).get("test_segments", [])
    metrics = _full_axis_metrics(labels, endpoints, test_score, threshold, test_segments, None)
    result = {"schema": "msl-target-only-aggregation-v1", "run_dir": str(run_dir),
              "target_variable": "col_0", "condition_variables": ["col_1..col_54"],
              "quantile": quantile, "saved_full_threshold": saved_threshold,
              "target_center": center, "target_scale": scale,
              "target_threshold": float(threshold), "metrics": metrics,
              "protocol": "post-hoc target-only aggregation; source four-view features and all inputs retained; no refit"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); print(json.dumps(run(args.run_dir, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
