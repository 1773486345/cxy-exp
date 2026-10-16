#!/usr/bin/env python
"""Fail the one-task shell command when benchmark artifacts are incomplete."""

import json
import math
import sys
from pathlib import Path

import pandas as pd


def main(run_path: str, task: str) -> None:
    root = Path(run_path)
    result_files = sorted((root / "results").glob("*_metrics.json"))
    if len(result_files) != 1:
        raise SystemExit(f"Expected one per-task metrics JSON, found {len(result_files)} in {root}")
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("task") != f"{task}.csv":
        raise SystemExit(f"Incomplete or mismatched task metrics: {payload}")
    for key in ("precision_at_calibrated_threshold", "recall_at_calibrated_threshold",
                "f1_at_calibrated_threshold", "normal_false_alarm_rate", "threshold"):
        if not math.isfinite(float(payload[key])):
            raise SystemExit(f"Non-finite {key} in {result_files[0]}")
    reports = sorted(root.glob("test_report*.csv"))
    if len(reports) != 1:
        raise SystemExit(f"Expected one test_report CSV, found {len(reports)} in {root}")
    report = pd.read_csv(reports[0])
    for metric in ("auc_roc", "auc_pr"):
        if metric not in report.columns or not math.isfinite(float(report.iloc[0][metric])):
            raise SystemExit(f"Missing or non-finite {metric} in {reports[0]}")
    print(f"MNRF-Lite V2 run complete: {root}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: validate_mnrf_lite_v2_run.py RUN_DIR TASK")
    main(sys.argv[1], sys.argv[2])
