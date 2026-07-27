#!/usr/bin/env python3
"""Read-only task and paper-level summary for completed TypeFusion-CATCH runs.

The tool never trains models, never rewrites source registries, and never reads
labels to derive metrics.  It accepts only completed score reports whose seed,
strategy and model parameters agree with the prepared registry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result/typefusion_catch_main"
ASD_PREFIX = "ASD_dataset_"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_report(path: Path) -> Dict[str, Any]:
    rows = read_csv(path)
    if len(rows) != 1:
        raise ValueError("expected exactly one metric row")
    row = rows[0]
    metric_columns = [key for key in row if key not in {"strategy_args", "metric_name"}]
    if len(metric_columns) != 1:
        raise ValueError("expected exactly one model metric column")
    model_column = metric_columns[0]
    prefix, params_text = model_column.split(";", 1)
    strategy = json.loads(row["strategy_args"])
    value = float(row[model_column])
    if not math.isfinite(value):
        raise ValueError("metric is not finite")
    return {
        "model_prefix": prefix,
        "params": json.loads(params_text),
        "strategy": strategy,
        "metric_name": row["metric_name"],
        "metric_value": value,
    }


def has_traceback(run_dir: Path) -> bool:
    for path in run_dir.glob("*.log"):
        if "traceback" in path.read_text(encoding="utf-8", errors="replace").lower():
            return True
    return False


def expected_params(source: Mapping[str, str]) -> Dict[str, Any]:
    return json.loads(source["typefusion_hyper_params_json"])


def report_matches_registry(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    # Batch size is deliberately handled separately to retain the OOM fairness
    # signal instead of accepting a changed batch as a fully comparable result.
    for key, value in expected.items():
        if key == "batch_size":
            continue
        if actual.get(key) != value:
            return False
    return True


def latest_report(task: str) -> Path | None:
    candidates = sorted(
        (ROOT / "result/score/TypeFusion-CATCH" / task).glob("run-*/test_report*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def mean_or_na(values: Iterable[float]) -> float | str:
    values = list(values)
    return sum(values) / len(values) if values else "N/A"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    source_rows = read_csv(RESULT_ROOT / "typefusion_catch_source_registry.csv")
    if len(source_rows) != 23:
        raise ValueError("source registry must contain 23 tasks")
    task_rows: List[Dict[str, Any]] = []
    for source in source_rows:
        task = source["task"]
        output: Dict[str, Any] = {
            "task": task,
            "paper_dataset": source["paper_dataset"],
            "baseline_metric_name": source["baseline_metric_name"],
            "catch_auc_roc": source["baseline_metric_value"]
            if source["baseline_metric_name"] == "auc_roc"
            else "N/A",
            "typefusion_auc_roc": "N/A",
            "auc_pr": "N/A",
            "typefusion_report": "",
            "typefusion_batch_size": "",
            "comparison_status": "missing_typefusion_report",
            "included_in_macro": False,
        }
        report_path = latest_report(task)
        if report_path is None or report_path.stat().st_size == 0:
            task_rows.append(output)
            continue
        try:
            if has_traceback(report_path.parent):
                raise ValueError("traceback in run log")
            parsed = parse_report(report_path)
            strategy = parsed["strategy"]
            if parsed["model_prefix"] != "TypeFusion-CATCH":
                raise ValueError("unexpected model prefix")
            if strategy.get("strategy_name") != "unfixed_detect_score":
                raise ValueError("unexpected strategy")
            if int(strategy.get("seed")) != 2021:
                raise ValueError("unexpected strategy seed")
            expected = expected_params(source)
            if not report_matches_registry(parsed["params"], expected):
                raise ValueError("model parameters differ from registry")
            actual_batch = parsed["params"].get("batch_size")
            output["typefusion_report"] = str(report_path)
            output["typefusion_batch_size"] = actual_batch
            if parsed["metric_name"] != "auc_roc":
                output["comparison_status"] = "unsupported_metric"
            elif source["baseline_requires_fair_rerun"] == "True" or source[
                "direct_baseline_comparable"
            ] != "True":
                output["typefusion_auc_roc"] = parsed["metric_value"]
                output["comparison_status"] = "incomparable_gecco_requires_seq192_catch"
            elif actual_batch != int(source["final_batch_size"]):
                output["typefusion_auc_roc"] = parsed["metric_value"]
                output["comparison_status"] = "incomparable_batch_mismatch"
            else:
                output["typefusion_auc_roc"] = parsed["metric_value"]
                output["comparison_status"] = "comparable"
                output["included_in_macro"] = True
        except Exception as exc:
            output["comparison_status"] = f"invalid_typefusion_report:{type(exc).__name__}"
        task_rows.append(output)

    paper_rows: List[Dict[str, Any]] = []
    paper_names = ["ASD"] + [
        task for task in ("CICIDS", "CalIt2", "Creditcard", "GECCO", "Genesis", "MSL", "NYC", "PSM", "SMAP", "SMD", "SWAT")
    ]
    for paper in paper_names:
        members = [row for row in task_rows if row["paper_dataset"] == paper]
        comparable = [row for row in members if row["included_in_macro"]]
        paper_rows.append(
            {
                "paper_dataset": paper,
                "task_count": len(members),
                "comparable_task_count": len(comparable),
                "catch_auc_roc": mean_or_na(
                    float(row["catch_auc_roc"]) for row in comparable if row["catch_auc_roc"] != "N/A"
                ),
                "typefusion_auc_roc": mean_or_na(
                    float(row["typefusion_auc_roc"])
                    for row in comparable
                    if row["typefusion_auc_roc"] != "N/A"
                ),
                "auc_pr": "N/A",
                "included_in_overall": bool(comparable and len(comparable) == len(members)),
            }
        )
    included_papers = [row for row in paper_rows if row["included_in_overall"]]
    overall_row = {
        "task": "overall_macro",
        "paper_dataset": "overall_macro",
        "catch_auc_roc": mean_or_na(
            float(row["catch_auc_roc"]) for row in included_papers if row["catch_auc_roc"] != "N/A"
        ),
        "typefusion_auc_roc": mean_or_na(
            float(row["typefusion_auc_roc"])
            for row in included_papers
            if row["typefusion_auc_roc"] != "N/A"
        ),
        "auc_pr": "N/A",
        "comparison_status": "overall_macro_12_paper_datasets"
        if len(included_papers) == 12
        else "overall_incomplete",
    }
    comparison_rows = [
        {
            "task": row["task"],
            "paper_dataset": row["paper_dataset"],
            "catch_auc_roc": row["catch_auc_roc"],
            "typefusion_auc_roc": row["typefusion_auc_roc"],
            "auc_pr": "N/A",
            "comparison_status": row["comparison_status"],
        }
        for row in task_rows
    ]
    comparison_rows.append(overall_row)
    for name, rows in (
        ("typefusion_task_metrics.csv", task_rows),
        ("typefusion_paper_metrics.csv", paper_rows),
        ("typefusion_comparison.csv", comparison_rows),
    ):
        with (RESULT_ROOT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
