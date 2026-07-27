"""Paper-level aggregation keeps ASD at one equal-weight paper-level row."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_typefusion_catch_real import macro_summary, paper_metric_rows  # noqa: E402


def row(task: str, paper: str, value: float, status: str = "exact_shared_config", comparable: bool = True) -> dict:
    return {
        "task": task,
        "paper_dataset": paper,
        "comparison_config_status": status,
        "comparable": comparable,
        "catch_auc_pr": value,
        "catch_auc_roc": value,
        "typefusion_auc_pr": value + 0.1,
        "typefusion_auc_roc": value + 0.1,
        "delta_auc_pr": 0.1,
        "delta_auc_roc": 0.1,
    }


class TypeFusionSummaryMetricTests(unittest.TestCase):
    def complete_tasks(self) -> list:
        tasks = [row(f"ASD_dataset_{index}", "ASD", float(index)) for index in range(1, 13)]
        for index, name in enumerate(
            ("CICIDS", "CalIt2", "Creditcard", "GECCO", "Genesis", "MSL", "NYC", "PSM", "SMAP", "SMD", "SWAT"),
            start=13,
        ):
            tasks.append(row(name, name, float(index)))
        return tasks

    def test_asd_equal_macro_and_overall_paper_weighting(self) -> None:
        tasks = self.complete_tasks()
        papers = paper_metric_rows(tasks)
        asd = next(item for item in papers if item["paper_dataset"] == "ASD")
        self.assertEqual(asd["catch_auc_roc"], 6.5)
        summary = macro_summary(papers, tasks)
        expected = (6.5 + sum(range(13, 24))) / 12
        self.assertAlmostEqual(summary["formal_overall"]["catch_auc_roc"], expected)
        self.assertEqual(summary["formal_overall"]["status"], "complete")

    def test_missing_gecco_and_override_sensitivity_are_explicit(self) -> None:
        tasks = self.complete_tasks()
        tasks[0]["comparison_config_status"] = "architecture_compatibility_override"
        papers = paper_metric_rows(tasks)
        summary = macro_summary(papers, tasks)
        self.assertEqual(summary["including_architecture_compatibility_override"]["status"], "complete")
        self.assertEqual(summary["exact_shared_config_only"]["status"], "incomplete")
        gecco = next(item for item in tasks if item["task"] == "GECCO")
        gecco["comparable"] = False
        gecco["comparison_config_status"] = "fairness_rerun_required"
        incomplete = macro_summary(paper_metric_rows(tasks), tasks)
        self.assertEqual(incomplete["formal_overall"]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
