"""Regression coverage for read-only full CATCH archive metric registration."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_catch_archive_metrics import audit_row  # noqa: E402
from typefusion_catch_result_utils import read_registry  # noqa: E402


class CatchArchiveMetricAuditTests(unittest.TestCase):
    def test_all_fixed_catch_archives_have_complete_finite_primary_metrics(self) -> None:
        rows = read_registry(ROOT / "result/typefusion_catch_main/typefusion_catch_source_registry.csv")
        self.assertEqual(len(rows), 23)
        for source in rows:
            update, audit = audit_row(dict(source))
            self.assertTrue(update["baseline_archive_valid"], source["task"])
            self.assertTrue(update["baseline_archive_sha256"], source["task"])
            self.assertTrue(math.isfinite(float(update["baseline_auc_roc"])), source["task"])
            self.assertTrue(math.isfinite(float(update["baseline_auc_pr"])), source["task"])
            self.assertIn(update["baseline_archive_validation_reason"], {"ok", "report_archive_auc_roc_conflict"})
            self.assertNotIn("detect_label", str(audit).lower())
            if update["report_archive_metric_conflict"]:
                self.assertNotEqual(
                    update["baseline_report_auc_roc"], update["baseline_archive_auc_roc"]
                )


if __name__ == "__main__":
    unittest.main()
