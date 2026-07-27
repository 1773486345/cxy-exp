"""Historical CATCH source audit must preserve unavailable-source evidence."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CatchHistoricalCodeAuditTests(unittest.TestCase):
    def test_every_registry_commit_has_an_explicit_audit_outcome(self) -> None:
        with (ROOT / "result/typefusion_catch_main/typefusion_catch_source_registry.csv").open() as handle:
            registry = list(csv.DictReader(handle))
        with (ROOT / "result/typefusion_catch_main/catch_historical_code_audit.csv").open() as handle:
            audits = list(csv.DictReader(handle))
        self.assertEqual(
            {row["source_catch_master_commit"] for row in registry},
            {row["archive_commit"] for row in audits},
        )
        for audit in audits:
            self.assertIn(audit["audit_status"], {"ok", "failed"})
            self.assertIn(audit["matches_current_catch"], {"True", "False"})
            if audit["audit_status"] == "ok":
                self.assertTrue(audit["aggregate_sha256"])
                if audit["matches_current_catch"] == "False":
                    self.assertTrue(audit["changed_files"])
            else:
                self.assertTrue(audit["audit_reason"])
        unavailable = [row for row in audits if row["audit_status"] == "failed"]
        self.assertTrue(unavailable, "an unresolvable commit must not be silently passed")
        unavailable_commits = {row["archive_commit"] for row in unavailable}
        for source in registry:
            if source["source_catch_master_commit"] in unavailable_commits:
                self.assertEqual(source["historical_code_comparable"], "False")
                self.assertTrue(source["historical_code_difference"].startswith("audit_unavailable:"))


if __name__ == "__main__":
    unittest.main()
