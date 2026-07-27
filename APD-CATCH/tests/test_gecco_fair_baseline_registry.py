"""GECCO fairness registration rejects old or geometry-mismatched results."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from register_gecco_fair_baseline import fair_script_contract, validate_candidate  # noqa: E402
from tests.typefusion_result_fixture import write_archive, write_report  # noqa: E402


class GeccoFairBaselineRegistryTests(unittest.TestCase):
    def test_current_registry_is_explicitly_unavailable(self) -> None:
        content = (ROOT / "result/typefusion_catch_main/gecco_fair_baseline_registry.csv").read_text()
        self.assertIn("GECCO,False,not_run_or_no_valid_result", content)

    def test_candidate_geometry_and_full_contract_validation(self) -> None:
        contract = fair_script_contract()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run-20260101T000000Z-1"
            run.mkdir()
            (run / "metadata.txt").write_text("catch_master_commit=abc\n", encoding="utf-8")
            wrong = dict(contract["params"])
            wrong["seq_len"] = 96
            write_archive(run / "CATCH.bad.csv.tar.gz", "CATCH", "GECCO.csv", wrong)
            write_report(run / "test_report.bad.csv", "CATCH", wrong)
            with self.assertRaisesRegex(ValueError, "config_mismatch"):
                validate_candidate(run, contract)
            (run / "CATCH.bad.csv.tar.gz").unlink()
            (run / "test_report.bad.csv").unlink()
            write_archive(run / "CATCH.good.csv.tar.gz", "CATCH", "GECCO.csv", contract["params"])
            write_report(run / "test_report.good.csv", "CATCH", contract["params"])
            result = validate_candidate(run, contract)
            self.assertTrue(result["fair_baseline_available"])
            self.assertEqual(result["fair_seq_len"], 192)
            self.assertEqual(result["fair_patch_size"], 16)
            self.assertEqual(result["fair_patch_stride"], 8)


if __name__ == "__main__":
    unittest.main()
