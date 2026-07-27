"""The benchmark must use the class fallback name, not the adapter display name."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from summarize_typefusion_catch_real import select_latest_valid_run  # noqa: E402
from tests.test_typefusion_run_selection import source_row, write_valid_run  # noqa: E402
from ts_benchmark.models.model_loader import get_models  # noqa: E402


class TypeFusionBenchmarkOutputNameTests(unittest.TestCase):
    def test_loader_name_drives_report_archive_and_selection(self) -> None:
        factories = get_models(
            {
                "models": [
                    {
                        "model_name": "typefusion_catch.TypeFusionCATCH",
                        "model_hyper_params": {},
                    }
                ]
            }
        )
        self.assertEqual(len(factories), 1)
        self.assertEqual(factories[0].model_name, "TypeFusionCATCH")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "PSM" / "run-20260101T000000Z-1"
            write_valid_run(
                run,
                archive_prefix=factories[0].model_name,
                archive_model_name=factories[0].model_name,
                report_model_name=factories[0].model_name,
            )
            selected, audit = select_latest_valid_run("PSM", source_row(), root)
            self.assertEqual(selected["archive_path"], str(run / "TypeFusionCATCH.result.csv.tar.gz"))
            self.assertTrue(audit[0]["selected"])


if __name__ == "__main__":
    unittest.main()
