"""Data and command contracts for the added CATCH external tasks."""

from __future__ import annotations

import csv
import gc
import math
import subprocess
import sys
import tarfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ts_benchmark.data.data_source import LocalExternalAnomalyDetectDataSource  # noqa: E402


WINDOW_LENGTH = 192
BASELINE_SCRIPTS = (
    "AnomalyTransformer.sh",
    "AutoEncoder.sh",
    "CATCH.sh",
    "DCdetector.sh",
    "DLinear.sh",
    "DualTF.sh",
    "IsolationForest.sh",
    "ModernTCN.sh",
    "NLinear.sh",
    "PatchTST.sh",
    "TFAD.sh",
    "TimesNet.sh",
    "hbosski.sh",
    "iTransformer.sh",
    "ocsvmski.sh",
    "pcaodetectorski.sh",
)
REPORTED_BASELINES = (
    "AnomalyTransformer",
    "CATCH",
    "DCdetector",
    "DLinear",
    "DualTF",
    "IsolationForest",
    "ModernTCN",
    "PatchTST",
    "TimesNet",
    "iTransformer",
)
INVALID_REPORTS = {
    ("HAI20_07", "DualTF"),
    ("BATADAL", "DualTF"),
}


class TestExternalCatchData(unittest.TestCase):
    def setUp(self) -> None:
        self.data_root = ROOT / "dataset" / "external_validation"
        self.metadata = pd.read_csv(self.data_root / "EXTERNAL_DETECT_META.csv").set_index(
            "file_name", drop=False
        )

    def test_metadata_matches_prepared_data_and_loader(self) -> None:
        prepared = {path.name for path in (self.data_root / "data").glob("*.csv")}
        self.assertEqual(prepared, set(self.metadata.index))
        self.assertEqual(len(prepared), 20)

        source = LocalExternalAnomalyDetectDataSource()
        for filename, row in self.metadata.iterrows():
            frame = source._load_series(filename)
            feature_columns = [column for column in frame.columns if column != "label"]
            self.assertGreaterEqual(len(feature_columns), 2, filename)
            self.assertTrue(np.isfinite(frame[feature_columns].to_numpy()).all(), filename)
            self.assertEqual(set(frame["label"].unique()), {0, 1}, filename)
            self.assertEqual(len(frame), int(row["length"]), filename)
            self.assertEqual(
                int(row["train_lens"]) + int(row["test_lens"]), len(frame), filename
            )
            self.assertGreaterEqual(int(row["train_lens"]), WINDOW_LENGTH, filename)
            self.assertGreaterEqual(int(row["test_lens"]), WINDOW_LENGTH, filename)
            del frame
            gc.collect()

    def test_standard_baseline_scripts_are_single_task_commands(self) -> None:
        for filename in self.metadata.index:
            task = Path(filename).stem
            for script_name in BASELINE_SCRIPTS:
                script = (
                    ROOT
                    / "scripts"
                    / "multivariate_detection"
                    / "detect_score"
                    / f"{task}_script"
                    / script_name
                )
                self.assertTrue(script.is_file(), script)
                subprocess.check_call(["bash", "-n", str(script)])
                command = script.read_text(encoding="utf-8")
                self.assertEqual(command.count("run_benchmark.py"), 1, script)
                self.assertIn('--data-set-name "external_detect"', command)
                self.assertIn(f'--data-name-list "{filename}"', command)

    def test_retained_reports_and_archives_are_finite_auc_roc(self) -> None:
        result_root = ROOT / "result" / "score" / "external_validation"
        for filename in self.metadata.index:
            task = Path(filename).stem
            for baseline in REPORTED_BASELINES:
                directory = result_root / task / baseline
                reports = list(directory.glob("test_report.*.csv"))
                archives = list(directory.glob("*.tar.gz"))
                if (task, baseline) in INVALID_REPORTS:
                    self.assertEqual(reports, [], (task, baseline))
                    self.assertEqual(archives, [], (task, baseline))
                    continue
                self.assertEqual(len(reports), 1, (task, baseline))
                self.assertEqual(len(archives), 1, (task, baseline))
                with tarfile.open(archives[0], "r:gz") as archive:
                    self.assertGreater(len(archive.getmembers()), 0, (task, baseline))
                content = reports[0].read_text(encoding="utf-8")
                self.assertNotIn("Traceback", content)
                rows = list(csv.reader(content.splitlines()))
                self.assertEqual(len(rows), 2, (task, baseline))
                self.assertEqual(rows[1][1], "auc_roc", (task, baseline))
                self.assertTrue(rows[0][2].startswith(f"{baseline};"), (task, baseline))
                self.assertTrue(math.isfinite(float(rows[1][2])), (task, baseline))


if __name__ == "__main__":
    unittest.main()
