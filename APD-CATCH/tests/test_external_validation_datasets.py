"""Integrity checks for the fixed external validation protocol.

Run after the data-preparation commands have completed. The small invariant tests
remain useful before download; data-dependent tests are skipped until the prepared
files exist rather than fabricating substitute data.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_SCRIPT_DIR = ROOT / "scripts" / "data_preparation" / "external_validation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXTERNAL_SCRIPT_DIR))

from common import (  # noqa: E402
    BATADAL_ATTACK_INTERVALS,
    DATA_ROOT,
    METADATA_PATH,
    METRO_FAULT_INTERVALS,
    MTSBENCH_REPO_ID,
    MTSBENCH_REVISION,
    MTSBENCH_RELATIVE_PATHS,
    MTSBENCH_MISSING_PATHS,
    MTSBENCH_VERIFICATION_PATH,
    PAPER_DATASET,
    REGISTRY_PATH,
    RESULT_ROOT,
    TASK_ORDER,
    load_prepared_task,
    mbench_url,
    metropt3_split_masks,
    validate_mtsbench_source_dir,
)


class TestExternalValidationInvariants(unittest.TestCase):
    def test_fixed_task_cardinality_and_macro_mapping(self):
        self.assertEqual(len(TASK_ORDER), 20)
        self.assertEqual(sum(task.startswith("MTSB_OPPORTUNITY_") for task in TASK_ORDER), 13)
        self.assertEqual(sum(task.startswith("MTSB_OCCUPANCY_") for task in TASK_ORDER), 2)
        counts = pd.Series(PAPER_DATASET).value_counts()
        self.assertEqual(counts["OPPORTUNITY"], 13)
        self.assertEqual(counts["Occupancy"], 2)
        self.assertEqual(len(counts), 7)

    def test_batadal_intervals_are_hourly_endpoint_inclusive(self):
        expected_counts = [70, 65, 31, 31, 100, 80, 30]
        counts = []
        for start, end in BATADAL_ATTACK_INTERVALS:
            hours = (pd.to_datetime(end, format="%d/%m/%Y %H") - pd.to_datetime(start, format="%d/%m/%Y %H")).total_seconds() / 3600
            counts.append(int(hours) + 1)
        self.assertEqual(counts, expected_counts)

    def test_metro_fault_intervals_are_fixed(self):
        self.assertEqual(len(METRO_FAULT_INTERVALS), 4)
        self.assertEqual(METRO_FAULT_INTERVALS[0], ("2020-04-18 00:00", "2020-04-18 23:59"))
        self.assertEqual(METRO_FAULT_INTERVALS[-1], ("2020-07-15 14:30", "2020-07-15 19:00"))

    def test_metro_split_excludes_the_preceding_month(self):
        timestamps = pd.Series(pd.to_datetime([
            "2020-02-28 23:59:50", "2020-03-01 00:00:00", "2020-03-31 23:59:59", "2020-04-01 00:00:00",
        ]))
        train, test, train_start, test_start = metropt3_split_masks(timestamps, pd.Period("2020-03", freq="M"))
        self.assertEqual(train_start, pd.Timestamp("2020-03-01 00:00:00"))
        self.assertEqual(test_start, pd.Timestamp("2020-04-01 00:00:00"))
        self.assertEqual(train.tolist(), [False, True, True, False])
        self.assertEqual(test.tolist(), [False, False, False, True])

    def test_mtsbench_download_path_is_pinned_and_excludes_validation_files(self):
        self.assertEqual(MTSBENCH_REPO_ID, "PLAN-Lab/mTSBench")
        self.assertEqual(MTSBENCH_REVISION, "9ea52adfa86373576f446a7f3f26395e506f1b8b")
        self.assertEqual(len(MTSBENCH_RELATIVE_PATHS), 34)
        self.assertFalse(any(path.endswith("_val.csv") for path in MTSBENCH_RELATIVE_PATHS))
        self.assertIn("/resolve/" + MTSBENCH_REVISION + "/", mbench_url(MTSBENCH_RELATIVE_PATHS[0]))
        self.assertNotIn("resolve/main", mbench_url(MTSBENCH_RELATIVE_PATHS[0]))

    def test_external_command_list_has_exactly_forty_independent_tasks(self):
        commands = [
            line.strip()
            for line in (ROOT / "EXTERNAL_VALIDATION_COMMANDS.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("sh ./scripts/multivariate_detection/detect_score/")
        ]
        self.assertEqual(len(commands), 40)
        self.assertEqual(len(set(commands)), 40)
        self.assertTrue(all(command.endswith(("/CATCH.sh", "/MSDCATCH.sh")) for command in commands))

    def test_external_wide_loader_excludes_timestamp_and_keeps_label(self):
        from ts_benchmark.data.data_source import LocalExternalAnomalyDetectDataSource

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.DataFrame(
                {
                    "timestamp": ["2020-01-01T00:00:00.000000", "2020-01-01T00:01:00.000000"],
                    "sensor_a": [1.0, 2.0],
                    "sensor_b": [3.0, 4.0],
                    "label": [0, 1],
                }
            )
            frame.to_csv(root / "sample.csv", index=False)
            source = object.__new__(LocalExternalAnomalyDetectDataSource)
            source.local_data_path = str(root)
            loaded = source._load_series("sample.csv")
            self.assertEqual(list(loaded.columns), ["sensor_a", "sensor_b", "label"])
            self.assertEqual(loaded.iloc[:, :2].to_numpy().dtype, np.float32)
            self.assertEqual(loaded["label"].tolist(), [0, 1])

    def test_local_mtsbench_artifact_accepts_only_the_frozen_missing_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index, relative in enumerate(MTSBENCH_MISSING_PATHS):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"source-{index}".encode("ascii")
                path.write_bytes(payload)
                rows.append(
                    {
                        "repo_relative_path": relative,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            pd.DataFrame(rows).to_csv(root / "mtsbench_missing_sha256.csv", index=False)
            self.assertEqual(
                [row["repo_relative_path"] for row in validate_mtsbench_source_dir(root)],
                list(MTSBENCH_MISSING_PATHS),
            )
            rows[-1]["repo_relative_path"] = "swan/swan_sf_val.csv"
            pd.DataFrame(rows).to_csv(root / "mtsbench_missing_sha256.csv", index=False)
            with self.assertRaises(ValueError):
                validate_mtsbench_source_dir(root)


@unittest.skipUnless(METADATA_PATH.exists() and REGISTRY_PATH.exists(), "run external download, preparation, and descriptor freeze first")
class TestPreparedExternalValidationData(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.read_csv(METADATA_PATH).set_index("file_name")
        self.registry = pd.read_csv(REGISTRY_PATH).set_index("task")
        self.registry["status"] = self.registry.get("status", pd.Series("valid", index=self.registry.index)).fillna("valid")
        self.valid_tasks = [task for task in TASK_ORDER if self.registry.loc[task, "status"] == "valid"]

    def test_all_twenty_tasks_are_registered_and_discoverable(self):
        self.assertEqual(set(self.registry.index), set(TASK_ORDER))
        self.assertEqual(set(self.metadata.index), {f"{task}.csv" for task in self.valid_tasks})
        self.assertTrue(all((DATA_ROOT / f"{task}.csv").exists() for task in self.valid_tasks))
        from ts_benchmark.data.data_source import LocalExternalAnomalyDetectDataSource

        source = LocalExternalAnomalyDetectDataSource()
        self.assertEqual(set(source.dataset.metadata.index), {f"{task}.csv" for task in self.valid_tasks})

    def test_shapes_labels_and_numeric_contracts(self):
        for task in self.valid_tasks:
            with self.subTest(task=task):
                train, test, train_labels, test_labels, frame = load_prepared_task(task)
                self.assertGreaterEqual(train.shape[1], 2)
                self.assertEqual(train.shape[1], test.shape[1])
                self.assertGreaterEqual(len(train), 192)
                self.assertGreaterEqual(len(test), 192)
                self.assertEqual(train.dtype, np.float32)
                self.assertTrue(train.flags.c_contiguous)
                self.assertTrue(test.flags.c_contiguous)
                self.assertTrue(np.isfinite(train).all())
                self.assertTrue(np.isfinite(test).all())
                self.assertEqual(test_labels.ndim, 1)
                self.assertEqual(set(np.unique(test_labels)), {0, 1})
                self.assertEqual(len(train_labels), len(train))
                self.assertEqual(list(frame.columns)[0], "timestamp")
                self.assertEqual(list(frame.columns)[-1], "label")
                train_length = len(train)
                timestamps = pd.to_datetime(frame["timestamp"])
                self.assertTrue(timestamps.iloc[:train_length].is_monotonic_increasing)
                self.assertTrue(timestamps.iloc[train_length:].is_monotonic_increasing)
                if task == "MetroPT3":
                    self.assertEqual(len(train), 230448)
                    self.assertEqual(len(test), 1071650)
                    self.assertTrue((timestamps.iloc[:train_length] >= pd.Timestamp("2020-03-01")).all())
                    self.assertTrue((timestamps.iloc[:train_length] < pd.Timestamp("2020-04-01")).all())
                    self.assertTrue((timestamps.iloc[train_length:] >= pd.Timestamp("2020-04-01")).all())
                    self.assertFalse((timestamps.dt.month == 2).any())

    def test_batadal_and_metro_official_interval_labels(self):
        batadal = self.registry.loc["BATADAL"]
        batadal_counts = json.loads(batadal["official_attack_interval_counts"])
        self.assertEqual(len(batadal_counts), 7)
        self.assertTrue(all(count > 0 for count in batadal_counts))
        metro = self.registry.loc["MetroPT3"]
        if metro["status"] == "valid":
            metro_counts = json.loads(metro["official_fault_interval_counts"])
            self.assertEqual(len(metro_counts), 4)
            self.assertTrue(all(count > 0 for count in metro_counts))
        else:
            self.assertEqual(metro["status"], "excluded_integrity_rule")
            self.assertEqual(metro["exclusion_reason"], "no_complete_calendar_month")

    def test_metro_calendar_audit_and_status_are_consistent(self):
        coverage_path = RESULT_ROOT / "metropt3_calendar_coverage.csv"
        audit_path = RESULT_ROOT / "METROPT3_SPLIT_AUDIT.md"
        self.assertTrue(coverage_path.exists())
        self.assertTrue(audit_path.exists())
        coverage = pd.read_csv(coverage_path)
        required = {
            "year_month", "first_timestamp", "last_timestamp", "row_count",
            "observed_natural_day_count", "calendar_day_count", "missing_natural_dates",
            "definition_a_calendar_boundary", "definition_b_every_natural_day",
            "definition_c_every_theoretical_sample", "theoretical_sample_count",
            "actual_to_theoretical_ratio",
        }
        self.assertTrue(required.issubset(coverage.columns))
        self.assertEqual(coverage.loc[coverage["definition_a_calendar_boundary"], "year_month"].iloc[0], "2020-03")
        self.assertEqual(coverage.loc[coverage["definition_b_every_natural_day"], "year_month"].iloc[0], "2020-03")
        self.assertFalse(coverage["definition_c_every_theoretical_sample"].any())
        metro = self.registry.loc["MetroPT3"]
        self.assertEqual(metro["status"], "valid")
        self.assertEqual(metro["first_complete_calendar_month"], "2020-03")
        self.assertEqual(metro["test_start"], "2020-04-01 00:00:00")
        self.assertEqual(json.loads(metro["official_fault_interval_counts"]), [8657, 2360, 17315, 1622])
        self.assertEqual(int(metro["test_anomaly_count"]), 29954)

    def test_mtsbench_task_counts_and_macro_means(self):
        self.assertEqual(sum(self.registry["paper_dataset"] == "OPPORTUNITY"), 13)
        self.assertEqual(sum(self.registry["paper_dataset"] == "Occupancy"), 2)
        values = pd.DataFrame({"paper_dataset": ["OPPORTUNITY"] * 13, "metric": np.arange(13, dtype=float)})
        self.assertEqual(values.groupby("paper_dataset")["metric"].mean().iloc[0], 6.0)
        verification = pd.read_csv(MTSBENCH_VERIFICATION_PATH)
        self.assertEqual(len(verification), 34)
        self.assertTrue((verification["status"] == "match").all())
        mbench = self.registry[self.registry["paper_dataset"].isin(["OPPORTUNITY", "Occupancy", "Metro", "SWAN-SF"])]
        self.assertEqual(set(mbench["source_repo_id"]), {MTSBENCH_REPO_ID})
        self.assertEqual(set(mbench["source_revision"]), {MTSBENCH_REVISION})

    def test_descriptor_freeze_is_pre_result_and_train_only(self):
        path = RESULT_ROOT / "external_descriptor_freeze.json"
        self.assertTrue(path.exists())
        freeze = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(freeze["model_results_present_at_freeze"])
        self.assertEqual(freeze["seq_len"], 192)
        self.assertEqual(set(freeze["candidate_expected_delta_auc_roc_directions"]), {
            "mean_drift",
            "low_frequency_energy_ratio_mean",
            "periodicity_top3_ratio",
            "correlation_drift",
        })
        self.assertEqual(freeze["planned_task_count"], 20)
        self.assertEqual(freeze["valid_task_count"], len(self.valid_tasks))
        self.assertEqual(freeze["planned_dataset_count"], 7)
        self.assertEqual(freeze["valid_dataset_count"], len({PAPER_DATASET[task] for task in self.valid_tasks}))
        descriptors = pd.read_csv(RESULT_ROOT / "external_task_descriptors.csv")
        self.assertEqual(set(descriptors["task"]), set(self.valid_tasks))

    def test_frozen_model_directories_remain_unmodified(self):
        changed = subprocess.check_output(
            ["git", "-C", str(ROOT.parent), "diff", "--name-only"], text=True
        ).splitlines()
        protected = (
            "APD-CATCH/ts_benchmark/baselines/catch/",
            "APD-CATCH/ts_benchmark/baselines/msd_catch/",
            "APD-CATCH/ts_benchmark/baselines/bhd_msd_catch/",
            "APD-CATCH/ts_benchmark/baselines/ra_msd_catch/",
        )
        self.assertFalse(any(path.startswith(protected) for path in changed))


if __name__ == "__main__":
    unittest.main()
