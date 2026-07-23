"""Integrity checks for the fixed external validation protocol.

Run after the data-preparation commands have completed. The small invariant tests
remain useful before download; data-dependent tests are skipped until the prepared
files exist rather than fabricating substitute data.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
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
from external_baseline_assets import (  # noqa: E402
    BASELINE_SPECS,
    command_for,
    script_path,
)
from summarize_external_validation import (  # noqa: E402
    ALL_METHODS,
    aggregate_all_method_results,
    collect_all_method_task_results,
)


def catch_script_hyper_parameters(task: str, model: str) -> tuple[dict, str]:
    path = ROOT / "scripts" / "multivariate_detection" / "detect_score" / f"{task}_script" / f"{model}.sh"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"--model-hyper-params '([^']+)'", text)
    if match is None:
        raise AssertionError(f"missing model hyperparameters: {path}")
    return json.loads(match.group(1)), text


def catch_training_batch_summary(train_length: int, batch_size: int, seq_len: int = 192) -> tuple[int, int, int, int]:
    train_split_length = int(train_length * 0.8)
    train_window_count = train_split_length - seq_len + 1
    train_batch_count = math.ceil(train_window_count / batch_size)
    catch_update_step = min(int(train_batch_count / 10), 100)
    return train_split_length, train_window_count, train_batch_count, catch_update_step


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

    def test_frozen_baseline_set_and_static_external_scripts(self):
        expected = {
            "ModernTCN", "iTransformer", "DualTF", "AnomalyTransformer", "DCdetector",
            "TimesNet", "PatchTST", "DLinear", "NLinear", "TFAD", "AutoEncoder",
            "OCSVM", "IsolationForest", "PCA", "HBOS",
        }
        self.assertEqual({spec["paper_name"] for spec in BASELINE_SPECS}, expected)
        self.assertFalse(expected & {"TranAD", "GDN", "USAD"})
        self.assertEqual(len(BASELINE_SPECS), 15)
        planned = []
        for task in TASK_ORDER:
            for spec in BASELINE_SPECS:
                path = script_path(task, spec)
                self.assertTrue(path.is_file(), path)
                subprocess.check_call(["bash", "-n", str(path)])
                text = path.read_text(encoding="utf-8")
                self.assertEqual(text.count("run_benchmark.py"), 1)
                self.assertEqual(text.count("--model-name"), 1)
                self.assertIn('--data-set-name "external_detect"', text)
                self.assertIn(f'--data-name-list "{task}.csv"', text)
                self.assertIn("--seed 2021", text)
                self.assertIn("unfixed_detect_score_multi_config.json", text)
                self.assertIn(f'score/external_validation/{task}/{spec["result_name"]}', text)
                self.assertNotIn("nohup", text)
                self.assertNotIn(" &", text)
                self.assertNotIn("for ", text)
                self.assertNotIn("while ", text)
                planned.append(path)
        self.assertEqual(len(planned), 300)
        self.assertEqual(len(set(planned)), 300)
        commands = [
            line.strip()
            for line in (ROOT / "EXTERNAL_BASELINE_COMMANDS.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("sh ./scripts/multivariate_detection/detect_score/")
        ]
        self.assertEqual(len(commands), 300)
        self.assertEqual(len(set(commands)), 300)
        self.assertTrue(all(" &" not in command and "nohup" not in command for command in commands))

    def test_baseline_command_templates_are_fixed_and_model_specific(self):
        transformer_names = {"iTransformer", "TimesNet", "PatchTST", "DLinear", "NLinear"}
        for spec in BASELINE_SPECS:
            with self.subTest(spec=spec["paper_name"]):
                command = command_for("HAI20_07", spec)
                self.assertIn(spec["framework_model_name"], command)
                self.assertIn("--seed 2021", command)
                self.assertIn("--data-set-name \"external_detect\"", command)
                self.assertIn("--save-path \"score/external_validation/HAI20_07/", command)
                self.assertEqual(spec["adapter"] == "transformer_adapter", spec["paper_name"] in transformer_names)
                self.assertTrue((ROOT / spec["source_existing_script"]).is_file())
                self.assertNotIn("label", str(spec["model_hyper_params"]).lower())

    def test_occupancy_catch_and_msd_share_the_recorded_compatibility_batch_size(self):
        for task in ("MTSB_OCCUPANCY_01", "MTSB_OCCUPANCY_02"):
            commands = {}
            for model in ("CATCH", "MSDCATCH"):
                script = ROOT / "scripts" / "multivariate_detection" / "detect_score" / f"{task}_script" / f"{model}.sh"
                text = script.read_text(encoding="utf-8")
                match = re.search(r"--model-hyper-params '([^']+)'", text)
                self.assertIsNotNone(match, script)
                commands[model] = json.loads(match.group(1))
            self.assertEqual(commands["CATCH"], commands["MSDCATCH"])
            self.assertEqual(commands["CATCH"]["batch_size"], 64)
            self.assertEqual(commands["CATCH"]["seq_len"], 192)

    def test_all_baseline_tasks_keep_their_frozen_psm_template_parameters(self):
        for task in TASK_ORDER:
            for spec in BASELINE_SPECS:
                command = command_for(task, spec)
                params = json.loads(re.search(r"--model-hyper-params '([^']+)'", command).group(1))
                self.assertEqual(params, spec["model_hyper_params"])
                if "batch_size" in spec["model_hyper_params"]:
                    self.assertLessEqual(params["batch_size"], spec["model_hyper_params"]["batch_size"])
                else:
                    self.assertNotIn("batch_size", params)

        occupancy = {spec["paper_name"]: json.loads(re.search(r"--model-hyper-params '([^']+)'", command_for("MTSB_OCCUPANCY_01", spec)).group(1)) for spec in BASELINE_SPECS}
        self.assertEqual(occupancy["DualTF"]["batch_size"], 8)
        self.assertEqual(occupancy["iTransformer"]["batch_size"], 64)
        for model in ("ModernTCN", "AnomalyTransformer", "DCdetector", "TimesNet", "PatchTST", "DLinear", "NLinear"):
            self.assertEqual(occupancy[model]["batch_size"], 128)
        self.assertNotIn("batch_size", occupancy["TFAD"])
        self.assertNotIn("batch_size", occupancy["AutoEncoder"])
        overrides = pd.read_csv(RESULT_ROOT / "external_baseline_batch_overrides.csv")
        self.assertEqual(
            list(overrides.columns),
            ["task", "paper_name", "original_batch_size", "final_batch_size", "reason"],
        )
        self.assertTrue(overrides.empty)

    def test_all_method_macro_and_ranking_leave_missing_results_unranked(self):
        records = pd.DataFrame(
            [
                {"task": "MTSB_OPPORTUNITY_01", "paper_dataset": "OPPORTUNITY", "method": "CATCH", "auc_pr": 0.4, "auc_roc": 0.6, "status": "valid"},
                {"task": "MTSB_OPPORTUNITY_01", "paper_dataset": "OPPORTUNITY", "method": "MSDCATCH", "auc_pr": 0.5, "auc_roc": 0.7, "status": "valid"},
                {"task": "MTSB_OPPORTUNITY_02", "paper_dataset": "OPPORTUNITY", "method": "CATCH", "auc_pr": 0.6, "auc_roc": 0.8, "status": "valid"},
                {"task": "MTSB_OPPORTUNITY_02", "paper_dataset": "OPPORTUNITY", "method": "MSDCATCH", "auc_pr": np.nan, "auc_roc": np.nan, "status": "missing"},
                {"task": "HAI20_07", "paper_dataset": "HAI20_07", "method": "CATCH", "auc_pr": 0.7, "auc_roc": 0.9, "status": "valid"},
                {"task": "HAI20_07", "paper_dataset": "HAI20_07", "method": "MSDCATCH", "auc_pr": 0.8, "auc_roc": 0.95, "status": "valid"},
            ]
        )
        self.assertIn("CATCH", ALL_METHODS)
        dataset, summary = aggregate_all_method_results(records)
        catch = dataset[(dataset["paper_dataset"] == "OPPORTUNITY") & (dataset["method"] == "CATCH")].iloc[0]
        msd = dataset[(dataset["paper_dataset"] == "OPPORTUNITY") & (dataset["method"] == "MSDCATCH")].iloc[0]
        self.assertEqual(catch["provisional_auc_pr"], 0.5)
        self.assertTrue(np.isnan(catch["auc_pr"]))
        self.assertTrue(np.isnan(catch["auc_roc"]))
        self.assertTrue(np.isnan(catch["auc_pr_rank"]))
        self.assertTrue(np.isnan(catch["auc_roc_rank"]))
        self.assertEqual(catch["valid_task_count"], 2)
        self.assertEqual(catch["expected_task_count"], 13)
        self.assertFalse(catch["complete"])
        self.assertEqual(msd["valid_task_count"], 1)
        self.assertFalse(msd["complete"])
        self.assertTrue(np.isnan(msd["auc_pr"]))
        self.assertTrue(np.isnan(msd["auc_roc"]))
        self.assertTrue(np.isnan(msd["auc_pr_rank"]))
        self.assertTrue(np.isnan(msd["auc_roc_rank"]))
        complete = dataset[(dataset["paper_dataset"] == "HAI20_07") & (dataset["method"] == "MSDCATCH")].iloc[0]
        self.assertTrue(complete["complete"])
        self.assertTrue(np.isfinite(complete["auc_pr_rank"]))
        missing = dataset[(dataset["paper_dataset"] == "HAI20_07") & (dataset["method"] == "CATCH")].iloc[0]
        self.assertTrue(np.isfinite(missing["auc_pr_rank"]))
        summary_catch = summary.loc[summary["method"] == "CATCH"].iloc[0]
        self.assertEqual(summary_catch["expected_dataset_count"], 7)
        self.assertEqual(summary_catch["provisional_dataset_count"], 2)
        self.assertEqual(summary_catch["complete_dataset_count"], 1)

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

    def test_catch_mask_interval_compatibility_batch_size_exception(self):
        expected_step_zero_at_default = {"MTSB_OCCUPANCY_01", "MTSB_OCCUPANCY_02"}
        default_step_zero = set()
        configured_batch_sizes = {}
        for task in self.valid_tasks:
            with self.subTest(task=task):
                catch_params, catch_text = catch_script_hyper_parameters(task, "CATCH")
                msd_params, msd_text = catch_script_hyper_parameters(task, "MSDCATCH")
                self.assertEqual(catch_params["batch_size"], msd_params["batch_size"])
                self.assertEqual(catch_params["seq_len"], 192)
                self.assertEqual(msd_params["seq_len"], 192)
                self.assertEqual(catch_params["patch_size"], 16)
                self.assertEqual(msd_params["patch_size"], 16)
                self.assertIn("--seed 2021", catch_text)
                self.assertIn("--seed 2021", msd_text)
                configured_batch_sizes[task] = catch_params["batch_size"]

                train_length = int(self.registry.loc[task, "train_length"])
                _, _, _, default_step = catch_training_batch_summary(train_length, 128)
                if default_step == 0:
                    default_step_zero.add(task)

        self.assertEqual(default_step_zero, expected_step_zero_at_default)
        self.assertEqual(
            {task for task, batch_size in configured_batch_sizes.items() if batch_size == 64},
            expected_step_zero_at_default,
        )
        self.assertTrue(
            all(batch_size == 128 for task, batch_size in configured_batch_sizes.items() if task not in expected_step_zero_at_default)
        )

        expected = {
            "MTSB_OCCUPANCY_01": (1073, 858, 667, 11, 1),
            "MTSB_OCCUPANCY_02": (1492, 1193, 1002, 16, 1),
        }
        for task, (train_length, split_length, window_count, batch_count, update_step) in expected.items():
            with self.subTest(task=task):
                self.assertEqual(int(self.registry.loc[task, "train_length"]), train_length)
                self.assertEqual(
                    catch_training_batch_summary(train_length, 64),
                    (split_length, window_count, batch_count, update_step),
                )

    def test_failed_attempts_are_outside_formal_result_discovery(self):
        failed_root = (
            ROOT / "result" / "score" / "external_validation" / "_failed_attempts"
            / "MTSB_OCCUPANCY_01" / "CATCH"
        )
        reason = failed_root / "failure_reason.txt"
        if reason.exists():
            self.assertIn("catch mask update step=0", reason.read_text(encoding="utf-8").lower())
        records = collect_all_method_task_results(tasks=["MTSB_OCCUPANCY_01"])
        self.assertFalse(records["archive"].fillna("").str.contains("_failed_attempts", regex=False).any())

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

    def test_baseline_model_source_directories_remain_unmodified(self):
        changed = subprocess.check_output(
            ["git", "-C", str(ROOT.parent), "diff", "--name-only"], text=True
        ).splitlines()
        protected = (
            "APD-CATCH/ts_benchmark/baselines/self_impl/",
            "APD-CATCH/ts_benchmark/baselines/time_series_library/",
            "APD-CATCH/ts_benchmark/baselines/merlion/",
            "APD-CATCH/ts_benchmark/baselines/tods/",
        )
        self.assertFalse(any(path.startswith(protected) for path in changed))


if __name__ == "__main__":
    unittest.main()
