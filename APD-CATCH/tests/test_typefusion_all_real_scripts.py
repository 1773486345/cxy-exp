"""Static audit for the fixed 23 real TypeFusion-CATCH task scripts."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result/typefusion_catch_main"
TASKS = (
    "ASD_dataset_1", "ASD_dataset_2", "ASD_dataset_3", "ASD_dataset_4",
    "ASD_dataset_5", "ASD_dataset_6", "ASD_dataset_7", "ASD_dataset_8",
    "ASD_dataset_9", "ASD_dataset_10", "ASD_dataset_11", "ASD_dataset_12",
    "CICIDS", "CalIt2", "Creditcard", "GECCO", "Genesis", "MSL", "NYC",
    "PSM", "SMAP", "SMD", "SWAT",
)
COMMON_PARAMS = (
    "batch_size", "cf_dim", "d_ff", "d_model", "dropout", "e_layers",
    "head_dim", "lr", "n_heads", "patience", "seq_len", "patch_size", "patch_stride",
)
FIXED_TYPEFUSION_PARAMS = {
    "seed": 2021,
    "fit_mode": "three_stage",
    "training_budget_mode": "equal_total_steps",
    "joint_finetune_lr_scale": 0.1,
    "lambda_freq": 0.1,
    "lambda_mask": 0.1,
    "temporal_hidden_dim": 128,
    "temporal_layers": 3,
    "memory_size": 32,
    "memory_topk": 4,
    "branch_dim": 128,
    "fusion_layers": 2,
    "fusion_heads": 4,
    "relation_mask_groups": 4,
    "pattern_mask_ratio": 0.25,
}
TEMPLATE_PATTERN = re.compile(r"^MODEL_HYPER_PARAMS='([^']+)'$", re.MULTILINE)


def csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def script_params(text: str, batch_size: int):
    match = TEMPLATE_PATTERN.search(text)
    if match is None:
        raise AssertionError("missing static model parameter template")
    return json.loads(match.group(1).replace("__BATCH_SIZE__", str(batch_size)))


class TypeFusionAllRealScriptsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_rows = csv_rows(RESULT_ROOT / "typefusion_catch_source_registry.csv")
        cls.data_rows = csv_rows(RESULT_ROOT / "typefusion_real_data_registry.csv")
        cls.source_by_task = {row["task"]: row for row in cls.source_rows}

    def test_fixed_real_registry_is_complete_and_healthy(self) -> None:
        self.assertEqual(tuple(row["task"] for row in self.source_rows), TASKS)
        self.assertEqual(tuple(row["task"] for row in self.data_rows), TASKS)
        self.assertEqual(sum(task.startswith("ASD_dataset_") for task in TASKS), 12)
        self.assertFalse(any("synthetic" in task.lower() for task in TASKS))
        for row in self.source_rows:
            self.assertEqual(row["source_complete"], "True", row["task"])
            self.assertEqual(row["source_conflict"], "False", row["task"])
            self.assertEqual(row["baseline_report_valid"], "True", row["task"])
            self.assertEqual(row["baseline_metric_available"], "True", row["task"])
        for row in self.data_rows:
            self.assertEqual(row["integrity_status"], "ok", row["task"])
            self.assertEqual(row["metadata_discoverable"], "True", row["task"])

    def test_each_script_preserves_audited_configuration_and_is_safe_shell(self) -> None:
        save_paths = set()
        for task in TASKS:
            source = self.source_by_task[task]
            path = ROOT / "scripts/multivariate_detection/detect_score" / f"{task}_script/TypeFusionCATCH.sh"
            self.assertTrue(path.is_file(), task)
            self.assertTrue(os.access(path, os.X_OK), task)
            syntax = subprocess.run(["bash", "-n", str(path)], check=False)
            self.assertEqual(syntax.returncode, 0, task)
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n"), task)
            self.assertIn(
                f'# Source CATCH official_CATCH.sh: {source["official_source_script"]}', text
            )
            self.assertIn(f'# Source CATCH test report: {source["baseline_report"]}', text)
            self.assertIn(
                f'# Source CATCH archive commit: {source["source_catch_master_commit"]}', text
            )
            self.assertIn("# Configuration audit date (UTC):", text)
            self.assertIn(
                f'# GECCO fairness override: {str(task == "GECCO").lower()}', text
            )
            self.assertIn(f'--data-name-list "{source["data_name"]}"', text)
            self.assertIn('--model-name "typefusion_catch.TypeFusionCATCH"', text)
            self.assertIn("--seed 2021", text)
            self.assertIn('--gpus "$GPU_ID"', text)
            self.assertIn("--num-workers 1", text)
            self.assertIn("--timeout 60000", text)
            self.assertIn(f"score/TypeFusion-CATCH/{task}/run-", text)
            self.assertIn('RUN_CONFIG_PATH="$ROOT_DIR/result/$SAVE_PATH/typefusion_run_config.json"', text)
            self.assertIn('export TYPEFUSION_RUN_CONFIG_PATH="$RUN_CONFIG_PATH"', text)
            self.assertIn('"model_name":"typefusion_catch.TypeFusionCATCH"', text)
            self.assertIn('"seed":2021', text)
            save_paths.add(task)
            self.assertNotIn("detect_label", text)
            self.assertNotIn("threshold", text.lower())
            self.assertNotIn("score fusion", text.lower())
            self.assertNotIn("test_label", text.lower())
            self.assertNotIn("&", text)
            self.assertIsNone(re.search(r"\b(?:for|while|until|do|done)\b", text), task)

            params = script_params(text, int(source["original_batch_size"]))
            effective = json.loads(source["effective_catch_model_hyper_params_json"])
            for key, value in FIXED_TYPEFUSION_PARAMS.items():
                self.assertEqual(params[key], value, f"{task}:{key}")
            self.assertEqual(params["catch_train_epochs"], effective["num_epochs"])
            for field in COMMON_PARAMS:
                if task == "GECCO" and field == "seq_len":
                    self.assertEqual(params[field], 192)
                elif task == "ASD_dataset_1" and field == "n_heads":
                    self.assertEqual(params[field], 4)
                    self.assertIn("n_heads:16->4", source["typefusion_compatibility_override"])
                else:
                    self.assertEqual(params[field], effective[field], f"{task}:{field}")
            if task != "ASD_dataset_1":
                self.assertEqual(source["typefusion_compatibility_override"], "", task)
            if task != "GECCO":
                self.assertEqual(source["gecco_fairness_override"], "False", task)
        self.assertEqual(len(save_paths), 23)

    def test_gecco_override_and_command_document_are_explicit(self) -> None:
        gecco = self.source_by_task["GECCO"]
        self.assertEqual(gecco["baseline_requires_fair_rerun"], "True")
        self.assertEqual(gecco["direct_baseline_comparable"], "False")
        fair_script = ROOT / "scripts/multivariate_detection/detect_score/GECCO_script/CATCH_GECCO_FAIR.sh"
        self.assertTrue(fair_script.is_file())
        self.assertTrue(os.access(fair_script, os.X_OK))
        self.assertEqual(subprocess.run(["bash", "-n", str(fair_script)], check=False).returncode, 0)
        self.assertIn('"seq_len":192', fair_script.read_text(encoding="utf-8"))

        commands = (ROOT / "TYPEFUSION_CATCH_ALL_REAL_COMMANDS.md").read_text(encoding="utf-8")
        command_lines = [line for line in commands.splitlines() if line.startswith("GPU_ID=0 sh ")]
        self.assertEqual(len(command_lines), 23)
        self.assertFalse(any("synthetic" in line.lower() for line in command_lines))
        self.assertFalse(any("&" in line or ";" in line for line in command_lines))
        self.assertEqual(
            command_lines,
            [
                f"GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/{task}_script/TypeFusionCATCH.sh"
                for task in TASKS[:12]
            ]
            + [
                f"GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/{task}_script/TypeFusionCATCH.sh"
                for task in ("CICIDS", "SWAT", "GECCO", "Genesis", "CalIt2", "Creditcard", "MSL", "NYC", "PSM", "SMAP", "SMD")
            ],
        )

    def test_original_catch_directory_is_unchanged(self) -> None:
        diff = subprocess.run(
            ["git", "diff", "--", "ts_benchmark/baselines/catch/"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(diff.stdout, "")


if __name__ == "__main__":
    unittest.main()
