"""Newest-valid TypeFusion run selection must retain every rejection reason."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from summarize_typefusion_catch_real import FIXED_TYPEFUSION_KEYS, select_latest_valid_run  # noqa: E402
from tests.typefusion_result_fixture import (  # noqa: E402
    FORMAL_TYPEFUSION_OUTPUT_NAME,
    write_archive,
    write_report,
)
from ts_benchmark.baselines.typefusion_catch.TypeFusionCATCH import TypeFusionCATCH  # noqa: E402


def expected_params() -> dict:
    return {
        "batch_size": 4,
        "catch_train_epochs": 1,
        "cf_dim": 4,
        "d_ff": 8,
        "d_model": 8,
        "dropout": 0.0,
        "e_layers": 1,
        "head_dim": 4,
        "lr": 0.001,
        "n_heads": 2,
        "patience": 1,
        "seq_len": 8,
        "patch_size": 4,
        "patch_stride": 2,
        **FIXED_TYPEFUSION_KEYS,
    }


def source_row() -> dict:
    return {"task": "PSM", "data_name": "PSM.csv", "typefusion_hyper_params_json": json.dumps(expected_params())}


def write_valid_run(
    path: Path,
    *,
    seed: int = 2021,
    params: dict | None = None,
    archive_prefix: str = FORMAL_TYPEFUSION_OUTPUT_NAME,
    archive_model_name: str = FORMAL_TYPEFUSION_OUTPUT_NAME,
    report_model_name: str = FORMAL_TYPEFUSION_OUTPUT_NAME,
) -> None:
    params = params or expected_params()
    path.mkdir(parents=True)
    write_archive(
        path / f"{archive_prefix}.result.csv.tar.gz",
        archive_model_name,
        "PSM.csv",
        params,
        seed=seed,
    )
    write_report(path / "test_report.result.csv", report_model_name, params, seed=seed)
    (path / "typefusion_run_config.json").write_text(
        json.dumps(
            {
                "data_name": "PSM.csv",
                "model_name": "typefusion_catch.TypeFusionCATCH",
                "seed": seed,
                "model_hyper_params": params,
                "training_budget_summary": {
                    "mode": "equal_total_steps",
                    "reference_total_steps": 6,
                    "branch_pretrain_steps": 2,
                    "fusion_train_steps": 2,
                    "joint_finetune_steps": 2,
                    "actual_branch_pretrain_steps": 2,
                    "actual_fusion_train_steps": 2,
                    "actual_joint_finetune_steps": 2,
                },
                "completed_stages": ["branch_pretrain", "fusion_train", "joint_finetune"],
            }
        ),
        encoding="utf-8",
    )


class TypeFusionRunSelectionTests(unittest.TestCase):
    def test_adapter_writes_existing_stage_accounting_to_opt_in_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "typefusion_run_config.json"
            path.write_text("{}", encoding="utf-8")
            adapter = TypeFusionCATCH()
            adapter.training_budget_summary = {"mode": "equal_total_steps", "actual_total_steps": 3}
            adapter.stage_optimizer_steps = {"branch_pretrain": 1, "fusion_train": 1, "joint_finetune": 1}
            adapter.stage_validation_losses = {"branch_pretrain": 1.0}
            adapter.stage_best_states = {
                "branch_pretrain": {}, "fusion_train": {}, "joint_finetune": {}
            }
            previous = os.environ.get("TYPEFUSION_RUN_CONFIG_PATH")
            os.environ["TYPEFUSION_RUN_CONFIG_PATH"] = str(path)
            try:
                adapter._write_result_audit_metadata()
            finally:
                if previous is None:
                    del os.environ["TYPEFUSION_RUN_CONFIG_PATH"]
                else:
                    os.environ["TYPEFUSION_RUN_CONFIG_PATH"] = previous
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["training_budget_summary"]["actual_total_steps"], 3)
            self.assertEqual(payload["completed_stages"], ["branch_pretrain", "fusion_train", "joint_finetune"])

    def test_newer_damaged_run_does_not_replace_older_valid_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_root = root / "PSM"
            valid = task_root / "run-20260101T000000Z-1"
            broken = task_root / "run-20260102T000000Z-2"
            write_valid_run(valid)
            broken.mkdir(parents=True)
            selected, audit = select_latest_valid_run("PSM", source_row(), root)
            self.assertEqual(selected["run_path"], str(valid))
            self.assertEqual(sum(bool(row["selected"]) for row in audit), 1)
            self.assertEqual(audit[0]["rejection_reason"], "run_config_missing")
            self.assertTrue(audit[1]["selected"])

    def test_latest_valid_and_common_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "PSM" / "run-20260102T000000Z-2"
            write_valid_run(valid)
            selected, audit = select_latest_valid_run("PSM", source_row(), root)
            self.assertEqual(selected["run_path"], str(valid))
            self.assertTrue(audit[0]["selected"])
        for name, mutator, expected_reason in (
            ("oom", lambda run: (run / "run.log").write_text("CUDA out of memory"), "failure_marker"),
            ("seed", lambda run: write_valid_run(run, seed=2022), "archive_seed_mismatch"),
            ("config", lambda run: write_valid_run(run, params={**expected_params(), "lr": 0.5}), "model_param_mismatch"),
            (
                "nan",
                lambda run: (
                    write_valid_run(run),
                    write_archive(
                        run / "TypeFusionCATCH.result.csv.tar.gz",
                        "TypeFusionCATCH",
                        "PSM.csv",
                        expected_params(),
                        auc_roc=float("nan"),
                    ),
                ),
                "archive_primary_metric_not_finite",
            ),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = root / "PSM" / "run-20260101T000000Z-1"
                if name == "oom":
                    write_valid_run(run)
                    mutator(run)
                else:
                    mutator(run)
                selected, audit = select_latest_valid_run("PSM", source_row(), root)
                self.assertIsNone(selected)
                self.assertIn(expected_reason, audit[0]["rejection_reason"])

    def test_legacy_archive_name_is_accepted_but_duplicate_candidates_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "PSM" / "run-20260101T000000Z-1"
            write_valid_run(
                legacy,
                archive_prefix="TypeFusion-CATCH",
                archive_model_name="TypeFusion-CATCH",
                report_model_name="TypeFusion-CATCH",
            )
            selected, audit = select_latest_valid_run("PSM", source_row(), root)
            self.assertEqual(selected["archive_path"], str(legacy / "TypeFusion-CATCH.result.csv.tar.gz"))
            self.assertTrue(audit[0]["selected"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "PSM" / "run-20260101T000000Z-1"
            write_valid_run(duplicate)
            write_archive(
                duplicate / "TypeFusion-CATCH.legacy.csv.tar.gz",
                FORMAL_TYPEFUSION_OUTPUT_NAME,
                "PSM.csv",
                expected_params(),
            )
            selected, audit = select_latest_valid_run("PSM", source_row(), root)
            self.assertIsNone(selected)
            self.assertIn("multiple_typefusion_archives_conflict", audit[0]["rejection_reason"])

    def test_wrong_result_display_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "PSM" / "run-20260101T000000Z-1"
            write_valid_run(run, archive_model_name="WrongModel", report_model_name="WrongModel")
            selected, audit = select_latest_valid_run("PSM", source_row(), root)
            self.assertIsNone(selected)
            self.assertIn("archive_model_name_mismatch:WrongModel", audit[0]["rejection_reason"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "PSM" / "run-20260101T000000Z-1"
            write_valid_run(run, report_model_name="WrongModel")
            selected, audit = select_latest_valid_run("PSM", source_row(), root)
            self.assertIsNone(selected)
            self.assertEqual(audit[0]["rejection_reason"], "test_report_model_name_mismatch")


if __name__ == "__main__":
    unittest.main()
