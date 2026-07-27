"""CATCH-style progress printing must not alter TypeFusion training work."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from ts_benchmark.baselines.typefusion_catch.TypeFusionCATCH import TypeFusionCATCH
from ts_benchmark.baselines.typefusion_catch.tests.common import small_normal_frame, tiny_fit_kwargs
from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel


class TrainingProgressLoggingTests(unittest.TestCase):
    def test_catch_style_stdout_preserves_steps_metadata_and_forward_count(self) -> None:
        frame = small_normal_frame()
        adapter = TypeFusionCATCH(**tiny_fit_kwargs(catch_train_epochs=3, patience=3))
        original_forward = TypeFusionCATCHModel.forward
        forward_calls = 0

        def counted_forward(model, *args, **kwargs):
            nonlocal forward_calls
            forward_calls += 1
            return original_forward(model, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "typefusion_run_config.json"
            config_path.write_text(json.dumps({"data_name": "unit-test"}), encoding="utf-8")
            captured = io.StringIO()
            with mock.patch.object(TypeFusionCATCHModel, "forward", new=counted_forward):
                with mock.patch.dict(
                    os.environ,
                    {
                        "TYPEFUSION_RUN_CONFIG_PATH": str(config_path),
                        "TYPEFUSION_PROFILE_TIMING": "",
                    },
                ):
                    with redirect_stdout(captured):
                        adapter.detect_fit(frame)
            metadata = json.loads(config_path.read_text(encoding="utf-8"))

        output = captured.getvalue()
        for token in (
            ">>>>>>> Stage:",
            "iters:",
            "speed:",
            "left time:",
            "Epoch:",
            "Train Loss:",
            "Vali Loss:",
            ">>>>>>> TypeFusion-CATCH training completed",
        ):
            self.assertIn(token, output)
        self.assertIn("First batch: stage=branch_pretrain", output)
        self.assertNotIn("data_wait_seconds", output)

        branch_pretrain = output.split(">>>>>>> Stage: fusion_train", maxsplit=1)[0]
        fusion_train = output.split(">>>>>>> Stage: fusion_train", maxsplit=1)[1].split(
            ">>>>>>> Stage: joint_finetune", maxsplit=1
        )[0]
        joint_finetune = output.split(">>>>>>> Stage: joint_finetune", maxsplit=1)[1]
        self.assertNotIn("joint loss:", branch_pretrain)
        self.assertNotIn("branch mask loss:", branch_pretrain)
        self.assertIn("joint loss:", fusion_train)
        self.assertIn("branch mask loss:", fusion_train)
        for token in (
            "state loss:",
            "evolution loss:",
            "pattern time loss:",
            "pattern freq loss:",
            "relation loss:",
            "joint loss:",
            "branch mask loss:",
        ):
            self.assertIn(token, joint_finetune)

        self.assertEqual(forward_calls, 6)
        self.assertEqual(
            adapter.stage_optimizer_steps,
            {
                "branch_pretrain": 1,
                "fusion_train": 1,
                "joint_finetune": 1,
            },
        )
        self.assertEqual(adapter.training_budget_summary["reference_total_steps"], 3)
        self.assertEqual(adapter.training_budget_summary["actual_total_steps"], 3)
        self.assertEqual(
            metadata["completed_stages"],
            ["branch_pretrain", "fusion_train", "joint_finetune"],
        )
        self.assertEqual(metadata["training_budget_summary"]["actual_total_steps"], 3)


if __name__ == "__main__":
    unittest.main()
