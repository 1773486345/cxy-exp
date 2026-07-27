"""Small-DataFrame smoke test for complete three-stage state continuity."""

import unittest

import torch

from ts_benchmark.baselines.typefusion_catch.TypeFusionCATCH import TypeFusionCATCH
from ts_benchmark.baselines.typefusion_catch.tests.common import small_normal_frame, tiny_fit_kwargs


def _changed(before, after, prefix: str) -> bool:
    return any(
        not torch.equal(before[name], after[name])
        for name in before
        if name.startswith(prefix)
    )


class ThreeStageStateContinuityTests(unittest.TestCase):
    def test_stage_checkpoints_are_restored_before_the_next_stage(self) -> None:
        frame = small_normal_frame()
        adapter = TypeFusionCATCH(**tiny_fit_kwargs())
        adapter.detect_fit(frame)

        stage_one = adapter.stage_best_states["branch_pretrain"]
        stage_two_start = adapter.stage_start_states["fusion_train"]
        for name, value in stage_one.items():
            if name.startswith(("state_branch.", "evolution_branch.", "pattern_branch.", "relation_branch.")):
                torch.testing.assert_close(stage_two_start[name], value, rtol=0.0, atol=0.0)

        stage_two = adapter.stage_best_states["fusion_train"]
        self.assertTrue(_changed(stage_two_start, stage_two, "branch_fusion."))
        self.assertTrue(_changed(stage_two_start, stage_two, "joint_decoder."))

        stage_three_start = adapter.stage_start_states["joint_finetune"]
        for name, value in stage_two.items():
            if name.startswith("branch_fusion."):
                torch.testing.assert_close(stage_three_start[name], value, rtol=0.0, atol=0.0)

        final_state = adapter.stage_best_states["joint_finetune"]
        self.assertEqual(set(adapter.best_state), set(final_state))
        for name in final_state:
            torch.testing.assert_close(adapter.best_state[name], final_state[name], rtol=0.0, atol=0.0)

        score, duplicate = adapter.detect_score(frame)
        self.assertGreater(score.size, 0)
        self.assertTrue(torch.isfinite(torch.from_numpy(score)).all())
        self.assertTrue((score == duplicate).all())

        no_checkpoint = TypeFusionCATCH(
            **tiny_fit_kwargs(fit_mode="single_stage", training_stage="fusion_train")
        )
        with self.assertRaisesRegex(ValueError, "requires an explicit prior checkpoint"):
            no_checkpoint.detect_fit(frame)

        resumed = TypeFusionCATCH(
            **tiny_fit_kwargs(fit_mode="single_stage", training_stage="fusion_train")
        )
        resumed.detect_fit(
            frame,
            previous_checkpoint=stage_one,
            previous_scaler=adapter.scaler,
        )
        for name, value in stage_one.items():
            if name.startswith(("state_branch.", "evolution_branch.", "pattern_branch.", "relation_branch.")):
                torch.testing.assert_close(
                    resumed.stage_start_states["fusion_train"][name], value, rtol=0.0, atol=0.0
                )


if __name__ == "__main__":
    unittest.main()
