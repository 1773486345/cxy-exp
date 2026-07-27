"""The default adapter fit mode must train all three stages."""

import unittest

from ts_benchmark.baselines.typefusion_catch.TypeFusionCATCH import TypeFusionCATCH
from ts_benchmark.baselines.typefusion_catch.tests.common import small_normal_frame, tiny_fit_kwargs
from ts_benchmark.baselines.typefusion_catch.tests.test_three_stage_state_continuity import _changed


class DefaultFitTrainsFusionAndDecoderTests(unittest.TestCase):
    def test_default_fit_runs_complete_stage_chain(self) -> None:
        adapter = TypeFusionCATCH(**tiny_fit_kwargs())
        self.assertEqual(adapter.config.fit_mode, "three_stage")
        adapter.detect_fit(small_normal_frame())
        self.assertEqual(
            set(adapter.stage_best_states),
            {"branch_pretrain", "fusion_train", "joint_finetune"},
        )
        fusion_start = adapter.stage_start_states["fusion_train"]
        fusion_best = adapter.stage_best_states["fusion_train"]
        self.assertTrue(_changed(fusion_start, fusion_best, "branch_fusion."))
        self.assertTrue(_changed(fusion_start, fusion_best, "joint_decoder."))


if __name__ == "__main__":
    unittest.main()
