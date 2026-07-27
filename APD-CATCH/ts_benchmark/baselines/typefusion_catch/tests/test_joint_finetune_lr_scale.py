"""Fixed low learning-rate protocol for Stage 3."""

import unittest

from ts_benchmark.baselines.typefusion_catch.TypeFusionCATCH import TypeFusionCATCH
from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.tests.common import small_normal_frame, tiny_fit_kwargs


class JointFinetuneLearningRateTests(unittest.TestCase):
    def test_stage_optimizers_use_the_configured_fixed_rates(self) -> None:
        scale = 0.2
        adapter = TypeFusionCATCH(**tiny_fit_kwargs(joint_finetune_lr_scale=scale))
        adapter.detect_fit(small_normal_frame())
        self.assertEqual(adapter.stage_optimizer_lrs["branch_pretrain"], adapter.config.lr)
        self.assertEqual(adapter.stage_optimizer_lrs["fusion_train"], adapter.config.lr)
        self.assertEqual(
            adapter.stage_optimizer_lrs["joint_finetune"], adapter.config.lr * scale
        )

    def test_invalid_stage_three_scale_is_rejected(self) -> None:
        for scale in (0.0, -0.1, 1.1):
            with self.assertRaisesRegex(ValueError, "joint_finetune_lr_scale"):
                TypeFusionConfig(joint_finetune_lr_scale=scale).validate()


if __name__ == "__main__":
    unittest.main()
