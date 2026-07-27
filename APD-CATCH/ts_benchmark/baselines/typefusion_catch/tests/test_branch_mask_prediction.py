"""Tests for branch-token masking and vectorised leave-one-out inference."""

import unittest

import torch

from ts_benchmark.baselines.typefusion_catch.layers.branch_fusion_transformer import BranchFusionTransformer
from ts_benchmark.baselines.typefusion_catch.tests.common import tiny_config


class BranchMaskPredictionTests(unittest.TestCase):
    def test_mask_prediction_and_batched_leave_one_out(self) -> None:
        torch.manual_seed(23)
        config = tiny_config("fusion_train")
        fusion = BranchFusionTransformer(config).train()
        q = torch.randn(3, config.num_patches, 4, config.branch_dim)
        masked = fusion.masked_branch_prediction(q)
        self.assertEqual(masked["prediction"].shape, q.shape)
        self.assertEqual(masked["branch_mask"].shape, (3, 4))
        self.assertTrue(masked["branch_mask"].any(dim=1).all())
        self.assertTrue(torch.isfinite(masked["loss"]))
        leave_one_out = fusion.leave_one_out(q)
        self.assertEqual(leave_one_out["q_normal"].shape, q.shape)
        self.assertEqual(int(leave_one_out["expanded_batch"]), q.size(0) * 4)
        self.assertEqual(leave_one_out["loo_masks"].shape, (q.size(0) * 4, 4))
        self.assertTrue(torch.equal(leave_one_out["loo_masks"][:4], torch.eye(4, dtype=torch.bool)))


if __name__ == "__main__":
    unittest.main()
