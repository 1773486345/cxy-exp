"""Relation masking must remain leak-free through the full model path."""

import unittest

import torch

from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel
from ts_benchmark.baselines.typefusion_catch.tests.common import tiny_config


class FullModelRelationMaskNoLeakageTests(unittest.TestCase):
    def test_train_and_eval_relation_reconstruction_mask_the_target(self) -> None:
        torch.manual_seed(53)
        config = tiny_config("branch_pretrain")
        model = TypeFusionCATCHModel(config)
        x = torch.randn(2, config.seq_len, config.c_in)
        target_time = 17
        target_channel = 2
        changed = x.clone()
        changed[:, target_time, target_channel] += 100.0

        for training in (True, False):
            model.train(training)
            # Training relation groups are random partitions.  Resetting the
            # seed makes both complete-path calls use the same target group.
            torch.manual_seed(59)
            with torch.no_grad():
                baseline = model(x, compute_joint=False)["branches"]["relation"]["x_hat"][:, target_time, target_channel]
            torch.manual_seed(59)
            with torch.no_grad():
                changed_prediction = model(changed, compute_joint=False)["branches"]["relation"]["x_hat"][:, target_time, target_channel]
            torch.testing.assert_close(baseline, changed_prediction, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
