"""Regression test for channel-masked conditional reconstruction."""

import unittest

import torch

from ts_benchmark.baselines.typefusion_catch.layers.masked_relation_branch import MaskedRelationBranch
from ts_benchmark.baselines.typefusion_catch.tests.common import tiny_config


class RelationMaskNoLeakageTests(unittest.TestCase):
    def test_masked_target_channel_cannot_change_its_reconstruction(self) -> None:
        torch.manual_seed(19)
        config = tiny_config("branch_pretrain")
        branch = MaskedRelationBranch(config).eval()
        x = torch.randn(2, config.seq_len, config.c_in)
        target_channel = 2
        changed = x.clone()
        changed[:, :, target_channel] += torch.randn_like(changed[:, :, target_channel]) * 100.0
        with torch.no_grad():
            prediction = branch(x, randomize_groups=False)["x_hat"][:, :, target_channel]
            changed_prediction = branch(changed, randomize_groups=False)["x_hat"][:, :, target_channel]
        torch.testing.assert_close(prediction, changed_prediction, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
