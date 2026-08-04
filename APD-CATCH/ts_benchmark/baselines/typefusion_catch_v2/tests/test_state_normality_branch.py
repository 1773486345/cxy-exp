import unittest

import torch

from .common import tiny_config
from ts_benchmark.baselines.typefusion_catch_v2.layers.state_normality_branch import StateNormalityBranch


class StateTests(unittest.TestCase):
    def test_differentiable_usage_and_state_hidden(self):
        config = tiny_config()
        branch = StateNormalityBranch(config)
        output = branch(torch.randn(2, config.seq_len, config.d_model))
        output["task_loss"].backward()
        self.assertTrue(torch.isfinite(branch.prototypes.grad).all())
        self.assertTrue(branch.prototypes.grad.abs().sum().item() > 0)
        self.assertEqual(output["z"].shape[-1], config.branch_dim)
