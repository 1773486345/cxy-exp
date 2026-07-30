import unittest

import torch

from .common import tiny_config
from ts_benchmark.baselines.typefusion_catch_v2.layers.state_normality_branch import StateNormalityBranch


class StateBranchTests(unittest.TestCase):
    def test_finite_prototype_distance_and_gradient(self):
        config = tiny_config()
        branch = StateNormalityBranch(config)
        output = branch(torch.randn(2, config.seq_len, config.c_in))
        self.assertTrue(torch.isfinite(output["raw_error"]).all())
        output["task_loss"].backward()
        self.assertIsNotNone(branch.prototypes.grad)
        self.assertTrue(torch.isfinite(branch.prototypes.grad).all())
