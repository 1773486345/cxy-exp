import unittest

import torch

from .common import tiny_config
from ts_benchmark.baselines.typefusion_catch_v2.layers.evolution_normality_branch import EvolutionNormalityBranch


class EvolutionTests(unittest.TestCase):
    def test_target_and_future_do_not_change_previous_prediction(self):
        config = tiny_config()
        branch = EvolutionNormalityBranch(config).eval()
        x = torch.randn(2, config.seq_len, config.c_in)
        t = 7
        first = branch(x)["prediction"][:, t - 1]
        changed = x.clone()
        changed[:, t:] += 100
        second = branch(changed)["prediction"][:, t - 1]
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))
        branch.train()
        loss = branch(x)["task_loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(next(branch.parameters()).grad).all())
