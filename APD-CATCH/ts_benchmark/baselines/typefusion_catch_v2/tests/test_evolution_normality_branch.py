import unittest

import torch

from .common import tiny_config
from ts_benchmark.baselines.typefusion_catch_v2.layers.evolution_normality_branch import EvolutionNormalityBranch


class EvolutionTests(unittest.TestCase):
    def test_train_eval_strict_causality(self):
        config = tiny_config()
        branch = EvolutionNormalityBranch(config.c_in, config.branch_dim, config.temporal_layers, 0.0)
        x = torch.randn(2, config.seq_len, config.c_in)
        for mode in (True, False):
            branch.train(mode)
            reference = branch(x)["prediction"][:, 5]
            changed = x.clone()
            changed[:, 6:] += 100.0
            changed[:, 5] += 100.0
            actual = branch(changed)["prediction"][:, 5]
            self.assertTrue(torch.allclose(reference, actual, atol=1e-6, rtol=1e-6))
            self.assertTrue(torch.equal(branch(x)["raw_error"][:, 0], torch.zeros(2)))
