import unittest

import torch

from .common import tiny_config
from ts_benchmark.baselines.typefusion_catch_v2.layers.relation_normality_branch import RelationNormalityBranch


class RelationTests(unittest.TestCase):
    def test_masked_target_current_value_has_no_effect(self):
        config = tiny_config()
        branch = RelationNormalityBranch(config)
        x = torch.randn(2, config.seq_len, config.c_in)
        target = 1
        first = branch(x)["prediction"][:, :, target]
        changed = x.clone()
        changed[:, :, target] += 100
        second = branch(changed)["prediction"][:, :, target]
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))
