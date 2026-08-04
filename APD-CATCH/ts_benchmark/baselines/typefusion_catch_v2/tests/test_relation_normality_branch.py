import unittest

import torch

from .common import tiny_config
from ts_benchmark.baselines.typefusion_catch_v2.layers.relation_normality_branch import RelationNormalityBranch


class RelationTests(unittest.TestCase):
    def test_target_mask_before_channel_mixing_and_chunk_bound(self):
        config = tiny_config(max_relation_attention_rows=8)
        branch = RelationNormalityBranch(config)
        x = torch.randn(2, config.seq_len, config.c_in)
        first = branch(x)
        changed = x.clone()
        changed[:, :, 1] += 100.0
        second = branch(changed)
        self.assertTrue(torch.allclose(first["prediction"][:, :, 1], second["prediction"][:, :, 1], atol=1e-6, rtol=1e-6))
        self.assertEqual(int(first["condition_batch_chunk"]), 1)
