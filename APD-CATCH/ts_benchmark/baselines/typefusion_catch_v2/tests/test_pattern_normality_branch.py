import unittest

import torch

from .common import tiny_config
from ts_benchmark.baselines.typefusion_catch_v2.layers.pattern_normality_branch import PatternNormalityBranch


class PatternTests(unittest.TestCase):
    def test_mask_before_encode_and_two_pass_coverage(self):
        config = tiny_config()
        branch = PatternNormalityBranch(config)
        x = torch.randn(2, config.seq_len, config.c_in)
        reference = branch(x)
        changed = x.clone()
        changed[:, :config.patch_size] += 100.0
        actual = branch(changed)
        self.assertTrue(torch.allclose(reference["masked_patch_predictions"][:, 0], actual["masked_patch_predictions"][:, 0], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.all(reference["masked_patch_mask"].any(dim=0)))
