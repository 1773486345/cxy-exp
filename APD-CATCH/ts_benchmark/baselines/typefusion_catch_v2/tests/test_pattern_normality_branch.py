import unittest

import torch

from .common import tiny_config
from ts_benchmark.baselines.typefusion_catch_v2.layers.pattern_normality_branch import PatternNormalityBranch


class PatternTests(unittest.TestCase):
    def test_two_masks_cover_all_patches(self):
        config = tiny_config()
        output = PatternNormalityBranch(config)(torch.randn(2, config.seq_len, config.c_in))
        masks = output["masked_patch_mask"]
        self.assertTrue(torch.all(masks.any(dim=0)))
        self.assertTrue(torch.isfinite(output["raw_error"]).all())
