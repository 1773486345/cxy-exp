import unittest

import torch

from .common import FakeCatch
from ts_benchmark.baselines.typefusion_catch_v2.layers.catch_anchor import CATCHAnchor


class CatchAnchorTests(unittest.TestCase):
    def test_forward_parity_and_freeze(self):
        torch.manual_seed(1)
        original = FakeCatch(3).eval()
        anchor = CATCHAnchor(catch_model=original).eval()
        x = torch.randn(2, 8, 3)
        expected = original(x)
        actual = anchor(x)
        self.assertTrue(torch.allclose(expected[0], actual["anchor_reconstruction"], atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.equal(expected[1], actual["anchor_spectrum"]))
        self.assertTrue(torch.equal(expected[2], actual["anchor_dc_loss"]))
        anchor.freeze()
        self.assertTrue(all(not p.requires_grad for p in anchor.parameters()))
