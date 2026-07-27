"""Layer-level relation masking must exclude the selected target channel value."""

import unittest

import torch

from ts_benchmark.baselines.typefusion_catch.layers.masked_relation_branch import MaskedRelationBranch
from ts_benchmark.baselines.typefusion_catch.tests.common import tiny_config


class RelationMaskNoLeakageTests(unittest.TestCase):
    def test_masked_target_channel_cannot_change_its_reconstruction(self) -> None:
        torch.manual_seed(703)
        branch = MaskedRelationBranch(tiny_config()).eval()
        x = torch.randn(2, 32, 4)
        changed = x.clone()
        changed[:, 11, 2] += 100.0
        baseline = branch(x, randomize_groups=False)["x_hat"][:, 11, 2]
        altered = branch(changed, randomize_groups=False)["x_hat"][:, 11, 2]
        torch.testing.assert_close(baseline, altered, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
