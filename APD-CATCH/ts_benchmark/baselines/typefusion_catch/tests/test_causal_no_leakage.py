"""Regression test for the evolution branch's strict causal boundary."""

import unittest

import torch

from ts_benchmark.baselines.typefusion_catch.layers.causal_evolution_branch import CausalEvolutionBranch
from ts_benchmark.baselines.typefusion_catch.tests.common import tiny_config


class CausalNoLeakageTests(unittest.TestCase):
    def test_target_value_cannot_change_its_prediction(self) -> None:
        torch.manual_seed(11)
        config = tiny_config("branch_pretrain")
        branch = CausalEvolutionBranch(config).eval()
        x = torch.randn(2, config.seq_len, config.c_in)
        target_time = 17
        changed = x.clone()
        changed[:, target_time, :] += 100.0
        with torch.no_grad():
            prediction = branch(x)["x_hat"][:, target_time, :]
            changed_prediction = branch(changed)["x_hat"][:, target_time, :]
        torch.testing.assert_close(prediction, changed_prediction, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
