"""Finite-loss and finite-gradient smoke test."""

import unittest

import torch

from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel
from ts_benchmark.baselines.typefusion_catch.tests.common import tiny_config


class BackwardFiniteTests(unittest.TestCase):
    def test_all_losses_and_key_gradients_are_finite(self) -> None:
        torch.manual_seed(29)
        config = tiny_config("joint_finetune")
        model = TypeFusionCATCHModel(config).train()
        output = model(torch.randn(2, config.seq_len, config.c_in))
        for loss in output["losses"].values():
            self.assertTrue(torch.isfinite(loss).all())
        output["losses"]["total"].backward()
        parameters = [
            model.state_branch.patch_decoder[-1].weight,
            model.evolution_branch.prediction_head.weight,
            model.pattern_branch.channel_decoder[-1].weight,
            model.relation_branch.output_head[-1].weight,
            model.branch_fusion.prediction_head[-1].weight,
            model.joint_decoder.patch_output[-1].weight,
        ]
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
