"""Causality regression through the complete TypeFusion model path."""

import unittest

import torch
from torch import nn

from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel
from ts_benchmark.baselines.typefusion_catch.tests.common import tiny_config


class FullModelCausalNoLeakageTests(unittest.TestCase):
    def test_train_and_eval_paths_are_strictly_causal(self) -> None:
        torch.manual_seed(37)
        config = tiny_config("branch_pretrain")
        model = TypeFusionCATCHModel(config)
        self.assertTrue(all(isinstance(block.norm, nn.LayerNorm) for block in model.evolution_branch.blocks))
        x = torch.randn(2, config.seq_len, config.c_in)
        target_time = 17
        changed_target = x.clone()
        changed_target[:, target_time, :] += 100.0
        changed_future = x.clone()
        changed_future[:, target_time + 1 :, :] += 100.0
        for training in (True, False):
            model.train(training)
            with torch.no_grad():
                baseline = model(x, compute_joint=False)["branches"]["evolution"]["x_hat"][:, target_time, :]
                target_prediction = model(changed_target, compute_joint=False)["branches"]["evolution"]["x_hat"][:, target_time, :]
                future_prediction = model(changed_future, compute_joint=False)["branches"]["evolution"]["x_hat"][:, target_time, :]
            torch.testing.assert_close(baseline, target_prediction, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(baseline, future_prediction, rtol=1e-6, atol=1e-6)

        model.train()
        model.zero_grad(set_to_none=True)
        output = model(x, compute_joint=False)
        output["losses"]["evolution"].backward()
        parameters = [
            model.evolution_branch.blocks[0].norm.weight,
            model.evolution_branch.blocks[0].depthwise.weight,
            model.evolution_branch.prediction_head.weight,
        ]
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
