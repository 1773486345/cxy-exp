"""Each trainable parameter belongs to the selected stage objective."""

import unittest

import torch

from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel
from ts_benchmark.baselines.typefusion_catch.tests.common import tiny_config


class NoUnusedTrainableParametersTests(unittest.TestCase):
    def test_stage_trainable_parameters_receive_finite_gradients(self) -> None:
        torch.manual_seed(43)
        for stage in ("branch_pretrain", "fusion_train", "joint_finetune"):
            model = TypeFusionCATCHModel(tiny_config(stage)).train()
            output = model(torch.randn(2, model.config.seq_len, model.config.c_in))
            output["losses"]["total"].backward()
            trainable = {
                name: parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertFalse(any("frequency_projection" in name or "shared_latent" in name for name in trainable))
            missing_gradients = [name for name, parameter in trainable.items() if parameter.grad is None]
            self.assertEqual(missing_gradients, [], msg=f"{stage} unused trainable parameters: {missing_gradients}")
            self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in trainable.values()))


if __name__ == "__main__":
    unittest.main()
