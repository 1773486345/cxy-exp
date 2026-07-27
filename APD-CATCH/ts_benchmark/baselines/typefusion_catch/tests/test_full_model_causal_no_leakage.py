"""Causality regression through the complete TypeFusion model path."""

import unittest

import torch

from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel
from ts_benchmark.baselines.typefusion_catch.tests.common import tiny_config


class FullModelCausalNoLeakageTests(unittest.TestCase):
    def test_revin_path_cannot_change_evolution_prediction_at_target(self) -> None:
        torch.manual_seed(37)
        config = tiny_config("joint_finetune")
        model = TypeFusionCATCHModel(config).eval()
        x = torch.randn(2, config.seq_len, config.c_in)
        target_time = 17
        changed_target = x.clone()
        changed_target[:, target_time, :] += 100.0
        changed_future = x.clone()
        changed_future[:, target_time + 1 :, :] -= 100.0
        with torch.no_grad():
            baseline = model(x)["branches"]["evolution"]["x_hat"][:, target_time, :]
            target_prediction = model(changed_target)["branches"]["evolution"]["x_hat"][:, target_time, :]
            future_prediction = model(changed_future)["branches"]["evolution"]["x_hat"][:, target_time, :]
        torch.testing.assert_close(baseline, target_prediction, rtol=0.0, atol=0.0)
        torch.testing.assert_close(baseline, future_prediction, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
