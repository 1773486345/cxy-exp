"""Validation must use deterministic LOO branch mask supervision."""

import unittest

import torch

from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel
from ts_benchmark.baselines.typefusion_catch.tests.common import tiny_config


class FusionValidationMaskLossTests(unittest.TestCase):
    def test_eval_loo_mask_loss_is_finite_and_nonzero(self) -> None:
        torch.manual_seed(41)
        config = tiny_config("fusion_train")
        model = TypeFusionCATCHModel(config).eval()
        output = model(torch.randn(2, config.seq_len, config.c_in))
        mask_loss = output["branch_mask_loss"]
        self.assertTrue(torch.isfinite(mask_loss))
        self.assertGreater(float(mask_loss), 0.0)
        self.assertIsNone(output["branch_mask_prediction"])
        self.assertEqual(int(output["leave_one_out"]["expanded_batch"]), 8)
        expected = output["losses"]["joint"] + config.lambda_mask * mask_loss
        torch.testing.assert_close(output["losses"]["total"], expected)


if __name__ == "__main__":
    unittest.main()
