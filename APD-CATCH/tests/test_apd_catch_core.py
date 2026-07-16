import unittest

import numpy as np
import torch

from ts_benchmark.baselines.apd_catch.APDCATCH import (
    APDCATCHConfig,
    NextPointDataset,
)
from ts_benchmark.baselines.apd_catch.models.APDCATCH_model import (
    APDCATCHModel,
    gaussian_nll,
)


def _model(variant: str) -> APDCATCHModel:
    config = APDCATCHConfig(
        variant=variant,
        seq_len=32,
        patch_size=8,
        patch_stride=4,
        cf_dim=8,
        d_model=8,
        d_ff=16,
        e_layers=1,
        n_heads=2,
        head_dim=4,
        dropout=0.0,
        head_dropout=0.0,
    )
    config.c_in = 3
    return APDCATCHModel(config)


class TestAPDCATCHCore(unittest.TestCase):
    def test_next_point_dataset_keeps_target_out_of_history(self):
        values = np.arange(60, dtype=np.float32).reshape(20, 3)
        dataset = NextPointDataset(values, history_length=8)
        history, target = dataset[4]
        np.testing.assert_array_equal(history.numpy(), values[4:12])
        np.testing.assert_array_equal(target.numpy(), values[12])

    def test_variants_share_parameter_budget_and_valid_distribution(self):
        torch.manual_seed(7)
        history = torch.randn(2, 32, 3)
        target = torch.randn(2, 3)
        parameter_counts = []
        for variant in ("causal_catch", "state", "state_scale"):
            model = _model(variant)
            parameter_counts.append(sum(p.numel() for p in model.parameters()))
            output = model(history)
            self.assertEqual(output["mean"].shape, target.shape)
            self.assertEqual(output["scale"].shape, target.shape)
            self.assertTrue(torch.all(output["scale"] > 0))
            self.assertTrue(
                torch.isfinite(
                    gaussian_nll(target, output["mean"], output["scale"])
                ).all()
            )
        self.assertEqual(len(set(parameter_counts)), 1)

    def test_state_scale_eval_is_deterministic(self):
        torch.manual_seed(11)
        model = _model("state_scale").eval()
        history = torch.randn(2, 32, 3)
        first = model(history)
        second = model(history.clone())
        for name in ("mean", "scale", "state_mean", "innovation_scale"):
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_reference_normalization_keeps_constant_history_scale_positive(self):
        model = _model("state_scale").eval()
        location = torch.tensor([10.0, 20.0, 30.0])
        scale = torch.tensor([1.0, 2.0, 4.0])
        model.set_reference_normalization(location, scale)
        output = model(torch.zeros(2, 32, 3))
        self.assertTrue(torch.all(output["scale"] > 0))
        self.assertEqual(model.state_span, 5)
        torch.testing.assert_close(model.reference_location, location, rtol=0, atol=0)
        torch.testing.assert_close(model.reference_scale, scale, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
