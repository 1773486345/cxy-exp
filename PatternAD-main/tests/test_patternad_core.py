import unittest

import numpy as np
import pandas as pd
import torch

from ts_benchmark.baselines.PatternAD.PatternAD import (
    PatternAD,
    PatternADConfig,
    PatternADNet,
)


class PatternADNetTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.config = PatternADConfig.from_kwargs(
            {
                "seq_len": 12,
                "d_model": 16,
                "graph_dim": 8,
                "d_ff": 32,
                "n_heads": 4,
                "e_layers": 1,
                "temporal_kernels": (1, 3, 5),
                "device": "cpu",
            }
        )

    def test_masked_terminal_is_target_blind(self):
        model = PatternADNet(4, self.config).eval()
        value = torch.randn(3, 12, 4)
        mask = torch.zeros_like(value, dtype=torch.bool)
        mask[:, -1, 1] = True
        changed = value.clone()
        changed[:, -1, 1] += 100.0
        with torch.no_grad():
            original = model(value, mask)
            counterfactual = model(changed, mask)
        self.assertTrue(
            torch.allclose(
                original["mean"][:, -1, 1],
                counterfactual["mean"][:, -1, 1],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_multiscale_conditional_outputs_are_finite(self):
        model = PatternADNet(4, self.config).eval()
        value = torch.randn(2, 12, 4)
        mask = torch.zeros_like(value, dtype=torch.bool)
        mask[:, -1, 0] = True
        with torch.no_grad():
            outputs = model(value, mask)
        for name in ("mean", "scale", "temporal_mean", "graph_mean", "temporal_reliability"):
            self.assertEqual(outputs[name].shape, value.shape)
            self.assertTrue(torch.isfinite(outputs[name]).all())
        self.assertTrue(torch.all(outputs["scale"] > 0.0))
        self.assertGreaterEqual(float(outputs["temporal_reliability"].min()), 0.0)
        self.assertLessEqual(float(outputs["temporal_reliability"].max()), 1.0)

    def test_no_graph_ablation_has_no_relation_messages(self):
        config = PatternADConfig.from_kwargs(
            {
                "seq_len": 12,
                "d_model": 16,
                "graph_dim": 8,
                "d_ff": 32,
                "n_heads": 4,
                "e_layers": 1,
                "temporal_kernels": (1, 3, 5),
                "relation_mode": "no_graph",
                "device": "cpu",
            }
        )
        model = PatternADNet(4, config).eval()
        value = torch.randn(2, 12, 4)
        mask = torch.zeros_like(value, dtype=torch.bool)
        mask[:, -1, 2] = True
        with torch.no_grad():
            outputs = model(value, mask)
        self.assertEqual(len(model.relation_layers), 0)
        self.assertEqual(float(outputs["graph_entropy"]), 0.0)
        self.assertEqual(float(outputs["relation_consistency"]), 0.0)

    def test_pattern_context_ablation_removes_context_projection(self):
        config = PatternADConfig.from_kwargs(
            {
                "seq_len": 12,
                "d_model": 16,
                "graph_dim": 8,
                "d_ff": 32,
                "n_heads": 4,
                "e_layers": 1,
                "temporal_kernels": (1, 3, 5),
                "use_pattern_context": False,
                "device": "cpu",
            }
        )
        model = PatternADNet(4, config).eval()
        value = torch.randn(2, 12, 4)
        mask = torch.zeros_like(value, dtype=torch.bool)
        context = model.encoder._visible_pattern_context(value, mask)
        self.assertEqual(float(context.abs().max()), 0.0)


class PatternADInterfaceTest(unittest.TestCase):
    def test_cpu_fit_and_score_return_one_score_per_timestamp(self):
        torch.set_num_threads(1)
        generator = np.random.default_rng(9)
        time_index = np.arange(42, dtype=np.float64)
        data = pd.DataFrame(
            {
                "x0": np.sin(time_index / 3.0),
                "x1": np.sin(time_index / 3.0 + 0.2) + 0.03 * generator.standard_normal(len(time_index)),
                "x2": 0.7 * np.sin(time_index / 6.0) + 0.1 * generator.standard_normal(len(time_index)),
            }
        )
        detector = PatternAD(
            device="cpu",
            seq_len=8,
            d_model=8,
            graph_dim=4,
            d_ff=16,
            n_heads=2,
            e_layers=1,
            temporal_kernels=(1, 3),
            batch_size=16,
            score_conditioning_batch_size=24,
            num_epochs=2,
            patience=2,
            validation_fraction=0.2,
        )
        detector.detect_fit(data)
        scores = detector.detect_score(data)
        self.assertEqual(scores.shape, (len(data),))
        self.assertTrue(np.isfinite(scores).all())
        components = detector.get_last_score_components()
        self.assertEqual(components["variable_nll"].shape, (len(data), data.shape[1]))
        self.assertGreater(detector.get_diagnostics()["parameter_count"], 0)


if __name__ == "__main__":
    unittest.main()
