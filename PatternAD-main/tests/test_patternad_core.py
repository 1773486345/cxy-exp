import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import torch

from ts_benchmark.baselines.PatternAD.PatternAD import (
    PatternAD,
    PatternADConfig,
    PatternADNet,
)
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.strategy.anomaly_detect import UnFixedDetectLabel
from ts_benchmark.evaluation.strategy.constants import FieldNames


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

    def test_future_values_do_not_change_past_states(self):
        model = PatternADNet(4, self.config).eval()
        value = torch.randn(2, 12, 4)
        mask = torch.zeros_like(value, dtype=torch.bool)
        changed = value.clone()
        changed[:, 8:, :] += 100.0
        with torch.no_grad():
            original = model(value, mask)
            counterfactual = model(changed, mask)
        self.assertTrue(
            torch.allclose(
                original["mean"][:, :8, :],
                counterfactual["mean"][:, :8, :],
                atol=1e-6,
                rtol=1e-6,
            )
        )

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

    def test_removed_handcrafted_context_controls_are_rejected(self):
        with self.assertRaises(TypeError):
            PatternADConfig.from_kwargs({"use_pattern_context": False})


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
        terminal_output = io.StringIO()
        with redirect_stdout(terminal_output):
            with self.assertLogs(
                "ts_benchmark.baselines.PatternAD.PatternAD", level="INFO"
            ) as captured_logs:
                detector.detect_fit(data)
                scores = detector.detect_score(data)
        log_output = "\n".join(captured_logs.output)
        self.assertIn("PatternAD fitting started", log_output)
        self.assertIn("PatternAD epoch 1/2", log_output)
        self.assertIn("delta=", log_output)
        self.assertIn("PatternAD scoring progress", log_output)
        self.assertIn("PatternAD scoring complete", log_output)
        self.assertIn("iters:", terminal_output.getvalue())
        self.assertIn("speed:", terminal_output.getvalue())
        self.assertIn("left time:", terminal_output.getvalue())
        self.assertEqual(scores.shape, (len(data),))
        self.assertTrue(np.isfinite(scores).all())
        components = detector.get_last_score_components()
        self.assertEqual(components["variable_nll"].shape, (len(data), data.shape[1]))
        self.assertGreater(detector.get_diagnostics()["parameter_count"], 0)

    def test_multi_interface_validates_and_records_auxiliary_input(self):
        torch.set_num_threads(1)
        generator = np.random.default_rng(21)
        data = pd.DataFrame(generator.normal(size=(36, 3)), columns=["x0", "x1", "x2"])
        static_text = pd.DataFrame({"text": ["audited static system description"]})
        aligned_text = pd.DataFrame({"text": [f"state-{index % 3}" for index in range(len(data))]})
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
            num_epochs=1,
            patience=1,
            validation_fraction=0.2,
        )
        detector.detect_multi_fit(data, static_text)
        scores = detector.detect_multi_score(data, aligned_text)
        self.assertEqual(scores.shape, (len(data),))
        self.assertTrue(np.isfinite(scores).all())
        diagnostics = detector.get_diagnostics()
        self.assertFalse(diagnostics["auxiliary_input"]["used_for_scoring"])
        self.assertFalse(diagnostics["score_calls"][-1]["auxiliary_input"]["used_for_scoring"])

    def test_strict_multi_strategy_runs_the_patternad_interface(self):
        torch.set_num_threads(1)
        generator = np.random.default_rng(31)
        train_data = pd.DataFrame(generator.normal(size=(72, 3)), columns=["x0", "x1", "x2"])
        train_label = pd.DataFrame({"label": np.zeros(len(train_data), dtype=int)})
        test_data = pd.DataFrame(generator.normal(size=(14, 3)), columns=train_data.columns)
        test_label = pd.DataFrame({"label": [0] * 7 + [1] * 7})
        static_text = pd.DataFrame({"text": ["static system description"]})
        strategy = UnFixedDetectLabel(
            {
                "strategy_name": "unfixed_detect_label",
                "evaluation_protocol": "train_calibration",
                "anomaly_ratios": [20],
                "calibration_fraction": 0.2,
                "seed": 7,
            },
            Evaluator([]),
        )
        strategy.split_multi_data = lambda series, text: (
            train_data,
            static_text,
            train_label,
            test_data,
            static_text,
            test_label,
        )
        result = strategy.multi_execute(
            "synthetic.csv",
            "synthetic_text.csv",
            lambda: PatternAD(
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
                num_epochs=1,
                patience=1,
                validation_fraction=0.2,
            ),
        )
        log_index = strategy.field_names.index(FieldNames.LOG_INFO)
        self.assertEqual(result[0][log_index], "")
        diagnostics_index = strategy.field_names.index(FieldNames.MODEL_DIAGNOSTICS)
        self.assertIn('"used_for_scoring":false', result[0][diagnostics_index])


if __name__ == "__main__":
    unittest.main()
