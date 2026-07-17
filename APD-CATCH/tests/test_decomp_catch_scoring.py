from __future__ import annotations

import copy
import io
import inspect
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from ts_benchmark.baselines.catch.CATCH import CATCH
from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel
from ts_benchmark.baselines.decomp_catch.decomposition import (
    decompose_slow_fast,
    moving_average,
    resolve_moving_average_window,
)
from ts_benchmark.baselines.decomp_catch.scoring import CATCHDecompositionScorer


SEQ_LEN = 8
N_CHANNELS = 2
SCORING_SEED = 20260717


def _frame(length: int, offset: float = 0.0) -> pd.DataFrame:
    values = np.arange(length * N_CHANNELS, dtype=np.float32).reshape(length, N_CHANNELS)
    values = values / 10.0 + offset
    return pd.DataFrame(values, columns=["channel_0", "channel_1"])


def _detector() -> CATCH:
    detector = CATCH(
        seq_len=SEQ_LEN,
        patch_size=4,
        patch_stride=4,
        inference_patch_size=4,
        inference_patch_stride=4,
        cf_dim=4,
        d_model=4,
        d_ff=8,
        e_layers=1,
        n_heads=1,
        head_dim=4,
        dropout=0.0,
        head_dropout=0.0,
        batch_size=1,
        affine=False,
    )
    detector.device = torch.device("cpu")
    train = _frame(24, offset=-1.0)
    detector.detect_hyper_param_tune(train)
    detector.config.task_name = "anomaly_detection"
    detector.config.c_in = N_CHANNELS
    detector.model = CATCHModel(detector.config).to(detector.device).eval()
    detector.scaler.fit(train.values)
    detector.early_stopping = SimpleNamespace(
        check_point=copy.deepcopy(detector.model.state_dict())
    )
    return detector


def _stats(scorer: CATCHDecompositionScorer) -> dict:
    return scorer.fit_normalization_stats(_frame(24, offset=-0.5), "validation", SCORING_SEED)


class DecompositionPrimitiveTest(unittest.TestCase):
    def test_moving_average_shape_dtype_and_time_only_behavior(self):
        x = torch.tensor(
            [[[0.0, 0.0], [0.0, 9.0], [0.0, 0.0]]], dtype=torch.float64
        )
        averaged = moving_average(x, 3)
        self.assertEqual(averaged.shape, x.shape)
        self.assertEqual(averaged.dtype, x.dtype)
        torch.testing.assert_close(averaged[:, :, 0], torch.zeros((1, 3), dtype=torch.float64))
        torch.testing.assert_close(
            averaged[:, :, 1], torch.full((1, 3), 3.0, dtype=torch.float64)
        )

    def test_replicate_padding_and_fast_reconstruction(self):
        x = torch.tensor([[[1.0], [2.0], [3.0]]])
        slow, fast = decompose_slow_fast(x, 3)
        expected = torch.tensor([[[4.0 / 3.0], [2.0], [8.0 / 3.0]]])
        torch.testing.assert_close(slow, expected)
        torch.testing.assert_close(slow + fast, x)
        self.assertIs(moving_average(x, 1), x)

    def test_window_rule_is_deterministic_and_nearest_legal_odd(self):
        self.assertEqual(resolve_moving_average_window(16, 192), 15)
        self.assertEqual(resolve_moving_average_window(16, 16), 15)
        self.assertEqual(resolve_moving_average_window(4, 5), 3)
        self.assertEqual(resolve_moving_average_window(2, 3), 1)

    def test_invalid_windows_raise_explicit_errors(self):
        x = torch.zeros(1, 3, 1)
        for invalid in (0, 2, 5):
            with self.assertRaises(ValueError):
                moving_average(x, invalid)
        with self.assertRaises(ValueError):
            moving_average(torch.zeros(1, 3), 1)
        with self.assertRaises(ValueError):
            resolve_moving_average_window(0, 3)


class CATCHDecompositionScorerTest(unittest.TestCase):
    def test_scorer_has_no_trainable_parameters_and_requires_reference_stats(self):
        scorer = CATCHDecompositionScorer(_detector())
        self.assertEqual(list(scorer.parameters()), [])
        with self.assertRaises(ValueError):
            scorer.score_dataframe(_frame(SEQ_LEN), scoring_seed=SCORING_SEED)

    def test_one_forward_per_batch_produces_all_scores_from_that_forward(self):
        detector = _detector()
        scorer = CATCHDecompositionScorer(detector)
        stats = _stats(scorer)
        captured = []
        original_forward = detector.model.forward

        def capture(batch_x):
            output = original_forward(batch_x)
            captured.append((batch_x.detach().clone(), output[0].detach().clone()))
            return output

        values = _frame(SEQ_LEN * 2)
        with patch.object(detector.model, "forward", side_effect=capture):
            result = scorer.score_dataframe(values, stats, SCORING_SEED)

        self.assertEqual(result["forward_calls"], 2)
        self.assertEqual(len(captured), 2)
        expected_time = torch.cat(
            [torch.mean((x - x_hat) ** 2, dim=-1) for x, x_hat in captured], dim=0
        ).reshape(-1).numpy()
        np.testing.assert_allclose(result["time_score"], expected_time, rtol=0, atol=0)
        for x, x_hat in captured:
            slow_x, fast_x = decompose_slow_fast(x, result["window_size"])
            slow_hat, fast_hat = decompose_slow_fast(x_hat, result["window_size"])
            torch.testing.assert_close(slow_x + fast_x, x)
            torch.testing.assert_close(slow_hat + fast_hat, x_hat)
        self.assertEqual(result["decomposition_reconstruction_max_error"], 0.0)

    def test_original_score_matches_original_formula_and_detect_score(self):
        detector = _detector()
        scorer = CATCHDecompositionScorer(detector)
        stats = _stats(scorer)
        values = _frame(SEQ_LEN * 2, offset=0.25)

        torch.manual_seed(SCORING_SEED)
        with redirect_stdout(io.StringIO()):
            baseline_score, _ = detector.detect_score(values)
        result = scorer.score_dataframe(values, stats, SCORING_SEED)

        expected = result["time_score"] + detector.config.score_lambda * result["frequency_score"]
        np.testing.assert_allclose(result["original_score"], expected, rtol=0, atol=0)
        np.testing.assert_allclose(result["original_score"], baseline_score, rtol=1e-6, atol=1e-6)

    def test_model_state_labels_and_score_alignment_are_preserved(self):
        detector = _detector()
        scorer = CATCHDecompositionScorer(detector)
        stats = _stats(scorer)
        original_state = copy.deepcopy(detector.model.state_dict())
        values = _frame(SEQ_LEN * 2 + 2)
        normal_labels = np.zeros(len(values), dtype=np.int64)
        changed_labels = np.ones(len(values), dtype=np.int64)

        def score_for_external_labels(labels: np.ndarray) -> dict:
            self.assertEqual(len(labels), len(values))
            return scorer.score_dataframe(values, stats, SCORING_SEED)

        self.assertNotIn("labels", inspect.signature(scorer.score_dataframe).parameters)
        first = score_for_external_labels(normal_labels)
        second = score_for_external_labels(changed_labels)

        for name, parameter in detector.model.state_dict().items():
            torch.testing.assert_close(parameter, original_state[name], rtol=0, atol=0)
        for name in (
            "original_score",
            "time_score",
            "frequency_score",
            "slow_score",
            "fast_score",
            "fusion_score",
        ):
            self.assertEqual(len(first[name]), len(first["time_index"]))
            np.testing.assert_array_equal(first[name], second[name])
            self.assertEqual(len(first[name]), len(normal_labels) - 2)
        self.assertEqual(first["dropped_tail_length"], 2)
        self.assertEqual(first["scored_length"], len(normal_labels) - 2)
        np.testing.assert_array_equal(first["time_index"], values.index[: len(normal_labels) - 2])

    def test_normalization_comes_only_from_reference_and_fusion_is_fixed(self):
        scorer = CATCHDecompositionScorer(_detector())
        stats = _stats(scorer)
        frozen_stats = copy.deepcopy(stats)
        values = _frame(SEQ_LEN * 2, offset=9.0)
        result = scorer.score_dataframe(values, stats, SCORING_SEED)

        self.assertEqual(stats, frozen_stats)
        self.assertEqual(stats["source"], "validation")
        self.assertEqual(stats["source_length"], 24)
        self.assertEqual(stats["scored_length"], 24)
        self.assertEqual(stats["dropped_tail_length"], 0)
        self.assertIn("zero_scale_diagnostics", stats)
        expected_fusion = 0.5 * (
            (result["slow_score"] - stats["slow_location"])
            / (stats["slow_scale"] + 1e-8)
        ) + 0.5 * (
            (result["fast_score"] - stats["fast_location"])
            / (stats["fast_scale"] + 1e-8)
        )
        np.testing.assert_allclose(result["fusion_score"], expected_fusion, rtol=0, atol=0)
        for name in (
            "original_score",
            "time_score",
            "frequency_score",
            "slow_score",
            "fast_score",
            "fusion_score",
        ):
            self.assertTrue(np.isfinite(result[name]).all())
        self.assertFalse(result["uses_cuda"])


if __name__ == "__main__":
    unittest.main()
