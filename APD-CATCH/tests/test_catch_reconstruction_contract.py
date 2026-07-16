"""Lightweight contract checks for the unmodified original CATCH baseline."""

from __future__ import annotations

import copy
import io
import inspect
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from ts_benchmark.baselines.catch.CATCH import CATCH
from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel


SEQ_LEN = 8
N_CHANNELS = 2


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
        batch_size=2,
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


def _continuous_scores(
    detector: CATCH, values: pd.DataFrame, labels: np.ndarray
) -> np.ndarray:
    if len(values) != len(labels):
        raise AssertionError("fixture labels must align with the input time axis")
    with redirect_stdout(io.StringIO()):
        scores, duplicate_scores = detector.detect_score(values)
    np.testing.assert_array_equal(scores, duplicate_scores)
    return scores


class CATCHReconstructionContractTest(unittest.TestCase):
    def test_existing_catch_configuration_initializes_a_model(self):
        detector = _detector()
        self.assertIsInstance(detector, CATCH)
        self.assertIsInstance(detector.model, CATCHModel)
        self.assertEqual(detector.config.seq_len, SEQ_LEN)
        self.assertEqual(detector.config.c_in, N_CHANNELS)

    def test_forward_reconstructs_a_full_window_with_input_shape(self):
        detector = _detector()
        window = torch.tensor(_frame(SEQ_LEN).values).unsqueeze(0)
        torch.manual_seed(11)
        with torch.no_grad():
            reconstruction, frequency_output, channel_loss = detector.model(window)
        self.assertEqual(reconstruction.shape, window.shape)
        self.assertEqual(frequency_output.shape, window.shape)
        self.assertEqual(channel_loss.ndim, 0)

    def test_evaluation_forward_is_reproducible_with_restored_rng_state(self):
        detector = _detector()
        window = torch.tensor(_frame(SEQ_LEN).values).unsqueeze(0)
        with torch.no_grad():
            torch.manual_seed(23)
            first, _, _ = detector.model(window)
            torch.manual_seed(23)
            second, _, _ = detector.model(window)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_continuous_scores_align_and_ignore_external_labels(self):
        detector = _detector()
        values = _frame(32)
        original_parameters = copy.deepcopy(detector.model.state_dict())
        normal_labels = np.zeros(len(values), dtype=np.int64)
        changed_labels = np.ones(len(values), dtype=np.int64)

        self.assertNotIn("labels", inspect.signature(detector.detect_score).parameters)
        torch.manual_seed(37)
        first = _continuous_scores(detector, values, normal_labels)
        torch.manual_seed(37)
        second = _continuous_scores(detector, values, changed_labels)

        self.assertEqual(first.shape, (len(normal_labels),))
        np.testing.assert_array_equal(first, second)
        for name, parameter in detector.model.state_dict().items():
            torch.testing.assert_close(parameter, original_parameters[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
