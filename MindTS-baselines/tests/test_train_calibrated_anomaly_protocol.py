import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.strategy.train_calibrated_anomaly_detect import (
    TrainCalibratedUnFixedDetectLabel,
)
from ts_benchmark.baselines.self_impl.Anomaly_trans.AnomalyTransformer import (
    AnomalyTransformer,
)


class RecordingModel:
    def __init__(self):
        self.config = SimpleNamespace(seq_len=3)
        self.fit_data = None
        self.fit_label = None
        self.score_inputs = []

    def detect_fit(self, data, label):
        self.fit_data = data.copy()
        self.fit_label = label.copy()

    def detect_score(self, data):
        self.score_inputs.append(data.copy())
        score = data["value"].to_numpy(dtype=float)
        return score, score


class TrainCalibratedProtocolTest(unittest.TestCase):
    def strategy(self):
        return TrainCalibratedUnFixedDetectLabel(
            {
                "strategy_name": "train_calibrated_unfixed_detect_label",
                "evaluation_protocol": "train_calibration",
                "anomaly_ratios": [25],
                "calibration_fraction": 0.2,
                "seed": 7,
            },
            Evaluator([]),
        )

    def test_temporal_holdout_is_the_only_calibration_source(self):
        strategy = self.strategy()
        train_data = pd.DataFrame({"value": np.arange(15, dtype=float)})
        train_label = pd.DataFrame({"label": np.zeros(15, dtype=int)})
        test_data = pd.DataFrame({"value": [20.0, 21.0]})
        test_label = pd.DataFrame({"label": [0, 1]})
        strategy.split_data = lambda _: (
            train_data,
            train_label,
            test_data,
            test_label,
        )

        model = RecordingModel()
        strategy.execute("series", lambda: model)

        self.assertEqual(model.fit_data["value"].tolist(), list(range(10)))
        self.assertEqual(model.score_inputs[0]["value"].tolist(), [12.0, 13.0, 14.0])
        self.assertEqual(model.score_inputs[1]["value"].tolist(), [20.0, 21.0])

    def test_threshold_does_not_change_when_test_tail_changes(self):
        strategy = self.strategy()
        calibration = np.array([0.0, 1.0, 2.0, 3.0])
        short_test = np.array([2.2, 2.3])
        extended_test = np.array([2.2, 2.3, 100.0, 1000.0])

        short_prediction, _ = strategy._threshold_from_calibration_scores(
            calibration, short_test
        )
        extended_prediction, _ = strategy._threshold_from_calibration_scores(
            calibration, extended_test
        )

        np.testing.assert_array_equal(
            short_prediction[25.0], extended_prediction[25.0][:2]
        )

    def test_contaminated_official_train_data_is_rejected(self):
        strategy = self.strategy()
        train_data = pd.DataFrame({"value": np.arange(15, dtype=float)})
        train_label = pd.DataFrame({"label": [0] * 14 + [1]})

        with self.assertRaisesRegex(ValueError, "anomaly-free official train"):
            strategy._split_fit_and_calibration(RecordingModel(), train_data, train_label)


class AnomalyTransformerScoreAlignmentTest(unittest.TestCase):
    def test_trailing_overlap_restores_the_original_input_length(self):
        complete_scores = np.arange(200, dtype=float)
        final_window_scores = np.arange(198, 298, dtype=float)

        aligned = AnomalyTransformer._append_trailing_scores(
            complete_scores, final_window_scores, input_length=298
        )

        self.assertEqual(len(aligned), 298)
        np.testing.assert_array_equal(aligned[:200], complete_scores)
        np.testing.assert_array_equal(aligned[200:], np.arange(200, 298, dtype=float))


if __name__ == "__main__":
    unittest.main()
