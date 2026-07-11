import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.strategy.anomaly_detect import (
    AnomalyDetect,
    AllDetectLabel,
    LEGACY_TEST_CONTAMINATED_PROTOCOL,
    UnFixedDetectLabel,
)
from ts_benchmark.report.utils.leaderboard import get_leaderboard


class RecordingMultiModel:
    def __init__(self):
        self.config = SimpleNamespace(seq_len=3, anomaly_ratio=[25])
        self.fit_data = None
        self.fit_text = None
        self.fit_label = None
        self.score_inputs = []

    def detect_multi_fit(self, data, text, label):
        self.fit_data = data.copy()
        self.fit_text = text.copy()
        self.fit_label = label.copy()

    def detect_multi_score(self, data, text):
        self.score_inputs.append(data.copy())
        score = data["value"].to_numpy(dtype=float)
        return score, score

    def detect_multi_label(self, data, text):
        raise AssertionError("Strict protocol must not call detect_multi_label")


class AnomalyProtocolTest(unittest.TestCase):
    def _strict_strategy(self):
        return UnFixedDetectLabel(
            {
                "strategy_name": "unfixed_detect_label",
                "evaluation_protocol": "train_calibration",
                "anomaly_ratios": [25],
                "calibration_fraction": 0.2,
                "seed": {"__default__": 7, "series": 13},
            },
            Evaluator([]),
        )

    def test_temporal_calibration_split_and_series_seed(self):
        strategy = self._strict_strategy()
        train_data = pd.DataFrame({"value": np.arange(15, dtype=float)})
        train_text = pd.DataFrame({"text": np.arange(15)})
        train_label = pd.DataFrame({"label": np.zeros(15, dtype=int)})
        test_data = pd.DataFrame({"value": [20.0, 21.0]})
        test_text = pd.DataFrame({"text": [20, 21]})
        test_label = pd.DataFrame({"label": [0, 1]})
        strategy.split_multi_data = lambda series, text: (
            train_data,
            train_text,
            train_label,
            test_data,
            test_text,
            test_label,
        )

        model = RecordingMultiModel()
        with patch(
            "ts_benchmark.evaluation.strategy.anomaly_detect.fix_random_seed"
        ) as seed_mock:
            strategy.multi_execute("series", "text", lambda: model)

        seed_mock.assert_called_once_with(13)
        self.assertEqual(model.fit_data["value"].tolist(), list(range(10)))
        self.assertEqual(model.fit_text["text"].tolist(), list(range(10)))
        self.assertEqual(len(model.fit_label), 10)
        self.assertEqual(
            model.score_inputs[0]["value"].tolist(), [12.0, 13.0, 14.0]
        )
        self.assertEqual(model.score_inputs[1]["value"].tolist(), [20.0, 21.0])

    def test_static_text_is_reused_without_materializing_time_length_rows(self):
        static_text = pd.DataFrame({"text": ["global system description"]})
        train_text, test_text = AnomalyDetect._split_multi_text(
            static_text, total_length=1000, train_length=800
        )
        self.assertEqual(train_text.to_dict("records"), static_text.to_dict("records"))
        self.assertEqual(test_text.to_dict("records"), static_text.to_dict("records"))

        strategy = self._strict_strategy()
        train_data = pd.DataFrame({"value": np.arange(15, dtype=float)})
        train_label = pd.DataFrame({"label": np.zeros(15, dtype=int)})
        _, _, fit_text, _, calibration_text = strategy._split_fit_and_calibration(
            train_data, train_label, static_text
        )
        self.assertEqual(len(fit_text), 1)
        self.assertEqual(len(calibration_text), 1)

    def test_nonstatic_short_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one static row"):
            AnomalyDetect._split_multi_text(
                pd.DataFrame({"text": ["a", "b"]}),
                total_length=10,
                train_length=6,
            )

    def test_strict_protocol_rejects_a_contaminated_official_train_split(self):
        strategy = self._strict_strategy()
        train_data = pd.DataFrame({"value": np.arange(15, dtype=float)})
        train_label = pd.DataFrame(
            {"label": [0] * 14 + [1]}
        )

        with self.assertRaisesRegex(ValueError, "anomaly-free official train"):
            strategy._split_fit_and_calibration(train_data, train_label)

    def test_missing_protocol_keeps_legacy_compatibility(self):
        strategy = UnFixedDetectLabel(
            {"strategy_name": "unfixed_detect_label", "seed": 2021},
            Evaluator([]),
        )
        self.assertEqual(
            strategy._evaluation_protocol(), LEGACY_TEST_CONTAMINATED_PROTOCOL
        )

    def test_strict_protocol_requires_score_api(self):
        strategy = self._strict_strategy()
        strategy.model = object()
        strategy.calibration_data = pd.DataFrame({"value": [0.0]})
        with self.assertRaisesRegex(TypeError, "detect_score"):
            strategy._detect_train_calibrated_label(
                pd.DataFrame({"value": [1.0]})
            )

    def test_all_detect_label_rejects_strict_protocol(self):
        with self.assertRaisesRegex(ValueError, "no disjoint training/calibration"):
            AllDetectLabel(
                {
                    "strategy_name": "all_detect_label",
                    "evaluation_protocol": "train_calibration",
                    "seed": 2021,
                },
                Evaluator([]),
            )

    def test_threshold_is_independent_of_test_tail(self):
        strategy = self._strict_strategy()
        calibration = np.array([0.0, 1.0, 2.0, 3.0])
        short_test = np.array([2.2, 2.3])
        extended_test = np.array([2.2, 2.3, 100.0, 1000.0])

        short_predictions, _ = strategy._threshold_from_calibration_scores(
            calibration, short_test
        )
        extended_predictions, _ = strategy._threshold_from_calibration_scores(
            calibration, extended_test
        )

        np.testing.assert_array_equal(
            short_predictions[25.0], extended_predictions[25.0][:2]
        )

    def test_conformal_threshold_is_conservative_when_calibration_is_too_small(self):
        strategy = UnFixedDetectLabel(
            {
                "strategy_name": "unfixed_detect_label",
                "evaluation_protocol": "train_calibration",
                "anomaly_ratios": [1],
                "seed": 2021,
            },
            Evaluator([]),
        )
        predictions, _ = strategy._threshold_from_calibration_scores(
            np.arange(20, dtype=float), np.array([1e6])
        )
        np.testing.assert_array_equal(predictions[1.0], np.array([0]))

    def test_leaderboard_keeps_anomaly_ratios_separate(self):
        strategy_args = json.dumps(
            {
                "strategy_name": "unfixed_detect_label",
                "evaluation_protocol": "train_calibration",
            }
        )
        log_data = pd.DataFrame(
            {
                "model_name": ["M"] * 4,
                "model_params": ["{}"] * 4,
                "strategy_args": [strategy_args] * 4,
                "file_name": ["a", "a", "b", "b"],
                "typical_anomaly_ratio": [1.0, 5.0, 1.0, 5.0],
                "metric": [0.2, 0.9, 0.4, 0.8],
            }
        )

        result = get_leaderboard(
            log_data,
            report_metrics="metric",
            aggregate_type="mean",
            fill_type="mean_value",
            nan_threshold=0.3,
        )

        self.assertAlmostEqual(result["M;{};anomaly_ratio=1"].iloc[0], 0.3)
        self.assertAlmostEqual(result["M;{};anomaly_ratio=5"].iloc[0], 0.85)


if __name__ == "__main__":
    unittest.main()
