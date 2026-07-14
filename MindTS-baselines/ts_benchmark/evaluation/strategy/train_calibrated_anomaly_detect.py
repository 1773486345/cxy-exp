# -*- coding: utf-8 -*-
"""Strict, train-calibrated anomaly-label evaluation for baseline runs."""

import logging
import time
import traceback
from typing import Any, List

import numpy as np
import pandas as pd

from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.evaluation.metrics import classification_metrics_label
from ts_benchmark.evaluation.strategy.anomaly_detect import AnomalyDetect
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.utils.data_processing import split_before
from ts_benchmark.utils.random_utils import fix_random_seed


TRAIN_CALIBRATION_PROTOCOL = "train_calibration"
_OPTIONAL_CONFIGS = {"calibration_gap"}


class TrainCalibratedUnFixedDetectLabel(AnomalyDetect):
    """Evaluate labels using a threshold calibrated only on normal train data."""

    REQUIRED_CONFIGS = [
        "evaluation_protocol",
        "anomaly_ratios",
        "calibration_fraction",
        "seed",
    ]

    def _check_config(self):
        provided_args = set(self.strategy_config)
        required_args = set(self.get_required_configs())
        missing_args = required_args - provided_args
        if missing_args:
            raise RuntimeError(f"Missing options: {', '.join(sorted(missing_args))}")
        extra_args = provided_args - required_args - _OPTIONAL_CONFIGS
        if extra_args:
            logging.warning("Unknown options: %s", ", ".join(sorted(extra_args)))
        if self.strategy_config["evaluation_protocol"] != TRAIN_CALIBRATION_PROTOCOL:
            raise ValueError(
                "train_calibrated_unfixed_detect_label requires "
                "evaluation_protocol='train_calibration'."
            )

    @staticmethod
    def _as_1d_array(values: Any, value_name: str) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == 0:
            raise ValueError(f"{value_name} must be an array, got a scalar.")
        return array.reshape(-1)

    @staticmethod
    def _unpack_prediction_output(output: Any):
        if isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError(
                    "Detection output tuples must contain exactly "
                    "(prediction, anomaly_score)."
                )
            return output
        return output, output

    def _threshold_ratios(self) -> List[float]:
        ratios = self.strategy_config["anomaly_ratios"]
        if np.isscalar(ratios):
            ratios = [ratios]
        validated = []
        for ratio in ratios:
            value = float(ratio)
            if not np.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(
                    f"Invalid anomaly ratio {ratio!r}; expected a percentage in [0, 100]."
                )
            validated.append(value)
        if not validated:
            raise ValueError("anomaly_ratios must contain at least one value.")
        return validated

    @staticmethod
    def _context_length(model) -> int:
        config = getattr(model, "config", None)
        lengths = []
        for name in ("seq_len", "win_size", "slide_win", "window_length"):
            value = getattr(config, name, None)
            if value is None:
                continue
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                lengths.append(value)
        return max(lengths, default=1)

    def _split_fit_and_calibration(
        self, model, train_data: pd.DataFrame, train_label: pd.DataFrame
    ):
        fraction = float(self.strategy_config["calibration_fraction"])
        if not 0 < fraction < 1:
            raise ValueError(
                f"calibration_fraction must be between 0 and 1, got {fraction}."
            )

        if train_label is None or len(train_label) != len(train_data):
            raise ValueError("Strict evaluation requires aligned official train labels.")
        try:
            label_values = np.asarray(train_label, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("Strict evaluation requires numeric official train labels.") from exc
        if not np.all(np.isfinite(label_values)):
            raise ValueError("Strict evaluation requires finite official train labels.")
        anomaly_count = int(np.count_nonzero(label_values))
        if anomaly_count:
            raise ValueError(
                "Strict evaluation requires an anomaly-free official train split, "
                f"but found {anomaly_count} non-zero labels."
            )

        context_length = self._context_length(model)
        default_gap = max(context_length - 1, 0)
        gap = int(self.strategy_config.get("calibration_gap", default_gap))
        if gap < 0:
            raise ValueError(f"calibration_gap must be non-negative, got {gap}.")

        total_length = len(train_data)
        calibration_length = int(np.ceil(total_length * fraction))
        calibration_start = total_length - calibration_length
        fit_end = calibration_start - gap
        if fit_end < context_length or calibration_length < context_length:
            raise ValueError(
                "Official train segment is too short for a disjoint temporal "
                "fit/calibration split: "
                f"total={total_length}, fit={fit_end}, gap={gap}, "
                f"calibration={calibration_length}, required_segment_length="
                f"{context_length}."
            )
        return (
            train_data.iloc[:fit_end],
            train_label.iloc[:fit_end],
            train_data.iloc[calibration_start:],
        )

    def _threshold_from_calibration_scores(self, calibration_output: Any, test_output: Any):
        calibration_score, _ = self._unpack_prediction_output(calibration_output)
        test_score, test_metric_score = self._unpack_prediction_output(test_output)
        calibration_score = self._as_1d_array(
            calibration_score, "calibration anomaly score"
        ).astype(float)
        test_score = self._as_1d_array(test_score, "test anomaly score").astype(float)
        test_metric_score = self._as_1d_array(
            test_metric_score, "test metric score"
        ).astype(float)
        if calibration_score.size == 0:
            raise ValueError("Calibration anomaly score is empty.")
        if not np.isfinite(calibration_score).all():
            raise ValueError("Calibration anomaly score contains NaN or infinity.")

        predictions = {}
        for ratio in self._threshold_ratios():
            alpha = ratio / 100.0
            if alpha <= 0:
                threshold = np.inf
            elif alpha >= 1:
                threshold = -np.inf
            else:
                rank = int(np.ceil((calibration_score.size + 1) * (1 - alpha)))
                threshold = (
                    np.inf
                    if rank > calibration_score.size
                    else np.partition(calibration_score, rank - 1)[rank - 1]
                )
            predictions[ratio] = (test_score > threshold).astype(int)
        return predictions, test_metric_score

    def _require_aligned(self, values: Any, expected_length: int, value_name: str):
        values = self._as_1d_array(values, value_name)
        if len(values) != expected_length:
            raise ValueError(
                f"Length mismatch for {value_name}: expected {expected_length}, "
                f"got {len(values)}. Strict evaluation does not silently pad or truncate."
            )
        return values

    def execute(self, series_name: str, model_factory):
        fix_random_seed(self._get_scalar_config_value("seed", series_name))
        model = model_factory()
        try:
            self.model = model
            train_data, train_label, test_data, test_label = self.split_data(series_name)
            fit_data, fit_label, calibration_data = self._split_fit_and_calibration(
                model, train_data, train_label
            )
            start_fit_time = time.time()
            if hasattr(model, "detect_fit"):
                model.detect_fit(fit_data, fit_label)
            else:
                model.fit(fit_data, fit_label)
            end_fit_time = time.time()

            detect_score = getattr(model, "detect_score", None)
            if not callable(detect_score):
                raise TypeError(
                    "Strict evaluation requires the model to implement detect_score()."
                )
            predictions, metric_score = self._threshold_from_calibration_scores(
                detect_score(calibration_data), detect_score(test_data)
            )
            end_inference_time = time.time()
            actual = self._as_1d_array(test_label.to_numpy(), "test label").astype(float)
            metric_score = self._require_aligned(
                metric_score, len(actual), "anomaly score"
            ).astype(float)

            results = []
            for ratio, predicted in predictions.items():
                predicted = self._require_aligned(
                    predicted, len(actual), f"prediction (anomaly_ratio={ratio})"
                ).astype(float)
                metric_results, log_info = self.evaluator.evaluate_with_log(
                    actual=actual, predicted=predicted, another=metric_score
                )
                suffix = [series_name]
                if getattr(FieldNames, "FIT_TIME", None) in self.field_names:
                    suffix.extend(
                        [end_fit_time - start_fit_time, end_inference_time - end_fit_time]
                    )
                suffix.extend([ratio, "", "", log_info])
                results.append(metric_results + suffix)
            return results
        except Exception as error:
            log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{error}"
            return [self.get_default_result(**{FieldNames.LOG_INFO: log})]

    def split_data(self, series_name: str):
        data = DataPool().get_pool().get_series(series_name).reset_index(drop=True)
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data = train.loc[:, train.columns != "label"]
        test_data = test.loc[:, test.columns != "label"]
        return train_data, train.loc[:, ["label"]], test_data, test.loc[:, ["label"]]

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__
