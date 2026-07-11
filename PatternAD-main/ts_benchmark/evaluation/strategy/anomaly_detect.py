# -*- coding: utf-8 -*-
import logging
import traceback
from typing import List, Any

import numpy as np
import pandas as pd
import torch

from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import classification_metrics_label
from ts_benchmark.evaluation.metrics import classification_metrics_score
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.strategy import Strategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.utils.data_processing import split_before
from ts_benchmark.utils.random_utils import fix_random_seed


logger = logging.getLogger(__name__)

TRAIN_CALIBRATION_PROTOCOL = "train_calibration"
LEGACY_TEST_CONTAMINATED_PROTOCOL = "legacy_test_contaminated"
SUPPORTED_EVALUATION_PROTOCOLS = {
    TRAIN_CALIBRATION_PROTOCOL,
    LEGACY_TEST_CONTAMINATED_PROTOCOL,
}


SCORE_METRIC_NAMES = {
    "auc_roc",
    "auc_pr",
    "R_AUC_ROC",
    "R_AUC_PR",
    "VUS_ROC",
    "VUS_PR",
}


class AnomalyDetect(Strategy):
    """
    异常检测类，用于在时间序列数据上执行异常检测。
    """

    REQUIRED_CONFIGS = ["seed"]
    USES_TRAIN_CALIBRATED_LABELS = False
    OPTIONAL_CONFIGS = {
        "anomaly_ratios",
        "calibration_fraction",
        "calibration_gap",
        "evaluation_protocol",
        "verbose_result",
    }

    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        """
        初始化子类实例。

        :param strategy_config: 模型评估配置。
        """
        super().__init__(strategy_config, evaluator)
        self.model = None
        self.data_lens = None
        self.calibration_data = None
        self.calibration_text = None

        protocol = self._evaluation_protocol()
        if protocol not in SUPPORTED_EVALUATION_PROTOCOLS:
            raise ValueError(
                f"Unknown evaluation_protocol {protocol!r}. Expected one of "
                f"{sorted(SUPPORTED_EVALUATION_PROTOCOLS)}."
            )

    def _check_config(self):
        provided_args = set(self.strategy_config)
        required_args = set(self.get_required_configs())
        missing_args = required_args - provided_args
        if missing_args:
            raise RuntimeError(f"Missing options: {', '.join(sorted(missing_args))} ")

        extra_args = provided_args - required_args - self.OPTIONAL_CONFIGS
        if extra_args:
            logger.warning("Unknown options: %s ", ", ".join(sorted(extra_args)))

    def _evaluation_protocol(self) -> str:
        # Existing benchmark configs predate protocol provenance and must retain
        # their historical API/length behavior. Strict calibration is opt-in and
        # is explicit in the PatternAD experiment configs.
        return self.strategy_config.get(
            "evaluation_protocol", LEGACY_TEST_CONTAMINATED_PROTOCOL
        )

    def _uses_legacy_test_contaminated_threshold(self) -> bool:
        return self._evaluation_protocol() == LEGACY_TEST_CONTAMINATED_PROTOCOL

    def _verbose_result(self) -> bool:
        return bool(self.strategy_config.get("verbose_result", False))

    def _require_model_method(self, method_name: str):
        method = getattr(self.model, method_name, None)
        if not callable(method):
            raise TypeError(
                f"evaluation_protocol='{TRAIN_CALIBRATION_PROTOCOL}' requires "
                f"the model to implement {method_name}(). Use a score-capable "
                "adapter or select the explicit legacy protocol for reproduction."
            )
        return method

    @staticmethod
    def _split_multi_text(
        text: pd.DataFrame, total_length: int, train_length: int
    ):
        """Split aligned text or reuse one row as a static description."""
        text = text.reset_index(drop=True)
        if len(text) == 1:
            return text.copy(), text.copy()
        if len(text) < total_length:
            raise ValueError(
                "Text must contain either one static row or at least one row per "
                f"time point; got text={len(text)}, time={total_length}."
            )
        aligned_text = text.iloc[:total_length].reset_index(drop=True)
        return split_before(aligned_text, train_length)

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
        ratios = self.strategy_config.get("anomaly_ratios")
        if ratios is None:
            model_config = getattr(self.model, "config", None)
            ratios = getattr(model_config, "anomaly_ratio", None)
        if ratios is None:
            raise ValueError(
                "Train-calibrated label evaluation requires anomaly_ratios in "
                "strategy_args or anomaly_ratio in the model config."
            )
        if np.isscalar(ratios):
            ratios = [ratios]

        validated_ratios = []
        for ratio in ratios:
            numeric_ratio = float(ratio)
            if not np.isfinite(numeric_ratio) or not 0 <= numeric_ratio <= 100:
                raise ValueError(
                    f"Invalid anomaly ratio {ratio!r}; expected a finite percentage "
                    "between 0 and 100."
                )
            validated_ratios.append(numeric_ratio)
        if not validated_ratios:
            raise ValueError("anomaly_ratios must contain at least one value.")
        return validated_ratios

    def _split_fit_and_calibration(
        self,
        train_data: pd.DataFrame,
        train_label: pd.DataFrame,
        train_text: pd.DataFrame = None,
    ):
        if (
            not self.USES_TRAIN_CALIBRATED_LABELS
            or self._uses_legacy_test_contaminated_threshold()
        ):
            return train_data, train_label, train_text, train_data, train_text

        fraction = float(self.strategy_config.get("calibration_fraction", 0.2))
        if not 0 < fraction < 1:
            raise ValueError(
                f"calibration_fraction must be between 0 and 1, got {fraction}."
            )

        model_config = getattr(self.model, "config", None)
        sequence_length = int(getattr(model_config, "seq_len", 1) or 1)
        default_gap = max(sequence_length - 1, 0)
        gap = int(self.strategy_config.get("calibration_gap", default_gap))
        if gap < 0:
            raise ValueError(f"calibration_gap must be non-negative, got {gap}.")

        total_length = len(train_data)
        if train_label is not None and len(train_label) != total_length:
            raise ValueError(
                "Official train data/label lengths differ: "
                f"{total_length} != {len(train_label)}."
            )
        if train_label is not None:
            try:
                train_label_values = np.asarray(
                    train_label, dtype=float
                ).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "train_calibration requires numeric official-train labels."
                ) from exc
            if not np.all(np.isfinite(train_label_values)):
                raise ValueError(
                    "train_calibration requires finite official-train labels."
                )
            anomaly_count = int(np.count_nonzero(train_label_values))
            if anomaly_count:
                raise ValueError(
                    "train_calibration requires an anomaly-free official train "
                    f"split, but found {anomaly_count} non-zero labels. Redefine "
                    "or clean the official train split before strict evaluation."
                )
        static_text = train_text is not None and len(train_text) == 1
        if (
            train_text is not None
            and not static_text
            and len(train_text) != total_length
        ):
            raise ValueError(
                "Official train data/text lengths differ: "
                f"{total_length} != {len(train_text)}."
            )

        calibration_length = int(np.ceil(total_length * fraction))
        calibration_start = total_length - calibration_length
        fit_end = calibration_start - gap
        minimum_length = max(sequence_length, 1)
        if fit_end < minimum_length or calibration_length < minimum_length:
            raise ValueError(
                "Official train segment is too short for a disjoint temporal "
                "fit/calibration split: "
                f"total={total_length}, fit={fit_end}, gap={gap}, "
                f"calibration={calibration_length}, required_segment_length="
                f"{minimum_length}."
            )

        fit_data = train_data.iloc[:fit_end]
        calibration_data = train_data.iloc[calibration_start:]
        fit_label = None if train_label is None else train_label.iloc[:fit_end]
        if train_text is None:
            fit_text = calibration_text = None
        elif static_text:
            fit_text = train_text.copy()
            calibration_text = train_text.copy()
        else:
            fit_text = train_text.iloc[:fit_end]
            calibration_text = train_text.iloc[calibration_start:]
        return (
            fit_data,
            fit_label,
            fit_text,
            calibration_data,
            calibration_text,
        )

    def _threshold_from_calibration_scores(
        self, calibration_output: Any, test_output: Any
    ):
        calibration_score, _ = self._unpack_prediction_output(calibration_output)
        test_score, another = self._unpack_prediction_output(test_output)
        calibration_score = self._as_1d_array(
            calibration_score, "calibration anomaly score"
        ).astype(float)
        test_score = self._as_1d_array(test_score, "test anomaly score").astype(float)
        another = self._as_1d_array(another, "test metric score").astype(float)

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
        return predictions, another

    def _detect_train_calibrated_label(self, test_data: pd.DataFrame):
        if self._uses_legacy_test_contaminated_threshold():
            return self.model.detect_label(test_data)
        detect_score = self._require_model_method("detect_score")
        return self._threshold_from_calibration_scores(
            detect_score(self.calibration_data),
            detect_score(test_data),
        )

    def _detect_train_calibrated_multi_label(
        self, test_data: pd.DataFrame, test_text: pd.DataFrame
    ):
        if self._uses_legacy_test_contaminated_threshold():
            return self.model.detect_multi_label(test_data, test_text)
        detect_multi_score = self._require_model_method("detect_multi_score")
        return self._threshold_from_calibration_scores(
            detect_multi_score(self.calibration_data, self.calibration_text),
            detect_multi_score(test_data, test_text),
        )

    def _detect_train_calibrated_mmd_label(
        self, test_data: pd.DataFrame, test_text: pd.DataFrame
    ):
        if self._uses_legacy_test_contaminated_threshold():
            return self.model.detect_timeMMD_label(test_data, test_text)
        detect_mmd_score = self._require_model_method("detect_timeMMD_score")
        return self._threshold_from_calibration_scores(
            detect_mmd_score(self.calibration_data, self.calibration_text),
            detect_mmd_score(test_data, test_text),
        )

    def _align_result_vector(
        self, values: Any, actual_length: int, value_name: str
    ) -> np.ndarray:
        values = self._as_1d_array(values, value_name)
        if len(values) == actual_length:
            return values

        if not self._uses_legacy_test_contaminated_threshold():
            raise ValueError(
                f"Length mismatch for {value_name}: expected {actual_length}, "
                f"got {len(values)}. The default protocol does not silently pad "
                "or truncate anomaly outputs."
            )

        logger.warning(
            "Legacy protocol is aligning %s from length %d to %d. This may distort metrics.",
            value_name,
            len(values),
            actual_length,
        )
        if len(values) < actual_length:
            return np.pad(values, (0, actual_length - len(values)), mode="constant")
        return values[:actual_length]

    def _evaluate_predictions(
        self,
        series_name: str,
        test_label: pd.DataFrame,
        prediction_output: Any,
    ) -> List[List[Any]]:
        predict_labels, another = self._unpack_prediction_output(prediction_output)
        if not isinstance(predict_labels, dict):
            predict_labels = {None: predict_labels}

        actual_label = self._as_1d_array(
            test_label.to_numpy(), "test label"
        ).astype(float)
        score_metric_cache = {}
        results = []
        for ratio, predict_label in predict_labels.items():
            predict_label = self._align_result_vector(
                predict_label,
                len(actual_label),
                f"prediction (anomaly_ratio={ratio})",
            ).astype(float)
            metric_score = self._align_result_vector(
                another,
                len(actual_label),
                "anomaly score",
            ).astype(float)

            metric_results, log_info = self._evaluate_with_cached_score_metrics(
                actual=actual_label,
                predicted=predict_label,
                another=metric_score,
                score_metric_cache=score_metric_cache,
            )
            if self._verbose_result():
                print(metric_results)
            metric_results += [series_name, ratio, "", "", log_info]
            results.append(metric_results)
        return results

    def _error_result(self, series_name: str, error: Exception):
        log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{error}"
        return [self.get_default_result(**{FieldNames.LOG_INFO: log})]

    def _evaluate_with_cached_score_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        another: np.ndarray,
        score_metric_cache: dict,
    ):
        evaluate_result = []
        log_info = ""
        for metric_name, metric_func in zip(
            self.evaluator.metric_names, self.evaluator.metric_funcs
        ):
            base_metric_name = metric_name.split(";", 1)[0]
            if base_metric_name in score_metric_cache:
                evaluate_result.append(score_metric_cache[base_metric_name])
                continue
            try:
                value = metric_func(actual, predicted, another)
            except Exception as e:
                value = np.nan
                func_name = getattr(metric_func, "__name__", repr(metric_func))
                log_info += (
                    f"Error in calculating {func_name}: "
                    f"{traceback.format_exc()}\n{e}\n"
                )
            if base_metric_name in SCORE_METRIC_NAMES:
                score_metric_cache[base_metric_name] = value
            evaluate_result.append(value)
        return evaluate_result, log_info


    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        """
        执行异常检测策略。

        :param series_name: 要执行异常检测的序列名称。
        :param model_factory: 模型对象的构造/工厂函数。
        :return: 评估结果。
        """
        fix_random_seed(self._get_scalar_config_value("seed", series_name))

        model = model_factory()
        try:
            self.model = model
            train_data, train_label, test_data, test_label = self.split_data(
                series_name
            )
            (
                fit_data,
                fit_label,
                _,
                self.calibration_data,
                _,
            ) = self._split_fit_and_calibration(train_data, train_label)
            if hasattr(model, "detect_fit"):
                self.model.detect_fit(fit_data, fit_label)
            else:
                self.model.fit(fit_data, fit_label)
            single_series_results_list = self._evaluate_predictions(
                series_name, test_label, self.detect(test_data)
            )
        except Exception as e:
            single_series_results_list = self._error_result(series_name, e)
        return single_series_results_list


    def multi_execute(
        self, series_name: str, text_name: str, model_factory: ModelFactory
    ) -> Any:
        """
        执行异常检测策略。

        :param series_name: 要执行异常检测的序列名称。
        :param model_factory: 模型对象的构造/工厂函数。
        :return: 评估结果。
        """
        fix_random_seed(self._get_scalar_config_value("seed", series_name))

        model = model_factory()
        try:
            self.model = model
            (
                train_data,
                train_text,
                train_label,
                test_data,
                test_text,
                test_label,
            ) = self.split_multi_data(series_name, text_name)
            (
                fit_data,
                fit_label,
                fit_text,
                self.calibration_data,
                self.calibration_text,
            ) = self._split_fit_and_calibration(
                train_data, train_label, train_text
            )

            torch.cuda.empty_cache()
            self.model.detect_multi_fit(fit_data, fit_text, fit_label)
            single_series_results_list = self._evaluate_predictions(
                series_name,
                test_label,
                self.multi_detect(test_data, test_text),
            )
        except Exception as e:
            single_series_results_list = self._error_result(series_name, e)
        return single_series_results_list

    def mmd_execute(
        self, series_name: str, text_name: str, model_factory: ModelFactory
    ) -> Any:
        """
        执行异常检测策略。

        :param series_name: 要执行异常检测的序列名称。
        :param model_factory: 模型对象的构造/工厂函数。
        :return: 评估结果。
        """
        fix_random_seed(self._get_scalar_config_value("seed", series_name))

        model = model_factory()
        try:
            self.model = model
            (
                train_data,
                train_text,
                train_label,
                test_data,
                test_text,
                test_label,
            ) = self.split_multi_data(series_name, text_name)
            (
                fit_data,
                fit_label,
                fit_text,
                self.calibration_data,
                self.calibration_text,
            ) = self._split_fit_and_calibration(
                train_data, train_label, train_text
            )

            torch.cuda.empty_cache()
            self.model.detect_timeMMD_fit(fit_data, fit_text, fit_label)
            single_series_results_list = self._evaluate_predictions(
                series_name,
                test_label,
                self.mmd_detect(test_data, test_text),
            )
        except Exception as e:
            single_series_results_list = self._error_result(series_name, e)
        return single_series_results_list


    def split_data(self, data: str):
        raise NotImplementedError

    def detect(self, test_data: pd.DataFrame):
        raise NotImplementedError

    def multi_detect(self, test_data: pd.DataFrame, test_text: pd.DataFrame):
        raise NotImplementedError

    def mmd_detect(self, test_data: pd.DataFrame, test_text: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def accepted_metrics():
        raise NotImplementedError

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.ANOMALY_RATIO,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]


class FixedDetectScore(AnomalyDetect):
    REQUIRED_CONFIGS = ["train_test_split"]

    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(self.strategy_config["train_test_split"] * self.data_lens)
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class FixedDetectLabel(AnomalyDetect):
    REQUIRED_CONFIGS = ["train_test_split"]
    USES_TRAIN_CALIBRATED_LABELS = True

    def split_data(self, series_name: str):
        data = DataPool().get_pool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(self.strategy_config["train_test_split"] * self.data_lens)
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self._detect_train_calibrated_label(test_data)

    def multi_detect(self, test_data, test_text):
        return self._detect_train_calibrated_multi_label(test_data, test_text)

    def mmd_detect(self, test_data, test_text):
        return self._detect_train_calibrated_mmd_label(test_data, test_text)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class UnFixedDetectScore(AnomalyDetect):
    def split_data(self, series_name: str):
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )

        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def split_multi_data(self, series_name, text_name):
        data_pool = DataPool().get_pool()
        data = data_pool.get_series(series_name).reset_index(drop=True)
        text = data_pool.get_text(text_name).reset_index(drop=True)
        train_length = int(
            data_pool.get_series_meta_info(series_name)["train_lens"].item()
        )
        if not 0 < train_length < len(data):
            raise ValueError(
                f"Invalid train_lens={train_length} for series length {len(data)}."
            )
        train_time, test_time = split_before(data, train_length)
        train_text, test_text = self._split_multi_text(
            text, len(data), train_length
        )
        train_data, train_label = (
            train_time.loc[:, train_time.columns != "label"],
            train_time.loc[:, ["label"]],
        )
        test_data, test_label = (
            test_time.loc[:, test_time.columns != "label"],
            test_time.loc[:, ["label"]],
        )
        return train_data, train_text, train_label, test_data, test_text, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    def multi_detect(self, test_data, test_text):
        return self.model.detect_multi_score(test_data, test_text)

    def mmd_detect(self, test_data, test_text):
        return self.model.detect_timeMMD_score(test_data, test_text)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class UnFixedDetectLabel(AnomalyDetect):
    USES_TRAIN_CALIBRATED_LABELS = True

    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def split_multi_data(self, series_name, text_name):
        data = DataPool().get_pool().get_series(series_name)
        text = DataPool().get_pool().get_text(text_name)
        data = data.reset_index(drop=True)
        text = text.reset_index(drop=True)

        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        if not 0 < train_length < len(data):
            raise ValueError(
                f"Invalid train_lens={train_length} for series length {len(data)}."
            )
        train_time, test_time = split_before(data, train_length)
        train_text, test_text = self._split_multi_text(
            text, len(data), train_length
        )

        train_data, train_label = (
            train_time.loc[:, train_time.columns != "label"],
            train_time.loc[:, ["label"]],
        )
        test_data, test_label = (
            test_time.loc[:, test_time.columns != "label"],
            test_time.loc[:, ["label"]],
        )
        return train_data, train_text, train_label, test_data, test_text, test_label

    def detect(self, test_data):
        return self._detect_train_calibrated_label(test_data)

    def multi_detect(self, test_data, test_text):
        return self._detect_train_calibrated_multi_label(test_data, test_text)

    def mmd_detect(self, test_data, test_text):
        return self._detect_train_calibrated_mmd_label(test_data, test_text)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class AllDetectScore(AnomalyDetect):
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != "label"], None
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    def multi_detect(self, test_data, test_text):
        return self.model.detect_multi_score(test_data, test_text)

    def mmd_detect(self, test_data, test_text):
        return self.model.detect_timeMMD_score(test_data, test_text)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class AllDetectLabel(AnomalyDetect):
    USES_TRAIN_CALIBRATED_LABELS = True

    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        super().__init__(strategy_config, evaluator)
        if not self._uses_legacy_test_contaminated_threshold():
            raise ValueError(
                "all_detect_label has no disjoint training/calibration segment. "
                "Use a fixed/unfixed label strategy for train-calibrated thresholds, "
                "or explicitly select evaluation_protocol='legacy_test_contaminated'."
            )

    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != "label"], None
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self._detect_train_calibrated_label(test_data)

    def multi_detect(self, test_data, test_text):
        return self._detect_train_calibrated_multi_label(test_data, test_text)

    def mmd_detect(self, test_data, test_text):
        return self._detect_train_calibrated_mmd_label(test_data, test_text)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__
