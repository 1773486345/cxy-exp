from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


DEFAULT_ANOMALY_RATIOS = (0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0)


def _as_2d_float_array(series: pd.DataFrame) -> np.ndarray:
    data = series.copy()
    if "date" in data.columns:
        data = data.drop(columns=["date"])
    data = data.apply(pd.to_numeric, errors="coerce")
    return data.to_numpy(dtype=float)


def _threshold_by_ratio(scores: np.ndarray, ratio: float) -> np.ndarray:
    ratio = float(np.clip(ratio, 0.0, 100.0))
    if scores.size == 0 or ratio <= 0:
        return np.zeros_like(scores, dtype=int)
    threshold = np.percentile(scores, 100.0 - ratio)
    return (scores > threshold).astype(int)


class ClassicADAdapter:
    def __init__(
        self,
        model_name: str,
        model_builder: Callable[[], object],
        score_method: str,
        anomaly_ratios: Iterable[float] = DEFAULT_ANOMALY_RATIOS,
    ):
        self.model_name = model_name
        self.model_builder = model_builder
        self.score_method = score_method
        self.anomaly_ratios = tuple(float(ratio) for ratio in anomaly_ratios)
        self.model = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def detect_fit(
        self, train_data: pd.DataFrame, train_label: Optional[pd.DataFrame] = None
    ) -> object:
        x_train = _as_2d_float_array(train_data)
        x_train = self.imputer.fit_transform(x_train)
        x_train = self.scaler.fit_transform(x_train)
        self.model = self.model_builder()
        self.model.fit(x_train)
        return self.model

    def detect_score(self, test_data: pd.DataFrame):
        if self.model is None:
            raise RuntimeError(f"{self.model_name} has not been fitted.")
        x_test = _as_2d_float_array(test_data)
        x_test = self.imputer.transform(x_test)
        x_test = self.scaler.transform(x_test)
        scores = self._score(x_test)
        return scores, scores

    def detect_label(self, test_data: pd.DataFrame):
        scores, _ = self.detect_score(test_data)
        labels = {
            ratio: _threshold_by_ratio(scores, ratio) for ratio in self.anomaly_ratios
        }
        return labels, scores

    def _score(self, x_test: np.ndarray) -> np.ndarray:
        if self.score_method == "reconstruction_error":
            x_reconstruct = self.model.inverse_transform(self.model.transform(x_test))
            return np.mean(np.square(x_test - x_reconstruct), axis=1)
        if self.score_method == "negative_score_samples":
            return -self.model.score_samples(x_test)
        if self.score_method == "negative_decision_function":
            return -self.model.decision_function(x_test).reshape(-1)
        raise ValueError(f"Unsupported score method: {self.score_method}")

    def __repr__(self):
        return self.model_name


def _factory(
    model_name: str,
    model_builder: Callable[..., object],
    score_method: str,
    default_params: Optional[Dict] = None,
):
    default_params = default_params or {}

    def build_model(**kwargs):
        anomaly_ratios = kwargs.pop("anomaly_ratio", DEFAULT_ANOMALY_RATIOS)
        params = dict(default_params)
        params.update(kwargs)
        return ClassicADAdapter(
            model_name=model_name,
            model_builder=lambda: model_builder(**params),
            score_method=score_method,
            anomaly_ratios=anomaly_ratios,
        )

    return {"model_factory": build_model, "required_hyper_params": {}}


PCA = _factory(
    "PCA",
    SklearnPCA,
    "reconstruction_error",
    {"n_components": 0.95, "svd_solver": "full"},
)

IsolationForest = _factory(
    "IsolationForest",
    SklearnIsolationForest,
    "negative_score_samples",
    {"n_estimators": 200, "contamination": "auto", "random_state": 2021, "n_jobs": 1},
)

LOF = _factory(
    "LOF",
    LocalOutlierFactor,
    "negative_score_samples",
    {"n_neighbors": 20, "novelty": True, "contamination": "auto", "n_jobs": 1},
)

OCSVM = _factory(
    "OCSVM",
    OneClassSVM,
    "negative_decision_function",
    {"kernel": "rbf", "gamma": "scale", "nu": 0.05},
)
