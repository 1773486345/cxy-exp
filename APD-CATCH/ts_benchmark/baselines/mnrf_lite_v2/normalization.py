"""Training-only fixed input and feature statistics."""

import numpy as np


class FixedInputStandardizer:
    def fit(self, values: np.ndarray) -> "FixedInputStandardizer":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("MNRF-Lite V2 requires [time, variables] with D >= 2")
        if not np.isfinite(values).all():
            raise ValueError("input contains non-finite values")
        self.mean_ = values.mean(axis=0).astype(np.float32)
        self.scale_ = np.maximum(values.std(axis=0), 1e-6).astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.mean_.shape[0]:
            raise ValueError("input variable count/order does not match fitted input statistics")
        if not np.isfinite(values).all():
            raise ValueError("input contains non-finite values")
        return ((values - self.mean_) / self.scale_).astype(np.float32, copy=False)


class FixedFeatureStandardizer:
    def __init__(self, floor: float = 0.05):
        self.floor = float(floor)

    def fit(self, features: np.ndarray, chunk_size: int = 512) -> "FixedFeatureStandardizer":
        if features.ndim != 3 or features.shape[1] < 2 or features.shape[2] != 25:
            raise ValueError("features must have shape [windows, D>=2, 25]")
        count = 0
        total = np.zeros(features.shape[1:], dtype=np.float64)
        total_sq = np.zeros_like(total)
        for start in range(0, len(features), chunk_size):
            chunk = np.asarray(features[start:start + chunk_size], dtype=np.float64)
            if not np.isfinite(chunk).all():
                raise ValueError("feature cache contains non-finite values")
            count += chunk.shape[0]
            total += chunk.sum(axis=0)
            total_sq += np.square(chunk).sum(axis=0)
        if count == 0:
            raise ValueError("no training windows available for feature standardization")
        self.mean_ = (total / count).astype(np.float32)
        variance = np.maximum(total_sq / count - np.square(total / count), 0.0)
        self.scale_ = np.maximum(np.sqrt(variance), self.floor).astype(np.float32)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if features.shape[-2:] != self.mean_.shape:
            raise ValueError("feature shape does not match fitted feature statistics")
        return ((features - self.mean_) / self.scale_).astype(np.float32, copy=False)
