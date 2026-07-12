"""A2-M4 explicit landmark and direction support compatibility.

M4 represents a candidate future by its strongest internal-change landmark and
the cross-channel direction at that landmark. Event-pre support is retrieved
from time-disjoint normal reference windows using only recent observable state
increments and the event-pre terminal state. It is neither a trajectory density,
learned pair energy, nor a learned finite codebook.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from ts_benchmark.baselines.A2TransitionCompatibility.A2TransitionCompatibility import (
    ReferenceUpperTail,
    TrajectoryStandardizer,
    _finite_windows,
)


class A2LandmarkCompatibility:
    """Reference-normal support for conditional change landmark and direction."""

    raw_score_key = "landmark_direction_surprisal"
    raw_score_name = "event_pre_landmark_direction_surprisal"

    def __init__(
        self,
        dimensions: int,
        history_length: int,
        horizon_length: int,
        condition_on_event_pre: bool = True,
        outer_alpha: float = 0.10,
        reliability_bin_count: int = 1,
        neighbor_count: int = 32,
        state_increment_length: int = 8,
        landmark_smoothing: float = 1.0,
        direction_weight: float = 1.0,
        device: str | torch.device = "cpu",
        **unused: Any,
    ) -> None:
        if unused:
            raise ValueError(f"Unsupported A2-M4 model arguments: {sorted(unused)}")
        if min(dimensions, history_length, horizon_length, neighbor_count) < 1:
            raise ValueError("M4 dimensions, lengths, and neighbor_count must be positive.")
        if horizon_length < 2:
            raise ValueError("M4 requires at least two future samples.")
        if not 2 <= state_increment_length <= history_length:
            raise ValueError("state_increment_length must be between two and history_length.")
        if not 0.0 < outer_alpha < 1.0:
            raise ValueError("outer_alpha must be in (0, 1).")
        if reliability_bin_count != 1:
            raise ValueError("M4 uses one global reference stratum; state matching is pre-score support.")
        if landmark_smoothing <= 0.0 or direction_weight < 0.0:
            raise ValueError("landmark_smoothing must be positive and direction_weight non-negative.")
        self.dimensions = int(dimensions)
        self.history_length = int(history_length)
        self.horizon_length = int(horizon_length)
        self.condition_on_event_pre = bool(condition_on_event_pre)
        self.outer_alpha = float(outer_alpha)
        self.reliability_bin_count = 1
        self.neighbor_count = int(neighbor_count)
        self.state_increment_length = int(state_increment_length)
        self.landmark_smoothing = float(landmark_smoothing)
        self.direction_weight = float(direction_weight)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
        self.normalizer = TrajectoryStandardizer()
        self.tail: Optional[ReferenceUpperTail] = None
        self.outer_threshold_: Optional[float] = None
        self.reference_features_: Optional[np.ndarray] = None
        self.reference_futures_: Optional[np.ndarray] = None
        self.reference_landmarks_: Optional[np.ndarray] = None
        self.reference_directions_: Optional[np.ndarray] = None
        self.fit_metadata_: Dict[str, Any] = {}

    @property
    def window_length(self) -> int:
        return self.history_length + self.horizon_length

    def _validate_windows(self, windows: np.ndarray, name: str) -> np.ndarray:
        array = _finite_windows(windows, name)
        if array.shape[1:] != (self.window_length, self.dimensions):
            raise ValueError(
                f"{name} must have shape [samples, {self.window_length}, {self.dimensions}]."
            )
        return array

    def _split(self, normalized_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            normalized_windows[:, : self.history_length],
            normalized_windows[:, self.history_length :],
        )

    def _state_features(self, event_pre: np.ndarray) -> np.ndarray:
        if event_pre.ndim != 3 or event_pre.shape[1:] != (
            self.history_length,
            self.dimensions,
        ):
            raise ValueError("event_pre has the wrong M4 history shape.")
        recent = event_pre[:, -self.state_increment_length :]
        increments = np.diff(recent, axis=1)
        features = np.concatenate((recent[:, -1], increments.reshape(len(recent), -1)), axis=1)
        if self.condition_on_event_pre:
            return features.astype(np.float64, copy=False)
        return np.zeros_like(features, dtype=np.float64)

    def _future_landmarks(self, future: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if future.ndim != 3 or future.shape[1:] != (self.horizon_length, self.dimensions):
            raise ValueError("future has the wrong M4 horizon shape.")
        increments = np.diff(future, axis=1)
        strengths = np.linalg.norm(increments, axis=2)
        landmarks = np.argmax(strengths, axis=1).astype(np.int64)
        chosen = increments[np.arange(len(increments)), landmarks]
        norms = np.linalg.norm(chosen, axis=1, keepdims=True)
        directions = chosen / np.maximum(norms, 1e-12)
        return landmarks, directions.astype(np.float64, copy=False)

    def _require_reference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = (
            self.reference_features_,
            self.reference_futures_,
            self.reference_landmarks_,
            self.reference_directions_,
        )
        if any(value is None for value in values):
            raise RuntimeError("M4 reference-normal support must be fitted before scoring.")
        return values  # type: ignore[return-value]

    def _neighbors(
        self, features: np.ndarray, exclude_reference_self: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        reference_features, _, _, _ = self._require_reference()
        if len(reference_features) <= self.neighbor_count:
            raise ValueError("M4 reference split must exceed neighbor_count.")
        squared = np.sum(
            np.square(features[:, None, :] - reference_features[None, :, :]), axis=2
        )
        if exclude_reference_self:
            if len(features) != len(reference_features):
                raise ValueError("M4 leave-one-out scoring requires the reference ordering.")
            squared[np.arange(len(features)), np.arange(len(features))] = np.inf
        selected = np.argpartition(squared, self.neighbor_count - 1, axis=1)[
            :, : self.neighbor_count
        ]
        selected_distances = np.take_along_axis(squared, selected, axis=1)
        order = np.argsort(selected_distances, axis=1, kind="stable")
        return (
            np.take_along_axis(selected, order, axis=1),
            np.take_along_axis(selected_distances, order, axis=1),
        )

    def _raw_scores_from_normalized(
        self, normalized_windows: np.ndarray, exclude_reference_self: bool = False
    ) -> np.ndarray:
        _, future = self._split(normalized_windows)
        features = self._state_features(normalized_windows[:, : self.history_length])
        indices, _ = self._neighbors(features, exclude_reference_self)
        _, _, reference_landmarks, reference_directions = self._require_reference()
        landmarks, directions = self._future_landmarks(future)
        neighbor_landmarks = reference_landmarks[indices]
        matches = neighbor_landmarks == landmarks[:, None]
        landmark_count = np.sum(matches, axis=1)
        landmark_probability = (landmark_count + self.landmark_smoothing) / (
            self.neighbor_count
            + self.landmark_smoothing * float(self.horizon_length - 1)
        )
        landmark_surprisal = -np.log(landmark_probability)
        neighbor_directions = reference_directions[indices]
        cosine = np.sum(neighbor_directions * directions[:, None, :], axis=2)
        masked_cosine = np.where(matches, cosine, -np.inf)
        best_cosine = np.max(masked_cosine, axis=1)
        direction_surprisal = np.where(
            np.isfinite(best_cosine), 1.0 - np.clip(best_cosine, -1.0, 1.0), 0.0
        )
        return (landmark_surprisal + self.direction_weight * direction_surprisal).astype(
            np.float64, copy=False
        )

    @staticmethod
    def _finite_sample_upper_threshold(scores: np.ndarray, alpha: float) -> float:
        ordered = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
        rank = int(math.ceil((len(ordered) + 1) * (1.0 - alpha))) - 1
        return float(ordered[min(max(rank, 0), len(ordered) - 1)])

    def fit(
        self,
        optimization_windows: np.ndarray,
        validation_windows: np.ndarray,
        reference_windows: np.ndarray,
        outer_calibration_windows: np.ndarray,
        seed: int,
    ) -> "A2LandmarkCompatibility":
        optimization = self._validate_windows(optimization_windows, "optimization_windows")
        validation = self._validate_windows(validation_windows, "validation_windows")
        reference = self._validate_windows(reference_windows, "reference_windows")
        outer_calibration = self._validate_windows(
            outer_calibration_windows, "outer_calibration_windows"
        )
        self.normalizer.fit(optimization)
        optimization_z = self.normalizer.transform(optimization)
        validation_z = self.normalizer.transform(validation)
        reference_z = self.normalizer.transform(reference)
        outer_calibration_z = self.normalizer.transform(outer_calibration)
        reference_pre, reference_future = self._split(reference_z)
        self.reference_features_ = self._state_features(reference_pre)
        self.reference_futures_ = reference_future.astype(np.float64, copy=True)
        self.reference_landmarks_, self.reference_directions_ = self._future_landmarks(
            reference_future
        )
        reference_scores = self._raw_scores_from_normalized(
            reference_z, exclude_reference_self=True
        )
        validation_scores = self._raw_scores_from_normalized(validation_z)
        outer_scores = self._raw_scores_from_normalized(outer_calibration_z)
        self.tail = ReferenceUpperTail().fit(reference_scores)
        self.outer_threshold_ = self._finite_sample_upper_threshold(
            self.tail.transform(outer_scores), self.outer_alpha
        )
        landmark_counts = np.bincount(
            self.reference_landmarks_, minlength=self.horizon_length - 1
        )
        self.fit_metadata_ = {
            "seed": int(seed),
            "optimization_windows": int(len(optimization)),
            "validation_windows": int(len(validation)),
            "reference_windows": int(len(reference)),
            "outer_calibration_windows": int(len(outer_calibration)),
            "normalizer": self.normalizer.metadata(),
            "calibration_kind": "global_landmark_direction_support",
            "reliability_boundaries": [],
            "reference_tails": {"0": self.tail.metadata()},
            "outer_alpha": self.outer_alpha,
            "reliability_bin_count": 1,
            "outer_thresholds": {"0": float(self.outer_threshold_)},
            "neighbor_count": self.neighbor_count,
            "state_increment_length": self.state_increment_length,
            "landmark_smoothing": self.landmark_smoothing,
            "direction_weight": self.direction_weight,
            "reference_landmark_counts": landmark_counts.astype(int).tolist(),
            "validation_mean_raw_score": float(np.mean(validation_scores)),
            "condition_on_event_pre": self.condition_on_event_pre,
            "parameter_count": 0,
        }
        return self

    def _require_fitted(self) -> None:
        if self.tail is None or self.outer_threshold_ is None:
            raise RuntimeError("A2LandmarkCompatibility must be fitted before scoring.")
        self._require_reference()

    def score_windows(self, windows: np.ndarray) -> Dict[str, np.ndarray]:
        self._require_fitted()
        normalized = self.normalizer.transform(self._validate_windows(windows, "windows"))
        raw_score = self._raw_scores_from_normalized(normalized)
        tail = self.tail.transform(raw_score)
        bins = np.zeros(len(raw_score), dtype=np.int64)
        thresholds = np.full(len(raw_score), float(self.outer_threshold_), dtype=np.float64)
        return {
            self.raw_score_key: raw_score,
            "compatibility_tail": tail,
            "reliability_bin": bins,
            "outer_threshold": thresholds,
            "outer_exceedance": (tail > thresholds).astype(np.int64),
        }

    def predict_mean_trajectory(self, windows: np.ndarray) -> np.ndarray:
        self._require_fitted()
        normalized = self.normalizer.transform(self._validate_windows(windows, "windows"))
        features = self._state_features(normalized[:, : self.history_length])
        indices, squared_distances = self._neighbors(features)
        _, reference_futures, _, _ = self._require_reference()
        weights = 1.0 / np.maximum(squared_distances, 1e-8)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        prediction_z = np.sum(reference_futures[indices] * weights[:, :, None, None], axis=1)
        return (
            prediction_z * self.normalizer.std_[None, None, :]
            + self.normalizer.mean_[None, None, :]
        ).astype(np.float32, copy=False)

    def event_pre_state(self, windows: np.ndarray) -> np.ndarray:
        self._require_fitted()
        normalized = self.normalizer.transform(self._validate_windows(windows, "windows"))
        return self._state_features(normalized[:, : self.history_length]).astype(
            np.float32, copy=False
        )

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        self._require_fitted()
        features, futures, landmarks, directions = self._require_reference()
        return {
            "reference_features": torch.from_numpy(features.astype(np.float32, copy=True)),
            "reference_futures": torch.from_numpy(futures.astype(np.float32, copy=True)),
            "reference_landmarks": torch.from_numpy(landmarks.astype(np.int64, copy=True)),
            "reference_directions": torch.from_numpy(directions.astype(np.float32, copy=True)),
        }
