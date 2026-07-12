"""Normal-only, input-restricted reliability calibration for Direction B1.

The calibration conditions on how variable each evidence source is allowed to
be, rather than on an oracle regime or the observed target terminal value.
It does not fuse the evidence scores or backpropagate into either repair head.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ts_benchmark.baselines.MultiEvidenceRepair.MultiEvidenceRepair import (
    EmpiricalUpperTail,
)


SCORE_COMPONENTS: Sequence[str] = (
    "temporal_residual",
    "cross_residual",
    "disagreement",
)


def conformal_upper_threshold(calibration_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample upper threshold based on calibration scores only."""
    scores = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("calibration scores must be non-empty and finite.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("target_fpr must be between zero and one.")
    if alpha == 0.0:
        return math.inf
    if alpha == 1.0:
        return -math.inf
    rank = int(math.ceil((len(scores) + 1) * (1.0 - alpha)))
    if rank > len(scores):
        return math.inf
    return float(np.partition(scores, rank - 1)[rank - 1])


class EvidenceReliabilityCalibration:
    """Fit component-wise tail maps conditional on allowed evidence reliability.

    ``global`` exactly reproduces B0's single reference-tail/outer-cutoff
    scheme. ``input_energy_stratified`` introduces B1's three (or more)
    reliability strata. Boundaries are fitted from optimization-normal windows;
    tails and thresholds remain independently fitted on reference and outer
    calibration windows, respectively.
    """

    def __init__(
        self,
        dimensions: int,
        target_index: int,
        target_fpr: float,
        mode: str = "global",
        reliability_strata: int = 1,
        min_reference_per_stratum: int = 8,
    ) -> None:
        if dimensions < 2 or not 0 <= target_index < dimensions:
            raise ValueError("invalid B1 dimensions or target_index.")
        if not 0.0 < target_fpr < 1.0:
            raise ValueError("target_fpr must be between zero and one.")
        if mode not in {"global", "input_energy_stratified"}:
            raise ValueError(f"Unsupported reliability calibration mode: {mode!r}")
        if reliability_strata < 1:
            raise ValueError("reliability_strata must be positive.")
        if mode == "global" and reliability_strata != 1:
            raise ValueError("global calibration must use exactly one stratum.")
        if mode == "input_energy_stratified" and reliability_strata < 2:
            raise ValueError("input_energy_stratified requires at least two strata.")
        self.dimensions = int(dimensions)
        self.target_index = int(target_index)
        self.target_fpr = float(target_fpr)
        self.mode = str(mode)
        self.reliability_strata = int(reliability_strata)
        self.min_reference_per_stratum = int(min_reference_per_stratum)
        self.boundaries_: Dict[str, np.ndarray] = {}
        self.tails_: Dict[str, Dict[int, EmpiricalUpperTail]] = {}
        self.thresholds_: Dict[str, Dict[int, float]] = {}
        self.reference_counts_: Dict[str, Dict[int, int]] = {}
        self.outer_counts_: Dict[str, Dict[int, int]] = {}

    def _validate_windows(self, windows: np.ndarray) -> np.ndarray:
        array = np.asarray(windows, dtype=np.float64)
        if array.ndim != 3 or array.shape[-1] != self.dimensions or array.shape[1] < 2:
            raise ValueError("windows must have shape [samples, history_plus_one, dimensions].")
        if len(array) == 0 or not np.isfinite(array).all():
            raise ValueError("windows must be non-empty and finite.")
        return array

    def features(self, windows: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute scales using only the information visible to each component."""
        array = self._validate_windows(windows)
        temporal = array[:, :-1, self.target_index]
        drivers = np.concatenate(
            (array[:, :, : self.target_index], array[:, :, self.target_index + 1 :]),
            axis=-1,
        )
        # Local innovation scales are computable from the exact evidence that
        # each repair head sees. They do not inspect target[t].
        temporal_energy = np.sqrt(
            np.mean(np.square(np.diff(temporal, axis=1)), axis=1) + 1e-6
        )
        cross_energy = np.sqrt(
            np.mean(np.square(np.diff(drivers, axis=1)), axis=(1, 2)) + 1e-6
        )
        return {
            "temporal_residual": temporal_energy,
            "cross_residual": cross_energy,
            "disagreement": np.sqrt(
                np.square(temporal_energy) + np.square(cross_energy)
            ),
        }

    def _strata_for_features(
        self, features: Mapping[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if not self.boundaries_:
            raise RuntimeError("Reliability calibration has not fitted boundaries.")
        return {
            component: np.searchsorted(
                self.boundaries_[component], np.asarray(values, dtype=np.float64), side="right"
            ).astype(np.int64)
            for component, values in features.items()
        }

    def _fit_boundaries(self, optimization_windows: np.ndarray) -> None:
        features = self.features(optimization_windows)
        if self.mode == "global":
            self.boundaries_ = {
                component: np.empty(0, dtype=np.float64)
                for component in SCORE_COMPONENTS
            }
            return
        quantiles = np.arange(1, self.reliability_strata, dtype=np.float64)
        quantiles /= float(self.reliability_strata)
        self.boundaries_ = {
            component: np.quantile(values, quantiles, method="linear")
            for component, values in features.items()
        }

    @staticmethod
    def _raw_component_scores(
        raw_scores: Mapping[str, np.ndarray], component: str, expected_length: int
    ) -> np.ndarray:
        if component not in raw_scores:
            raise ValueError(f"Missing raw B score component: {component}")
        values = np.asarray(raw_scores[component], dtype=np.float64).reshape(-1)
        if len(values) != expected_length or not np.isfinite(values).all():
            raise ValueError(f"Invalid raw B score component: {component}")
        return values

    def fit(
        self,
        optimization_windows: np.ndarray,
        reference_windows: np.ndarray,
        reference_raw_scores: Mapping[str, np.ndarray],
        outer_windows: np.ndarray,
        outer_raw_scores: Mapping[str, np.ndarray],
    ) -> "EvidenceReliabilityCalibration":
        self._fit_boundaries(optimization_windows)
        reference = self._validate_windows(reference_windows)
        outer = self._validate_windows(outer_windows)
        reference_strata = self._strata_for_features(self.features(reference))
        self.tails_ = {}
        self.reference_counts_ = {}
        for component in SCORE_COMPONENTS:
            raw = self._raw_component_scores(reference_raw_scores, component, len(reference))
            tails: Dict[int, EmpiricalUpperTail] = {}
            counts: Dict[int, int] = {}
            for stratum in range(self.reliability_strata):
                mask = reference_strata[component] == stratum
                count = int(mask.sum())
                if count < self.min_reference_per_stratum:
                    raise ValueError(
                        f"Reference split has only {count} {component} samples in reliability "
                        f"stratum {stratum}; need at least {self.min_reference_per_stratum}."
                    )
                tails[stratum] = EmpiricalUpperTail().fit(raw[mask])
                counts[stratum] = count
            self.tails_[component] = tails
            self.reference_counts_[component] = counts
        outer_scored = self.transform(outer, outer_raw_scores)
        self.thresholds_ = {}
        self.outer_counts_ = {}
        for component in SCORE_COMPONENTS:
            tail_component = f"{component}_tail"
            self.thresholds_[component] = {
                0: conformal_upper_threshold(
                    outer_scored[tail_component], self.target_fpr
                )
            }
            self.outer_counts_[component] = {0: int(len(outer))}
        return self

    def transform(
        self, windows: np.ndarray, raw_scores: Mapping[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if not self.tails_:
            raise RuntimeError("Reliability calibration must be fitted before transform.")
        array = self._validate_windows(windows)
        result = {
            name: np.asarray(values, dtype=np.float64).copy()
            for name, values in raw_scores.items()
        }
        features = self.features(array)
        strata = self._strata_for_features(features)
        for component in SCORE_COMPONENTS:
            raw = self._raw_component_scores(result, component, len(array))
            tail = np.empty(len(array), dtype=np.float64)
            for stratum, calibrator in self.tails_[component].items():
                mask = strata[component] == stratum
                tail[mask] = calibrator.transform(raw[mask])
            result[f"{component}_tail"] = tail
            result[f"{component}_reliability_feature"] = features[component]
            result[f"{component}_reliability_stratum"] = strata[component]
        return result

    def threshold_array(
        self, scored: Mapping[str, np.ndarray], tail_component: str
    ) -> np.ndarray:
        if not tail_component.endswith("_tail"):
            raise ValueError("tail_component must end in _tail.")
        component = tail_component[: -len("_tail")]
        if component not in self.thresholds_:
            raise RuntimeError("Reliability calibration thresholds are not fitted.")
        strata = np.asarray(
            scored[f"{component}_reliability_stratum"], dtype=np.int64
        ).reshape(-1)
        return np.full(
            len(strata), self.thresholds_[component][0], dtype=np.float64
        )

    def exceeds(self, scored: Mapping[str, np.ndarray], tail_component: str) -> np.ndarray:
        values = np.asarray(scored[tail_component], dtype=np.float64).reshape(-1)
        return values > self.threshold_array(scored, tail_component)

    def row_exceeds(self, row: Mapping[str, Any], tail_component: str) -> bool:
        if not tail_component.endswith("_tail"):
            raise ValueError("tail_component must end in _tail.")
        component = tail_component[: -len("_tail")]
        return bool(float(row[tail_component]) > self.thresholds_[component][0])

    def metadata(self) -> Dict[str, Any]:
        if not self.thresholds_:
            raise RuntimeError("Reliability calibration is not fitted.")
        return {
            "mode": self.mode,
            "reliability_strata": self.reliability_strata,
            "feature": "allowed_input_adjacent_innovation_rms",
            "boundary_fit_split": "optimization_normal",
            "tail_fit_split": "reference_normal",
            "threshold_fit_split": "outer_calibration_normal_global_after_stratified_tail",
            "threshold_scope": "global",
            "target_terminal_used_by_features": False,
            "oracle_regime_used": False,
            "boundaries": {
                component: values.astype(float).tolist()
                for component, values in self.boundaries_.items()
            },
            "reference_counts": {
                component: {str(key): value for key, value in counts.items()}
                for component, counts in self.reference_counts_.items()
            },
            "outer_counts": {
                component: {str(key): value for key, value in counts.items()}
                for component, counts in self.outer_counts_.items()
            },
            "thresholds": {
                component: {str(key): value for key, value in thresholds.items()}
                for component, thresholds in self.thresholds_.items()
            },
        }
