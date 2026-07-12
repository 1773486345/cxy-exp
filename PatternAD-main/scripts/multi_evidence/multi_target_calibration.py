"""Target-specific B2a reliability calibration without matrix flattening."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from scripts.multi_evidence.reliability_calibration import (
    SCORE_COMPONENTS,
    EvidenceReliabilityCalibration,
)


class MultiTargetEvidenceReliabilityCalibration:
    """Own one independent ECRC object for every designated target channel."""

    def __init__(
        self,
        dimensions: int,
        target_indices: Sequence[int],
        target_fpr: float,
        mode: str = "input_energy_stratified",
        reliability_strata: int = 3,
        min_reference_per_stratum: int = 50,
    ) -> None:
        targets = tuple(int(target) for target in target_indices)
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("target_indices must be a non-empty unique sequence.")
        self.dimensions = int(dimensions)
        self.target_indices = targets
        self.target_fpr = float(target_fpr)
        self.mode = str(mode)
        self.reliability_strata = int(reliability_strata)
        self.min_reference_per_stratum = int(min_reference_per_stratum)
        self.calibrators: Dict[int, EvidenceReliabilityCalibration] = {}

    @property
    def target_count(self) -> int:
        return len(self.target_indices)

    def _validate_matrix_scores(
        self, scores: Mapping[str, np.ndarray], expected_rows: int
    ) -> None:
        for component in SCORE_COMPONENTS:
            if component not in scores:
                raise ValueError(f"Missing B2 raw component: {component}")
            array = np.asarray(scores[component], dtype=np.float64)
            if array.shape != (expected_rows, self.target_count):
                raise ValueError(
                    f"{component} must have shape [{expected_rows}, {self.target_count}]."
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{component} contains non-finite values.")

    @staticmethod
    def _component_column(
        scores: Mapping[str, np.ndarray], position: int
    ) -> Dict[str, np.ndarray]:
        return {
            component: np.asarray(scores[component], dtype=np.float64)[:, position]
            for component in SCORE_COMPONENTS
        }

    def fit(
        self,
        optimization_windows: np.ndarray,
        reference_windows: np.ndarray,
        reference_raw_scores: Mapping[str, np.ndarray],
        outer_windows: np.ndarray,
        outer_raw_scores: Mapping[str, np.ndarray],
    ) -> "MultiTargetEvidenceReliabilityCalibration":
        self._validate_matrix_scores(reference_raw_scores, len(reference_windows))
        self._validate_matrix_scores(outer_raw_scores, len(outer_windows))
        self.calibrators = {}
        for position, target_index in enumerate(self.target_indices):
            self.calibrators[target_index] = EvidenceReliabilityCalibration(
                dimensions=self.dimensions,
                target_index=target_index,
                target_fpr=self.target_fpr,
                mode=self.mode,
                reliability_strata=self.reliability_strata,
                min_reference_per_stratum=self.min_reference_per_stratum,
            ).fit(
                optimization_windows,
                reference_windows,
                self._component_column(reference_raw_scores, position),
                outer_windows,
                self._component_column(outer_raw_scores, position),
            )
        return self

    def _require_fitted(self) -> None:
        if set(self.calibrators) != set(self.target_indices):
            raise RuntimeError("Multi-target reliability calibration is not fitted.")

    def transform(
        self, windows: np.ndarray, raw_scores: Mapping[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        self._require_fitted()
        self._validate_matrix_scores(raw_scores, len(windows))
        result = {
            name: np.asarray(values, dtype=np.float64).copy()
            for name, values in raw_scores.items()
        }
        transformed_columns: Dict[str, list[np.ndarray]] = {}
        for position, target_index in enumerate(self.target_indices):
            transformed = self.calibrators[target_index].transform(
                windows, self._component_column(raw_scores, position)
            )
            for component in SCORE_COMPONENTS:
                for suffix in ("_tail", "_reliability_feature", "_reliability_stratum"):
                    name = f"{component}{suffix}"
                    transformed_columns.setdefault(name, []).append(
                        np.asarray(transformed[name], dtype=np.float64)
                    )
        result.update(
            {
                name: np.stack(columns, axis=1)
                for name, columns in transformed_columns.items()
            }
        )
        return result

    def exceeds(
        self, scored: Mapping[str, np.ndarray], tail_component: str
    ) -> np.ndarray:
        self._require_fitted()
        if not tail_component.endswith("_tail"):
            raise ValueError("tail_component must end in _tail.")
        raw_component = tail_component[: -len("_tail")]
        values = np.asarray(scored[tail_component], dtype=np.float64)
        strata = np.asarray(
            scored[f"{raw_component}_reliability_stratum"], dtype=np.int64
        )
        if values.shape != strata.shape or values.shape[1] != self.target_count:
            raise ValueError("B2 tail and stratum matrices have incompatible shapes.")
        output = np.empty(values.shape, dtype=bool)
        for position, target_index in enumerate(self.target_indices):
            threshold = self.calibrators[target_index].threshold_array(
                {
                    tail_component: values[:, position],
                    f"{raw_component}_reliability_stratum": strata[:, position],
                },
                tail_component,
            )
            output[:, position] = values[:, position] > threshold
        return output

    def metadata(self) -> Dict[str, Any]:
        self._require_fitted()
        return {
            "target_indices": list(self.target_indices),
            "no_target_flattening": True,
            "calibration_by_target": {
                str(target_index): self.calibrators[target_index].metadata()
                for target_index in self.target_indices
            },
        }
