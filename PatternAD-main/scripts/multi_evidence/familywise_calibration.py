"""B2c family-wise, reliability-stratified evidence calibration.

This module intentionally leaves B1's global-after-stratified ECRC unchanged.
It supplies a separately named calibration object for B2c, where the normal
alert rule inspects the union of cross and disagreement evidence.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from scripts.multi_evidence.reliability_calibration import (
    SCORE_COMPONENTS,
    EvidenceReliabilityCalibration,
    EmpiricalUpperTail,
    conformal_upper_threshold,
)


DEFAULT_COMPONENT_ALPHAS = {
    "temporal_residual": 0.05,
    "cross_residual": 0.025,
    "disagreement": 0.025,
}


class FamilyWiseEvidenceReliabilityCalibration(EvidenceReliabilityCalibration):
    """Use independent outer-normal conformal cutoffs per component and bin.

    Temporal evidence remains at alpha=0.05. Cross and disagreement evidence
    each receive alpha=0.025, a pre-declared Bonferroni allocation for their
    `cross OR disagreement` normal-alert family. The raw scores and reference
    tails are unchanged from ECRC; only the outer-normal cutoff scope changes.
    """

    def __init__(
        self,
        dimensions: int,
        target_index: int,
        component_alphas: Mapping[str, float] | None = None,
        mode: str = "input_energy_stratified",
        reliability_strata: int = 3,
        min_reference_per_stratum: int = 50,
        min_outer_per_stratum: int = 50,
    ) -> None:
        alphas = dict(DEFAULT_COMPONENT_ALPHAS if component_alphas is None else component_alphas)
        if set(alphas) != set(SCORE_COMPONENTS):
            raise ValueError("component_alphas must define every evidence component exactly once.")
        if any(not 0.0 < float(alpha) < 1.0 for alpha in alphas.values()):
            raise ValueError("Every B2c component alpha must be strictly between zero and one.")
        if min_outer_per_stratum < 1:
            raise ValueError("min_outer_per_stratum must be positive.")
        super().__init__(
            dimensions=dimensions,
            target_index=target_index,
            target_fpr=max(float(alpha) for alpha in alphas.values()),
            mode=mode,
            reliability_strata=reliability_strata,
            min_reference_per_stratum=min_reference_per_stratum,
        )
        self.component_alphas = {
            component: float(alphas[component]) for component in SCORE_COMPONENTS
        }
        self.min_outer_per_stratum = int(min_outer_per_stratum)

    def fit(
        self,
        optimization_windows: np.ndarray,
        reference_windows: np.ndarray,
        reference_raw_scores: Mapping[str, np.ndarray],
        outer_windows: np.ndarray,
        outer_raw_scores: Mapping[str, np.ndarray],
    ) -> "FamilyWiseEvidenceReliabilityCalibration":
        self._fit_boundaries(optimization_windows)
        reference = self._validate_windows(reference_windows)
        outer = self._validate_windows(outer_windows)
        reference_strata = self._strata_for_features(self.features(reference))
        self.tails_ = {}
        self.reference_counts_ = {}
        for component in SCORE_COMPONENTS:
            raw = self._raw_component_scores(
                reference_raw_scores, component, len(reference)
            )
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
            strata = np.asarray(
                outer_scored[f"{component}_reliability_stratum"], dtype=np.int64
            )
            thresholds: Dict[int, float] = {}
            counts: Dict[int, int] = {}
            for stratum in range(self.reliability_strata):
                mask = strata == stratum
                count = int(mask.sum())
                if count < self.min_outer_per_stratum:
                    raise ValueError(
                        f"Outer calibration has only {count} {component} samples in reliability "
                        f"stratum {stratum}; need at least {self.min_outer_per_stratum}."
                    )
                thresholds[stratum] = conformal_upper_threshold(
                    outer_scored[tail_component][mask], self.component_alphas[component]
                )
                counts[stratum] = count
            self.thresholds_[component] = thresholds
            self.outer_counts_[component] = counts
        return self

    def threshold_array(
        self, scored: Mapping[str, np.ndarray], tail_component: str
    ) -> np.ndarray:
        if not tail_component.endswith("_tail"):
            raise ValueError("tail_component must end in _tail.")
        component = tail_component[: -len("_tail")]
        if component not in self.thresholds_:
            raise RuntimeError("B2c calibration is not fitted.")
        strata = np.asarray(
            scored[f"{component}_reliability_stratum"], dtype=np.int64
        ).reshape(-1)
        if np.any(strata < 0) or np.any(strata >= self.reliability_strata):
            raise ValueError("B2c reliability stratum is outside the fitted range.")
        return np.asarray(
            [self.thresholds_[component][int(stratum)] for stratum in strata],
            dtype=np.float64,
        )

    def row_exceeds(self, row: Mapping[str, Any], tail_component: str) -> bool:
        component = tail_component[: -len("_tail")]
        stratum = int(row[f"{component}_reliability_stratum"])
        return bool(float(row[tail_component]) > self.thresholds_[component][stratum])

    def metadata(self) -> Dict[str, Any]:
        if not self.thresholds_:
            raise RuntimeError("B2c calibration is not fitted.")
        return {
            "mode": self.mode,
            "reliability_strata": self.reliability_strata,
            "feature": "allowed_input_adjacent_innovation_rms",
            "boundary_fit_split": "optimization_normal",
            "tail_fit_split": "reference_normal",
            "threshold_fit_split": "outer_calibration_normal_per_reliability_stratum",
            "threshold_scope": "per_reliability_stratum",
            "minimum_outer_per_stratum": self.min_outer_per_stratum,
            "component_alphas": self.component_alphas,
            "familywise_control": {
                "family": ["cross_residual", "disagreement"],
                "method": "bonferroni",
                "family_alpha": 0.05,
                "per_component_alpha": 0.025,
            },
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
