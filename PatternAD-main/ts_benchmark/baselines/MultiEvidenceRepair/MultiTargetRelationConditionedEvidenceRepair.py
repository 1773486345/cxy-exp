"""Independent per-target wrapper for B3a relation-conditioned repair."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from ts_benchmark.baselines.MultiEvidenceRepair.RelationConditionedEvidenceRepair import (
    RelationConditionedEvidenceRepair,
)


class MultiTargetRelationConditionedEvidenceRepair:
    """Own a fully independent B3a relation-conditioned repair pair per target."""

    def __init__(
        self,
        dimensions: int,
        target_indices: Sequence[int],
        temporal_d_model: int = 32,
        cross_d_model: int = 22,
        cross_head_d_model: int = 20,
        dropout: float = 0.0,
        learning_rate: float = 3e-3,
        epochs: int = 20,
        patience: int = 4,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
    ) -> None:
        if dimensions < 2:
            raise ValueError("B3a requires at least two channels.")
        targets = tuple(int(target) for target in target_indices)
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("target_indices must be a non-empty unique sequence.")
        if any(target < 0 or target >= dimensions for target in targets):
            raise ValueError("target_indices contains a channel outside dimensions.")
        self.dimensions = int(dimensions)
        self.target_indices = targets
        self.device = torch.device(device)
        self.model_kwargs = {
            "temporal_d_model": int(temporal_d_model),
            "cross_d_model": int(cross_d_model),
            "cross_head_d_model": int(cross_head_d_model),
            "dropout": float(dropout),
            "learning_rate": float(learning_rate),
            "epochs": int(epochs),
            "patience": int(patience),
            "batch_size": int(batch_size),
            "device": self.device,
        }
        self.models: Dict[int, RelationConditionedEvidenceRepair] = {}
        self.fit_metadata_: Dict[str, Any] = {}

    def _build_model(
        self, target_index: int, seed: int
    ) -> RelationConditionedEvidenceRepair:
        cuda_devices = []
        if self.device.type == "cuda" and self.device.index is not None:
            cuda_devices = [self.device.index]
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            return RelationConditionedEvidenceRepair(
                dimensions=self.dimensions,
                target_index=target_index,
                **self.model_kwargs,
            )

    def fit(
        self,
        optimization_windows: np.ndarray,
        validation_windows: np.ndarray,
        reference_windows: np.ndarray,
        seed: int,
        frozen_temporal_states: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    ) -> "MultiTargetRelationConditionedEvidenceRepair":
        self.models = {}
        metadata: Dict[str, Any] = {}
        if frozen_temporal_states is not None and set(frozen_temporal_states) != {
            str(target) for target in self.target_indices
        }:
            raise ValueError("Frozen temporal checkpoint must provide exactly every target.")
        for target_index in self.target_indices:
            target_seed = int(seed) + 100_003 * (target_index + 1)
            model = self._build_model(target_index, target_seed)
            frozen_hash = None
            if frozen_temporal_states is not None:
                frozen_hash = model.load_and_freeze_temporal_state(
                    frozen_temporal_states[str(target_index)]
                )
            self.models[target_index] = model.fit(
                optimization_windows,
                validation_windows,
                reference_windows,
                target_seed,
            )
            metadata[str(target_index)] = {
                "seed": target_seed,
                "fit": model.fit_metadata_,
                "frozen_temporal_state_sha256": frozen_hash,
            }
        self.fit_metadata_ = {
            "target_indices": list(self.target_indices),
            "targets": metadata,
            **self.parameter_isolation_report(),
        }
        return self

    def _require_fitted(self) -> None:
        if set(self.models) != set(self.target_indices):
            raise RuntimeError("B3a models must be fitted before scoring.")

    def score_windows(
        self, windows: np.ndarray, include_tails: bool = False
    ) -> Dict[str, np.ndarray]:
        self._require_fitted()
        collected: Dict[str, list[np.ndarray]] = {}
        for target_index in self.target_indices:
            scores = self.models[target_index].score_windows(
                windows, include_tails=include_tails
            )
            for name, values in scores.items():
                collected.setdefault(name, []).append(np.asarray(values, dtype=np.float64))
        return {
            name: np.stack(values, axis=1) for name, values in collected.items()
        }

    def parameter_isolation_report(self) -> Dict[str, Any]:
        parameter_sets: Dict[str, set[int]] = {}
        for target_index, model in self.models.items():
            branch_sets = model.net.branch_parameter_ids()
            for branch, identifiers in branch_sets.items():
                parameter_sets[f"target_{target_index}_{branch}"] = identifiers
        seen: set[int] = set()
        overlaps = []
        for name, identifiers in parameter_sets.items():
            overlap = seen & identifiers
            if overlap:
                overlaps.append({"branch": name, "overlap_count": len(overlap)})
            seen.update(identifiers)
        return {
            "all_branch_parameter_sets_disjoint": not overlaps,
            "branch_parameter_counts": {
                name: len(identifiers) for name, identifiers in parameter_sets.items()
            },
            "parameter_overlaps": overlaps,
        }

    def evidence_isolation_report(self, windows: np.ndarray) -> Dict[str, Any]:
        self._require_fitted()
        reports = {
            str(target_index): self.models[target_index].evidence_isolation_report(windows)
            for target_index in self.target_indices
        }
        return {"per_target": reports, **self.parameter_isolation_report()}

    def state_dict(self) -> Dict[str, Mapping[str, torch.Tensor]]:
        self._require_fitted()
        return {
            str(target_index): {
                name: tensor.detach().cpu()
                for name, tensor in model.net.state_dict().items()
            }
            for target_index, model in self.models.items()
        }

    def temporal_state_sha256(self) -> Dict[str, str]:
        """Return the final temporal checkpoint fingerprint for every target."""
        self._require_fitted()
        return {
            str(target_index): model.temporal_state_sha256()
            for target_index, model in self.models.items()
        }
