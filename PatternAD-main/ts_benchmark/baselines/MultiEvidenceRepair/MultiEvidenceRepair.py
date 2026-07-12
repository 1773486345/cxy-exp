"""Auditable dual-evidence repair model for the Direction B0 experiment.

This module deliberately does not inherit from PatternAD.  Its two branches
receive different tensors, use disjoint parameters, and expose their scores
separately.  It is an experiment harness, not a benchmark registration.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _as_float_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def terminal_windows(values: np.ndarray, history_length: int) -> np.ndarray:
    """Return one terminal-point window per valid timestamp.

    The row at index ``k`` represents original terminal point ``k + H``.  No
    overlap aggregation is performed anywhere in B0.
    """
    series = _as_float_array(values, "values")
    if series.ndim != 2:
        raise ValueError("values must have shape [time, dimensions].")
    if history_length < 1:
        raise ValueError("history_length must be positive.")
    if len(series) <= history_length:
        raise ValueError("values must contain more rows than history_length.")
    return np.stack(
        [series[end - history_length : end + 1] for end in range(history_length, len(series))],
        axis=0,
    )


class ChannelStandardizer:
    """Per-channel normalizer fitted only on the optimization-normal split."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray) -> "ChannelStandardizer":
        array = _as_float_array(values, "standardizer values")
        if array.ndim != 2 or len(array) < 2:
            raise ValueError("standardizer values must have shape [time, dimensions].")
        self.mean_ = array.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.std_ = array.std(axis=0, dtype=np.float64).astype(np.float32)
        self.std_ = np.maximum(self.std_, np.float32(1e-6))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ChannelStandardizer must be fitted before transform.")
        array = _as_float_array(values, "values")
        if array.shape[-1] != len(self.mean_):
            raise ValueError("values have a different channel count from the fitted normalizer.")
        return ((array - self.mean_) / self.std_).astype(np.float32, copy=False)

    def metadata(self) -> Dict[str, Any]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ChannelStandardizer is not fitted.")
        return {"mean": self.mean_.astype(float).tolist(), "std": self.std_.astype(float).tolist()}


class EvidenceRepairNet(nn.Module):
    """Two non-sharing GRU repair branches with enforced information contracts."""

    def __init__(
        self,
        dimensions: int,
        target_index: int,
        d_model: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dimensions < 2:
            raise ValueError("EvidenceRepairNet requires at least two channels.")
        if not 0 <= target_index < dimensions:
            raise ValueError("target_index is outside dimensions.")
        if d_model < 1:
            raise ValueError("d_model must be positive.")
        if dropout != 0.0:
            raise ValueError("B0 forbids dropout so evidence paths remain deterministic.")
        self.dimensions = int(dimensions)
        self.target_index = int(target_index)
        self.d_model = int(d_model)
        self.temporal_gru = nn.GRU(input_size=1, hidden_size=d_model, batch_first=True)
        self.cross_gru = nn.GRU(
            input_size=dimensions - 1, hidden_size=d_model, batch_first=True
        )
        self.temporal_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        self.cross_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def evidence_inputs(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract B0 inputs without allowing a branch to see forbidden cells."""
        if windows.ndim != 3 or windows.shape[-1] != self.dimensions:
            raise ValueError(
                "windows must have shape [batch, history_plus_one, dimensions]."
            )
        if windows.shape[1] < 2:
            raise ValueError("windows must contain a past point and a terminal point.")
        target = windows[:, -1, self.target_index]
        temporal = windows[:, :-1, self.target_index : self.target_index + 1]
        before = windows[:, :, : self.target_index]
        after = windows[:, :, self.target_index + 1 :]
        cross = torch.cat((before, after), dim=-1)
        return temporal, cross, target

    def forward(self, windows: torch.Tensor) -> Dict[str, torch.Tensor]:
        temporal, cross, target = self.evidence_inputs(windows)
        temporal_hidden, _ = self.temporal_gru(temporal)
        cross_hidden, _ = self.cross_gru(cross)
        mu_temporal = self.temporal_head(temporal_hidden[:, -1]).squeeze(-1)
        mu_cross = self.cross_head(cross_hidden[:, -1]).squeeze(-1)
        return {
            "target": target,
            "mu_temporal": mu_temporal,
            "mu_cross": mu_cross,
        }

    def branch_parameter_ids(self) -> Dict[str, set[int]]:
        temporal = {
            id(parameter)
            for module in (self.temporal_gru, self.temporal_head)
            for parameter in module.parameters()
        }
        cross = {
            id(parameter)
            for module in (self.cross_gru, self.cross_head)
            for parameter in module.parameters()
        }
        return {"temporal": temporal, "cross": cross}


class EmpiricalUpperTail:
    """Reference-only empirical upper-tail surprisal map."""

    def __init__(self) -> None:
        self._reference: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray) -> "EmpiricalUpperTail":
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(array) < 2 or not np.isfinite(array).all():
            raise ValueError("tail reference must contain at least two finite values.")
        self._reference = np.sort(array.copy())
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self._reference is None:
            raise RuntimeError("EmpiricalUpperTail must be fitted before transform.")
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("tail values must be finite.")
        lower_count = np.searchsorted(self._reference, array, side="left")
        survival = (len(self._reference) - lower_count + 1.0) / (
            len(self._reference) + 1.0
        )
        return -np.log(survival)

    def metadata(self) -> Dict[str, Any]:
        if self._reference is None:
            raise RuntimeError("EmpiricalUpperTail is not fitted.")
        return {
            "count": int(len(self._reference)),
            "minimum": float(self._reference[0]),
            "maximum": float(self._reference[-1]),
        }


class MultiEvidenceRepair:
    """Fit and score B0's strictly separated temporal and cross repair paths."""

    SCORE_COMPONENTS: Sequence[str] = (
        "temporal_residual",
        "cross_residual",
        "disagreement",
    )

    def __init__(
        self,
        dimensions: int,
        target_index: int,
        d_model: int = 32,
        dropout: float = 0.0,
        learning_rate: float = 3e-3,
        epochs: int = 20,
        patience: int = 4,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if epochs < 1 or patience < 1 or batch_size < 1:
            raise ValueError("epochs, patience, and batch_size must be positive.")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
        self.net = EvidenceRepairNet(dimensions, target_index, d_model, dropout).to(self.device)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.tails: Dict[str, EmpiricalUpperTail] = {}
        self.fit_metadata_: Dict[str, Any] = {}

    @property
    def dimensions(self) -> int:
        return self.net.dimensions

    @property
    def target_index(self) -> int:
        return self.net.target_index

    @staticmethod
    def _validate_windows(windows: np.ndarray, dimensions: int, name: str) -> np.ndarray:
        array = _as_float_array(windows, name)
        if array.ndim != 3 or array.shape[-1] != dimensions or array.shape[1] < 2:
            raise ValueError(
                f"{name} must have shape [samples, history_plus_one, {dimensions}]."
            )
        if len(array) == 0:
            raise ValueError(f"{name} must contain at least one window.")
        return array

    def _prediction_loss(self, prediction: Mapping[str, torch.Tensor]) -> torch.Tensor:
        temporal = F.smooth_l1_loss(prediction["mu_temporal"], prediction["target"])
        cross = F.smooth_l1_loss(prediction["mu_cross"], prediction["target"])
        return temporal + cross

    def _loss_on_windows(self, windows: np.ndarray) -> float:
        self.net.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(windows), self.batch_size):
                batch = torch.as_tensor(windows[start : start + self.batch_size], device=self.device)
                total += float(self._prediction_loss(self.net(batch)).item()) * len(batch)
                count += len(batch)
        return total / max(count, 1)

    def fit(
        self,
        optimization_windows: np.ndarray,
        validation_windows: np.ndarray,
        reference_windows: np.ndarray,
        seed: int,
    ) -> "MultiEvidenceRepair":
        optimization = self._validate_windows(
            optimization_windows, self.dimensions, "optimization_windows"
        )
        validation = self._validate_windows(
            validation_windows, self.dimensions, "validation_windows"
        )
        reference = self._validate_windows(
            reference_windows, self.dimensions, "reference_windows"
        )
        history = optimization.shape[1]
        if validation.shape[1] != history or reference.shape[1] != history:
            raise ValueError("all B0 splits must use the same history length.")
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        best_loss = math.inf
        best_epoch = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        stale_epochs = 0
        history_rows = []
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            ordering = torch.randperm(len(optimization), generator=generator).numpy()
            epoch_loss = 0.0
            seen = 0
            for start in range(0, len(ordering), self.batch_size):
                indices = ordering[start : start + self.batch_size]
                batch = torch.as_tensor(optimization[indices], device=self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = self._prediction_loss(self.net(batch))
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(indices)
                seen += len(indices)
            validation_loss = self._loss_on_windows(validation)
            history_rows.append(
                {
                    "epoch": epoch,
                    "optimization_loss": epoch_loss / max(seen, 1),
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1e-9:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.net.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("B0 training did not produce a valid checkpoint.")
        self.net.load_state_dict(best_state)
        reference_scores = self.score_windows(reference, include_tails=False)
        self.tails = {
            component: EmpiricalUpperTail().fit(reference_scores[component])
            for component in self.SCORE_COMPONENTS
        }
        parameter_sets = self.net.branch_parameter_ids()
        self.fit_metadata_ = {
            "optimization_windows": int(len(optimization)),
            "validation_windows": int(len(validation)),
            "reference_windows": int(len(reference)),
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_loss),
            "training_history": history_rows,
            "parameter_counts": {
                "temporal": int(sum(parameter.numel() for parameter in self.net.temporal_gru.parameters()) + sum(parameter.numel() for parameter in self.net.temporal_head.parameters())),
                "cross": int(sum(parameter.numel() for parameter in self.net.cross_gru.parameters()) + sum(parameter.numel() for parameter in self.net.cross_head.parameters())),
            },
            "parameter_sets_disjoint": not bool(
                parameter_sets["temporal"] & parameter_sets["cross"]
            ),
            "reference_tail_metadata": {
                component: tail.metadata() for component, tail in self.tails.items()
            },
        }
        return self

    def score_windows(
        self, windows: np.ndarray, include_tails: bool = True
    ) -> Dict[str, np.ndarray]:
        array = self._validate_windows(windows, self.dimensions, "windows")
        outputs: Dict[str, list[np.ndarray]] = {
            "target": [],
            "mu_temporal": [],
            "mu_cross": [],
        }
        self.net.eval()
        with torch.no_grad():
            for start in range(0, len(array), self.batch_size):
                batch = torch.as_tensor(array[start : start + self.batch_size], device=self.device)
                prediction = self.net(batch)
                for name in outputs:
                    outputs[name].append(prediction[name].detach().cpu().numpy())
        result = {name: np.concatenate(chunks).astype(np.float64) for name, chunks in outputs.items()}
        result["temporal_residual"] = np.square(result["target"] - result["mu_temporal"])
        result["cross_residual"] = np.square(result["target"] - result["mu_cross"])
        result["disagreement"] = np.square(result["mu_temporal"] - result["mu_cross"])
        if include_tails:
            if set(self.tails) != set(self.SCORE_COMPONENTS):
                raise RuntimeError("B0 tails are not fitted. Call fit before score_windows.")
            for component in self.SCORE_COMPONENTS:
                result[f"{component}_tail"] = self.tails[component].transform(
                    result[component]
                )
        return result

    def evidence_isolation_report(self, windows: np.ndarray) -> Dict[str, float | bool]:
        """Numerically verify the branch input contracts on a supplied window."""
        array = self._validate_windows(windows, self.dimensions, "windows")[:1].copy()
        self.net.eval()

        def predict(value: np.ndarray) -> Dict[str, np.ndarray]:
            return self.score_windows(value, include_tails=False)

        base = predict(array)
        changed_drivers = array.copy()
        changed_drivers[:, :, [index for index in range(self.dimensions) if index != self.target_index]] += 3.0
        changed_target_terminal = array.copy()
        changed_target_terminal[:, -1, self.target_index] += 3.0
        changed_target_all = array.copy()
        changed_target_all[:, :, self.target_index] += 3.0
        temporal_driver_delta = float(
            np.max(np.abs(base["mu_temporal"] - predict(changed_drivers)["mu_temporal"]))
        )
        temporal_terminal_delta = float(
            np.max(
                np.abs(
                    base["mu_temporal"]
                    - predict(changed_target_terminal)["mu_temporal"]
                )
            )
        )
        cross_target_delta = float(
            np.max(np.abs(base["mu_cross"] - predict(changed_target_all)["mu_cross"]))
        )
        parameter_sets = self.net.branch_parameter_ids()
        return {
            "temporal_driver_delta": temporal_driver_delta,
            "temporal_terminal_target_delta": temporal_terminal_delta,
            "cross_target_column_delta": cross_target_delta,
            "parameter_sets_disjoint": not bool(
                parameter_sets["temporal"] & parameter_sets["cross"]
            ),
        }
