"""Terminal-blind relation-history conditioned repair for Direction B3a.

The cross path may observe the designated target only before the repaired
terminal. This lets it encode current target-driver compatibility without
leaking the value whose cross prediction is evaluated.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ts_benchmark.baselines.MultiEvidenceRepair.MultiEvidenceRepair import (
    EmpiricalUpperTail,
    _as_float_array,
)


class RelationConditionedEvidenceRepairNet(nn.Module):
    """Separate temporal, relation-history, and all-driver encoders."""

    def __init__(
        self,
        dimensions: int,
        target_index: int,
        temporal_d_model: int,
        cross_d_model: int,
        cross_head_d_model: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dimensions < 2:
            raise ValueError("B3a requires at least two channels.")
        if not 0 <= target_index < dimensions:
            raise ValueError("target_index is outside dimensions.")
        if temporal_d_model < 1 or cross_d_model < 1 or cross_head_d_model < 1:
            raise ValueError("B3a model widths must be positive.")
        if dropout != 0.0:
            raise ValueError("B3a forbids dropout so information checks are deterministic.")
        self.dimensions = int(dimensions)
        self.target_index = int(target_index)
        self.temporal_d_model = int(temporal_d_model)
        self.cross_d_model = int(cross_d_model)
        self.cross_head_d_model = int(cross_head_d_model)
        self.temporal_gru = nn.GRU(
            input_size=1, hidden_size=temporal_d_model, batch_first=True
        )
        # B3a must start its unchanged temporal branch from exactly the B2a-GC
        # state for the same target seed. The B2a-GC constructor initialized a
        # width-matched driver GRU between temporal_gru and temporal_head. Build
        # and discard that legacy module solely to consume the same RNG stream.
        # It is not registered, optimized, or used by B3a.
        nn.GRU(
            input_size=dimensions - 1,
            hidden_size=temporal_d_model,
            batch_first=True,
        )
        self.temporal_head = nn.Sequential(
            nn.Linear(temporal_d_model, temporal_d_model),
            nn.ReLU(),
            nn.Linear(temporal_d_model, 1),
        )
        self.relation_history_gru = nn.GRU(
            input_size=dimensions, hidden_size=cross_d_model, batch_first=True
        )
        self.driver_gru = nn.GRU(
            input_size=dimensions - 1, hidden_size=cross_d_model, batch_first=True
        )
        self.cross_head = nn.Sequential(
            nn.Linear(2 * cross_d_model, cross_head_d_model),
            nn.ReLU(),
            nn.Linear(cross_head_d_model, 1),
        )

    def evidence_inputs(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract B3a inputs while excluding the target terminal from cross state."""
        if windows.ndim != 3 or windows.shape[-1] != self.dimensions:
            raise ValueError(
                "windows must have shape [batch, history_plus_one, dimensions]."
            )
        if windows.shape[1] < 2:
            raise ValueError("windows must contain a past point and a terminal point.")
        target = windows[:, -1, self.target_index]
        temporal_history = windows[:, :-1, self.target_index : self.target_index + 1]
        # The full pre-terminal state is the observable relation evidence.
        relation_history = windows[:, :-1, :]
        drivers = torch.cat(
            (
                windows[:, :, : self.target_index],
                windows[:, :, self.target_index + 1 :],
            ),
            dim=-1,
        )
        return target, temporal_history, relation_history, drivers

    def forward(self, windows: torch.Tensor) -> Dict[str, torch.Tensor]:
        target, temporal_history, relation_history, drivers = self.evidence_inputs(windows)
        temporal_hidden, _ = self.temporal_gru(temporal_history)
        relation_hidden, _ = self.relation_history_gru(relation_history)
        driver_hidden, _ = self.driver_gru(drivers)
        mu_temporal = self.temporal_head(temporal_hidden[:, -1]).squeeze(-1)
        cross_state = torch.cat((relation_hidden[:, -1], driver_hidden[:, -1]), dim=-1)
        mu_cross = self.cross_head(cross_state).squeeze(-1)
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
            for module in (self.relation_history_gru, self.driver_gru, self.cross_head)
            for parameter in module.parameters()
        }
        return {"temporal": temporal, "cross": cross}


class RelationConditionedEvidenceRepair:
    """B3a scalar target repair with terminal-blind relation conditioning."""

    SCORE_COMPONENTS: Sequence[str] = (
        "temporal_residual",
        "cross_residual",
        "disagreement",
    )

    def __init__(
        self,
        dimensions: int,
        target_index: int,
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
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if epochs < 1 or patience < 1 or batch_size < 1:
            raise ValueError("epochs, patience, and batch_size must be positive.")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
        self.net = RelationConditionedEvidenceRepairNet(
            dimensions,
            target_index,
            temporal_d_model,
            cross_d_model,
            cross_head_d_model,
            dropout,
        ).to(self.device)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.tails: Dict[str, EmpiricalUpperTail] = {}
        self.fit_metadata_: Dict[str, Any] = {}

    def temporal_state_sha256(self) -> str:
        """Hash the temporal modules in a portable, tensor-content form."""
        digest = hashlib.sha256()
        for name, tensor in sorted(self.net.state_dict().items()):
            if not name.startswith(("temporal_gru.", "temporal_head.")):
                continue
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(repr(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def load_and_freeze_temporal_state(
        self, source_state: Mapping[str, torch.Tensor]
    ) -> str:
        """Load only compatible temporal tensors and make them immutable.

        B3a's relation-conditioned cross path is evaluated against a B2a-GC
        control.  Freezing the selected B2a-GC temporal modules prevents the
        modified cross path or its checkpoint rule from changing that control.
        """
        expected = {
            name: tensor
            for name, tensor in self.net.state_dict().items()
            if name.startswith(("temporal_gru.", "temporal_head."))
        }
        received = {
            name: tensor.detach().cpu()
            for name, tensor in source_state.items()
            if name.startswith(("temporal_gru.", "temporal_head."))
        }
        if set(received) != set(expected):
            missing = sorted(set(expected) - set(received))
            unexpected = sorted(set(received) - set(expected))
            raise ValueError(
                "Frozen temporal checkpoint has incompatible tensor names: "
                f"missing={missing}; unexpected={unexpected}."
            )
        for name, tensor in expected.items():
            source = received[name]
            if tuple(source.shape) != tuple(tensor.shape) or source.dtype != tensor.dtype:
                raise ValueError(
                    "Frozen temporal checkpoint tensor is incompatible: "
                    f"{name} source={tuple(source.shape)}/{source.dtype}; "
                    f"expected={tuple(tensor.shape)}/{tensor.dtype}."
                )
        temporal_gru = {
            name[len("temporal_gru.") :]: value
            for name, value in received.items()
            if name.startswith("temporal_gru.")
        }
        temporal_head = {
            name[len("temporal_head.") :]: value
            for name, value in received.items()
            if name.startswith("temporal_head.")
        }
        self.net.temporal_gru.load_state_dict(temporal_gru, strict=True)
        self.net.temporal_head.load_state_dict(temporal_head, strict=True)
        for module in (self.net.temporal_gru, self.net.temporal_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        return self.temporal_state_sha256()

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

    @staticmethod
    def _prediction_loss(prediction: Mapping[str, torch.Tensor]) -> torch.Tensor:
        temporal = F.smooth_l1_loss(prediction["mu_temporal"], prediction["target"])
        cross = F.smooth_l1_loss(prediction["mu_cross"], prediction["target"])
        return temporal + cross

    @staticmethod
    def _cross_prediction_loss(prediction: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return F.smooth_l1_loss(prediction["mu_cross"], prediction["target"])

    def _validation_losses(self, windows: np.ndarray) -> Dict[str, float]:
        self.net.eval()
        totals = {"temporal": 0.0, "cross": 0.0}
        count = 0
        with torch.no_grad():
            for start in range(0, len(windows), self.batch_size):
                batch = torch.as_tensor(
                    windows[start : start + self.batch_size], device=self.device
                )
                prediction = self.net(batch)
                totals["temporal"] += float(
                    F.smooth_l1_loss(
                        prediction["mu_temporal"], prediction["target"]
                    ).item()
                ) * len(batch)
                totals["cross"] += float(
                    F.smooth_l1_loss(
                        prediction["mu_cross"], prediction["target"]
                    ).item()
                ) * len(batch)
                count += len(batch)
        temporal = totals["temporal"] / max(count, 1)
        cross = totals["cross"] / max(count, 1)
        return {"temporal": temporal, "cross": cross, "joint": temporal + cross}

    def _loss_on_windows(self, windows: np.ndarray) -> float:
        return self._validation_losses(windows)["joint"]

    def fit(
        self,
        optimization_windows: np.ndarray,
        validation_windows: np.ndarray,
        reference_windows: np.ndarray,
        seed: int,
    ) -> "RelationConditionedEvidenceRepair":
        optimization = self._validate_windows(
            optimization_windows, self.dimensions, "optimization_windows"
        )
        validation = self._validate_windows(
            validation_windows, self.dimensions, "validation_windows"
        )
        reference = self._validate_windows(
            reference_windows, self.dimensions, "reference_windows"
        )
        temporal_frozen = not any(
            parameter.requires_grad
            for module in (self.net.temporal_gru, self.net.temporal_head)
            for parameter in module.parameters()
        )
        trainable_parameters = [
            parameter for parameter in self.net.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("B3a has no trainable cross-path parameters.")
        optimizer = torch.optim.Adam(trainable_parameters, lr=self.learning_rate)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        best_state = None
        best_loss = math.inf
        best_epoch = 0
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
                prediction = self.net(batch)
                loss = (
                    self._cross_prediction_loss(prediction)
                    if temporal_frozen
                    else self._prediction_loss(prediction)
                )
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(indices)
                seen += len(indices)
            validation_losses = self._validation_losses(validation)
            validation_loss = (
                validation_losses["cross"]
                if temporal_frozen
                else validation_losses["joint"]
            )
            history_rows.append(
                {
                    "epoch": epoch,
                    "optimization_loss": epoch_loss / max(seen, 1),
                    "validation_loss": validation_loss,
                    "temporal_validation_loss": validation_losses["temporal"],
                    "cross_validation_loss": validation_losses["cross"],
                    "joint_validation_loss": validation_losses["joint"],
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
            raise RuntimeError("B3a training did not produce a valid checkpoint.")
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
            "selection_metric": (
                "cross_validation_loss_with_frozen_temporal"
                if temporal_frozen
                else "joint_validation_loss"
            ),
            "temporal_frozen": bool(temporal_frozen),
            "temporal_state_sha256": self.temporal_state_sha256(),
            "training_history": history_rows,
            "parameter_counts": {
                "temporal": int(sum(parameter.numel() for parameter in self.net.temporal_gru.parameters()) + sum(parameter.numel() for parameter in self.net.temporal_head.parameters())),
                "cross": int(sum(parameter.numel() for parameter in self.net.relation_history_gru.parameters()) + sum(parameter.numel() for parameter in self.net.driver_gru.parameters()) + sum(parameter.numel() for parameter in self.net.cross_head.parameters())),
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
        result = {
            name: np.concatenate(chunks).astype(np.float64) for name, chunks in outputs.items()
        }
        result["temporal_residual"] = np.square(result["target"] - result["mu_temporal"])
        result["cross_residual"] = np.square(result["target"] - result["mu_cross"])
        result["disagreement"] = np.square(result["mu_temporal"] - result["mu_cross"])
        if include_tails:
            if set(self.tails) != set(self.SCORE_COMPONENTS):
                raise RuntimeError("B3a tails are not fitted. Call fit before score_windows.")
            for component in self.SCORE_COMPONENTS:
                result[f"{component}_tail"] = self.tails[component].transform(
                    result[component]
                )
        return result

    def evidence_isolation_report(self, windows: np.ndarray) -> Dict[str, float | bool]:
        """Verify branch blindness and the explicit pre-terminal relation input."""
        array = self._validate_windows(windows, self.dimensions, "windows")[:1].copy()
        self.net.eval()

        def predict(value: np.ndarray) -> Dict[str, np.ndarray]:
            return self.score_windows(value, include_tails=False)

        base = predict(array)
        driver_indices = [
            index for index in range(self.dimensions) if index != self.target_index
        ]
        changed_drivers = array.copy()
        changed_drivers[:, :, driver_indices] += 3.0
        changed_target_terminal = array.copy()
        changed_target_terminal[:, -1, self.target_index] += 3.0
        changed_target_history = array.copy()
        changed_target_history[:, :-1, self.target_index] += 3.0
        with torch.no_grad():
            base_inputs = self.net.evidence_inputs(
                torch.as_tensor(array, device=self.device)
            )
            terminal_inputs = self.net.evidence_inputs(
                torch.as_tensor(changed_target_terminal, device=self.device)
            )
            history_inputs = self.net.evidence_inputs(
                torch.as_tensor(changed_target_history, device=self.device)
            )
        return {
            "temporal_driver_delta": float(
                np.max(np.abs(base["mu_temporal"] - predict(changed_drivers)["mu_temporal"]))
            ),
            "temporal_terminal_target_delta": float(
                np.max(
                    np.abs(
                        base["mu_temporal"] - predict(changed_target_terminal)["mu_temporal"]
                    )
                )
            ),
            "cross_terminal_target_delta": float(
                np.max(
                    np.abs(base["mu_cross"] - predict(changed_target_terminal)["mu_cross"])
                )
            ),
            "relation_history_terminal_target_input_delta": float(
                torch.max(torch.abs(base_inputs[2] - terminal_inputs[2])).item()
            ),
            "relation_history_target_history_input_delta": float(
                torch.max(torch.abs(base_inputs[2] - history_inputs[2])).item()
            ),
            "parameter_sets_disjoint": not bool(
                self.net.branch_parameter_ids()["temporal"]
                & self.net.branch_parameter_ids()["cross"]
            ),
        }
