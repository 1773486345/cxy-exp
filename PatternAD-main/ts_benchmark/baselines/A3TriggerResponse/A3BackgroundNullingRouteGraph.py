"""A3-N1 background-nulling trigger-response route graph.

N1 fits a one-dimensional normal innovation subspace on ordinary optimization
increments only. It removes that direction from future raw increments before a
joint trigger-conditioned graph grammar scores the remaining route topology.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ts_benchmark.baselines.A2TransitionCompatibility.A2TransitionCompatibility import (
    ReferenceUpperTail,
    _finite_windows,
)
from ts_benchmark.baselines.A3TriggerResponse.A3ObservableGraphGrammar import (
    ObservableGraphGrammarNet,
    extract_trigger_states,
    response_graph_tokens,
)


class A3BackgroundNullingRouteGraph:
    """Normal-only grammar over future routes after all-channel factor removal."""

    def __init__(
        self,
        dimensions: int,
        history_length: int,
        horizon_length: int,
        token_energy_threshold: float,
        cue_length: int,
        minimum_trigger_amplitude: float,
        trigger_linear_tolerance: float,
        hidden_size: int = 32,
        condition_on_event_pre: bool = True,
        learning_rate: float = 3e-3,
        epochs: int = 80,
        patience: int = 10,
        batch_size: int = 64,
        outer_alpha: float = 0.05,
        device: str | torch.device = "cpu",
    ) -> None:
        if min(
            dimensions,
            history_length,
            horizon_length,
            cue_length,
            hidden_size,
            epochs,
            patience,
            batch_size,
        ) < 1:
            raise ValueError("A3-N1 dimensions and training parameters must be positive.")
        if cue_length > history_length or token_energy_threshold <= 0.0:
            raise ValueError("A3-N1 token or trigger horizon is invalid.")
        if minimum_trigger_amplitude <= 0.0 or trigger_linear_tolerance <= 0.0:
            raise ValueError("A3-N1 trigger extractor parameters are invalid.")
        if learning_rate <= 0.0 or not 0.0 < outer_alpha < 1.0:
            raise ValueError("A3-N1 learning rate or outer alpha is invalid.")
        self.dimensions = int(dimensions)
        self.history_length = int(history_length)
        self.horizon_length = int(horizon_length)
        self.token_energy_threshold = float(token_energy_threshold)
        self.cue_length = int(cue_length)
        self.minimum_trigger_amplitude = float(minimum_trigger_amplitude)
        self.trigger_linear_tolerance = float(trigger_linear_tolerance)
        self.hidden_size = int(hidden_size)
        self.condition_on_event_pre = bool(condition_on_event_pre)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.outer_alpha = float(outer_alpha)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
        self.trigger_state_count = 1 + 2 * self.dimensions
        self.net = ObservableGraphGrammarNet(
            self.dimensions,
            self.horizon_length,
            self.hidden_size,
            self.trigger_state_count,
        ).to(self.device)
        self.background_factor_: Optional[np.ndarray] = None
        self.background_factor_singular_values_: Optional[np.ndarray] = None
        self.tail: Optional[ReferenceUpperTail] = None
        self.outer_threshold_: Optional[float] = None
        self.fit_metadata_: Dict[str, Any] = {}

    @property
    def window_length(self) -> int:
        return self.history_length + self.horizon_length

    def _validate_windows(self, windows: np.ndarray, name: str) -> np.ndarray:
        values = _finite_windows(windows, name)
        if values.shape[1:] != (self.window_length, self.dimensions):
            raise ValueError(
                f"{name} must have shape [samples, {self.window_length}, {self.dimensions}]."
            )
        return values

    def _trigger_states(self, event_pre: np.ndarray) -> np.ndarray:
        extracted = extract_trigger_states(
            event_pre,
            cue_length=self.cue_length,
            minimum_amplitude=self.minimum_trigger_amplitude,
            linear_tolerance=self.trigger_linear_tolerance,
        )
        states = np.asarray(extracted[:, 0], dtype=np.int64)
        if not self.condition_on_event_pre:
            states.fill(0)
        return states

    def _fit_background_factor(self, normal_optimization_values: np.ndarray) -> None:
        values = np.asarray(normal_optimization_values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimensions or len(values) < 3:
            raise ValueError("normal_optimization_values must be [time, dimensions] with at least 3 rows.")
        if not np.isfinite(values).all():
            raise ValueError("normal_optimization_values must be finite.")
        increments = np.diff(values, axis=0)
        centered = increments - increments.mean(axis=0, keepdims=True)
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
        factor = right_vectors[0]
        norm = float(np.linalg.norm(factor))
        if not np.isfinite(factor).all() or norm <= 0.0:
            raise RuntimeError("A3-N1 background PCA did not produce a valid factor.")
        self.background_factor_ = (factor / norm).astype(np.float64, copy=False)
        self.background_factor_singular_values_ = singular_values.astype(np.float64, copy=False)

    def _project_future(self, future: np.ndarray) -> np.ndarray:
        if self.background_factor_ is None:
            raise RuntimeError("A3-N1 background factor must be fitted before projection.")
        values = np.asarray(future, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (self.horizon_length, self.dimensions):
            raise ValueError("future must be [samples, horizon, dimensions].")
        increments = np.diff(values, axis=1)
        coefficients = np.einsum("btd,d->bt", increments, self.background_factor_)
        projected_increments = increments - coefficients[:, :, None] * self.background_factor_[None, None, :]
        projected = np.zeros_like(values)
        projected[:, 1:] = np.cumsum(projected_increments, axis=1)
        return projected.astype(np.float32)

    def projected_future(self, windows: np.ndarray) -> np.ndarray:
        """Return the all-channel background-null trajectory derived from ``Y_t``."""
        values = self._validate_windows(windows, "windows")
        return self._project_future(values[:, self.history_length :])

    def _tokens_and_states(self, raw_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self._validate_windows(raw_windows, "windows")
        states = self._trigger_states(values[:, : self.history_length])
        projected = self._project_future(values[:, self.history_length :])
        return states, response_graph_tokens(projected, self.token_energy_threshold)

    def _build_net(self, seed: int) -> None:
        cuda_devices = (
            [self.device.index]
            if self.device.type == "cuda" and self.device.index is not None
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(int(seed))
            if cuda_devices:
                torch.cuda.manual_seed_all(int(seed))
            self.net = ObservableGraphGrammarNet(
                self.dimensions,
                self.horizon_length,
                self.hidden_size,
                self.trigger_state_count,
            ).to(self.device)

    def _node_losses(self, states: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.net(states, tokens)
        return F.cross_entropy(
            logits.reshape(-1, self.net.response_vocab_size),
            tokens.reshape(-1),
            reduction="none",
        ).reshape_as(tokens)

    def _loss_on_windows(self, raw_windows: np.ndarray) -> float:
        states, tokens = self._tokens_and_states(raw_windows)
        self.net.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                losses = self._node_losses(
                    torch.as_tensor(states[start:end], device=self.device),
                    torch.as_tensor(tokens[start:end], device=self.device),
                )
                total += float(losses.sum().item())
        return total / len(states)

    def _raw_scores(self, raw_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        states, tokens = self._tokens_and_states(raw_windows)
        values = []
        node_values = []
        self.net.eval()
        with torch.no_grad():
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                node_losses = self._node_losses(
                    torch.as_tensor(states[start:end], device=self.device),
                    torch.as_tensor(tokens[start:end], device=self.device),
                )
                node_values.append(node_losses.detach().cpu().numpy())
                values.append(node_losses.sum(dim=1).detach().cpu().numpy())
        return (
            np.concatenate(values).astype(np.float64, copy=False),
            np.concatenate(node_values).astype(np.float64, copy=False),
            states,
        )

    @staticmethod
    def _finite_sample_upper_threshold(scores: np.ndarray, alpha: float) -> float:
        ordered = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
        rank = int(math.ceil((len(ordered) + 1) * (1.0 - alpha))) - 1
        return float(ordered[min(max(rank, 0), len(ordered) - 1)])

    def fit(
        self,
        normal_optimization_values: np.ndarray,
        optimization_windows: np.ndarray,
        validation_windows: np.ndarray,
        reference_windows: np.ndarray,
        outer_calibration_windows: np.ndarray,
        seed: int,
    ) -> "A3BackgroundNullingRouteGraph":
        optimization = self._validate_windows(optimization_windows, "optimization_windows")
        validation = self._validate_windows(validation_windows, "validation_windows")
        reference = self._validate_windows(reference_windows, "reference_windows")
        outer_calibration = self._validate_windows(
            outer_calibration_windows, "outer_calibration_windows"
        )
        self._fit_background_factor(normal_optimization_values)
        optimization_states, optimization_tokens = self._tokens_and_states(optimization)
        self._build_net(int(seed))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        best_loss = math.inf
        best_epoch = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        stale_epochs = 0
        training_history = []
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            ordering = torch.randperm(len(optimization), generator=generator).numpy()
            total = 0.0
            for start in range(0, len(ordering), self.batch_size):
                indices = ordering[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                losses = self._node_losses(
                    torch.as_tensor(optimization_states[indices], device=self.device),
                    torch.as_tensor(optimization_tokens[indices], device=self.device),
                )
                loss = losses.sum(dim=1).mean()
                loss.backward()
                optimizer.step()
                total += float(losses.sum().item())
            validation_loss = self._loss_on_windows(validation)
            training_history.append(
                {
                    "epoch": int(epoch),
                    "optimization_loss": total / len(optimization),
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
            raise RuntimeError("A3-N1 training did not produce a valid checkpoint.")
        self.net.load_state_dict(best_state)
        reference_scores, _, _ = self._raw_scores(reference)
        outer_scores, _, _ = self._raw_scores(outer_calibration)
        self.tail = ReferenceUpperTail().fit(reference_scores)
        self.outer_threshold_ = self._finite_sample_upper_threshold(
            self.tail.transform(outer_scores), self.outer_alpha
        )
        if self.background_factor_ is None or self.background_factor_singular_values_ is None:
            raise RuntimeError("A3-N1 background factor is unexpectedly unavailable.")
        self.fit_metadata_ = {
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_loss),
            "training_history": training_history,
            "normal_optimization_value_count": int(len(normal_optimization_values)),
            "optimization_windows": int(len(optimization)),
            "validation_windows": int(len(validation)),
            "reference_windows": int(len(reference)),
            "outer_calibration_windows": int(len(outer_calibration)),
            "background_factor": self.background_factor_.tolist(),
            "background_factor_singular_values": self.background_factor_singular_values_.tolist(),
            "outer_alpha": self.outer_alpha,
            "reference_tail": self.tail.metadata(),
            "outer_threshold": float(self.outer_threshold_),
            "condition_on_event_pre": self.condition_on_event_pre,
            "response_vocab_size": self.net.response_vocab_size,
            "parameter_count": int(sum(parameter.numel() for parameter in self.net.parameters())),
        }
        return self

    def _require_fitted(self) -> None:
        if self.background_factor_ is None or self.tail is None or self.outer_threshold_ is None:
            raise RuntimeError("A3-N1 model must be fitted before scoring.")

    def event_pre_state(self, windows: np.ndarray) -> np.ndarray:
        self._require_fitted()
        values = self._validate_windows(windows, "windows")
        return self._trigger_states(values[:, : self.history_length]).astype(np.float64)[:, None]

    def background_factor(self) -> np.ndarray:
        self._require_fitted()
        if self.background_factor_ is None:
            raise RuntimeError("A3-N1 background factor is unavailable.")
        return self.background_factor_.copy()

    def score_windows(self, windows: np.ndarray) -> Dict[str, np.ndarray]:
        self._require_fitted()
        raw, node, states = self._raw_scores(windows)
        if self.tail is None or self.outer_threshold_ is None:
            raise RuntimeError("A3-N1 reference calibration is missing.")
        tail = self.tail.transform(raw)
        return {
            "null_route_surprisal": raw,
            "null_route_tail": tail,
            "null_route_threshold": np.full(len(raw), self.outer_threshold_, dtype=np.float64),
            "null_route_exceedance": (tail > self.outer_threshold_).astype(np.int64),
            "node_surprisal": node,
            "trigger_state": states.astype(np.int64),
        }

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        self._require_fitted()
        return self.net.state_dict()
