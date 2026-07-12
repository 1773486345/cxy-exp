"""A3-G3 counterfactual trigger-response effect-graph grammar.

The model separates ordinary continuation variation from an induced response.
It first predicts a normal multichannel continuation using event-pre values
only, adds a normal trigger-conditioned response template, and then encodes the
remaining effect as a fixed activation/onset/direction graph.  The final score
is joint graph surprisal, not a scalar forecast residual.
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


class A3CounterfactualEffectGraphGrammar:
    """Normal-only grammar over effect graphs after a past-only counterfactual.

    A ridge map predicts the normal raw continuation from the full event-pre
    window.  Fixed-state templates estimated from normal routed/no-trigger
    windows account for the normal response induced by an observable trigger.
    Neither component receives the judged future at inference time.
    """

    def __init__(
        self,
        dimensions: int,
        history_length: int,
        horizon_length: int,
        effect_token_energy_threshold: float,
        cue_length: int,
        minimum_trigger_amplitude: float,
        trigger_linear_tolerance: float,
        ridge_penalty: float = 1e-3,
        hidden_size: int = 32,
        condition_on_event_pre: bool = True,
        learning_rate: float = 3e-3,
        epochs: int = 80,
        patience: int = 10,
        batch_size: int = 64,
        outer_alpha: float = 0.10,
        device: str | torch.device = "cpu",
    ) -> None:
        if min(
            dimensions,
            history_length,
            horizon_length,
            hidden_size,
            cue_length,
            epochs,
            patience,
            batch_size,
        ) < 1:
            raise ValueError("A3-G3 dimensions and training parameters must be positive.")
        if cue_length > history_length or effect_token_energy_threshold <= 0.0:
            raise ValueError("A3-G3 effect-token extractor parameters are invalid.")
        if minimum_trigger_amplitude <= 0.0 or trigger_linear_tolerance <= 0.0:
            raise ValueError("A3-G3 trigger extractor parameters are invalid.")
        if ridge_penalty <= 0.0 or learning_rate <= 0.0 or not 0.0 < outer_alpha < 1.0:
            raise ValueError("A3-G3 fitting or calibration parameters are invalid.")
        self.dimensions = int(dimensions)
        self.history_length = int(history_length)
        self.horizon_length = int(horizon_length)
        self.effect_token_energy_threshold = float(effect_token_energy_threshold)
        self.cue_length = int(cue_length)
        self.minimum_trigger_amplitude = float(minimum_trigger_amplitude)
        self.trigger_linear_tolerance = float(trigger_linear_tolerance)
        self.ridge_penalty = float(ridge_penalty)
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
        self.ridge_weights_: Optional[np.ndarray] = None
        self.response_templates_: Optional[np.ndarray] = None
        self.template_counts_: Optional[np.ndarray] = None
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

    @staticmethod
    def _design_matrix(event_pre: np.ndarray) -> np.ndarray:
        values = np.asarray(event_pre, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("event_pre must be [samples, history, dimensions].")
        flattened = values.reshape(len(values), -1)
        return np.concatenate((flattened, np.ones((len(values), 1), dtype=np.float64)), axis=1)

    def _fit_counterfactual(
        self,
        ordinary_optimization_windows: np.ndarray,
        normal_event_optimization_windows: np.ndarray,
    ) -> None:
        ordinary = self._validate_windows(
            ordinary_optimization_windows, "ordinary_optimization_windows"
        )
        normal_events = self._validate_windows(
            normal_event_optimization_windows, "normal_event_optimization_windows"
        )
        ordinary_pre = ordinary[:, : self.history_length]
        ordinary_future = ordinary[:, self.history_length :]
        design = self._design_matrix(ordinary_pre)
        targets = ordinary_future.reshape(len(ordinary), -1).astype(np.float64)
        penalty = np.eye(design.shape[1], dtype=np.float64) * self.ridge_penalty
        penalty[-1, -1] = 0.0  # The intercept remains unpenalized.
        self.ridge_weights_ = np.linalg.solve(design.T @ design + penalty, design.T @ targets)

        event_pre = normal_events[:, : self.history_length]
        event_future = normal_events[:, self.history_length :]
        base = self._predict_normal_continuation(event_pre)
        states = self._trigger_states(event_pre)
        templates = np.zeros(
            (self.trigger_state_count, self.horizon_length, self.dimensions), dtype=np.float64
        )
        counts = np.bincount(states, minlength=self.trigger_state_count).astype(np.int64)
        residual = event_future.astype(np.float64) - base
        for state in np.flatnonzero(counts):
            templates[state] = residual[states == state].mean(axis=0)
        self.response_templates_ = templates
        self.template_counts_ = counts

    def _predict_normal_continuation(self, event_pre: np.ndarray) -> np.ndarray:
        if self.ridge_weights_ is None:
            raise RuntimeError("A3-G3 counterfactual dynamics must be fitted before prediction.")
        design = self._design_matrix(event_pre)
        predicted = design @ self.ridge_weights_
        return predicted.reshape(len(design), self.horizon_length, self.dimensions)

    def _counterfactual(self, raw_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self._validate_windows(raw_windows, "windows")
        if self.response_templates_ is None:
            raise RuntimeError("A3-G3 response templates must be fitted before prediction.")
        event_pre = values[:, : self.history_length]
        states = self._trigger_states(event_pre)
        baseline = self._predict_normal_continuation(event_pre) + self.response_templates_[states]
        return baseline.astype(np.float32), states

    def _tokens_and_states(self, raw_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = self._validate_windows(raw_windows, "windows")
        baseline, states = self._counterfactual(values)
        effect = values[:, self.history_length :] - baseline
        tokens = response_graph_tokens(effect, self.effect_token_energy_threshold)
        return states, tokens, effect

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
        states, tokens, _ = self._tokens_and_states(raw_windows)
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

    def _raw_scores(self, raw_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states, tokens, effect = self._tokens_and_states(raw_windows)
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
            effect.astype(np.float64, copy=False),
        )

    @staticmethod
    def _finite_sample_upper_threshold(scores: np.ndarray, alpha: float) -> float:
        ordered = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
        rank = int(math.ceil((len(ordered) + 1) * (1.0 - alpha))) - 1
        return float(ordered[min(max(rank, 0), len(ordered) - 1)])

    def fit(
        self,
        ordinary_optimization_windows: np.ndarray,
        normal_event_optimization_windows: np.ndarray,
        optimization_windows: np.ndarray,
        validation_windows: np.ndarray,
        reference_windows: np.ndarray,
        outer_calibration_windows: np.ndarray,
        seed: int,
    ) -> "A3CounterfactualEffectGraphGrammar":
        optimization = self._validate_windows(optimization_windows, "optimization_windows")
        validation = self._validate_windows(validation_windows, "validation_windows")
        reference = self._validate_windows(reference_windows, "reference_windows")
        outer_calibration = self._validate_windows(
            outer_calibration_windows, "outer_calibration_windows"
        )
        self._fit_counterfactual(ordinary_optimization_windows, normal_event_optimization_windows)
        optimization_states, optimization_tokens, _ = self._tokens_and_states(optimization)
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
            raise RuntimeError("A3-G3 training did not produce a valid checkpoint.")
        self.net.load_state_dict(best_state)
        reference_scores, _, _, _ = self._raw_scores(reference)
        outer_scores, _, _, _ = self._raw_scores(outer_calibration)
        self.tail = ReferenceUpperTail().fit(reference_scores)
        self.outer_threshold_ = self._finite_sample_upper_threshold(
            self.tail.transform(outer_scores), self.outer_alpha
        )
        if self.template_counts_ is None:
            raise RuntimeError("A3-G3 template counts are unexpectedly unavailable.")
        self.fit_metadata_ = {
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_loss),
            "training_history": training_history,
            "ordinary_optimization_windows": int(len(ordinary_optimization_windows)),
            "normal_event_optimization_windows": int(len(normal_event_optimization_windows)),
            "optimization_windows": int(len(optimization)),
            "validation_windows": int(len(validation)),
            "reference_windows": int(len(reference)),
            "outer_calibration_windows": int(len(outer_calibration)),
            "ridge_penalty": self.ridge_penalty,
            "template_state_counts": self.template_counts_.tolist(),
            "outer_alpha": self.outer_alpha,
            "reference_tail": self.tail.metadata(),
            "outer_threshold": float(self.outer_threshold_),
            "condition_on_event_pre": self.condition_on_event_pre,
            "response_vocab_size": self.net.response_vocab_size,
            "parameter_count": int(sum(parameter.numel() for parameter in self.net.parameters())),
        }
        return self

    def _require_fitted(self) -> None:
        if (
            self.ridge_weights_ is None
            or self.response_templates_ is None
            or self.tail is None
            or self.outer_threshold_ is None
        ):
            raise RuntimeError("A3-G3 model must be fitted before scoring.")

    def event_pre_state(self, windows: np.ndarray) -> np.ndarray:
        self._require_fitted()
        values = self._validate_windows(windows, "windows")
        return self._trigger_states(values[:, : self.history_length]).astype(np.float64)[:, None]

    def counterfactual_baseline(self, windows: np.ndarray) -> np.ndarray:
        """Return the future baseline, which is exclusively a function of ``P_t``."""
        self._require_fitted()
        baseline, _ = self._counterfactual(windows)
        return baseline

    def score_windows(self, windows: np.ndarray) -> Dict[str, np.ndarray]:
        self._require_fitted()
        raw, node, states, effect = self._raw_scores(windows)
        if self.tail is None or self.outer_threshold_ is None:
            raise RuntimeError("A3-G3 reference calibration is missing.")
        tail = self.tail.transform(raw)
        terminal_effect = effect[:, -1] - effect[:, 0]
        return {
            "effect_graph_surprisal": raw,
            "effect_graph_tail": tail,
            "effect_graph_threshold": np.full(len(raw), self.outer_threshold_, dtype=np.float64),
            "effect_graph_exceedance": (tail > self.outer_threshold_).astype(np.int64),
            "node_surprisal": node,
            "trigger_state": states.astype(np.int64),
            "effect_terminal_l2": np.linalg.norm(terminal_effect, axis=1).astype(np.float64),
        }

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        self._require_fitted()
        return self.net.state_dict()
