"""A3-G2 observable trigger-conditioned response-graph grammar.

Both the event-pre trigger state and the response graph nodes are extracted
from raw values by fixed rules. A small autoregressive grammar then models the
joint normal graph distribution conditioned on that observable trigger state.
No learned future codebook or raw-trajectory reconstruction is used.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.a3.generate_trigger_response_contract import extract_response_tokens
from ts_benchmark.baselines.A2TransitionCompatibility.A2TransitionCompatibility import (
    ReferenceUpperTail,
    _finite_windows,
)


def extract_trigger_states(
    event_pre: np.ndarray,
    cue_length: int,
    minimum_amplitude: float,
    linear_tolerance: float,
) -> np.ndarray:
    """Return `(state, linear_error)` from a raw event-pre window.

    State zero is no trigger. A nonzero state encodes the detected channel and
    sign as `1 + 2 * channel + sign_is_positive`. A valid trigger must follow
    the declared terminal linear shape over the whole cue interval. This rule
    receives no source-channel name, cue mode, role, or generator metadata.
    """
    values = np.asarray(event_pre, dtype=np.float32)
    was_single = values.ndim == 2
    if was_single:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1] < cue_length or cue_length < 2:
        raise ValueError("event_pre must be [samples, history, dimensions] with a valid cue length.")
    if minimum_amplitude <= 0.0 or linear_tolerance <= 0.0:
        raise ValueError("trigger amplitude and linear tolerance must be positive.")
    tail = values[:, -cue_length:]
    weights = np.linspace(0.0, 1.0, cue_length, dtype=np.float32)[None, :, None]
    line = tail[:, :1] + weights * (tail[:, -1:] - tail[:, :1])
    errors = np.max(np.abs(tail - line), axis=1)
    displacement = tail[:, -1] - tail[:, 0]
    accepted = (errors <= float(linear_tolerance)) & (
        np.abs(displacement) >= float(minimum_amplitude)
    )
    states = np.zeros(len(values), dtype=np.int64)
    selected_errors = np.min(errors, axis=1).astype(np.float64)
    for row in range(len(values)):
        candidates = np.flatnonzero(accepted[row])
        if len(candidates):
            channel = int(candidates[np.argmax(np.abs(displacement[row, candidates]))])
            states[row] = 1 + 2 * channel + int(displacement[row, channel] > 0.0)
            selected_errors[row] = float(errors[row, channel])
    output = np.stack((states, selected_errors), axis=1)
    return output[0] if was_single else output


def response_graph_tokens(future: np.ndarray, token_energy_threshold: float) -> np.ndarray:
    """Encode every channel's fixed `(active, onset, direction)` node as one token.

    Token zero is inactive. Active token `1 + 2 * onset + direction` has a
    vocabulary of `1 + 2 * horizon_length`, making the complete response graph
    a sequence of fixed observable node tokens.
    """
    values = np.asarray(future, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("future must have shape [samples, horizon, dimensions].")
    tokens = extract_response_tokens(values, token_energy_threshold)
    active = np.asarray(tokens["active"], dtype=np.int64)
    onset = np.asarray(tokens["onset"], dtype=np.int64)
    direction = np.asarray(tokens["direction"], dtype=np.int64)
    encoded = np.zeros_like(active, dtype=np.int64)
    mask = active.astype(bool)
    encoded[mask] = 1 + 2 * onset[mask] + direction[mask]
    return encoded


class ObservableGraphGrammarNet(nn.Module):
    """Autoregressive normal grammar over observable response-graph nodes."""

    def __init__(
        self,
        dimensions: int,
        horizon_length: int,
        hidden_size: int,
        trigger_state_count: int,
    ) -> None:
        super().__init__()
        if min(dimensions, horizon_length, hidden_size, trigger_state_count) < 1:
            raise ValueError("Graph-grammar dimensions must be positive.")
        self.dimensions = int(dimensions)
        self.horizon_length = int(horizon_length)
        self.hidden_size = int(hidden_size)
        self.trigger_state_count = int(trigger_state_count)
        self.response_vocab_size = 1 + 2 * self.horizon_length
        self.start_token = self.response_vocab_size
        self.trigger_embedding = nn.Embedding(self.trigger_state_count, self.hidden_size)
        self.token_embedding = nn.Embedding(self.response_vocab_size + 1, self.hidden_size)
        self.node_embedding = nn.Embedding(self.dimensions, self.hidden_size)
        self.decoder = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.output = nn.Linear(self.hidden_size, self.response_vocab_size)

    def forward(self, trigger_states: torch.Tensor, graph_tokens: torch.Tensor) -> torch.Tensor:
        if trigger_states.ndim != 1 or graph_tokens.ndim != 2:
            raise ValueError("trigger_states must be [batch] and graph_tokens must be [batch, dimensions].")
        if graph_tokens.shape[1] != self.dimensions or len(trigger_states) != len(graph_tokens):
            raise ValueError("A3-G2 batch shapes do not match graph dimensions.")
        if torch.any(trigger_states < 0) or torch.any(trigger_states >= self.trigger_state_count):
            raise ValueError("trigger state is outside the fixed grammar vocabulary.")
        if torch.any(graph_tokens < 0) or torch.any(graph_tokens >= self.response_vocab_size):
            raise ValueError("response graph token is outside the fixed grammar vocabulary.")
        start = torch.full(
            (len(graph_tokens), 1), self.start_token, dtype=torch.long, device=graph_tokens.device
        )
        previous = torch.cat((start, graph_tokens[:, :-1]), dim=1)
        positions = torch.arange(self.dimensions, device=graph_tokens.device)[None, :]
        inputs = (
            self.token_embedding(previous)
            + self.node_embedding(positions)
            + self.trigger_embedding(trigger_states)[:, None, :]
        )
        initial = self.trigger_embedding(trigger_states).unsqueeze(0)
        hidden, _ = self.decoder(inputs, initial)
        return self.output(hidden)


class A3ObservableGraphGrammar:
    """Normal-only conditional joint grammar for an observable response graph."""

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
        outer_alpha: float = 0.10,
        device: str | torch.device = "cpu",
    ) -> None:
        if min(dimensions, history_length, horizon_length, hidden_size, cue_length, epochs, patience, batch_size) < 1:
            raise ValueError("A3-G2 dimensions and training parameters must be positive.")
        if cue_length > history_length or token_energy_threshold <= 0.0:
            raise ValueError("A3-G2 cue length and token threshold are invalid.")
        if minimum_trigger_amplitude <= 0.0 or trigger_linear_tolerance <= 0.0:
            raise ValueError("A3-G2 trigger extractor parameters must be positive.")
        if learning_rate <= 0.0 or not 0.0 < outer_alpha < 1.0:
            raise ValueError("A3-G2 learning rate or outer alpha is invalid.")
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
            self.dimensions, self.horizon_length, self.hidden_size, self.trigger_state_count
        ).to(self.device)
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

    def _tokens_and_states(self, raw_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        windows = self._validate_windows(raw_windows, "windows")
        event_pre = windows[:, : self.history_length]
        future = windows[:, self.history_length :]
        return (
            self._trigger_states(event_pre),
            response_graph_tokens(future, self.token_energy_threshold),
        )

    def _build_net(self, seed: int) -> None:
        cuda_devices = (
            [self.device.index] if self.device.type == "cuda" and self.device.index is not None else []
        )
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(int(seed))
            if cuda_devices:
                torch.cuda.manual_seed_all(int(seed))
            self.net = ObservableGraphGrammarNet(
                self.dimensions, self.horizon_length, self.hidden_size, self.trigger_state_count
            ).to(self.device)

    def _node_losses(self, states: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.net(states, tokens)
        return F.cross_entropy(
            logits.reshape(-1, self.net.response_vocab_size), tokens.reshape(-1), reduction="none"
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
        optimization_windows: np.ndarray,
        validation_windows: np.ndarray,
        reference_windows: np.ndarray,
        outer_calibration_windows: np.ndarray,
        seed: int,
    ) -> "A3ObservableGraphGrammar":
        optimization = self._validate_windows(optimization_windows, "optimization_windows")
        validation = self._validate_windows(validation_windows, "validation_windows")
        reference = self._validate_windows(reference_windows, "reference_windows")
        outer_calibration = self._validate_windows(outer_calibration_windows, "outer_calibration_windows")
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
            raise RuntimeError("A3-G2 training did not produce a valid checkpoint.")
        self.net.load_state_dict(best_state)
        reference_scores, _, _ = self._raw_scores(reference)
        outer_scores, _, _ = self._raw_scores(outer_calibration)
        self.tail = ReferenceUpperTail().fit(reference_scores)
        self.outer_threshold_ = self._finite_sample_upper_threshold(
            self.tail.transform(outer_scores), self.outer_alpha
        )
        state_counts = np.bincount(
            optimization_states, minlength=self.trigger_state_count
        ).astype(np.int64)
        self.fit_metadata_ = {
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_loss),
            "training_history": training_history,
            "optimization_windows": int(len(optimization)),
            "validation_windows": int(len(validation)),
            "reference_windows": int(len(reference)),
            "outer_calibration_windows": int(len(outer_calibration)),
            "outer_alpha": self.outer_alpha,
            "reference_tail": self.tail.metadata(),
            "outer_threshold": float(self.outer_threshold_),
            "condition_on_event_pre": self.condition_on_event_pre,
            "trigger_state_counts": state_counts.tolist(),
            "response_vocab_size": self.net.response_vocab_size,
            "parameter_count": int(sum(parameter.numel() for parameter in self.net.parameters())),
        }
        return self

    def _require_fitted(self) -> None:
        if self.tail is None or self.outer_threshold_ is None:
            raise RuntimeError("A3-G2 model must be fitted before scoring.")

    def event_pre_state(self, windows: np.ndarray) -> np.ndarray:
        self._require_fitted()
        values = self._validate_windows(windows, "windows")
        return self._trigger_states(values[:, : self.history_length]).astype(np.float64)[:, None]

    def score_windows(self, windows: np.ndarray) -> Dict[str, np.ndarray]:
        self._require_fitted()
        raw, node, states = self._raw_scores(windows)
        if self.tail is None or self.outer_threshold_ is None:
            raise RuntimeError("A3-G2 reference calibration is missing.")
        tail = self.tail.transform(raw)
        return {
            "joint_graph_surprisal": raw,
            "joint_graph_tail": tail,
            "joint_graph_threshold": np.full(len(raw), self.outer_threshold_, dtype=np.float64),
            "joint_graph_exceedance": (tail > self.outer_threshold_).astype(np.int64),
            "node_surprisal": node,
            "trigger_state": states.astype(np.int64),
        }

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        self._require_fitted()
        return self.net.state_dict()
