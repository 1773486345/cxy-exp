"""A2-M2 contrastive compatibility energy for event-pre/future pairs.

M2 learns the relation between an event-pre state and a normal future, rather
than assigning a generative density to the future alone. Training positives are
normal windows ``(P_t, Y_t)``; in-batch mismatched normal futures are the only
negative pairs. The model never receives episode role, cue mode, onset, or
generator-regime metadata.
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
    TrajectoryStandardizer,
    _finite_windows,
)


class ContrastiveCompatibilityNet(nn.Module):
    """Encode ``P_t`` and a candidate trajectory's internal increments."""

    def __init__(
        self,
        dimensions: int,
        horizon_length: int,
        hidden_size: int,
        condition_on_event_pre: bool = True,
    ) -> None:
        super().__init__()
        if min(dimensions, horizon_length, hidden_size) < 1:
            raise ValueError("model dimensions and hidden_size must be positive.")
        if horizon_length < 2:
            raise ValueError("A2 contrastive compatibility requires horizon_length >= 2.")
        self.dimensions = int(dimensions)
        self.horizon_length = int(horizon_length)
        self.hidden_size = int(hidden_size)
        self.condition_on_event_pre = bool(condition_on_event_pre)
        if self.condition_on_event_pre:
            self.event_pre_encoder: nn.GRU | None = nn.GRU(
                dimensions, hidden_size, batch_first=True
            )
            self.unconditional_state = None
        else:
            self.event_pre_encoder = None
            self.unconditional_state = nn.Parameter(torch.zeros(hidden_size))
        self.future_increment_encoder = nn.GRU(dimensions, hidden_size, batch_first=True)
        self.event_pre_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.future_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon_length * dimensions),
        )

    def encode_event_pre(self, event_pre: torch.Tensor) -> torch.Tensor:
        if event_pre.ndim != 3 or event_pre.shape[-1] != self.dimensions:
            raise ValueError("event_pre must have shape [batch, history, dimensions].")
        if self.event_pre_encoder is None:
            if self.unconditional_state is None:
                raise RuntimeError("Unconditional A2-M2 state is missing.")
            state = self.unconditional_state.unsqueeze(0).expand(len(event_pre), -1)
        else:
            _, hidden = self.event_pre_encoder(event_pre)
            state = hidden[-1]
        return state

    def encode_candidate_future(self, future: torch.Tensor) -> torch.Tensor:
        if future.ndim != 3 or future.shape[-1] != self.dimensions:
            raise ValueError("future must have shape [batch, horizon, dimensions].")
        if future.shape[1] != self.horizon_length:
            raise ValueError("future has a different horizon from this A2-M2 model.")
        # Internal increments preserve timing and cross-channel coordination while
        # excluding a direct absolute endpoint/level shortcut.
        increments = future[:, 1:] - future[:, :-1]
        _, hidden = self.future_increment_encoder(increments)
        return hidden[-1]

    def compatibility_embeddings(
        self, event_pre: torch.Tensor, future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.encode_event_pre(event_pre)
        future_state = self.encode_candidate_future(future)
        event_pre_embedding = F.normalize(self.event_pre_projection(state), dim=-1)
        future_embedding = F.normalize(self.future_projection(future_state), dim=-1)
        return state, event_pre_embedding, future_embedding

    def forecast(self, state: torch.Tensor) -> torch.Tensor:
        return self.forecast_head(state).reshape(
            len(state), self.horizon_length, self.dimensions
        )


class A2ContrastiveCompatibility:
    """Reference-calibrated contrastive energy for the A2 task contract."""

    raw_score_key = "contrastive_energy"
    raw_score_name = "event_pre_future_contrastive_energy"

    def __init__(
        self,
        dimensions: int,
        history_length: int,
        horizon_length: int,
        hidden_size: int = 32,
        condition_on_event_pre: bool = True,
        learning_rate: float = 3e-3,
        epochs: int = 80,
        patience: int = 10,
        batch_size: int = 64,
        outer_alpha: float = 0.10,
        reliability_bin_count: int = 2,
        contrastive_temperature: float = 0.20,
        forecast_weight: float = 0.25,
        device: str | torch.device = "cpu",
        **unused: Any,
    ) -> None:
        if unused:
            raise ValueError(f"Unsupported A2-M2 model arguments: {sorted(unused)}")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if epochs < 1 or patience < 1 or batch_size < 2:
            raise ValueError("epochs, patience, and batch_size must be at least 2 where applicable.")
        if not 0.0 < outer_alpha < 1.0:
            raise ValueError("outer_alpha must be in (0, 1).")
        if reliability_bin_count < 1:
            raise ValueError("reliability_bin_count must be positive.")
        if contrastive_temperature <= 0.0:
            raise ValueError("contrastive_temperature must be positive.")
        if forecast_weight < 0.0:
            raise ValueError("forecast_weight must be non-negative.")
        self.dimensions = int(dimensions)
        self.history_length = int(history_length)
        self.horizon_length = int(horizon_length)
        self.hidden_size = int(hidden_size)
        self.condition_on_event_pre = bool(condition_on_event_pre)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.outer_alpha = float(outer_alpha)
        self.reliability_bin_count = int(reliability_bin_count)
        self.contrastive_temperature = float(contrastive_temperature)
        self.forecast_weight = float(forecast_weight)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
        self.net = ContrastiveCompatibilityNet(
            self.dimensions,
            self.horizon_length,
            self.hidden_size,
            self.condition_on_event_pre,
        ).to(self.device)
        self.normalizer = TrajectoryStandardizer()
        self.tails: Dict[int, ReferenceUpperTail] = {}
        self.reliability_boundaries_: Optional[np.ndarray] = None
        self.outer_thresholds_: Dict[int, float] = {}
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
            self.net = ContrastiveCompatibilityNet(
                self.dimensions,
                self.horizon_length,
                self.hidden_size,
                self.condition_on_event_pre,
            ).to(self.device)

    def _split(self, normalized_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            normalized_windows[:, : self.history_length],
            normalized_windows[:, self.history_length :],
        )

    def _reliability_values(self, raw_windows: np.ndarray) -> np.ndarray:
        windows = self._validate_windows(raw_windows, "windows")
        event_pre = np.asarray(windows[:, : self.history_length], dtype=np.float64)
        return np.sqrt(np.mean(np.square(np.diff(event_pre, axis=1)), axis=(1, 2)) + 1e-12)

    def _fit_reliability_boundaries(self, optimization_windows: np.ndarray) -> None:
        values = self._reliability_values(optimization_windows)
        if self.reliability_bin_count == 1:
            self.reliability_boundaries_ = np.empty(0, dtype=np.float64)
            return
        quantiles = np.arange(1, self.reliability_bin_count) / self.reliability_bin_count
        boundaries = np.quantile(values, quantiles)
        if np.any(np.diff(boundaries) <= 0.0):
            raise ValueError("Observable A2 reliability bins must have distinct boundaries.")
        self.reliability_boundaries_ = np.asarray(boundaries, dtype=np.float64)

    def _reliability_bins(self, raw_windows: np.ndarray) -> np.ndarray:
        if self.reliability_boundaries_ is None:
            raise RuntimeError("A2 reliability boundaries must be fitted before scoring.")
        return np.searchsorted(
            self.reliability_boundaries_, self._reliability_values(raw_windows), side="right"
        ).astype(np.int64)

    def _batch_loss(self, event_pre: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        if len(event_pre) < 2:
            raise ValueError("A2-M2 contrastive batches require at least two normal windows.")
        state, event_pre_embedding, future_embedding = self.net.compatibility_embeddings(
            event_pre, future
        )
        logits = event_pre_embedding @ future_embedding.transpose(0, 1)
        logits = logits / self.contrastive_temperature
        targets = torch.arange(len(event_pre), device=self.device)
        contrastive = 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets)
        )
        forecast = self.net.forecast(state)
        forecast_loss = F.mse_loss(forecast, future)
        return contrastive + self.forecast_weight * forecast_loss

    def _loss_on_normalized(self, normalized_windows: np.ndarray) -> float:
        self.net.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(normalized_windows), self.batch_size):
                batch = normalized_windows[start : start + self.batch_size]
                if len(batch) < 2:
                    continue
                event_pre, future = self._split(batch)
                loss = self._batch_loss(
                    torch.as_tensor(event_pre, device=self.device),
                    torch.as_tensor(future, device=self.device),
                )
                total += float(loss.item()) * len(batch)
                count += len(batch)
        if count == 0:
            raise ValueError("A2-M2 validation requires at least two normal windows.")
        return total / count

    def _raw_scores_from_normalized(self, normalized_windows: np.ndarray) -> np.ndarray:
        self.net.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(normalized_windows), self.batch_size):
                event_pre, future = self._split(normalized_windows[start : start + self.batch_size])
                _, event_pre_embedding, future_embedding = self.net.compatibility_embeddings(
                    torch.as_tensor(event_pre, device=self.device),
                    torch.as_tensor(future, device=self.device),
                )
                # A high energy means the candidate future is less compatible
                # with its own event-pre state; this is the calibrated A2 score.
                scores.append((1.0 - torch.sum(event_pre_embedding * future_embedding, dim=-1)).cpu().numpy())
        return np.concatenate(scores).astype(np.float64, copy=False)

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
    ) -> "A2ContrastiveCompatibility":
        optimization = self._validate_windows(optimization_windows, "optimization_windows")
        validation = self._validate_windows(validation_windows, "validation_windows")
        reference = self._validate_windows(reference_windows, "reference_windows")
        outer_calibration = self._validate_windows(
            outer_calibration_windows, "outer_calibration_windows"
        )
        self.normalizer.fit(optimization)
        self._fit_reliability_boundaries(optimization)
        optimization_z = self.normalizer.transform(optimization)
        validation_z = self.normalizer.transform(validation)
        reference_z = self.normalizer.transform(reference)
        outer_calibration_z = self.normalizer.transform(outer_calibration)
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
            ordering = torch.randperm(len(optimization_z), generator=generator).numpy()
            total = 0.0
            count = 0
            for start in range(0, len(ordering), self.batch_size):
                batch = optimization_z[ordering[start : start + self.batch_size]]
                if len(batch) < 2:
                    continue
                event_pre, future = self._split(batch)
                optimizer.zero_grad(set_to_none=True)
                loss = self._batch_loss(
                    torch.as_tensor(event_pre, device=self.device),
                    torch.as_tensor(future, device=self.device),
                )
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(batch)
                count += len(batch)
            if count == 0:
                raise ValueError("A2-M2 optimization requires at least two normal windows.")
            validation_loss = self._loss_on_normalized(validation_z)
            training_history.append(
                {
                    "epoch": epoch,
                    "optimization_loss": total / count,
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
            raise RuntimeError("A2-M2 training did not produce a valid checkpoint.")
        self.net.load_state_dict(best_state)
        reference_energy = self._raw_scores_from_normalized(reference_z)
        reference_bins = self._reliability_bins(reference)
        outer_energy = self._raw_scores_from_normalized(outer_calibration_z)
        outer_bins = self._reliability_bins(outer_calibration)
        self.tails = {}
        self.outer_thresholds_ = {}
        for bin_index in range(self.reliability_bin_count):
            reference_scores = reference_energy[reference_bins == bin_index]
            outer_scores = outer_energy[outer_bins == bin_index]
            if len(reference_scores) < 2 or len(outer_scores) < 2:
                raise ValueError(
                    f"A2 reliability bin {bin_index} lacks reference or outer-calibration support."
                )
            tail = ReferenceUpperTail().fit(reference_scores)
            self.tails[bin_index] = tail
            self.outer_thresholds_[bin_index] = self._finite_sample_upper_threshold(
                tail.transform(outer_scores), self.outer_alpha
            )
        self.fit_metadata_ = {
            "seed": int(seed),
            "optimization_windows": int(len(optimization)),
            "validation_windows": int(len(validation)),
            "reference_windows": int(len(reference)),
            "outer_calibration_windows": int(len(outer_calibration)),
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_loss),
            "training_history": training_history,
            "normalizer": self.normalizer.metadata(),
            "reliability_boundaries": self.reliability_boundaries_.astype(float).tolist(),
            "reference_tails": {
                str(bin_index): tail.metadata() for bin_index, tail in self.tails.items()
            },
            "outer_alpha": self.outer_alpha,
            "reliability_bin_count": self.reliability_bin_count,
            "outer_thresholds": {
                str(bin_index): float(value)
                for bin_index, value in self.outer_thresholds_.items()
            },
            "contrastive_temperature": self.contrastive_temperature,
            "forecast_weight": self.forecast_weight,
            "condition_on_event_pre": self.condition_on_event_pre,
            "parameter_count": int(sum(parameter.numel() for parameter in self.net.parameters())),
        }
        return self

    def _require_fitted(self) -> None:
        if (
            len(self.tails) != self.reliability_bin_count
            or len(self.outer_thresholds_) != self.reliability_bin_count
        ):
            raise RuntimeError("A2ContrastiveCompatibility must be fitted before scoring.")

    def score_windows(self, windows: np.ndarray) -> Dict[str, np.ndarray]:
        self._require_fitted()
        normalized = self.normalizer.transform(self._validate_windows(windows, "windows"))
        energy = self._raw_scores_from_normalized(normalized)
        bins = self._reliability_bins(windows)
        tail = np.empty(len(energy), dtype=np.float64)
        thresholds = np.empty(len(energy), dtype=np.float64)
        for bin_index, fitted_tail in self.tails.items():
            mask = bins == bin_index
            tail[mask] = fitted_tail.transform(energy[mask])
            thresholds[mask] = self.outer_thresholds_[bin_index]
        return {
            self.raw_score_key: energy,
            "compatibility_tail": tail,
            "reliability_bin": bins,
            "outer_threshold": thresholds,
            "outer_exceedance": (tail > thresholds).astype(np.int64),
        }

    def predict_mean_trajectory(self, windows: np.ndarray) -> np.ndarray:
        """Return the auxiliary normal forecast used only for the skill gate."""
        self._require_fitted()
        normalized = self.normalizer.transform(self._validate_windows(windows, "windows"))
        event_pre, _ = self._split(normalized)
        prediction = []
        self.net.eval()
        with torch.no_grad():
            for start in range(0, len(event_pre), self.batch_size):
                state = self.net.encode_event_pre(
                    torch.as_tensor(event_pre[start : start + self.batch_size], device=self.device)
                )
                prediction.append(self.net.forecast(state).cpu().numpy())
        normalized_prediction = np.concatenate(prediction, axis=0)
        return (
            normalized_prediction * self.normalizer.std_[None, None, :]
            + self.normalizer.mean_[None, None, :]
        ).astype(np.float32, copy=False)

    def event_pre_state(self, windows: np.ndarray) -> np.ndarray:
        """Return the state built before any candidate future is inspected."""
        self._require_fitted()
        normalized = self.normalizer.transform(self._validate_windows(windows, "windows"))
        event_pre, _ = self._split(normalized)
        self.net.eval()
        with torch.no_grad():
            state = self.net.encode_event_pre(torch.as_tensor(event_pre, device=self.device))
        return state.detach().cpu().numpy()

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        self._require_fitted()
        return {name: tensor.detach().cpu() for name, tensor in self.net.state_dict().items()}
