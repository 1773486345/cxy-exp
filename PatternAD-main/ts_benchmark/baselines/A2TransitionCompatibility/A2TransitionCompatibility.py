"""A2's first event-pre-conditioned trajectory compatibility model.

The encoder consumes only P_t. Its decoder produces a mixture over the full
judged horizon, and compatibility is the full-trajectory mixture likelihood of
Y_t. Generator metadata such as role, cue mode, and onset is not accepted by
this module.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn


def _finite_windows(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty [samples, time, dimensions] array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


class TrajectoryStandardizer:
    """Per-channel normalizer fitted on optimization normal transitions only."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, windows: np.ndarray) -> "TrajectoryStandardizer":
        array = _finite_windows(windows, "optimization_windows")
        self.mean_ = array.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        self.std_ = array.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        self.std_ = np.maximum(self.std_, np.float32(1e-6))
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TrajectoryStandardizer must be fitted before transform.")
        array = _finite_windows(windows, "windows")
        if array.shape[-1] != len(self.mean_):
            raise ValueError("windows have a different channel count from the fitted normalizer.")
        return ((array - self.mean_) / self.std_).astype(np.float32, copy=False)

    def metadata(self) -> Dict[str, Any]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TrajectoryStandardizer is not fitted.")
        return {"mean": self.mean_.astype(float).tolist(), "std": self.std_.astype(float).tolist()}


class ReferenceUpperTail:
    """Reference-only empirical upper-tail map for trajectory NLL."""

    def __init__(self) -> None:
        self.reference_: Optional[np.ndarray] = None

    def fit(self, scores: np.ndarray) -> "ReferenceUpperTail":
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("reference scores must contain at least two finite values.")
        self.reference_ = np.sort(values.copy())
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("ReferenceUpperTail must be fitted before transform.")
        values = np.asarray(scores, dtype=np.float64)
        lower_count = np.searchsorted(self.reference_, values, side="left")
        survival = (len(self.reference_) - lower_count + 1.0) / (len(self.reference_) + 1.0)
        return -np.log(survival)

    def metadata(self) -> Dict[str, Any]:
        if self.reference_ is None:
            raise RuntimeError("ReferenceUpperTail is not fitted.")
        return {
            "count": int(len(self.reference_)),
            "minimum": float(self.reference_[0]),
            "maximum": float(self.reference_[-1]),
        }


class TrajectoryCompatibilityNet(nn.Module):
    """Decode a conditional or unconditional mixture over whole trajectories."""

    def __init__(
        self,
        dimensions: int,
        horizon_length: int,
        hidden_size: int,
        mixture_components: int,
        condition_on_event_pre: bool = True,
    ) -> None:
        super().__init__()
        if min(dimensions, horizon_length, hidden_size, mixture_components) < 1:
            raise ValueError("model dimensions and component count must be positive.")
        self.dimensions = int(dimensions)
        self.horizon_length = int(horizon_length)
        self.hidden_size = int(hidden_size)
        self.mixture_components = int(mixture_components)
        self.condition_on_event_pre = bool(condition_on_event_pre)
        if self.condition_on_event_pre:
            self.encoder: nn.GRU | None = nn.GRU(dimensions, hidden_size, batch_first=True)
            self.unconditional_state = None
        else:
            self.encoder = None
            self.unconditional_state = nn.Parameter(torch.zeros(hidden_size))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(
                hidden_size,
                mixture_components * (horizon_length * dimensions + 1) + mixture_components,
            ),
        )

    def encode_event_pre(self, event_pre: torch.Tensor) -> torch.Tensor:
        if event_pre.ndim != 3 or event_pre.shape[-1] != self.dimensions:
            raise ValueError("event_pre must have shape [batch, history, dimensions].")
        if self.encoder is None:
            if self.unconditional_state is None:
                raise RuntimeError("Unconditional A2 state is missing.")
            return self.unconditional_state.unsqueeze(0).expand(len(event_pre), -1)
        _, hidden = self.encoder(event_pre)
        return hidden[-1]

    def decode_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.ndim != 2 or state.shape[-1] != self.hidden_size:
            raise ValueError("state must have shape [batch, hidden_size].")
        output = self.decoder(state)
        trajectory_size = self.horizon_length * self.dimensions
        means_end = self.mixture_components * trajectory_size
        log_variances_end = means_end + self.mixture_components
        means = output[:, :means_end].reshape(
            len(state), self.mixture_components, self.horizon_length, self.dimensions
        )
        log_variances = output[:, means_end:log_variances_end].clamp(-8.0, 4.0)
        logits = output[:, log_variances_end:]
        return means, log_variances, logits

    def forward(self, event_pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decode_state(self.encode_event_pre(event_pre))


class A2TransitionCompatibility:
    """Fit a full-trajectory conditional compatibility score under A2 splits."""

    def __init__(
        self,
        dimensions: int,
        history_length: int,
        horizon_length: int,
        hidden_size: int = 32,
        mixture_components: int = 3,
        condition_on_event_pre: bool = True,
        learning_rate: float = 3e-3,
        epochs: int = 80,
        patience: int = 10,
        batch_size: int = 64,
        outer_alpha: float = 0.10,
        reliability_bin_count: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if epochs < 1 or patience < 1 or batch_size < 1:
            raise ValueError("epochs, patience, and batch_size must be positive.")
        if not 0.0 < outer_alpha < 1.0:
            raise ValueError("outer_alpha must be in (0, 1).")
        if reliability_bin_count < 1:
            raise ValueError("reliability_bin_count must be positive.")
        self.dimensions = int(dimensions)
        self.history_length = int(history_length)
        self.horizon_length = int(horizon_length)
        self.hidden_size = int(hidden_size)
        self.mixture_components = int(mixture_components)
        self.condition_on_event_pre = bool(condition_on_event_pre)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.outer_alpha = float(outer_alpha)
        self.reliability_bin_count = int(reliability_bin_count)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
        self.net = TrajectoryCompatibilityNet(
            self.dimensions,
            self.horizon_length,
            self.hidden_size,
            self.mixture_components,
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
        cuda_devices = [self.device.index] if self.device.type == "cuda" and self.device.index is not None else []
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(int(seed))
            if cuda_devices:
                torch.cuda.manual_seed_all(int(seed))
            self.net = TrajectoryCompatibilityNet(
                self.dimensions,
                self.horizon_length,
                self.hidden_size,
                self.mixture_components,
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

    def _loss_on_normalized(self, normalized_windows: np.ndarray) -> float:
        self.net.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(normalized_windows), self.batch_size):
                event_pre, future = self._split(normalized_windows[start : start + self.batch_size])
                means, log_variances, logits = self.net(
                    torch.as_tensor(event_pre, device=self.device)
                )
                loss = self._negative_log_likelihood(
                    means, log_variances, logits, torch.as_tensor(future, device=self.device)
                ).mean()
                total += float(loss.item()) * len(event_pre)
                count += len(event_pre)
        return total / max(count, 1)

    def _negative_log_likelihood(
        self,
        means: torch.Tensor,
        log_variances: torch.Tensor,
        logits: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        """Per-window NLL under an isotropic-per-mode full-trajectory mixture."""
        if future.ndim != 3:
            raise ValueError("future must have shape [batch, horizon, dimensions].")
        residual_square = torch.sum(torch.square(future[:, None] - means), dim=(2, 3))
        cells = future.shape[1] * future.shape[2]
        component_log_density = -0.5 * (
            residual_square / torch.exp(log_variances)
            + cells * (log_variances + math.log(2.0 * math.pi))
        )
        return -torch.logsumexp(
            torch.log_softmax(logits, dim=-1) + component_log_density, dim=-1
        ) / cells

    def _raw_scores_from_normalized(self, normalized_windows: np.ndarray) -> np.ndarray:
        self.net.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(normalized_windows), self.batch_size):
                event_pre, future = self._split(normalized_windows[start : start + self.batch_size])
                means, log_variances, logits = self.net(
                    torch.as_tensor(event_pre, device=self.device)
                )
                nll = self._negative_log_likelihood(
                    means, log_variances, logits, torch.as_tensor(future, device=self.device)
                )
                scores.append(nll.detach().cpu().numpy())
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
    ) -> "A2TransitionCompatibility":
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
                event_pre, future = self._split(batch)
                optimizer.zero_grad(set_to_none=True)
                means, log_variances, logits = self.net(
                    torch.as_tensor(event_pre, device=self.device)
                )
                loss = self._negative_log_likelihood(
                    means, log_variances, logits, torch.as_tensor(future, device=self.device)
                ).mean()
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(batch)
                count += len(batch)
            validation_loss = self._loss_on_normalized(validation_z)
            training_history.append(
                {
                    "epoch": epoch,
                    "optimization_loss": total / max(count, 1),
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
            raise RuntimeError("A2 training did not produce a valid checkpoint.")
        self.net.load_state_dict(best_state)
        reference_nll = self._raw_scores_from_normalized(reference_z)
        reference_bins = self._reliability_bins(reference)
        outer_nll = self._raw_scores_from_normalized(outer_calibration_z)
        outer_bins = self._reliability_bins(outer_calibration)
        self.tails = {}
        self.outer_thresholds_ = {}
        for bin_index in range(self.reliability_bin_count):
            reference_scores = reference_nll[reference_bins == bin_index]
            outer_scores = outer_nll[outer_bins == bin_index]
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
            "mixture_components": self.mixture_components,
            "condition_on_event_pre": self.condition_on_event_pre,
            "parameter_count": int(sum(parameter.numel() for parameter in self.net.parameters())),
        }
        return self

    def _require_fitted(self) -> None:
        if len(self.tails) != self.reliability_bin_count or len(self.outer_thresholds_) != self.reliability_bin_count:
            raise RuntimeError("A2TransitionCompatibility must be fitted before scoring.")

    def score_windows(self, windows: np.ndarray) -> Dict[str, np.ndarray]:
        self._require_fitted()
        normalized = self.normalizer.transform(self._validate_windows(windows, "windows"))
        nll = self._raw_scores_from_normalized(normalized)
        bins = self._reliability_bins(windows)
        tail = np.empty(len(nll), dtype=np.float64)
        thresholds = np.empty(len(nll), dtype=np.float64)
        for bin_index, fitted_tail in self.tails.items():
            mask = bins == bin_index
            tail[mask] = fitted_tail.transform(nll[mask])
            thresholds[mask] = self.outer_thresholds_[bin_index]
        return {
            "trajectory_nll": nll,
            "compatibility_tail": tail,
            "reliability_bin": bins,
            "outer_threshold": thresholds,
            "outer_exceedance": (tail > thresholds).astype(np.int64),
        }

    def predict_mean_trajectory(self, windows: np.ndarray) -> np.ndarray:
        """Return the mixture-mean future for a normal-skill comparison only."""
        self._require_fitted()
        normalized = self.normalizer.transform(self._validate_windows(windows, "windows"))
        event_pre, _ = self._split(normalized)
        predicted = []
        self.net.eval()
        with torch.no_grad():
            for start in range(0, len(event_pre), self.batch_size):
                means, _, logits = self.net(
                    torch.as_tensor(event_pre[start : start + self.batch_size], device=self.device)
                )
                weights = torch.softmax(logits, dim=-1)[:, :, None, None]
                predicted.append(torch.sum(weights * means, dim=1).detach().cpu().numpy())
        normalized_prediction = np.concatenate(predicted, axis=0)
        return (
            normalized_prediction * self.normalizer.std_[None, None, :]
            + self.normalizer.mean_[None, None, :]
        ).astype(np.float32, copy=False)

    def event_pre_state(self, windows: np.ndarray) -> np.ndarray:
        """Return the encoder state; changing Y_t cannot change this result."""
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
