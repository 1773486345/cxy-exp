"""A2-M3 finite-code compatibility for event-pre/future transitions.

M3 replaces M2's continuous pairwise contrastive energy with a finite codebook
of normal within-horizon transition patterns. An event-pre encoder predicts a
normal code; a candidate future is assigned to its nearest normal increment
code. The anomaly score combines code mismatch with distance from normal code
support. No episode role, cue mode, onset, regime, or generator metadata is an
input to the model.
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


class TransitionCodeNet(nn.Module):
    """Encode event-pre states and future increments into a shared code space."""

    def __init__(
        self,
        dimensions: int,
        horizon_length: int,
        hidden_size: int,
        codebook_size: int,
        condition_on_event_pre: bool = True,
    ) -> None:
        super().__init__()
        if min(dimensions, horizon_length, hidden_size, codebook_size) < 1:
            raise ValueError("M3 dimensions, hidden_size, and codebook_size must be positive.")
        if horizon_length < 2:
            raise ValueError("M3 requires at least two future samples for increment codes.")
        self.dimensions = int(dimensions)
        self.horizon_length = int(horizon_length)
        self.hidden_size = int(hidden_size)
        self.codebook_size = int(codebook_size)
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
        self.event_code_head = nn.Linear(hidden_size, codebook_size)
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon_length * dimensions),
        )
        self.codebook = nn.Parameter(torch.empty(codebook_size, hidden_size))
        nn.init.normal_(self.codebook, mean=0.0, std=0.05)

    def encode_event_pre(self, event_pre: torch.Tensor) -> torch.Tensor:
        if event_pre.ndim != 3 or event_pre.shape[-1] != self.dimensions:
            raise ValueError("event_pre must have shape [batch, history, dimensions].")
        if self.event_pre_encoder is None:
            if self.unconditional_state is None:
                raise RuntimeError("Unconditional A2-M3 state is missing.")
            return self.unconditional_state.unsqueeze(0).expand(len(event_pre), -1)
        _, hidden = self.event_pre_encoder(event_pre)
        return hidden[-1]

    def encode_future_increments(self, future: torch.Tensor) -> torch.Tensor:
        if future.ndim != 3 or future.shape[1:] != (self.horizon_length, self.dimensions):
            raise ValueError("future must have shape [batch, horizon, dimensions].")
        increments = future[:, 1:] - future[:, :-1]
        _, hidden = self.future_increment_encoder(increments)
        return hidden[-1]

    def event_logits(self, event_pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encode_event_pre(event_pre)
        return state, self.event_code_head(state)

    def forecast(self, state: torch.Tensor) -> torch.Tensor:
        return self.forecast_head(state).reshape(
            len(state), self.horizon_length, self.dimensions
        )


class A2TransitionCodeCompatibility:
    """M3 discrete normal-transition support with global reference calibration."""

    raw_score_key = "transition_code_surprisal"
    raw_score_name = "event_pre_transition_code_surprisal"

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
        reliability_bin_count: int = 1,
        codebook_size: int = 5,
        code_loss_weight: float = 1.0,
        vq_loss_weight: float = 0.25,
        forecast_weight: float = 0.25,
        support_weight: float = 1.0,
        minimum_code_occupancy: int = 8,
        device: str | torch.device = "cpu",
        **unused: Any,
    ) -> None:
        if unused:
            raise ValueError(f"Unsupported A2-M3 model arguments: {sorted(unused)}")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if epochs < 1 or patience < 1 or batch_size < 2:
            raise ValueError("epochs, patience, and batch_size must be at least 2 where applicable.")
        if not 0.0 < outer_alpha < 1.0:
            raise ValueError("outer_alpha must be in (0, 1).")
        if reliability_bin_count != 1:
            raise ValueError(
                "M3 uses one global reference stratum: normal transition codes, not "
                "a post-hoc volatility bin, represent normal heterogeneity."
            )
        if min(codebook_size, hidden_size) < 1:
            raise ValueError("codebook_size and hidden_size must be positive.")
        if min(code_loss_weight, vq_loss_weight, forecast_weight, support_weight) < 0.0:
            raise ValueError("M3 loss and support weights must be non-negative.")
        if minimum_code_occupancy < 1:
            raise ValueError("minimum_code_occupancy must be positive.")
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
        self.reliability_bin_count = 1
        self.codebook_size = int(codebook_size)
        self.code_loss_weight = float(code_loss_weight)
        self.vq_loss_weight = float(vq_loss_weight)
        self.forecast_weight = float(forecast_weight)
        self.support_weight = float(support_weight)
        self.minimum_code_occupancy = int(minimum_code_occupancy)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
        self.net = TransitionCodeNet(
            self.dimensions,
            self.horizon_length,
            self.hidden_size,
            self.codebook_size,
            self.condition_on_event_pre,
        ).to(self.device)
        self.normalizer = TrajectoryStandardizer()
        self.tail: Optional[ReferenceUpperTail] = None
        self.outer_threshold_: Optional[float] = None
        self.support_scale_: Optional[float] = None
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
            self.net = TransitionCodeNet(
                self.dimensions,
                self.horizon_length,
                self.hidden_size,
                self.codebook_size,
                self.condition_on_event_pre,
            ).to(self.device)

    def _split(self, normalized_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            normalized_windows[:, : self.history_length],
            normalized_windows[:, self.history_length :],
        )

    @staticmethod
    def _nearest_codes(
        future_embedding: torch.Tensor, codebook: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        squared_distance = torch.sum(
            torch.square(future_embedding[:, None, :] - codebook[None, :, :]), dim=-1
        )
        distances, codes = torch.min(squared_distance, dim=1)
        return codes, distances

    def _initialize_codebook(self, normalized_windows: np.ndarray) -> None:
        """Use deterministic farthest-point seeds so every code starts supported."""
        self.net.eval()
        embedding_parts = []
        with torch.no_grad():
            for start in range(0, len(normalized_windows), self.batch_size):
                _, future = self._split(normalized_windows[start : start + self.batch_size])
                embedding_parts.append(
                    self.net.encode_future_increments(
                        torch.as_tensor(future, device=self.device)
                    ).detach().cpu()
                )
        embeddings = torch.cat(embedding_parts, dim=0)
        if len(embeddings) < self.codebook_size:
            raise ValueError("M3 optimization split has fewer windows than codebook entries.")
        selected = [0]
        nearest = torch.sum(torch.square(embeddings - embeddings[0]), dim=1)
        for _ in range(1, self.codebook_size):
            selected.append(int(torch.argmax(nearest).item()))
            next_distance = torch.sum(
                torch.square(embeddings - embeddings[selected[-1]]), dim=1
            )
            nearest = torch.minimum(nearest, next_distance)
        with torch.no_grad():
            self.net.codebook.copy_(embeddings[selected].to(self.device))

    def _batch_loss(self, event_pre: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        state, event_logits = self.net.event_logits(event_pre)
        future_embedding = self.net.encode_future_increments(future)
        codes, _ = self._nearest_codes(future_embedding, self.net.codebook)
        quantized = self.net.codebook[codes]
        code_loss = F.cross_entropy(event_logits, codes.detach())
        vq_loss = F.mse_loss(future_embedding, quantized.detach()) + F.mse_loss(
            future_embedding.detach(), quantized
        )
        forecast_loss = F.mse_loss(self.net.forecast(state), future)
        return (
            self.code_loss_weight * code_loss
            + self.vq_loss_weight * vq_loss
            + self.forecast_weight * forecast_loss
        )

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
            raise ValueError("M3 validation requires at least two normal windows.")
        return total / count

    def _support_statistics(self, normalized_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        codes = []
        distances = []
        with torch.no_grad():
            for start in range(0, len(normalized_windows), self.batch_size):
                _, future = self._split(normalized_windows[start : start + self.batch_size])
                future_embedding = self.net.encode_future_increments(
                    torch.as_tensor(future, device=self.device)
                )
                batch_codes, batch_distances = self._nearest_codes(
                    future_embedding, self.net.codebook
                )
                codes.append(batch_codes.cpu().numpy())
                distances.append(batch_distances.cpu().numpy())
        return np.concatenate(codes), np.concatenate(distances).astype(np.float64, copy=False)

    def _raw_scores_from_normalized(self, normalized_windows: np.ndarray) -> np.ndarray:
        if self.support_scale_ is None:
            raise RuntimeError("M3 support scale must be fitted before scoring.")
        self.net.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(normalized_windows), self.batch_size):
                event_pre, future = self._split(normalized_windows[start : start + self.batch_size])
                _, event_logits = self.net.event_logits(
                    torch.as_tensor(event_pre, device=self.device)
                )
                future_embedding = self.net.encode_future_increments(
                    torch.as_tensor(future, device=self.device)
                )
                codes, distances = self._nearest_codes(future_embedding, self.net.codebook)
                code_log_probability = torch.log_softmax(event_logits, dim=-1).gather(
                    1, codes[:, None]
                )[:, 0]
                score = -code_log_probability + self.support_weight * distances / self.support_scale_
                scores.append(score.cpu().numpy())
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
    ) -> "A2TransitionCodeCompatibility":
        optimization = self._validate_windows(optimization_windows, "optimization_windows")
        validation = self._validate_windows(validation_windows, "validation_windows")
        reference = self._validate_windows(reference_windows, "reference_windows")
        outer_calibration = self._validate_windows(
            outer_calibration_windows, "outer_calibration_windows"
        )
        self.normalizer.fit(optimization)
        optimization_z = self.normalizer.transform(optimization)
        validation_z = self.normalizer.transform(validation)
        reference_z = self.normalizer.transform(reference)
        outer_calibration_z = self.normalizer.transform(outer_calibration)
        self._build_net(int(seed))
        self._initialize_codebook(optimization_z)
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
                raise ValueError("M3 optimization requires at least two normal windows.")
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
            raise RuntimeError("M3 training did not produce a valid checkpoint.")
        self.net.load_state_dict(best_state)
        optimization_codes, optimization_distances = self._support_statistics(optimization_z)
        self.support_scale_ = float(max(np.quantile(optimization_distances, 0.75), 1e-6))
        reference_scores = self._raw_scores_from_normalized(reference_z)
        outer_scores = self._raw_scores_from_normalized(outer_calibration_z)
        self.tail = ReferenceUpperTail().fit(reference_scores)
        self.outer_threshold_ = self._finite_sample_upper_threshold(
            self.tail.transform(outer_scores), self.outer_alpha
        )
        usage = np.bincount(optimization_codes, minlength=self.codebook_size)
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
            "calibration_kind": "global_transition_code_support",
            "reliability_boundaries": [],
            "reference_tails": {"0": self.tail.metadata()},
            "outer_alpha": self.outer_alpha,
            "reliability_bin_count": 1,
            "outer_thresholds": {"0": float(self.outer_threshold_)},
            "codebook_size": self.codebook_size,
            "optimization_code_usage": usage.astype(int).tolist(),
            "minimum_code_occupancy": self.minimum_code_occupancy,
            "support_scale": self.support_scale_,
            "code_loss_weight": self.code_loss_weight,
            "vq_loss_weight": self.vq_loss_weight,
            "forecast_weight": self.forecast_weight,
            "support_weight": self.support_weight,
            "condition_on_event_pre": self.condition_on_event_pre,
            "parameter_count": int(sum(parameter.numel() for parameter in self.net.parameters())),
        }
        return self

    def additional_gates(self) -> Dict[str, Dict[str, Any]]:
        """Expose M3's non-collapse requirement to the shared A2 evaluator."""
        self._require_fitted()
        usage = list(self.fit_metadata_["optimization_code_usage"])
        return {
            "transition_code_coverage": {
                "optimization_code_usage": usage,
                "minimum_code_occupancy": self.minimum_code_occupancy,
                "passed": all(count >= self.minimum_code_occupancy for count in usage),
            }
        }

    def _require_fitted(self) -> None:
        if self.tail is None or self.outer_threshold_ is None or self.support_scale_ is None:
            raise RuntimeError("A2TransitionCodeCompatibility must be fitted before scoring.")

    def score_windows(self, windows: np.ndarray) -> Dict[str, np.ndarray]:
        self._require_fitted()
        normalized = self.normalizer.transform(self._validate_windows(windows, "windows"))
        raw_score = self._raw_scores_from_normalized(normalized)
        tail = self.tail.transform(raw_score)
        bins = np.zeros(len(raw_score), dtype=np.int64)
        thresholds = np.full(len(raw_score), float(self.outer_threshold_), dtype=np.float64)
        return {
            self.raw_score_key: raw_score,
            "compatibility_tail": tail,
            "reliability_bin": bins,
            "outer_threshold": thresholds,
            "outer_exceedance": (tail > thresholds).astype(np.int64),
        }

    def predict_mean_trajectory(self, windows: np.ndarray) -> np.ndarray:
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
