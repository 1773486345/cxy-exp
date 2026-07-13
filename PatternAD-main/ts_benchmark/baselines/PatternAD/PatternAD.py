"""PatternAD: dynamic relation-state conditional anomaly detection.

The detector keeps a standard temporal Transformer backbone, but makes the
cross-variable mechanism an end-to-end part of conditional reconstruction:

* a shared multi-scale temporal encoder produces a state for every variable;
* directed relation layers infer a history-conditioned graph at every scale;
* temporal and graph decoders form one conditional Gaussian through a learned
  reliability gate; and
* training uses masked counterfactual targets, so the graph decoder cannot read
  the value it is asked to explain.

The public class implements the benchmark's ``detect_fit``/``detect_score``
interface directly.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


DEFAULT_PATTERN_AD_HYPER_PARAMS = {
    "seq_len": 96,
    "d_model": 64,
    "graph_dim": 32,
    "d_ff": 128,
    "n_heads": 4,
    "e_layers": 2,
    "dropout": 0.1,
    "temporal_kernels": (1, 5, 11),
    "context_window": 9,
    "use_pattern_context": True,
    "relation_mode": "full",
    "graph_topk": 0,
    "graph_target_chunk_size": 16,
    "point_mask_ratio": 0.12,
    "variable_block_mask_ratio": 0.20,
    "max_mask_block_length": 8,
    "branch_loss_weight": 0.25,
    "relation_consistency_weight": 0.05,
    "min_scale": 0.03,
    "batch_size": 32,
    "score_conditioning_batch_size": 128,
    "num_epochs": 30,
    "patience": 5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "validation_fraction": 0.15,
    "score_top_k": 3,
    "device": None,
    "seed": None,
}


@dataclass
class PatternADConfig:
    """Validated configuration for a single PatternAD run."""

    seq_len: int = 96
    d_model: int = 64
    graph_dim: int = 32
    d_ff: int = 128
    n_heads: int = 4
    e_layers: int = 2
    dropout: float = 0.1
    temporal_kernels: Tuple[int, ...] = (1, 5, 11)
    context_window: int = 9
    use_pattern_context: bool = True
    relation_mode: str = "full"
    graph_topk: int = 0
    graph_target_chunk_size: int = 16
    point_mask_ratio: float = 0.12
    variable_block_mask_ratio: float = 0.20
    max_mask_block_length: int = 8
    branch_loss_weight: float = 0.25
    relation_consistency_weight: float = 0.05
    min_scale: float = 0.03
    batch_size: int = 32
    score_conditioning_batch_size: int = 128
    num_epochs: int = 30
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.15
    score_top_k: int = 3
    device: Optional[str] = None
    seed: Optional[int] = None
    n_features: Optional[int] = None

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, object]) -> "PatternADConfig":
        values = dict(DEFAULT_PATTERN_AD_HYPER_PARAMS)
        values.update(kwargs)
        kernels = tuple(int(kernel) for kernel in values["temporal_kernels"])
        config = cls(**{**values, "temporal_kernels": kernels})
        config.validate()
        return config

    def validate(self) -> None:
        if self.seq_len < 4:
            raise ValueError("seq_len must be at least 4.")
        if self.d_model < 4 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads.")
        if self.graph_dim < 1 or self.d_ff < self.d_model:
            raise ValueError("graph_dim must be positive and d_ff must be at least d_model.")
        if self.e_layers < 1 or self.n_heads < 1:
            raise ValueError("e_layers and n_heads must be positive.")
        if not self.temporal_kernels or any(kernel < 1 or kernel % 2 == 0 for kernel in self.temporal_kernels):
            raise ValueError("temporal_kernels must be nonempty positive odd integers.")
        if self.context_window < 2:
            raise ValueError("context_window must be at least 2.")
        self.use_pattern_context = bool(self.use_pattern_context)
        if self.relation_mode not in {"full", "single_scale", "no_graph"}:
            raise ValueError("relation_mode must be one of: full, single_scale, no_graph.")
        if self.graph_topk < 0 or self.graph_target_chunk_size < 1:
            raise ValueError("graph_topk must be nonnegative and graph_target_chunk_size positive.")
        for name in ("point_mask_ratio", "variable_block_mask_ratio", "validation_fraction", "dropout"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1).")
        if self.max_mask_block_length < 1:
            raise ValueError("max_mask_block_length must be positive.")
        if self.min_scale <= 0.0 or self.batch_size < 1 or self.score_conditioning_batch_size < 1:
            raise ValueError("min_scale, batch_size, and score_conditioning_batch_size must be positive.")
        if self.num_epochs < 1 or self.patience < 1:
            raise ValueError("num_epochs and patience must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
        if self.branch_loss_weight < 0.0 or self.relation_consistency_weight < 0.0:
            raise ValueError("loss weights must be nonnegative.")


def _sinusoidal_position_encoding(length: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a parameter-free positional code valid for any window length."""
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / max(width, 1))
    )
    encoding = torch.zeros(length, width, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * divisor)
    if width > 1:
        encoding[:, 1::2] = torch.cos(positions * divisor[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0).unsqueeze(2)


class PatternTemporalEncoder(nn.Module):
    """Encode visible temporal patterns at multiple scales for every variable."""

    def __init__(self, n_features: int, config: PatternADConfig):
        super().__init__()
        self.n_features = int(n_features)
        self.config = config
        self.d_model = int(config.d_model)
        self.stems = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, self.d_model, kernel_size=kernel, padding=kernel // 2),
                    nn.GELU(),
                    nn.Conv1d(self.d_model, self.d_model, kernel_size=1),
                )
                for kernel in config.temporal_kernels
            ]
        )
        self.context_proj = nn.Sequential(
            nn.Linear(5, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.variable_embedding = nn.Parameter(torch.empty(1, 1, self.n_features, self.d_model))
        self.mask_embedding = nn.Parameter(torch.empty(1, 1, 1, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.e_layers)
        self.history_gru = nn.GRU(self.d_model, self.d_model, batch_first=True)
        self.output_norm = nn.LayerNorm(self.d_model)
        nn.init.trunc_normal_(self.variable_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_embedding, std=0.02)

    @staticmethod
    def _causal_rolling_mean(values: torch.Tensor, valid: torch.Tensor, window: int) -> torch.Tensor:
        """Compute a mask-aware rolling mean using only the current and past values."""
        source = torch.where(valid, values, torch.zeros_like(values)).transpose(1, 2)
        weights = valid.to(dtype=values.dtype).transpose(1, 2)
        source = F.pad(source, (window - 1, 0))
        weights = F.pad(weights, (window - 1, 0))
        value_sum = F.avg_pool1d(source, kernel_size=window, stride=1) * window
        count = F.avg_pool1d(weights, kernel_size=window, stride=1) * window
        return (value_sum / count.clamp_min(1.0)).transpose(1, 2)

    def _visible_pattern_context(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Form target-blind level, scale, trend, and high-frequency context."""
        if not self.config.use_pattern_context:
            return torch.zeros(
                (*x.shape, self.d_model), dtype=x.dtype, device=x.device
            )
        valid = ~mask.bool()
        window = min(max(2, int(self.config.context_window)), x.shape[1])
        local_mean = self._causal_rolling_mean(x, valid, window)
        local_mean_square = self._causal_rolling_mean(x.square(), valid, window)
        local_std = (local_mean_square - local_mean.square()).clamp_min(0.0).sqrt()

        deltas = torch.zeros_like(x)
        delta_valid = torch.zeros_like(valid)
        if x.shape[1] > 1:
            deltas[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
            delta_valid[:, 1:, :] = valid[:, 1:, :] & valid[:, :-1, :]
        local_trend = self._causal_rolling_mean(deltas, delta_valid, window)
        high_frequency = torch.where(valid, x - local_mean, torch.zeros_like(x))
        features = torch.stack(
            (local_mean, local_std, local_trend, high_frequency, mask.to(dtype=x.dtype)),
            dim=-1,
        )
        return self.context_proj(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("Expected x with shape [batch, time, variables].")
        batch, time_steps, dimensions = x.shape
        if dimensions != self.n_features:
            raise ValueError(f"Expected {self.n_features} variables, received {dimensions}.")
        masked_x = torch.where(mask, torch.zeros_like(x), x)
        flat_x = masked_x.transpose(1, 2).reshape(batch * dimensions, 1, time_steps)
        position = _sinusoidal_position_encoding(time_steps, self.d_model, x.device, x.dtype)
        mask_code = mask.unsqueeze(-1).to(dtype=x.dtype) * self.mask_embedding
        pattern_context = self._visible_pattern_context(x, mask)
        scale_tokens: List[torch.Tensor] = []
        for stem in self.stems:
            token = stem(flat_x).reshape(batch, dimensions, self.d_model, time_steps)
            token = token.permute(0, 3, 1, 2)
            token = token + self.variable_embedding + position + mask_code + pattern_context
            scale_tokens.append(token)
        temporal_input = torch.stack(scale_tokens, dim=0).mean(dim=0)
        flattened = temporal_input.permute(0, 2, 1, 3).reshape(batch * dimensions, time_steps, self.d_model)
        temporal_state = self.temporal_encoder(flattened)
        temporal_state, _ = self.history_gru(temporal_state)
        temporal_state = self.output_norm(temporal_state)
        temporal_state = temporal_state.reshape(batch, dimensions, time_steps, self.d_model).permute(0, 2, 1, 3)
        return scale_tokens, temporal_state


class DynamicPatternGraphLayer(nn.Module):
    """Infer a directed target-from-source graph without materializing D x D graphs."""

    def __init__(self, d_model: int, graph_dim: int, dropout: float, target_chunk_size: int, top_k: int):
        super().__init__()
        self.query = nn.Linear(d_model, graph_dim, bias=False)
        self.key = nn.Linear(d_model, graph_dim, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.update = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.target_chunk_size = int(target_chunk_size)
        self.top_k = int(top_k)
        self.scale = float(graph_dim) ** -0.5

    def forward(self, history: torch.Tensor, scale_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time_steps, dimensions, width = history.shape
        if dimensions < 2:
            raise ValueError("PatternAD requires at least two variables.")
        query = self.query(history)
        key = self.key(history)
        values = self.value(scale_tokens)
        messages: List[torch.Tensor] = []
        entropy_sum = history.new_zeros(())
        entropy_count = 0
        source_index = torch.arange(dimensions, device=history.device)
        for start in range(0, dimensions, self.target_chunk_size):
            end = min(dimensions, start + self.target_chunk_size)
            logits = torch.einsum("btqh,btdh->btqd", query[:, :, start:end], key) * self.scale
            target_index = torch.arange(start, end, device=history.device)
            diagonal = target_index[:, None] == source_index[None, :]
            logits = logits.masked_fill(diagonal.view(1, 1, end - start, dimensions), float("-inf"))
            if 0 < self.top_k < dimensions - 1:
                cutoff = torch.topk(logits, k=self.top_k, dim=-1).values[..., -1:]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            attention = torch.softmax(logits, dim=-1)
            messages.append(torch.einsum("btqd,btdh->btqh", attention, values))
            entropy_sum = entropy_sum + (-(attention * attention.clamp_min(1e-8).log()).sum(dim=-1)).sum()
            entropy_count += attention.shape[0] * attention.shape[1] * attention.shape[2]
        message = torch.cat(messages, dim=2)
        relation = self.norm(scale_tokens + self.update(torch.cat((history, message), dim=-1)))
        return relation, message, entropy_sum / max(entropy_count, 1)


class PatternADNet(nn.Module):
    """Shared relation-state encoder and conditional dual-decoder detector."""

    def __init__(self, n_features: int, config: PatternADConfig):
        super().__init__()
        self.n_features = int(n_features)
        self.config = config
        self.encoder = PatternTemporalEncoder(self.n_features, config)
        if config.relation_mode == "full":
            self.active_scale_indices = tuple(range(len(config.temporal_kernels)))
        elif config.relation_mode == "single_scale":
            self.active_scale_indices = (len(config.temporal_kernels) // 2,)
        else:
            self.active_scale_indices = ()
        self.relation_layers = nn.ModuleList(
            [
                DynamicPatternGraphLayer(
                    config.d_model,
                    config.graph_dim,
                    config.dropout,
                    config.graph_target_chunk_size,
                    config.graph_topk,
                )
                for _ in self.active_scale_indices
            ]
        )
        self.scale_gate = (
            nn.Linear(config.d_model, len(self.active_scale_indices))
            if self.active_scale_indices
            else None
        )
        self.temporal_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, 2),
        )
        self.graph_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, 2),
        )
        self.reliability_gate = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        scale_tokens, temporal_state = self.encoder(x, mask)
        relation_states: List[torch.Tensor] = []
        messages: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        for layer, scale_index in zip(self.relation_layers, self.active_scale_indices):
            scale_token = scale_tokens[scale_index]
            relation, message, entropy = layer(temporal_state, scale_token)
            relation_states.append(relation)
            messages.append(message)
            entropies.append(entropy)
        if relation_states:
            if len(relation_states) == 1:
                graph_state = relation_states[0]
            else:
                if self.scale_gate is None:
                    raise RuntimeError("Relation scale gate is missing for a multi-scale graph.")
                weights = torch.softmax(self.scale_gate(temporal_state), dim=-1)
                graph_state = sum(
                    weights[..., index : index + 1] * relation
                    for index, relation in enumerate(relation_states)
                )
        else:
            # The temporal-only ablation retains the decoder and likelihood
            # protocol while removing every cross-variable message.
            graph_state = temporal_state
        temporal_parameters = self.temporal_head(temporal_state)
        graph_parameters = self.graph_head(graph_state)
        temporal_mean, temporal_raw_scale = temporal_parameters.unbind(dim=-1)
        graph_mean, graph_raw_scale = graph_parameters.unbind(dim=-1)
        temporal_scale = F.softplus(temporal_raw_scale) + self.config.min_scale
        graph_scale = F.softplus(graph_raw_scale) + self.config.min_scale
        temporal_reliability = torch.sigmoid(self.reliability_gate(torch.cat((temporal_state, graph_state), dim=-1)).squeeze(-1))
        mean = temporal_reliability * temporal_mean + (1.0 - temporal_reliability) * graph_mean
        second_moment = (
            temporal_reliability * (temporal_scale.square() + temporal_mean.square())
            + (1.0 - temporal_reliability) * (graph_scale.square() + graph_mean.square())
        )
        scale = (second_moment - mean.square()).clamp_min(self.config.min_scale ** 2).sqrt()
        if len(messages) > 1:
            normalized_messages = [F.normalize(message, dim=-1) for message in messages]
            message_center = torch.stack(normalized_messages, dim=0).mean(dim=0)
            relation_consistency = torch.stack(
                [(message - message_center).square().mean() for message in normalized_messages]
            ).mean()
        else:
            relation_consistency = temporal_state.new_zeros(())
        return {
            "mean": mean,
            "scale": scale,
            "temporal_mean": temporal_mean,
            "temporal_scale": temporal_scale,
            "graph_mean": graph_mean,
            "graph_scale": graph_scale,
            "temporal_reliability": temporal_reliability,
            "relation_consistency": relation_consistency,
            "graph_entropy": torch.stack(entropies).mean() if entropies else temporal_state.new_zeros(()),
        }


class _WindowDataset(Dataset):
    """Chronological sliding windows without materializing the full window matrix."""

    def __init__(self, values: np.ndarray, seq_len: int):
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError("values must be a nonempty [time, variables] array.")
        self.values = np.asarray(values, dtype=np.float32)
        self.seq_len = int(seq_len)
        if len(self.values) < self.seq_len:
            padding = np.repeat(self.values[:1], self.seq_len - len(self.values), axis=0)
            self.values = np.concatenate((padding, self.values), axis=0)
        self.count = len(self.values) - self.seq_len + 1

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.values[index : index + self.seq_len].copy())


def _fill_missing(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.astype(np.float64)
    return numeric.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)


class PatternAD:
    """Paper-oriented multivariate detector with a normal-only training protocol."""

    def __init__(self, **kwargs):
        self.config = PatternADConfig.from_kwargs(kwargs)
        requested_device = self.config.device
        if requested_device is None:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PatternAD was configured for CUDA, but CUDA is unavailable.")
        self.scaler = StandardScaler()
        self.model: Optional[PatternADNet] = None
        self.feature_names: Optional[List[str]] = None
        self.fit_diagnostics_: Optional[Dict[str, object]] = None
        self._last_score_components: Dict[str, np.ndarray] = {}

    @staticmethod
    def required_hyper_params() -> Dict[str, str]:
        return {}

    def detect_hyper_param_tune(self, train_data: pd.DataFrame) -> None:
        if train_data.shape[1] < 2:
            raise ValueError("PatternAD requires at least two observed variables.")
        self.config.n_features = int(train_data.shape[1])

    @staticmethod
    def _gaussian_nll(target: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return 0.5 * (((target - mean) / scale).square() + 2.0 * scale.log() + math.log(2.0 * math.pi))

    def _sample_training_mask(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, dimensions = x.shape
        point_mask = torch.rand_like(x) < self.config.point_mask_ratio
        selected = torch.rand(batch, dimensions, device=x.device) < self.config.variable_block_mask_ratio
        block_length = torch.randint(1, self.config.max_mask_block_length + 1, (batch, dimensions), device=x.device)
        block_end = torch.randint(1, time_steps + 1, (batch, dimensions), device=x.device)
        timeline = torch.arange(time_steps, device=x.device).view(1, time_steps, 1)
        block_start = (block_end - block_length).clamp_min(0).unsqueeze(1)
        block_end = block_end.unsqueeze(1)
        block_mask = selected.unsqueeze(1) & (timeline >= block_start) & (timeline < block_end)
        mask = point_mask | block_mask
        if not bool(mask.any()):
            mask[0, -1, 0] = True
        return mask

    @staticmethod
    def _validation_mask(x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, dimensions = x.shape
        batch_index = torch.arange(batch, device=x.device).view(batch, 1, 1)
        time_index = torch.arange(time_steps, device=x.device).view(1, time_steps, 1)
        dimension_index = torch.arange(dimensions, device=x.device).view(1, 1, dimensions)
        point_mask = (batch_index + 2 * time_index + 3 * dimension_index).remainder(11) == 0
        terminal_mask = (batch_index + dimension_index).remainder(2) == 0
        point_mask[:, -1:, :] |= terminal_mask
        return point_mask

    def _loss(self, target: torch.Tensor, mask: torch.Tensor, outputs: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        joint_nll = self._gaussian_nll(target, outputs["mean"], outputs["scale"])[mask].mean()
        temporal_nll = self._gaussian_nll(target, outputs["temporal_mean"], outputs["temporal_scale"])[mask].mean()
        graph_nll = self._gaussian_nll(target, outputs["graph_mean"], outputs["graph_scale"])[mask].mean()
        loss = (
            joint_nll
            + self.config.branch_loss_weight * 0.5 * (temporal_nll + graph_nll)
            + self.config.relation_consistency_weight * outputs["relation_consistency"]
        )
        diagnostics = {
            "joint_nll": float(joint_nll.detach().cpu()),
            "temporal_nll": float(temporal_nll.detach().cpu()),
            "graph_nll": float(graph_nll.detach().cpu()),
            "relation_consistency": float(outputs["relation_consistency"].detach().cpu()),
            "graph_entropy": float(outputs["graph_entropy"].detach().cpu()),
        }
        return loss, diagnostics

    def _validation_loss(self, loader: DataLoader) -> float:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        self.model.eval()
        weighted_loss = 0.0
        total = 0
        with torch.no_grad():
            for batch in loader:
                target = batch.to(self.device, non_blocking=self.device.type == "cuda")
                mask = self._validation_mask(target)
                outputs = self.model(target, mask)
                loss, _ = self._loss(target, mask, outputs)
                weighted_loss += float(loss.detach().cpu()) * len(target)
                total += len(target)
        return weighted_loss / max(total, 1)

    def detect_fit(self, train_data: pd.DataFrame, train_label: Optional[pd.DataFrame] = None) -> "PatternAD":
        del train_label
        self.detect_hyper_param_tune(train_data)
        frame = _fill_missing(train_data)
        self.feature_names = list(frame.columns)
        values = frame.to_numpy(dtype=np.float64, copy=True)
        split_index = int(round(len(values) * (1.0 - self.config.validation_fraction)))
        split_index = min(max(split_index, 1), len(values) - 1) if len(values) > 1 else 1
        optimization_values = values[:split_index]
        validation_values = values[split_index:] if split_index < len(values) else values[-1:]
        self.scaler.fit(optimization_values)
        optimization_values = self.scaler.transform(optimization_values).astype(np.float32)
        validation_values = self.scaler.transform(validation_values).astype(np.float32)
        train_loader = DataLoader(
            _WindowDataset(optimization_values, self.config.seq_len),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        validation_loader = DataLoader(
            _WindowDataset(validation_values, self.config.seq_len),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        self.model = PatternADNet(int(self.config.n_features), self.config).to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_validation = float("inf")
        stale_epochs = 0
        history: List[Dict[str, float]] = []
        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            loss_sum = 0.0
            total = 0
            epoch_graph_entropy = 0.0
            for batch in train_loader:
                target = batch.to(self.device, non_blocking=self.device.type == "cuda")
                mask = self._sample_training_mask(target)
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(target, mask)
                loss, diagnostics = self._loss(target, mask, outputs)
                loss.backward()
                if self.config.gradient_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(target)
                epoch_graph_entropy += diagnostics["graph_entropy"] * len(target)
                total += len(target)
            validation_loss = self._validation_loss(validation_loader)
            epoch_record = {
                "epoch": float(epoch),
                "train_loss": loss_sum / max(total, 1),
                "validation_loss": validation_loss,
                "graph_entropy": epoch_graph_entropy / max(total, 1),
            }
            history.append(epoch_record)
            if validation_loss < best_validation - 1e-6:
                best_validation = validation_loss
                stale_epochs = 0
                best_state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break
        if best_state is None:
            raise RuntimeError("PatternAD did not produce a valid checkpoint.")
        self.model.load_state_dict(best_state)
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        self.fit_diagnostics_ = {
            "model": "PatternAD",
            "n_features": int(self.config.n_features),
            "parameter_count": int(parameter_count),
            "device": str(self.device),
            "best_validation_loss": float(best_validation),
            "epochs_completed": int(len(history)),
            "history": history,
            "temporal_kernels": list(self.config.temporal_kernels),
            "graph_topk": int(self.config.graph_topk),
            "relation_mode": self.config.relation_mode,
        }
        return self

    def _window_batches(self, values: np.ndarray, window_batch_size: int) -> Iterable[np.ndarray]:
        dataset = _WindowDataset(values, self.config.seq_len)
        for start in range(0, len(dataset), window_batch_size):
            yield np.stack([dataset[index].numpy() for index in range(start, min(start + window_batch_size, len(dataset)))])

    def _conditional_terminal_nll(self, windows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        batch, _, dimensions = windows.shape
        repeated = windows.repeat_interleave(dimensions, dim=0)
        mask = torch.zeros_like(repeated, dtype=torch.bool)
        rows = torch.arange(batch * dimensions, device=windows.device)
        targets = torch.arange(dimensions, device=windows.device).repeat(batch)
        mask[rows, -1, targets] = True
        outputs = self.model(repeated, mask)
        target = repeated[rows, -1, targets]
        joint = self._gaussian_nll(target, outputs["mean"][rows, -1, targets], outputs["scale"][rows, -1, targets])
        temporal = self._gaussian_nll(target, outputs["temporal_mean"][rows, -1, targets], outputs["temporal_scale"][rows, -1, targets])
        graph = self._gaussian_nll(target, outputs["graph_mean"][rows, -1, targets], outputs["graph_scale"][rows, -1, targets])
        return joint.reshape(batch, dimensions), temporal.reshape(batch, dimensions), graph.reshape(batch, dimensions)

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.feature_names is None:
            raise ValueError("Model not trained. Call detect_fit before detect_score.")
        if list(test.columns) != self.feature_names:
            raise ValueError("Test columns must exactly match the training columns.")
        frame = _fill_missing(test)
        values = self.scaler.transform(frame.to_numpy(dtype=np.float64, copy=True)).astype(np.float32)
        dimensions = values.shape[1]
        if dimensions != self.config.n_features:
            raise ValueError("Test feature count does not match the fitted model.")
        window_batch_size = max(1, self.config.score_conditioning_batch_size // dimensions)
        joint_rows: List[np.ndarray] = []
        temporal_rows: List[np.ndarray] = []
        graph_rows: List[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch in self._window_batches(values, window_batch_size):
                windows = torch.as_tensor(batch, device=self.device)
                joint, temporal, graph = self._conditional_terminal_nll(windows)
                joint_rows.append(joint.detach().cpu().numpy())
                temporal_rows.append(temporal.detach().cpu().numpy())
                graph_rows.append(graph.detach().cpu().numpy())
        joint_nll = np.concatenate(joint_rows, axis=0)
        temporal_nll = np.concatenate(temporal_rows, axis=0)
        graph_nll = np.concatenate(graph_rows, axis=0)
        top_k = min(max(1, self.config.score_top_k), dimensions)
        endpoint_scores = np.partition(joint_nll, kth=dimensions - top_k, axis=1)[:, -top_k:].mean(axis=1)
        endpoint_temporal = np.partition(temporal_nll, kth=dimensions - top_k, axis=1)[:, -top_k:].mean(axis=1)
        endpoint_graph = np.partition(graph_nll, kth=dimensions - top_k, axis=1)[:, -top_k:].mean(axis=1)
        if len(values) < self.config.seq_len:
            scores = np.repeat(endpoint_scores[0], len(values))
            temporal_scores = np.repeat(endpoint_temporal[0], len(values))
            graph_scores = np.repeat(endpoint_graph[0], len(values))
            variable_nll = np.repeat(joint_nll[:1], len(values), axis=0)
        else:
            padding = self.config.seq_len - 1
            scores = np.concatenate((np.repeat(endpoint_scores[0], padding), endpoint_scores))
            temporal_scores = np.concatenate((np.repeat(endpoint_temporal[0], padding), endpoint_temporal))
            graph_scores = np.concatenate((np.repeat(endpoint_graph[0], padding), endpoint_graph))
            variable_nll = np.concatenate((np.repeat(joint_nll[:1], padding, axis=0), joint_nll), axis=0)
        self._last_score_components = {
            "score": scores.astype(np.float64, copy=False),
            "temporal_nll": temporal_scores.astype(np.float64, copy=False),
            "graph_nll": graph_scores.astype(np.float64, copy=False),
            "variable_nll": variable_nll.astype(np.float64, copy=False),
        }
        return self._last_score_components["score"].copy()

    def get_diagnostics(self) -> Dict[str, object]:
        return copy.deepcopy(self.fit_diagnostics_ or {"model": "PatternAD", "status": "unfitted"})

    def get_last_score_components(self) -> Dict[str, np.ndarray]:
        return {name: values.copy() for name, values in self._last_score_components.items()}
