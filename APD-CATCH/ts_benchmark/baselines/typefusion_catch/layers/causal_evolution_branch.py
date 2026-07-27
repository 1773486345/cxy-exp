"""Strictly causal evolution prediction branch."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.layers.shared_catch_stem import patchify_time


class CausalSeparableBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            dilation=dilation,
            groups=hidden_dim,
            padding=0,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        update = self.depthwise(F.pad(x, (self.left_padding, 0)))
        update = self.pointwise(update)
        return x + self.dropout(F.gelu(self.norm(update)))


class CausalEvolutionBranch(nn.Module):
    """Predicts each point from earlier points only.

    Input is shifted before every causal stack, so the output at time t has no
    computational path from x[t] or any future value.
    """

    def __init__(self, config: TypeFusionConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.temporal_hidden_dim
        self.input_projection = nn.Conv1d(config.c_in, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            CausalSeparableBlock(hidden, dilation=2**index, dropout=config.dropout)
            for index in range(config.temporal_layers)
        )
        self.token_projection = nn.Linear(hidden, config.d_model)
        self.prediction_head = nn.Conv1d(config.d_model, config.c_in, kernel_size=1)

    def forward(self, standardized_input: Tensor) -> Dict[str, Tensor]:
        # shift right once before the first causal operation; no target value is
        # ever supplied to the predictor for its own output position.
        history = F.pad(standardized_input[:, :-1, :], (0, 0, 1, 0))
        hidden = self.input_projection(history.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        token_sequence = self.token_projection(hidden.transpose(1, 2))
        prediction = self.prediction_head(token_sequence.transpose(1, 2)).transpose(1, 2)
        token_hidden = patchify_time(
            token_sequence, self.config.patch_size, self.config.patch_stride
        ).mean(dim=2)
        return {
            "z": token_hidden,
            "x_hat": prediction,
            "e": (standardized_input - prediction).abs(),
        }

    def unfreeze_tail(self) -> None:
        for module in (self.token_projection, self.prediction_head):
            for parameter in module.parameters():
                parameter.requires_grad = True
