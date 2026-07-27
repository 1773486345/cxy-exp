"""A small time-domain preservation path for the shared CATCH stem."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DepthwiseSeparableTemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size, padding=padding, groups=hidden_dim
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        update = self.pointwise(self.depthwise(x))
        update = self.norm(update)
        return x + self.dropout(F.gelu(update))


class LocalTemporalStem(nn.Module):
    """Retains local time-domain information without becoming another CATCH."""

    def __init__(
        self, c_in: int, hidden_dim: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(c_in, hidden_dim)
        self.blocks = nn.ModuleList(
            DepthwiseSeparableTemporalBlock(hidden_dim, kernel_size=3, dropout=dropout)
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # [B, T, C] -> [B, T, H]
        hidden = self.input_projection(x).transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_norm(hidden.transpose(1, 2))
