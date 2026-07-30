"""Encode frozen CATCH reconstructions into aligned normal-run context."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AnchorContextEncoder(nn.Module):
    """Small temporal encoder with no independent score path.

    The context has one token per input time point.  ``LayerNorm`` is applied
    point-wise, so no batch/time statistics are leaked into downstream branches.
    """

    def __init__(self, c_in: int, context_dim: int = 128, hidden_dim: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(c_in, int):
            config = c_in
            context_dim = int(getattr(config, "context_dim", getattr(config, "joint_dim", context_dim)))
            hidden_dim = int(getattr(config, "temporal_hidden_dim", hidden_dim or context_dim))
            dropout = float(getattr(config, "dropout", dropout))
            c_in = int(getattr(config, "c_in"))
        hidden_dim = int(hidden_dim or context_dim)
        self.input_projection = nn.Conv1d(c_in, hidden_dim, kernel_size=1)
        self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.pointwise = nn.Conv1d(hidden_dim, context_dim, kernel_size=1)
        self.norm = nn.LayerNorm(context_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, anchor_reconstruction: Tensor) -> Tensor:
        if anchor_reconstruction.ndim != 3:
            raise ValueError("anchor_reconstruction must have shape [B,T,C]")
        hidden = self.input_projection(anchor_reconstruction.transpose(1, 2))
        hidden = F.gelu(self.depthwise(hidden))
        hidden = self.pointwise(hidden).transpose(1, 2)
        return self.dropout(self.norm(hidden))
