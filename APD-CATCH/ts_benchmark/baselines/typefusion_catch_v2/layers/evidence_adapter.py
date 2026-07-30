"""Independent type token adapters with context-only FiLM conditioning."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EvidenceAdapter(nn.Module):
    def __init__(self, branch_dim: int, joint_dim: int = 128, context_dim: int = 128, type_index: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(branch_dim, int):
            config = branch_dim
            branch_dim = int(getattr(config, "branch_dim", 128))
            joint_dim = int(getattr(config, "joint_dim", joint_dim))
            context_dim = int(getattr(config, "context_dim", context_dim))
            dropout = float(getattr(config, "dropout", dropout))
        self.projection = nn.Sequential(
            nn.Linear(branch_dim + 2, joint_dim), nn.LayerNorm(joint_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.type_embedding = nn.Parameter(torch.randn(1, 1, joint_dim) * 0.02)
        self.gamma = nn.Linear(context_dim, joint_dim)
        self.beta = nn.Linear(context_dim, joint_dim)
        self.type_index = int(type_index)

    def forward(self, z: Tensor, raw_error: Tensor, evidence_logit: Tensor, anchor_context: Tensor) -> Tensor:
        features = torch.cat((z, torch.log1p(raw_error.clamp_min(0)).unsqueeze(-1), evidence_logit.unsqueeze(-1)), dim=-1)
        token = self.projection(features) + self.type_embedding
        return (1.0 + self.gamma(anchor_context)) * token + self.beta(anchor_context)
