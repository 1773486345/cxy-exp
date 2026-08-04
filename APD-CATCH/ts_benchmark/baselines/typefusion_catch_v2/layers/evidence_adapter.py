"""Independent adapters that align branch evidence into typed tokens."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import Tensor, nn


class EvidenceAdapter(nn.Module):
    def __init__(self, branch_dim: int, joint_dim: int = 128, type_index: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(branch_dim, int):
            config = branch_dim
            branch_dim = int(config.branch_dim)
            joint_dim = int(config.joint_dim)
            dropout = float(config.dropout)
        self.projection = nn.Sequential(nn.Linear(int(branch_dim) + 2, int(joint_dim)), nn.LayerNorm(int(joint_dim)), nn.GELU(), nn.Dropout(dropout))
        self.type_embedding = nn.Parameter(torch.randn(1, 1, int(joint_dim)) * 0.02)
        self.type_index = int(type_index)

    def forward(self, z: Tensor, raw_error: Tensor, evidence_logit: Tensor) -> Tensor:
        features = torch.cat((z, torch.log1p(raw_error.clamp_min(0)).unsqueeze(-1), evidence_logit.unsqueeze(-1)), dim=-1)
        return self.projection(features) + self.type_embedding


class EvidenceAdapterBank(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(EvidenceAdapter(config, type_index=index) for index in range(4))

    def forward(self, branches: Iterable[dict]) -> Tensor:
        tokens: List[Tensor] = []
        for adapter, branch in zip(self.adapters, branches):
            tokens.append(adapter(branch["z"], branch["raw_error"], branch["evidence_logit"]))
        return torch.stack(tokens, dim=2)
