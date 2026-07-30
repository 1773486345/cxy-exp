"""Prototype-based state normality test."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class StateNormalityBranch(nn.Module):
    def __init__(self, c_in: int, branch_dim: int = 128, memory_size: int = 32, topk: int = 4, dropout: float = 0.0, usage_weight: float = 0.01) -> None:
        super().__init__()
        if not isinstance(c_in, int):
            config = c_in
            branch_dim = int(getattr(config, "branch_dim", branch_dim))
            memory_size = int(getattr(config, "state_memory_size", getattr(config, "memory_size", memory_size)))
            topk = int(getattr(config, "state_topk", getattr(config, "memory_topk", topk)))
            dropout = float(getattr(config, "dropout", dropout))
            usage_weight = float(getattr(config, "lambda_state_usage", usage_weight))
            c_in = int(getattr(config, "c_in"))
        self.c_in = int(c_in)
        self.branch_dim = int(branch_dim)
        self.memory_size = int(memory_size)
        self.topk = max(1, min(int(topk), memory_size))
        self.usage_weight = float(usage_weight)
        self.encoder = nn.Sequential(
            nn.Conv1d(c_in, branch_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm(branch_dim),
            nn.Dropout(dropout),
        )
        self.prototypes = nn.Parameter(torch.randn(memory_size, branch_dim) * 0.02)
        self.evidence_head = nn.Linear(branch_dim, 1)

    def forward(self, x: Tensor, anchor_context: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.c_in:
            raise ValueError("x must have shape [B,T,c_in]")
        hidden = self.encoder[0](x.transpose(1, 2)).transpose(1, 2)
        hidden = self.encoder[1](hidden)
        hidden = self.encoder[2](hidden)
        hidden = self.encoder[3](hidden)
        if anchor_context is not None and anchor_context.shape[-1] == hidden.shape[-1]:
            hidden = hidden + anchor_context
        normal_proto = F.normalize(self.prototypes, dim=-1)
        query = F.normalize(hidden, dim=-1)
        distances = 1.0 - torch.matmul(query, normal_proto.transpose(0, 1))
        top_dist, top_idx = distances.topk(self.topk, dim=-1, largest=False)
        weights = torch.softmax(-top_dist, dim=-1)
        gather = self.prototypes[top_idx]
        z = (weights.unsqueeze(-1) * gather).sum(dim=-2)
        raw_error = (weights * top_dist).sum(dim=-1).clamp_min(0.0)
        evidence_logit = self.evidence_head(z).squeeze(-1) + raw_error
        usage = torch.bincount(top_idx.detach().reshape(-1), minlength=self.memory_size).to(x.dtype)
        usage = usage / usage.sum().clamp_min(1.0)
        usage_loss = ((usage - 1.0 / self.memory_size) ** 2).mean()
        compactness = raw_error.mean()
        return {
            "z": z,
            "raw_error": raw_error,
            "evidence_logit": evidence_logit,
            "evidence": F.softplus(evidence_logit),
            "task_loss": compactness + self.usage_weight * usage_loss,
            "prototype_usage_loss": usage_loss,
            "prototype_indices": top_idx,
        }
