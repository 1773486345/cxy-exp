"""Prototype normality test over the shared time-domain representation."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class StateNormalityBranch(nn.Module):
    def __init__(self, config_or_dim, branch_dim: int = 128, memory_size: int = 32, topk: int = 4, temperature: float = 1.0, usage_weight: float = 0.01, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(config_or_dim, int):
            config = config_or_dim
            input_dim = int(config.d_model)
            branch_dim = int(config.branch_dim)
            memory_size = int(config.state_memory_size)
            topk = int(config.state_topk)
            temperature = float(config.prototype_temperature)
            usage_weight = float(config.lambda_state_usage)
            dropout = float(config.dropout)
        else:
            input_dim = int(config_or_dim)
        self.branch_dim = int(branch_dim)
        self.memory_size = int(memory_size)
        self.topk = max(1, min(int(topk), self.memory_size))
        self.temperature = float(temperature)
        self.usage_weight = float(usage_weight)
        self.state_projection = nn.Sequential(nn.Linear(input_dim, self.branch_dim), nn.GELU(), nn.LayerNorm(self.branch_dim))
        self.prototypes = nn.Parameter(torch.randn(self.memory_size, self.branch_dim) * 0.02)
        self.z_projection = nn.Sequential(nn.Linear(self.branch_dim * 3, self.branch_dim), nn.GELU(), nn.LayerNorm(self.branch_dim), nn.Dropout(dropout))
        self.evidence_head = nn.Linear(self.branch_dim, 1)

    def forward(self, h_time: Tensor) -> Dict[str, Tensor]:
        state_hidden = self.state_projection(h_time)
        distances = (state_hidden.unsqueeze(2) - self.prototypes.view(1, 1, self.memory_size, self.branch_dim)).pow(2).mean(dim=-1)
        # Formal state matching remains sparse top-k: only this assignment is
        # allowed to affect context, raw error, and the returned state token.
        top_values, top_indices = torch.topk(-distances, self.topk, dim=-1)
        top_weights = torch.softmax(top_values / self.temperature, dim=-1)
        sparse_assignment = torch.zeros_like(distances).scatter(-1, top_indices, top_weights)
        prototype_context = torch.einsum("btm,md->btd", sparse_assignment, self.prototypes)
        raw_error = (sparse_assignment * distances).sum(dim=-1)
        z = self.z_projection(torch.cat((state_hidden, prototype_context, state_hidden - prototype_context), dim=-1))
        evidence_logit = self.evidence_head(z).squeeze(-1) + raw_error
        # Usage is deliberately dense and is never used to construct the
        # formal normal prototype context.  This gives every memory slot a
        # regularization gradient without weakening top-k state matching.
        usage_assignment = torch.softmax(-distances / self.temperature, dim=-1)
        usage = usage_assignment.mean(dim=(0, 1))
        usage_loss = (usage - 1.0 / self.memory_size).pow(2).mean()
        compactness = raw_error.mean()
        commitment = F.mse_loss(state_hidden, prototype_context.detach())
        return {
            "z": z,
            "raw_error": raw_error,
            "evidence_logit": evidence_logit,
            "evidence": F.softplus(evidence_logit),
            "task_loss": compactness + commitment + self.usage_weight * usage_loss,
            "prototype_usage_loss": usage_loss,
            "prototype_assignment": sparse_assignment,
            "prototype_usage_assignment": usage_assignment,
            "prototype_usage": usage,
            "prototype_indices": top_indices,
        }
