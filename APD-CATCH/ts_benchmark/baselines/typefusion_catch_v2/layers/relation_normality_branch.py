"""Memory-bounded channel-group conditional normality test."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RelationNormalityBranch(nn.Module):
    def __init__(self, c_in: int, groups: int = 4, branch_dim: int = 128, dropout: float = 0.0, use_activation_checkpoint: bool = True) -> None:
        super().__init__()
        if not isinstance(c_in, int):
            config = c_in
            groups = int(getattr(config, "relation_mask_groups", groups))
            branch_dim = int(getattr(config, "branch_dim", branch_dim))
            dropout = float(getattr(config, "dropout", dropout))
            c_in = int(getattr(config, "c_in"))
        self.c_in, self.groups = int(c_in), max(1, min(int(groups), int(c_in)))
        self.branch_dim = int(branch_dim)
        self.use_activation_checkpoint = bool(use_activation_checkpoint)
        self.value_projection = nn.Linear(1, branch_dim)
        self.channel_embedding = nn.Parameter(torch.randn(1, 1, c_in, branch_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(branch_dim, 4, branch_dim * 2, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.channel_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.temporal = nn.Conv1d(branch_dim, branch_dim, 3, padding=1, groups=branch_dim)
        self.output_head = nn.Sequential(nn.LayerNorm(branch_dim), nn.Linear(branch_dim, 1))
        self.evidence_head = nn.Linear(branch_dim, 1)

    def _condition(self, masked: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        b, t, c = masked.shape
        h = self.value_projection(masked.unsqueeze(-1)) + self.channel_embedding
        flat = h.reshape(b * t, c, self.branch_dim)
        if self.use_activation_checkpoint and torch.is_grad_enabled():
            encoded = checkpoint(self.channel_encoder, flat, use_reentrant=False)
        else:
            encoded = self.channel_encoder(flat)
        h = encoded.view(b, t, c, self.branch_dim)
        temporal = self.temporal(h.permute(0, 2, 3, 1).reshape(b * c, self.branch_dim, t))
        temporal = temporal.view(b, c, self.branch_dim, t).permute(0, 3, 1, 2)
        pred = self.output_head(temporal).squeeze(-1)
        target_pred = pred.index_select(2, targets)
        hidden = temporal.index_select(2, targets).mean(dim=2)
        return target_pred, hidden

    def forward(self, x: Tensor, anchor_context: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if x.ndim != 3 or x.size(-1) != self.c_in:
            raise ValueError("x must have shape [B,T,c_in]")
        assignments = torch.arange(self.c_in, device=x.device) % self.groups
        predictions = [None] * self.c_in
        hidden_sum = x.new_zeros(x.size(0), x.size(1), self.branch_dim)
        for group in range(self.groups):
            targets = (assignments == group).nonzero(as_tuple=False).flatten()
            masked = x.clone().masked_fill((assignments == group).view(1, 1, -1), 0.0)
            pred, hidden = self._condition(masked, targets)
            hidden_sum = hidden_sum + hidden
            for local, channel in enumerate(targets.tolist()):
                predictions[channel] = pred[:, :, local]
        reconstructed = torch.stack([item for item in predictions if item is not None], dim=-1)
        z = hidden_sum / float(self.c_in)
        raw_error = (x - reconstructed).abs().mean(dim=-1)
        evidence_logit = self.evidence_head(z).squeeze(-1) + raw_error
        return {
            "z": z,
            "prediction": reconstructed,
            "raw_error": raw_error,
            "evidence_logit": evidence_logit,
            "evidence": F.softplus(evidence_logit),
            "task_loss": raw_error.mean(),
            "channel_assignments": assignments,
        }
