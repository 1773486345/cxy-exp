"""Strictly causal normal transition test."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _CausalBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left = 2 * dilation
        self.depthwise = nn.Conv1d(hidden, hidden, 3, dilation=dilation, groups=hidden)
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        update = self.depthwise(F.pad(x, (self.left, 0)))
        update = self.pointwise(update).transpose(1, 2)
        update = self.norm(update).transpose(1, 2)
        return x + self.dropout(F.gelu(update))


class EvolutionNormalityBranch(nn.Module):
    """Predict x[t] from x[:t], with no temporal-statistics normalization."""

    def __init__(self, c_in: int, branch_dim: int = 128, layers: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(c_in, int):
            config = c_in
            branch_dim = int(getattr(config, "branch_dim", branch_dim))
            layers = int(getattr(config, "temporal_layers", layers))
            dropout = float(getattr(config, "dropout", dropout))
            c_in = int(getattr(config, "c_in"))
        self.c_in = int(c_in)
        self.input_projection = nn.Conv1d(c_in, branch_dim, 1)
        self.blocks = nn.ModuleList(_CausalBlock(branch_dim, 2 ** i, dropout) for i in range(layers))
        self.token_projection = nn.Linear(branch_dim, branch_dim)
        self.prediction_head = nn.Linear(branch_dim, c_in)
        self.evidence_head = nn.Linear(branch_dim, 1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.c_in:
            raise ValueError("x must have shape [B,T,c_in]")
        # The right shift is outside the stack, so even the first point has a
        # real empty history rather than a fabricated target value.
        history = F.pad(x[:, :-1], (0, 0, 1, 0))
        hidden = self.input_projection(history.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        z = self.token_projection(hidden.transpose(1, 2))
        prediction = self.prediction_head(z)
        raw_error = (x - prediction).abs().mean(dim=-1)
        valid_mask = torch.ones_like(raw_error, dtype=torch.bool)
        valid_mask[:, 0] = False
        valid = valid_mask.to(raw_error.dtype)
        raw_error = raw_error * valid
        # There is no observed history at t=0.  Do not expose a fabricated
        # evolution token to the adapter or scorer.
        z = z.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        evidence_logit = (self.evidence_head(z).squeeze(-1) + raw_error) * valid
        task_loss = ((x - prediction).abs().mean(dim=-1) * valid).sum() / valid.sum().clamp_min(1.0)
        return {
            "z": z,
            "prediction": prediction,
            "raw_error": raw_error,
            "evidence_logit": evidence_logit.masked_fill(valid == 0, 0.0),
            "evidence": F.softplus(evidence_logit).masked_fill(valid == 0, 0.0),
            "task_loss": task_loss,
            "valid_mask": valid_mask,
        }
