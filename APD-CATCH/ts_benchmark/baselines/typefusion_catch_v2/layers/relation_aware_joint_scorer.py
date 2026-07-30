"""Relation-aware scorer for four type evidence streams."""

from __future__ import annotations

import itertools
from typing import Dict, Optional, Sequence, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RelationAwareJointScorer(nn.Module):
    """Produce one joint logit from branch tokens and pairwise relations.

    There is deliberately no raw-input or score-weighting input.  The context
    only modulates pair tokens; the sufficient-evidence path remains branch
    driven and uses a fixed-temperature log-sum-exp.
    """

    def __init__(self, joint_dim: int = 128, context_dim: int = 128, layers: int = 2, heads: int = 4, sufficient_temperature: float = 1.0, relation_correction_cap: float = 2.0) -> None:
        super().__init__()
        if not isinstance(joint_dim, int):
            config = joint_dim
            joint_dim = int(getattr(config, "joint_dim", 128))
            context_dim = int(getattr(config, "context_dim", context_dim))
            layers = int(getattr(config, "joint_layers", layers))
            heads = int(getattr(config, "joint_heads", heads))
            sufficient_temperature = float(getattr(config, "sufficient_temperature", sufficient_temperature))
            relation_correction_cap = float(getattr(config, "relation_correction_cap", relation_correction_cap))
        # Accept the common positional wiring used by the v2 model
        # (joint_dim, joint_layers, joint_heads, context_dim, ...), while
        # retaining the explicit (joint_dim, context_dim, layers, heads, ...)
        # API for direct callers.
        elif context_dim <= 8 and layers <= 16 and heads >= joint_dim:
            context_dim, layers, heads = heads, context_dim, layers
        self.joint_dim = int(joint_dim)
        self.sufficient_temperature = float(sufficient_temperature)
        if self.sufficient_temperature <= 0:
            raise ValueError("sufficient_temperature must be positive")
        self.relation_correction_cap = float(relation_correction_cap)
        self.branch_heads = nn.ModuleList(nn.Linear(joint_dim, 1) for _ in range(4))
        self.evidence_projection = nn.Linear(1, joint_dim)
        self.pair_mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(4 * joint_dim, 2 * joint_dim), nn.GELU(), nn.Linear(2 * joint_dim, joint_dim))
            for _ in range(6)
        )
        self.context_gamma = nn.Linear(context_dim, joint_dim)
        self.context_beta = nn.Linear(context_dim, joint_dim)
        actual_heads = max(1, min(int(heads), joint_dim))
        while joint_dim % actual_heads:
            actual_heads -= 1
        layer = nn.TransformerEncoderLayer(joint_dim, actual_heads, 4 * joint_dim, dropout=0.0, batch_first=True, norm_first=True)
        self.relation_transformer = nn.TransformerEncoder(layer, num_layers=max(1, int(layers)))
        self.relation_head = nn.Sequential(nn.LayerNorm(joint_dim), nn.Linear(joint_dim, 1))

    def _stack_tokens(self, u: Union[Tensor, Sequence[Tensor]]) -> Tensor:
        if isinstance(u, (tuple, list)):
            u = torch.stack(tuple(u), dim=2)
        if u.ndim != 4 or u.shape[2] != 4 or u.shape[-1] != self.joint_dim:
            raise ValueError("u must have shape [B,T,4,joint_dim]")
        return u

    def forward(self, u: Union[Tensor, Sequence[Tensor]], evidence_logit: Optional[Tensor] = None, anchor_context: Optional[Tensor] = None) -> Dict[str, Tensor]:
        tokens = self._stack_tokens(u)
        b, t, _, d = tokens.shape
        branch_logits = torch.stack([head(tokens[:, :, i, :]).squeeze(-1) for i, head in enumerate(self.branch_heads)], dim=-1)
        sufficient = self.sufficient_temperature * torch.logsumexp(branch_logits / self.sufficient_temperature, dim=-1)
        relation_tokens = []
        for index, (i, j) in enumerate(itertools.combinations(range(4), 2)):
            left, right = tokens[:, :, i, :], tokens[:, :, j, :]
            pair = torch.cat((left, right, (left - right).abs(), left * right), dim=-1)
            pair = self.pair_mlps[index](pair)
            if anchor_context is not None:
                if anchor_context.shape[:2] != (b, t):
                    raise ValueError("anchor_context must align with u")
                pair = self.context_gamma(anchor_context) * pair + self.context_beta(anchor_context)
            relation_tokens.append(pair)
        # Branch tokens can receive scalar evidence as a feature, but evidence
        # never bypasses the scorer's branch paths.
        branch_tokens = tokens
        if evidence_logit is not None:
            if evidence_logit.shape != (b, t, 4):
                raise ValueError("evidence_logit must have shape [B,T,4]")
            branch_tokens = branch_tokens + self.evidence_projection(evidence_logit.unsqueeze(-1))
        all_tokens = torch.cat((branch_tokens, torch.stack(relation_tokens, dim=2)), dim=2)
        encoded = self.relation_transformer(all_tokens.reshape(b * t, 10, d)).reshape(b, t, 10, d)
        relation_delta_raw = self.relation_head(encoded[:, :, 4:, :].mean(dim=2)).squeeze(-1)
        relation_delta = self.relation_correction_cap * torch.tanh(relation_delta_raw)
        joint_logit = sufficient + relation_delta
        return {
            "joint_logit": joint_logit,
            "joint_score": F.softplus(joint_logit),
            "sufficient_logit": sufficient,
            "branch_logits": branch_logits,
            "branch_logit": branch_logits,
            "relation_delta_raw": relation_delta_raw,
            "relation_delta": relation_delta,
            "relation_token_count": torch.tensor(10, device=tokens.device),
        }
