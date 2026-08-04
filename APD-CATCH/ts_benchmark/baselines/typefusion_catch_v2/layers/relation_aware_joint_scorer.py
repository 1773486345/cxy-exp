"""Four-token sufficient evidence plus pairwise and short-term relations."""

from __future__ import annotations

import itertools
from typing import Dict, Sequence, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RelationAwareJointScorer(nn.Module):
    def __init__(self, config_or_dim, layers: int = 2, heads: int = 4, temperature: float = 1.0, correction_cap: float = 2.0) -> None:
        super().__init__()
        if not isinstance(config_or_dim, int):
            config = config_or_dim
            joint_dim = int(config.joint_dim)
            layers = int(config.joint_layers)
            heads = int(config.joint_heads)
            temperature = float(config.sufficient_temperature)
            correction_cap = float(config.relation_correction_cap)
            kernel = int(config.relation_temporal_kernel)
        else:
            joint_dim = int(config_or_dim)
            kernel = 5
        self.joint_dim = joint_dim
        self.sufficient_temperature = float(temperature)
        self.relation_correction_cap = float(correction_cap)
        self.branch_heads = nn.ModuleList(nn.Sequential(nn.LayerNorm(joint_dim), nn.Linear(joint_dim, 1)) for _ in range(4))
        self.evidence_projection = nn.Linear(1, joint_dim)
        self.pair_mlps = nn.ModuleList(nn.Sequential(nn.Linear(4 * joint_dim, 2 * joint_dim), nn.GELU(), nn.Linear(2 * joint_dim, joint_dim)) for _ in range(6))
        actual_heads = max(1, min(int(heads), joint_dim))
        while joint_dim % actual_heads:
            actual_heads -= 1
        layer = nn.TransformerEncoderLayer(joint_dim, actual_heads, 4 * joint_dim, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.relation_transformer = nn.TransformerEncoder(layer, num_layers=max(1, int(layers)))
        self.temporal_depthwise = nn.Conv1d(joint_dim, joint_dim, kernel, padding=kernel // 2, groups=joint_dim)
        self.temporal_pointwise = nn.Conv1d(joint_dim, joint_dim, 1)
        self.temporal_norm = nn.LayerNorm(joint_dim)
        self.relation_head = nn.Linear(joint_dim, 1)

    def _stack_tokens(self, tokens: Union[Tensor, Sequence[Tensor]]) -> Tensor:
        if isinstance(tokens, (tuple, list)):
            tokens = torch.stack(tuple(tokens), dim=2)
        if tokens.ndim != 4 or tokens.size(2) != 4 or tokens.size(-1) != self.joint_dim:
            raise ValueError("tokens must have shape [B,T,4,joint_dim]")
        return tokens

    def forward(self, tokens: Union[Tensor, Sequence[Tensor]], evidence_logits: Tensor) -> Dict[str, Tensor]:
        tokens = self._stack_tokens(tokens)
        batch, time, _, dim = tokens.shape
        if evidence_logits.shape != (batch, time, 4):
            raise ValueError("evidence_logits must have shape [B,T,4]")
        branch_logits = torch.stack([head(tokens[:, :, index]).squeeze(-1) for index, head in enumerate(self.branch_heads)], dim=-1)
        sufficient = self.sufficient_temperature * torch.logsumexp(branch_logits / self.sufficient_temperature, dim=-1)
        relation_tokens = []
        for index, (left_index, right_index) in enumerate(itertools.combinations(range(4), 2)):
            left, right = tokens[:, :, left_index], tokens[:, :, right_index]
            relation_tokens.append(self.pair_mlps[index](torch.cat((left, right, (left - right).abs(), left * right), dim=-1)))
        all_tokens = torch.cat((tokens + self.evidence_projection(evidence_logits.unsqueeze(-1)), torch.stack(relation_tokens, dim=2)), dim=2)
        encoded = self.relation_transformer(all_tokens.reshape(batch * time, 10, dim)).reshape(batch, time, 10, dim)
        summary = encoded.mean(dim=2)
        temporal = self.temporal_depthwise(summary.transpose(1, 2))
        temporal = self.temporal_pointwise(temporal).transpose(1, 2)
        temporal = F.gelu(self.temporal_norm(temporal))
        relation_delta_raw = self.relation_head(temporal).squeeze(-1)
        relation_delta = self.relation_correction_cap * torch.tanh(relation_delta_raw)
        joint_logit = sufficient + relation_delta
        return {
            "joint_logit": joint_logit,
            "joint_score": F.softplus(joint_logit),
            "sufficient_logit": sufficient,
            "branch_logits": branch_logits,
            "relation_delta_raw": relation_delta_raw,
            "relation_delta": relation_delta,
            "relation_tokens": torch.stack(relation_tokens, dim=2),
        }
