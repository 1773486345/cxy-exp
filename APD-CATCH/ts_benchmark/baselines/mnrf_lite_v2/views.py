"""Four causal, variable-preserving MNRF feature views."""

import torch
from torch import nn
import torch.nn.functional as F

from .backbone import PointEmbedding, CausalResidualTCN
from .layout import FeatureLayout


def mask_pattern_values(local, local_valid, starts, span=8):
    """Vectorized per-sample/per-variable contiguous masking before differencing."""
    count, length, variables = local.shape
    starts = starts.to(device=local.device, dtype=torch.long)
    offsets = torch.arange(span, device=local.device).view(1, 1, span)
    positions = starts.unsqueeze(-1) + offsets
    target = local.gather(1, positions.permute(0, 2, 1))
    time = torch.arange(length, device=local.device).view(1, length, 1)
    masked = (time >= starts.unsqueeze(1)) & (time < starts.unsqueeze(1) + span)
    return local.masked_fill(masked, 0), local_valid & ~masked, target.permute(0, 2, 1), (positions.to(local.dtype) / 31.0)


class FourViewEncoder(nn.Module):
    def __init__(self, variables: int, width: int = 32, view_dim: int = 8,
                 layout: FeatureLayout = None):
        super().__init__()
        layout = layout or FeatureLayout(view_dim, view_dim, view_dim)
        self.variables = variables
        self.width = width
        self.layout = layout
        self.view_dim = view_dim
        self.point = PointEmbedding(variables, width)
        self.tcn = CausalResidualTCN(width)
        self.evolution_adapter = nn.Linear(width, layout.evolution_dim)
        self.pattern_adapter = nn.Linear(3 * width, layout.pattern_dim)
        self.relation_source = nn.Linear(3 * width, width)
        self.relation_query = nn.Linear(width, width)
        self.relation_attention = nn.MultiheadAttention(width, 2, dropout=0.0, batch_first=True)
        self.relation_adapter = nn.Linear(width, layout.relation_dim)

    def _encode_sequence(self, values, valid=None):
        return self.tcn(self.point(values, valid))

    def encode_views(self, values, valid=None):
        if values.ndim != 3 or values.size(-1) != self.variables:
            raise ValueError("expected windows [N,L,D] in fitted variable order")
        if values.size(1) < 32:
            raise ValueError("window shorter than fixed pattern length 32")
        if valid is None:
            valid = torch.ones_like(values, dtype=torch.bool)
        hist = values[:, :-1, :]
        hist_valid = valid[:, :-1, :]
        h_hist = self._encode_sequence(hist, hist_valid)[:, -1, :, :]
        current = torch.where(valid[:, -1, :], values[:, -1, :], torch.zeros_like(values[:, -1, :]))
        # S is a direct value coordinate, with no trainable state network.
        state = current.unsqueeze(-1)
        evolution = self.evolution_adapter(h_hist)

        pattern_values = values[:, -32:, :]
        pattern_valid = valid[:, -32:, :]
        h_pattern = self._encode_sequence(pattern_values, pattern_valid)
        pattern = self.pattern_adapter(torch.cat((
            h_pattern.mean(dim=1),
            h_pattern.max(dim=1).values,
            h_pattern[:, -1],
        ), dim=-1))

        current_fields = torch.stack((
            current,
            torch.zeros_like(current),
            valid[:, -1, :].to(values.dtype),
            torch.zeros_like(current),
        ), dim=-1)
        current_phi = F.gelu(self.point.projection(current_fields))
        identity = self.point.variable_identity.view(1, self.variables, self.width).expand(values.size(0), -1, -1)
        source_input = torch.cat((h_hist, current_phi, identity), dim=-1)
        source = F.gelu(self.relation_source(source_input))
        relation = self._relation_from_source(source, identity)
        features = torch.cat((state, evolution, pattern, relation), dim=-1)
        if features.size(-1) != self.layout.total_dim:
            raise RuntimeError(f"four-view layout expected F={self.layout.total_dim}, got {features.size(-1)}")
        return {
            "features": features,
            "state": state,
            "evolution": evolution,
            "pattern": pattern,
            "relation": relation,
            "h_hist": h_hist,
        }

    def encode_pretrain(self, values, valid=None, task="evolution", mask_start=None):
        if task not in {"evolution", "pattern", "relation"}:
            raise ValueError(f"unsupported pretraining task: {task}")
        if valid is None:
            valid = torch.ones_like(values, dtype=torch.bool)
        variables = values.size(-1)
        if task == "evolution":
            hist = values[:, :-1, :]
            hist_valid = valid[:, :-1, :]
            h_hist = self._encode_sequence(hist, hist_valid)[:, -1, :, :]
            return self.evolution_adapter(h_hist), values[:, -1, :], torch.zeros(values.size(0), values.size(-1), device=values.device)
        if task == "pattern":
            count, variables = values.size(0), values.size(-1)
            starts = mask_start
            if starts is None:
                starts = torch.randint(0, 25, (count, variables), device=values.device)
            local = values[:, -32:, :].clone()
            local_valid = valid[:, -32:, :].clone()
            target = torch.zeros(count, variables, 8, device=values.device, dtype=values.dtype)
            local, local_valid, target, relative_positions = mask_pattern_values(local, local_valid, starts)
            h = self._encode_sequence(local, local_valid)
            z = self.pattern_adapter(torch.cat((h.mean(dim=1), h.max(dim=1).values, h[:, -1]), dim=-1))
            return z, target, relative_positions

        hist = values[:, :-1, :]
        hist_valid = valid[:, :-1, :]
        h_hist = self._encode_sequence(hist, hist_valid)[:, -1, :, :]
        current = torch.where(valid[:, -1, :], values[:, -1, :], torch.zeros_like(values[:, -1, :]))
        fields = torch.stack((current, torch.zeros_like(current), valid[:, -1, :].to(values.dtype), torch.zeros_like(current)), dim=-1)
        current_phi = F.gelu(self.point.projection(fields))
        identity = self.point.variable_identity.view(1, variables, self.width).expand(values.size(0), -1, -1)
        source = F.gelu(self.relation_source(torch.cat((h_hist, current_phi, identity), dim=-1)))
        z = self._relation_from_source(source, identity)
        return z, values[:, -1, :], torch.zeros(values.size(0), variables, device=values.device)

    def _relation_from_source(self, source, identity):
        # MHA's boolean attn_mask uses True for disallowed query-key pairs.
        query = self.relation_query(identity)
        diagonal_block = torch.eye(self.variables, dtype=torch.bool, device=source.device)
        attended, _ = self.relation_attention(
            query, source, source, attn_mask=diagonal_block, need_weights=False
        )
        return self.relation_adapter(attended)
