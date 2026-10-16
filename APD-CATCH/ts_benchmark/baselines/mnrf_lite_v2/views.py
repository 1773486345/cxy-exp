"""Four causal, variable-preserving MNRF feature views."""

import torch
from torch import nn
import torch.nn.functional as F

from .backbone import PointEmbedding, CausalResidualTCN


class FourViewEncoder(nn.Module):
    def __init__(self, variables: int, width: int = 32, view_dim: int = 8):
        super().__init__()
        self.variables = variables
        self.width = width
        self.view_dim = view_dim
        self.point = PointEmbedding(variables, width)
        self.tcn = CausalResidualTCN(width)
        self.evolution_adapter = nn.Linear(width, view_dim)
        self.pattern_adapter = nn.Linear(3 * width, view_dim)
        self.relation_source = nn.Linear(3 * width, width)
        self.relation_query = nn.Linear(width, width)
        self.relation_attention = nn.MultiheadAttention(width, 2, dropout=0.0, batch_first=True)
        self.relation_adapter = nn.Linear(width, view_dim)

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
        queries = self.relation_query(identity).unsqueeze(2)
        sources = source.unsqueeze(1).expand(-1, self.variables, -1, -1)
        flat_query = queries.reshape(-1, 1, self.width)
        flat_source = sources.reshape(-1, self.variables, self.width)
        self_mask = torch.eye(self.variables, dtype=torch.bool, device=values.device).repeat(values.size(0), 1)
        attended, _ = self.relation_attention(
            flat_query, flat_source, flat_source,
            key_padding_mask=self_mask,
            need_weights=False,
        )
        relation = self.relation_adapter(attended[:, 0].reshape(values.size(0), self.variables, self.width))
        features = torch.cat((state, evolution, pattern, relation), dim=-1)
        if features.size(-1) != 25:
            raise RuntimeError("four-view layout must be [S,E,P,R] with 25 coordinates")
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
            for n in range(count):
                for d in range(variables):
                    start = int(starts[n, d])
                    target[n, d] = local[n, start:start + 8, d]
                    local[n, start:start + 8, d] = 0
                    local_valid[n, start:start + 8, d] = False
            h = self._encode_sequence(local, local_valid)
            z = self.pattern_adapter(torch.cat((h.mean(dim=1), h.max(dim=1).values, h[:, -1]), dim=-1))
            offsets = torch.arange(8, device=values.device, dtype=values.dtype).view(1, 1, 8)
            rel = (starts.to(values.dtype).unsqueeze(-1) + offsets) / 31.0
            return z, target, rel

        hist = values[:, :-1, :]
        hist_valid = valid[:, :-1, :]
        h_hist = self._encode_sequence(hist, hist_valid)[:, -1, :, :]
        current = torch.where(valid[:, -1, :], values[:, -1, :], torch.zeros_like(values[:, -1, :]))
        fields = torch.stack((current, torch.zeros_like(current), valid[:, -1, :].to(values.dtype), torch.zeros_like(current)), dim=-1)
        current_phi = F.gelu(self.point.projection(fields))
        identity = self.point.variable_identity.view(1, variables, self.width).expand(values.size(0), -1, -1)
        source = F.gelu(self.relation_source(torch.cat((h_hist, current_phi, identity), dim=-1)))
        q = self.relation_query(identity).unsqueeze(2)
        kv = source.unsqueeze(1).expand(-1, variables, -1, -1)
        flat_q = q.reshape(-1, 1, self.width)
        flat_kv = kv.reshape(-1, variables, self.width)
        mask = torch.eye(variables, dtype=torch.bool, device=values.device).repeat(values.size(0), 1)
        attended, _ = self.relation_attention(flat_q, flat_kv, flat_kv, key_padding_mask=mask, need_weights=False)
        z = self.relation_adapter(attended[:, 0].reshape(values.size(0), variables, self.width))
        return z, values[:, -1, :], torch.zeros(values.size(0), variables, device=values.device)
