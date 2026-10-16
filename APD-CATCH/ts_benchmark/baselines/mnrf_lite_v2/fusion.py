"""Four invertible, alternating A/B attention-conditioned affine couplings."""

import math

import torch
from torch import nn
import torch.nn.functional as F


MECHANISM_SLICES = {"S": (0, 1), "E": (1, 9), "P": (9, 17), "R": (17, 25)}
A_MECHANISMS = ("S", "R")
B_MECHANISMS = ("E", "P")


class ConditionalCoupling(nn.Module):
    def __init__(self, source_names, target_names, dimensions, width=32):
        super().__init__()
        self.source_names = tuple(source_names)
        self.target_names = tuple(target_names)
        self.source_projection = nn.ModuleDict({
            name: nn.Linear(dimensions[name], width, bias=False)
            for name in self.source_names
        })
        self.query_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 2, dropout=0.0, batch_first=True)
        self.attention_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, width))
        self.output_norm = nn.LayerNorm(width)
        self.outputs = nn.ModuleDict({
            name: nn.Linear(width, 2 * dimensions[name]) for name in self.target_names
        })
        for layer in self.outputs.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def condition(self, values, variable_identity, fusion_identity, target_identities):
        batch, variables, _ = values.shape
        bias = variable_identity
        source_tokens = []
        for name in self.source_names:
            start, end = MECHANISM_SLICES[name]
            source_tokens.append(
                self.source_projection[name](values[..., start:end])
                + fusion_identity[name].view(1, 1, -1)
                + bias
            )
        source = self.source_norm(torch.stack(source_tokens, dim=2).reshape(batch * variables, 2, -1))
        target_ids = torch.stack([target_identities[name] for name in self.target_names])
        query = (target_ids.view(1, 1, len(self.target_names), -1) + bias.unsqueeze(2))
        query = self.query_norm(query.reshape(batch * variables, len(self.target_names), -1))
        attended, _ = self.attention(query, source, source, need_weights=False)
        hidden = self.attention_norm(query + attended)
        hidden = self.output_norm(hidden + self.feedforward(hidden))
        return {
            name: self.outputs[name](hidden[:, index]).reshape(batch, variables, -1)
            for index, name in enumerate(self.target_names)
        }


class AlternatingAffineFlow(nn.Module):
    """Maps 25-D per-variable features to a standard normal base density."""

    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.fusion_identity = nn.ParameterDict({
            name: nn.Parameter(torch.randn(width) * 0.02) for name in ("S", "E", "P", "R")
        })
        # Block-specific variable maps and coupling networks are independent.
        layouts = ((A_MECHANISMS, B_MECHANISMS), (B_MECHANISMS, A_MECHANISMS),
                   (A_MECHANISMS, B_MECHANISMS), (B_MECHANISMS, A_MECHANISMS))
        self.variable_maps = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in layouts)
        self.blocks = nn.ModuleList(
            ConditionalCoupling(src, dst, {"S": 1, "E": 8, "P": 8, "R": 8}, width)
            for src, dst in layouts
        )
        self.A = tuple([0] + list(range(17, 25)))
        self.B = tuple(range(1, 17))
        self.variable_count = None

    def _identity(self, encoder_identity, batch, variables):
        if encoder_identity.ndim == 2:
            encoder_identity = encoder_identity.unsqueeze(0).expand(batch, -1, -1)
        if encoder_identity.shape != (batch, variables, self.width):
            raise ValueError("encoder variable identity must have shape [D,32] or [N,D,32]")
        return encoder_identity

    def _couple(self, values, encoder_identity, reverse=False):
        batch, variables, dimensions = values.shape
        if dimensions != 25:
            raise ValueError("flow input must be [N,D,25]")
        identity = self._identity(encoder_identity, batch, variables)
        result = values
        logdet = values.new_zeros(batch, variables)
        order = range(3, -1, -1) if reverse else range(4)
        for block_index in order:
            block = self.blocks[block_index]
            variable = self.variable_maps[block_index](identity)
            params = block.condition(result, variable, self.fusion_identity, self.fusion_identity)
            for name in block.target_names:
                start, end = MECHANISM_SLICES[name]
                raw = params[name]
                size = end - start
                shift_raw, shift = raw[..., :size], raw[..., size:]
                scale = torch.tanh(shift_raw)
                updated = result[..., start:end]
                if reverse:
                    updated = (updated - shift) * torch.exp(-scale)
                    logdet = logdet - scale.sum(dim=-1)
                else:
                    updated = updated * torch.exp(scale) + shift
                    logdet = logdet + scale.sum(dim=-1)
                result = torch.cat((result[..., :start], updated, result[..., end:]), dim=-1)
        return result, logdet

    def forward(self, values, encoder_identity):
        z, logdet = self._couple(values, encoder_identity, reverse=False)
        nll = 0.5 * z.square().sum(dim=-1) + (25.0 / 2.0) * math.log(2.0 * math.pi) - logdet
        return {"z": z, "logdet": logdet, "nll": nll}

    def inverse(self, z, encoder_identity):
        return self._couple(z, encoder_identity, reverse=True)
