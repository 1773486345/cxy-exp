"""Per-variable point embedding and shared strictly causal TCN."""

import torch
from torch import nn
import torch.nn.functional as F


class PointEmbedding(nn.Module):
    def __init__(self, variables: int, width: int = 32):
        super().__init__()
        self.variables = int(variables)
        self.projection = nn.Linear(4, width)
        self.variable_identity = nn.Parameter(torch.randn(variables, width) * 0.02)

    def forward(self, values, valid=None):
        if values.ndim != 3 or values.size(-1) != self.variables:
            raise ValueError("values must have shape [N,L,D]")
        if valid is None:
            valid = torch.ones_like(values, dtype=torch.bool)
        observed = valid.to(values.dtype)
        safe = torch.where(valid, values, torch.zeros_like(values))
        difference = torch.zeros_like(safe)
        difference_valid = torch.zeros_like(valid)
        adjacent = valid[:, 1:, :] & valid[:, :-1, :]
        difference[:, 1:, :] = torch.where(
            adjacent, safe[:, 1:, :] - safe[:, :-1, :], torch.zeros_like(safe[:, 1:, :])
        )
        difference_valid[:, 1:, :] = adjacent
        fields = torch.stack((safe, difference, observed, difference_valid.to(values.dtype)), dim=-1)
        hidden = F.gelu(self.projection(fields))
        return hidden + self.variable_identity.view(1, 1, self.variables, -1)


class CausalDepthwiseLayer(nn.Module):
    def __init__(self, width: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.left = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(width, width, kernel_size, dilation=dilation, groups=width, bias=False)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        update = self.depthwise(F.pad(x, (self.left, 0)))
        update = self.pointwise(update).transpose(1, 2)
        update = self.norm(update).transpose(1, 2)
        return F.gelu(update)


class CausalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.layers = nn.Sequential(
            CausalDepthwiseLayer(width, dilation),
            CausalDepthwiseLayer(width, dilation),
        )

    def forward(self, x):
        return x + self.layers(x)


class CausalResidualTCN(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.dilations = (1, 2, 4, 8, 16, 32)
        self.blocks = nn.ModuleList(CausalResidualBlock(width, dilation) for dilation in self.dilations)

    @property
    def receptive_field(self):
        return 1 + 4 * sum(self.dilations)

    def forward(self, x):
        # [N,L,D,H] -> independent sequences [N*D,H,L]
        batch, time, variables, width = x.shape
        hidden = x.permute(0, 2, 3, 1).reshape(batch * variables, width, time)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden.reshape(batch, variables, width, time).permute(0, 3, 1, 2)
