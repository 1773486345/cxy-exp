"""Decode predicted normal branch tokens into a joint normal reconstruction."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.layers.shared_catch_stem import overlap_add


class JointNormalDecoder(nn.Module):
    """Consumes only leave-one-out normal tokens and learned decoding queries."""

    def __init__(self, config: TypeFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_queries = nn.Parameter(torch.randn(1, config.num_patches, 1, config.branch_dim) * 0.02)
        self.branch_attention = nn.MultiheadAttention(
            config.branch_dim, config.fusion_heads, dropout=config.dropout, batch_first=True
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.branch_dim,
            nhead=config.fusion_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_decoder = nn.TransformerEncoder(temporal_layer, num_layers=1)
        self.patch_output = nn.Sequential(
            nn.LayerNorm(config.branch_dim),
            nn.Linear(config.branch_dim, config.patch_size * config.c_in),
        )

    def forward(self, q_normal: Tensor) -> Tensor:
        batch, patches, branches, hidden = q_normal.shape
        if (patches, branches, hidden) != (self.config.num_patches, 4, self.config.branch_dim):
            raise ValueError("Q_normal shape does not match the configured decoder grid")
        query = self.patch_queries.expand(batch, -1, -1, -1).reshape(batch * patches, 1, hidden)
        branch_tokens = q_normal.reshape(batch * patches, branches, hidden)
        fused, _ = self.branch_attention(query, branch_tokens, branch_tokens, need_weights=False)
        fused = self.temporal_decoder(fused.view(batch, patches, hidden))
        patch_values = self.patch_output(fused).view(
            batch, patches, self.config.patch_size, self.config.c_in
        )
        return overlap_add(patch_values, self.config.seq_len, self.config.patch_stride)
