"""Prototype-memory branch for state-space normality."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.layers.shared_catch_stem import overlap_add, patchify_time


class MemoryStateBranch(nn.Module):
    """Reads sparse normal-state prototypes instead of a high-capacity identity AE."""

    def __init__(self, config: TypeFusionConfig) -> None:
        super().__init__()
        self.config = config
        patch_dim = config.patch_size * config.c_in
        self.local_projection = nn.Sequential(
            nn.Linear(patch_dim, config.d_model), nn.GELU(), nn.LayerNorm(config.d_model)
        )
        self.query_projection = nn.Linear(config.d_model * 2, config.d_model)
        self.normal_memory = nn.Parameter(torch.randn(config.memory_size, config.d_model) * 0.02)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.patch_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, patch_dim),
        )

    def forward(self, normalized_input: Tensor, temporal_latent: Tensor) -> Dict[str, Tensor]:
        local_patches = patchify_time(
            normalized_input, self.config.patch_size, self.config.patch_stride
        )
        batch, patch_count, _, _ = local_patches.shape
        local_state = self.local_projection(local_patches.reshape(batch, patch_count, -1))
        query = self.query_projection(torch.cat([local_state, temporal_latent], dim=-1))
        memory = F.normalize(self.normal_memory, dim=-1)
        scores = torch.matmul(F.normalize(query, dim=-1), memory.transpose(0, 1))
        top_values, top_indices = scores.topk(self.config.memory_topk, dim=-1)
        sparse_scores = torch.full_like(scores, float("-inf"))
        sparse_scores.scatter_(-1, top_indices, top_values)
        attention = torch.softmax(sparse_scores, dim=-1)
        normal_state = self.output_norm(torch.matmul(attention, self.normal_memory))
        reconstruction_patches = self.patch_decoder(normal_state).view(
            batch, patch_count, self.config.patch_size, self.config.c_in
        )
        reconstruction = overlap_add(
            reconstruction_patches, self.config.seq_len, self.config.patch_stride
        )
        return {
            "z": normal_state,
            "x_hat": reconstruction,
            "e": (normalized_input - reconstruction).abs(),
            "memory_attention": attention,
        }

    def unfreeze_tail(self) -> None:
        for module in (self.output_norm, self.patch_decoder):
            for parameter in module.parameters():
                parameter.requires_grad = True
