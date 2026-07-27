"""Convert heterogeneous branch evidence into aligned local tokens."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.layers.shared_catch_stem import patchify_time


class EvidenceAdapter(nn.Module):
    """Maps a branch representation and dense local evidence into [B, P, D]."""

    def __init__(self, config: TypeFusionConfig, branch_index: int) -> None:
        super().__init__()
        self.config = config
        self.branch_index = branch_index
        # Evidence stays local: three summary statistics are made per time patch,
        # never from a full-window scalar score.
        self.evidence_projection = nn.Sequential(
            nn.Linear(5, config.branch_dim), nn.GELU(), nn.LayerNorm(config.branch_dim)
        )
        self.representation_projection = nn.Linear(config.d_model, config.branch_dim)
        self.output_norm = nn.LayerNorm(config.branch_dim)
        self.branch_type_embedding = nn.Parameter(torch.randn(1, 1, config.branch_dim) * 0.02)
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.num_patches, config.branch_dim) * 0.02
        )

    def forward(self, z: Tensor, evidence: Tensor, reconstruction: Tensor) -> Tensor:
        evidence_patches = patchify_time(
            evidence, self.config.patch_size, self.config.patch_stride
        )
        reconstruction_patches = patchify_time(
            reconstruction, self.config.patch_size, self.config.patch_stride
        )
        evidence_features = torch.stack(
            [
                evidence_patches.mean(dim=(2, 3)),
                evidence_patches.amax(dim=(2, 3)),
                evidence_patches.square().mean(dim=(2, 3)),
                reconstruction_patches.mean(dim=(2, 3)),
                reconstruction_patches.square().mean(dim=(2, 3)),
            ],
            dim=-1,
        )
        if z.size(1) != self.config.num_patches:
            raise ValueError("Branch representation does not align with the configured patch grid")
        token = self.evidence_projection(evidence_features) + self.representation_projection(z)
        return self.output_norm(token + self.branch_type_embedding + self.position_embedding)
