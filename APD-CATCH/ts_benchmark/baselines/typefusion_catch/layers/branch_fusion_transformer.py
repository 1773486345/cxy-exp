"""Conflict-aware token fusion with masked normal branch prediction."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig


class BranchFusionTransformer(nn.Module):
    """Learns normal branch-token relationships, never scalar score weights."""

    branch_count = 4

    def __init__(self, config: TypeFusionConfig) -> None:
        super().__init__()
        self.config = config
        layer = nn.TransformerEncoderLayer(
            d_model=config.branch_dim,
            nhead=config.fusion_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.branch_transformer = nn.TransformerEncoder(layer, num_layers=config.fusion_layers)
        self.mask_branch_tokens = nn.Parameter(torch.randn(1, 1, self.branch_count, config.branch_dim) * 0.02)
        self.temporal_mixer = nn.Conv1d(
            config.branch_dim, config.branch_dim, kernel_size=3, padding=1, groups=config.branch_dim
        )
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(config.branch_dim),
            nn.Linear(config.branch_dim, config.branch_dim),
        )

    def _apply_mask(self, q: Tensor, branch_mask: Tensor) -> Tensor:
        if branch_mask.shape != (q.size(0), self.branch_count):
            raise ValueError(
                f"branch_mask must be [B, {self.branch_count}], got {tuple(branch_mask.shape)}"
            )
        return torch.where(
            branch_mask[:, None, :, None], self.mask_branch_tokens.expand(q.size(0), q.size(1), -1, -1), q
        )

    def forward(self, q: Tensor, branch_mask: Tensor) -> Tensor:
        """Predict all branch tokens after replacing requested branches with masks."""

        masked_q = self._apply_mask(q, branch_mask)
        batch, patches, branches, hidden = masked_q.shape
        branch_encoded = self.branch_transformer(masked_q.reshape(batch * patches, branches, hidden))
        branch_encoded = branch_encoded.view(batch, patches, branches, hidden)
        # This is deliberately shallow temporal interaction after the primary
        # branch-axis Transformer.
        temporal = branch_encoded.permute(0, 2, 3, 1).reshape(batch * branches, hidden, patches)
        temporal = self.temporal_mixer(temporal).view(batch, branches, hidden, patches).permute(0, 3, 1, 2)
        return self.prediction_head(temporal)

    def masked_branch_prediction(self, q: Tensor) -> Dict[str, Tensor]:
        """Training objective: mask one or two branch types per sample."""

        batch = q.size(0)
        counts = torch.randint(1, 3, (batch,), device=q.device)
        random_order = torch.rand(batch, self.branch_count, device=q.device).argsort(dim=-1)
        selected_ranks = torch.arange(self.branch_count, device=q.device).unsqueeze(0) < counts.unsqueeze(1)
        branch_mask = torch.zeros(batch, self.branch_count, dtype=torch.bool, device=q.device)
        branch_mask.scatter_(dim=1, index=random_order, src=selected_ranks)
        prediction = self.forward(q, branch_mask)
        mask = branch_mask[:, None, :, None].expand_as(prediction)
        target = q.detach()
        loss = (prediction[mask] - target[mask]).square().mean()
        return {"prediction": prediction, "branch_mask": branch_mask, "loss": loss}

    def leave_one_out(self, q: Tensor) -> Dict[str, Tensor]:
        """Predict each normal branch token in a single B*4 Transformer pass."""

        batch, patches, branches, hidden = q.shape
        if branches != self.branch_count:
            raise ValueError(f"Expected {self.branch_count} branch tokens, got {branches}")
        loo_masks = torch.eye(self.branch_count, dtype=torch.bool, device=q.device)
        expanded_q = q[:, None, :, :, :].expand(batch, self.branch_count, patches, branches, hidden)
        flattened_q = expanded_q.reshape(batch * self.branch_count, patches, branches, hidden)
        flattened_masks = loo_masks[None, :, :].expand(batch, -1, -1).reshape(batch * self.branch_count, branches)
        prediction = self.forward(flattened_q, flattened_masks)
        prediction = prediction.view(batch, self.branch_count, patches, branches, hidden)
        branch_indices = torch.arange(self.branch_count, device=q.device)
        selector = branch_indices.view(1, self.branch_count, 1, 1, 1).expand(
            batch, self.branch_count, patches, 1, hidden
        )
        normal_tokens = prediction.gather(dim=3, index=selector).squeeze(3).permute(0, 2, 1, 3).contiguous()
        return {
            "q_normal": normal_tokens,
            "expanded_batch": torch.tensor(batch * self.branch_count, device=q.device),
            "loo_masks": flattened_masks,
        }

    @staticmethod
    def leave_one_out_mask_loss(q: Tensor, leave_one_out: Dict[str, Tensor]) -> Tensor:
        """Deterministic validation loss on exactly the four masked tokens."""

        prediction = leave_one_out["q_normal"]
        return (prediction - q.detach()).square().mean()
