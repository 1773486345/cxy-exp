"""Channel-masked conditional reconstruction for multivariate relations."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.layers.shared_catch_stem import patchify_time


class MaskedRelationBranch(nn.Module):
    """Reconstructs each channel only in a forward pass where that channel is masked.

    All deterministic inference groups are expanded along the batch axis and
    evaluated in one vectorised invocation.  Selecting an output for a target
    channel always selects the group where that target was zero-masked.
    """

    def __init__(self, config: TypeFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.max_groups = min(config.relation_mask_groups, config.c_in)
        hidden = config.d_model
        self.value_projection = nn.Linear(1, hidden)
        self.channel_embedding = nn.Parameter(torch.randn(1, 1, config.c_in, hidden) * 0.02)
        attention = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.fusion_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_fusion = nn.TransformerEncoder(attention, num_layers=1)
        self.temporal_mixer = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.output_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.token_projection = nn.Linear(hidden, hidden)

    def _group_masks(self, device: torch.device, randomize: bool) -> Tuple[Tensor, Tensor]:
        channels = self.config.c_in
        groups = self.max_groups
        if randomize:
            order = torch.randperm(channels, device=device)
        else:
            order = torch.arange(channels, device=device)
        group_index = torch.empty(channels, dtype=torch.long, device=device)
        group_index[order] = torch.arange(channels, device=device) % groups
        masks = torch.zeros(groups, channels, dtype=torch.bool, device=device)
        masks[group_index, torch.arange(channels, device=device)] = True
        return masks, group_index

    def forward(self, normalized_input: Tensor, randomize_groups: bool) -> Dict[str, Tensor]:
        batch, time, channels = normalized_input.shape
        masks, group_index = self._group_masks(normalized_input.device, randomize_groups)
        groups = masks.size(0)

        # [B, G, T, C] contains all channel-masking conditions at once.  The
        # only tensor supplied to value_projection has target entries set to 0.
        masked_input = normalized_input[:, None, :, :].expand(batch, groups, time, channels)
        masked_input = masked_input.masked_fill(masks[None, :, None, :], 0.0)
        flat_input = masked_input.reshape(batch * groups, time, channels)
        hidden = self.value_projection(flat_input.unsqueeze(-1))
        hidden = hidden + self.channel_embedding
        hidden = self.channel_fusion(hidden.reshape(batch * groups * time, channels, -1))
        hidden = hidden.view(batch * groups, time, channels, -1)

        # Temporal mixing occurs independently for each channel after masking.
        temporal = hidden.permute(0, 2, 3, 1).reshape(batch * groups * channels, -1, time)
        temporal = self.temporal_mixer(temporal).view(batch * groups, channels, -1, time).permute(0, 3, 1, 2)
        prediction_all = self.output_head(temporal).squeeze(-1).view(batch, groups, time, channels)
        hidden_all = temporal.view(batch, groups, time, channels, -1)

        # A channel's selected prediction comes exclusively from the group where
        # that channel is masked.  No residual or concat path sees its value.
        selected_prediction = prediction_all.permute(0, 2, 1, 3).gather(
            2,
            group_index.view(1, 1, 1, channels).expand(batch, time, 1, channels),
        ).squeeze(2)
        selected_hidden = hidden_all.permute(0, 2, 1, 3, 4).gather(
            2,
            group_index.view(1, 1, 1, channels, 1).expand(batch, time, 1, channels, hidden_all.size(-1)),
        ).squeeze(2)
        z = self.token_projection(
            patchify_time(selected_hidden.mean(dim=2), self.config.patch_size, self.config.patch_stride).mean(dim=2)
        )
        return {
            "z": z,
            "x_hat": selected_prediction,
            "e": (normalized_input - selected_prediction).abs(),
            "channel_masks": masks,
            "group_index": group_index,
        }

    def unfreeze_tail(self) -> None:
        for module in (self.output_head, self.token_projection):
            for parameter in module.parameters():
                parameter.requires_grad = True
