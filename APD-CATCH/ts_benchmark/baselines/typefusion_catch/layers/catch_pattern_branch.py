"""Frequency-patch reconstruction branch for segment-level normal patterns."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.layers.shared_catch_stem import overlap_add


class CATCHPatternBranch(nn.Module):
    """Masked frequency-patch reconstruction on top of shared CATCH fusion."""

    def __init__(self, config: TypeFusionConfig) -> None:
        super().__init__()
        self.config = config
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.fusion_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.frequency_transformer = nn.TransformerEncoder(layer, num_layers=config.e_layers)
        self.mask_patch_token = nn.Parameter(torch.zeros(1, 1, 1, config.d_model))
        self.channel_decoder = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.patch_size * 2),
        )
        self.token_norm = nn.LayerNorm(config.d_model)

    def _sample_patch_mask(self, batch: int, patches: int, device: torch.device) -> Tensor:
        mask = torch.rand(batch, patches, device=device) < self.config.pattern_mask_ratio
        if self.config.pattern_mask_ratio > 0:
            empty = ~mask.any(dim=1)
            mask[empty, 0] = True
        return mask

    def forward(
        self,
        normalized_input: Tensor,
        frequency_channels: Tensor,
        training_mask: bool,
    ) -> Dict[str, Tensor]:
        batch, patches, channels, hidden = frequency_channels.shape
        if training_mask:
            patch_mask = self._sample_patch_mask(batch, patches, normalized_input.device)
            masked = torch.where(
                patch_mask[:, :, None, None], self.mask_patch_token, frequency_channels
            )
        else:
            patch_mask = torch.zeros(batch, patches, dtype=torch.bool, device=normalized_input.device)
            masked = frequency_channels

        sequence = masked.permute(0, 2, 1, 3).reshape(batch * channels, patches, hidden)
        encoded = self.frequency_transformer(sequence).view(batch, channels, patches, hidden).permute(0, 2, 1, 3)
        encoded = self.token_norm(encoded)
        z = encoded.mean(dim=2)
        decoded = self.channel_decoder(encoded).view(
            batch, patches, channels, self.config.patch_size, 2
        )
        component_patches = decoded.permute(0, 1, 3, 2, 4).reshape(
            batch, patches, self.config.patch_size, channels * 2
        )
        components = overlap_add(
            component_patches, self.config.seq_len, self.config.patch_stride
        ).view(batch, self.config.seq_len, channels, 2)
        predicted_spectrum = torch.view_as_complex(components.contiguous()).permute(0, 2, 1)
        reconstruction = torch.fft.ifft(predicted_spectrum, dim=-1).real.transpose(1, 2)
        return {
            "z": z,
            "x_hat": reconstruction,
            "e": (normalized_input - reconstruction).abs(),
            "predicted_spectrum": predicted_spectrum,
            "patch_mask": patch_mask,
        }

    def unfreeze_tail(self) -> None:
        for module in (self.channel_decoder, self.token_norm):
            for parameter in module.parameters():
                parameter.requires_grad = True
