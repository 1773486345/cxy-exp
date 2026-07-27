"""Information-preserving CATCH-inspired shared encoder.

The stem retains CATCH's RevIN, full FFT, frequency patching and
cross-channel Transformer idea.  It deliberately performs no anomaly
classification and exposes its intermediate normalised representation to the
specialised branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.layers.local_temporal_stem import LocalTemporalStem


@dataclass
class RevINStatistics:
    mean: Tensor
    stdev: Tensor


class RevIN(nn.Module):
    """Stateless-per-forward RevIN with explicitly returned recovery statistics."""

    def __init__(self, c_in: int, affine: bool, subtract_last: bool, eps: float = 1e-5) -> None:
        super().__init__()
        self.subtract_last = subtract_last
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(c_in))
            self.affine_bias = nn.Parameter(torch.zeros(c_in))

    def normalize(self, x: Tensor) -> Tuple[Tensor, RevINStatistics]:
        mean = x[:, -1:, :] if self.subtract_last else x.mean(dim=1, keepdim=True)
        mean = mean.detach()
        stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        normalized = (x - mean) / stdev
        if self.affine:
            normalized = normalized * self.affine_weight + self.affine_bias
        return normalized, RevINStatistics(mean=mean, stdev=stdev)

    def denormalize(self, x: Tensor, statistics: RevINStatistics) -> Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        return x * statistics.stdev + statistics.mean


def patchify_time(x: Tensor, patch_size: int, stride: int) -> Tensor:
    """Patch [B, T, C], right-padding only when CATCH's grid needs it."""

    length = x.size(1)
    patch_count = 1 if length <= patch_size else (length - patch_size + stride - 1) // stride + 1
    padded_length = (patch_count - 1) * stride + patch_size
    if padded_length > length:
        x = F.pad(x, (0, 0, 0, padded_length - length))
    # unfold yields [B, P, C, K]
    return x.unfold(dimension=1, size=patch_size, step=stride).permute(0, 1, 3, 2).contiguous()


def overlap_add(patches: Tensor, seq_len: int, stride: int) -> Tensor:
    """Overlap-add [B, P, K, C] back into [B, T, C] with average coverage."""

    batch, patch_count, patch_size, channels = patches.shape
    padded_length = (patch_count - 1) * stride + patch_size
    output = patches.new_zeros(batch, padded_length, channels)
    counts = patches.new_zeros(batch, padded_length, channels)
    for patch_index in range(patch_count):
        start = patch_index * stride
        end = start + patch_size
        output[:, start:end, :] += patches[:, patch_index, :, :]
        counts[:, start:end, :] += 1
    return (output / counts.clamp_min(1.0))[:, :seq_len, :]


class TransCChannelFusion(nn.Module):
    """CATCH-style channel fusion applied independently to every frequency patch."""

    def __init__(self, patch_dim: int, config: TypeFusionConfig) -> None:
        super().__init__()
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, config.cf_dim), nn.Dropout(config.dropout)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.cf_dim,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.e_layers)
        self.output_projection = nn.Linear(config.cf_dim, config.d_model)

    def forward(self, frequency_patch: Tensor) -> Tensor:
        # [B*P, C, 2K], with C as the Transformer sequence dimension.
        hidden = self.patch_embedding(frequency_patch)
        hidden = self.transformer(hidden)
        return self.output_projection(hidden)


class SharedCatchStem(nn.Module):
    def __init__(self, config: TypeFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.revin = RevIN(
            config.c_in, affine=bool(config.affine), subtract_last=bool(config.subtract_last)
        )
        self.channel_fusion = TransCChannelFusion(config.patch_size * 2, config)
        self.local_temporal = LocalTemporalStem(
            config.c_in,
            config.temporal_hidden_dim,
            config.temporal_layers,
            config.dropout,
        )
        self.temporal_projection = nn.Linear(config.temporal_hidden_dim, config.d_model)

    def forward(self, x: Tensor) -> Dict[str, Tensor | RevINStatistics]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, C] input, got shape {tuple(x.shape)}")
        if x.size(1) != self.config.seq_len or x.size(2) != self.config.c_in:
            raise ValueError(
                "Input shape must match config: "
                f"expected [B, {self.config.seq_len}, {self.config.c_in}], got {tuple(x.shape)}"
            )

        normalized, statistics = self.revin.normalize(x)
        spectrum = torch.fft.fft(normalized.transpose(1, 2), dim=-1)
        components = torch.view_as_real(spectrum.permute(0, 2, 1)).contiguous()
        component_patches = patchify_time(
            components.reshape(x.size(0), x.size(1), x.size(2) * 2),
            self.config.patch_size,
            self.config.patch_stride,
        )
        batch, patches, patch_size, _ = component_patches.shape
        frequency_patches = component_patches.view(
            batch, patches, patch_size, self.config.c_in, 2
        )
        channel_inputs = frequency_patches.permute(0, 1, 3, 2, 4).reshape(
            batch * patches, self.config.c_in, patch_size * 2
        )
        frequency_channels = self.channel_fusion(channel_inputs).view(
            batch, patches, self.config.c_in, self.config.d_model
        )
        temporal_sequence = self.local_temporal(normalized)
        temporal_latent = self.temporal_projection(
            patchify_time(
                temporal_sequence, self.config.patch_size, self.config.patch_stride
            ).mean(dim=2)
        )
        return {
            "normalized_input": normalized,
            "revin_statistics": statistics,
            "spectrum": spectrum,
            "frequency_channels": frequency_channels,
            "temporal_latent": temporal_latent,
        }
