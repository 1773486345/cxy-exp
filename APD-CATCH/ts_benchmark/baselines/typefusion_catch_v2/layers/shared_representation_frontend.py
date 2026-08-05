"""Shared time and CATCH-style frequency encoders for TypeFusion-CATCH v2."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import TypeFusionCATCHV2Config
from .patch_utils import patchify_time


class RevIN(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def normalize(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        mean = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - mean) / stdev, {"mean": mean, "stdev": stdev}


class SharedRepresentationFrontend(nn.Module):
    """Information-preserving paths with explicit time/frequency entry points.

    ``encode_time`` does not compute a frequency graph.  This is important for
    the State-only path: Pattern owns its two masked frequency computations and
    therefore cannot accidentally receive an unmasked target representation.
    """

    def __init__(self, config: TypeFusionCATCHV2Config) -> None:
        super().__init__()
        self.config = config
        self.time_input = nn.Linear(config.c_in, config.d_model)
        self.time_depthwise = nn.Conv1d(config.d_model, config.d_model, 3, padding=1, groups=config.d_model)
        self.time_pointwise = nn.Conv1d(config.d_model, config.d_model, 1)
        self.time_norm = nn.LayerNorm(config.d_model)
        self.revin = RevIN()
        self.frequency_patch_embedding = nn.Linear(config.patch_size * 2, config.cf_dim)
        frequency_layer = nn.TransformerEncoderLayer(
            d_model=config.cf_dim,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.frequency_channel_encoder = nn.TransformerEncoder(frequency_layer, num_layers=config.e_layers)
        self.frequency_projection = nn.Linear(config.cf_dim, config.d_model)

    def _validate(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.size(1) != self.config.seq_len or x.size(2) != self.config.c_in:
            raise ValueError(f"Expected [B,{self.config.seq_len},{self.config.c_in}], got {tuple(x.shape)}")
        return x.float().contiguous()

    def encode_time(self, x: Tensor) -> Dict[str, Tensor | Dict[str, Tensor]]:
        """Encode only local time information for the State branch."""
        x = self._validate(x)
        time = self.time_input(x)
        time = self.time_depthwise(time.transpose(1, 2))
        time = self.time_pointwise(time).transpose(1, 2)
        h_time = self.time_norm(F.gelu(time))
        normalized, statistics = self.revin.normalize(x)
        return {
            "h_time": h_time,
            "normalized_input": normalized.detach(),
            "revin_statistics": {key: value.detach() for key, value in statistics.items()},
        }

    def encode_frequency(self, x: Tensor) -> Dict[str, Tensor | Dict[str, Tensor]]:
        """Encode one already-masked view with public frequency patch stride."""
        x = self._validate(x)
        normalized, statistics = self.revin.normalize(x)
        spectrum = torch.fft.fft(normalized.transpose(1, 2), dim=-1)
        real_imag = torch.stack((spectrum.real, spectrum.imag), dim=-1).permute(0, 2, 1, 3)
        frequency_input = real_imag.reshape(x.size(0), x.size(1), x.size(2) * 2)
        patches = patchify_time(frequency_input, self.config.patch_size, self.config.patch_stride)
        batch, count, size, channels2 = patches.shape
        channels = self.config.c_in
        channel_patches = patches.view(batch, count, size, channels, 2).permute(0, 1, 3, 2, 4)
        channel_patches = channel_patches.reshape(batch * count, channels, size * 2)
        hidden = self.frequency_patch_embedding(channel_patches)
        hidden = self.frequency_channel_encoder(hidden)
        h_freq = self.frequency_projection(hidden).view(batch, count, channels, self.config.d_model)
        return {
            "h_freq": h_freq,
            "normalized_input": normalized.detach(),
            "revin_statistics": {key: value.detach() for key, value in statistics.items()},
        }

    def forward(self, x: Tensor) -> Dict[str, Tensor | Dict[str, Tensor]]:
        # Keep the legacy callable as the State/time-only path.  Frequency
        # encoding is intentionally explicit through encode_frequency().
        return self.encode_time(x)
