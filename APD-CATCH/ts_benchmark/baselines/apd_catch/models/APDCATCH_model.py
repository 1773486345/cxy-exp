"""Target-blind adaptive decomposition on top of CATCH frequency patches."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.channel_mask import channel_mask_generator
from ..layers.cross_channel_Transformer import Trans_C


class APDCATCHModel(nn.Module):
    """Predict the next multivariate observation from a strictly past-only window."""

    VALID_VARIANTS = {"causal_catch", "fixed", "adaptive"}

    def __init__(self, configs):
        super().__init__()
        if configs.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(self.VALID_VARIANTS)}, got {configs.variant!r}"
            )

        self.variant = configs.variant
        self.seq_len = configs.seq_len
        self.n_vars = configs.c_in
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.cutoff_min = configs.cutoff_min
        self.cutoff_max = configs.cutoff_max
        self.fixed_cutoff = configs.fixed_cutoff
        self.cutoff_temperature = configs.cutoff_temperature
        self.minimum_scale = configs.minimum_scale
        self.maximum_scale = configs.maximum_scale
        self.register_buffer("scale_floor", torch.ones(self.n_vars))

        spectrum_size = self.seq_len // 2 + 1
        if self.patch_size > spectrum_size:
            raise ValueError(
                f"patch_size={self.patch_size} exceeds rFFT size={spectrum_size}"
            )
        remainder = (spectrum_size - self.patch_size) % self.patch_stride
        self.spectrum_padding = (self.patch_stride - remainder) % self.patch_stride
        self.patch_num = (
            (spectrum_size + self.spectrum_padding - self.patch_size)
            // self.patch_stride
            + 1
        )

        normalized_frequency = torch.fft.rfftfreq(self.seq_len)
        normalized_frequency = normalized_frequency / normalized_frequency[-1].clamp_min(1e-8)
        self.register_buffer("normalized_frequency", normalized_frequency)

        router_hidden = max(configs.cf_dim, 16)
        self.cutoff_router = nn.Sequential(
            nn.LayerNorm(spectrum_size),
            nn.Linear(spectrum_size, router_hidden),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(router_hidden, 1),
        )

        self.mask_generator = channel_mask_generator(
            input_size=self.patch_size,
            n_vars=self.n_vars,
        )
        self.frequency_transformer = Trans_C(
            dim=configs.cf_dim,
            depth=configs.e_layers,
            heads=configs.n_heads,
            mlp_dim=configs.d_ff,
            dim_head=configs.head_dim,
            dropout=configs.dropout,
            patch_dim=self.patch_size * 2,
            horizon=2,
            d_model=configs.d_model * 2,
            regular_lambda=configs.regular_lambda,
            temperature=configs.temperature,
        )

        encoded_dim = configs.d_model * 2
        flattened_dim = encoded_dim * self.patch_num
        head_hidden = max(configs.d_ff, encoded_dim)
        self.low_head = nn.Sequential(
            nn.LayerNorm(flattened_dim),
            nn.Linear(flattened_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(configs.head_dropout),
            nn.Linear(head_hidden, 1),
        )
        self.high_head = nn.Sequential(
            nn.LayerNorm(flattened_dim),
            nn.Linear(flattened_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(configs.head_dropout),
            nn.Linear(head_hidden, 1),
        )
        self.scale_head = nn.Sequential(
            nn.LayerNorm(encoded_dim * 2 + 1),
            nn.Linear(encoded_dim * 2 + 1, head_hidden),
            nn.GELU(),
            nn.Dropout(configs.head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def _cutoff(self, spectrum: torch.Tensor) -> torch.Tensor:
        magnitude = torch.log1p(spectrum.abs())
        router_input = magnitude.reshape(-1, magnitude.shape[-1])
        adaptive = torch.sigmoid(self.cutoff_router(router_input)).reshape(
            spectrum.shape[0], spectrum.shape[1]
        )
        adaptive = self.cutoff_min + (self.cutoff_max - self.cutoff_min) * adaptive
        if self.variant == "adaptive":
            return adaptive
        return torch.full_like(adaptive, self.fixed_cutoff)

    def set_scale_floor(self, scale_floor: torch.Tensor) -> None:
        """Set fixed, training-derived per-variable scale floors."""
        scale_floor = torch.as_tensor(
            scale_floor, dtype=self.scale_floor.dtype, device=self.scale_floor.device
        )
        if scale_floor.shape != self.scale_floor.shape:
            raise ValueError(
                f"scale_floor must have shape {tuple(self.scale_floor.shape)}, "
                f"got {tuple(scale_floor.shape)}"
            )
        if not torch.isfinite(scale_floor).all() or torch.any(scale_floor <= 0):
            raise ValueError("scale_floor must be finite and strictly positive")
        self.scale_floor.copy_(scale_floor)

    def _decompose(
        self, spectrum: torch.Tensor, cutoff: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.variant == "causal_catch":
            return spectrum, torch.zeros_like(spectrum), spectrum.real.new_zeros(())

        frequency = self.normalized_frequency.view(1, 1, -1)
        low_mask = torch.sigmoid(
            (cutoff.unsqueeze(-1) - frequency) / self.cutoff_temperature
        )
        high_mask = 1.0 - low_mask
        partition_error = (low_mask + high_mask - 1.0).abs().max()
        return spectrum * low_mask, spectrum * high_mask, partition_error

    def _frequency_patches(self, spectrum: torch.Tensor) -> torch.Tensor:
        if self.spectrum_padding:
            padding = spectrum[..., -1:].expand(
                *spectrum.shape[:-1], self.spectrum_padding
            )
            spectrum = torch.cat((spectrum, padding), dim=-1)
        real = spectrum.real.unfold(-1, self.patch_size, self.patch_stride)
        imag = spectrum.imag.unfold(-1, self.patch_size, self.patch_stride)
        patches = torch.cat((real, imag), dim=-1)
        return patches.permute(0, 2, 1, 3).contiguous()

    def _encode_component(
        self, spectrum: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patches = self._frequency_patches(spectrum)
        batch_size, patch_num, n_vars, patch_dim = patches.shape
        flattened = patches.reshape(batch_size * patch_num, n_vars, patch_dim)
        channel_mask = self.mask_generator(flattened)
        encoded, channel_loss = self.frequency_transformer(
            flattened, channel_mask
        )
        encoded = encoded.reshape(batch_size, patch_num, n_vars, -1)
        return encoded.permute(0, 2, 1, 3).contiguous(), channel_loss

    def forward(self, history: torch.Tensor) -> Dict[str, torch.Tensor]:
        if history.ndim != 3:
            raise ValueError(
                f"history must have shape [batch, time, variable], got {history.shape}"
            )
        if history.shape[1] != self.seq_len or history.shape[2] != self.n_vars:
            raise ValueError(
                "history shape does not match configured sequence length and variables: "
                f"expected [*, {self.seq_len}, {self.n_vars}], got {history.shape}"
            )

        location = history.mean(dim=1, keepdim=True).detach()
        history_median = history.median(dim=1, keepdim=True).values
        local_mad = (history - history_median).abs().median(dim=1, keepdim=True).values
        local_scale = 1.4826 * local_mad
        scale = torch.sqrt(
            local_scale.square() + self.scale_floor.view(1, 1, -1).square()
        ).detach()
        normalized = (history - location) / scale
        spectrum = torch.fft.rfft(normalized.permute(0, 2, 1), dim=-1)

        cutoff = self._cutoff(spectrum)
        low_spectrum, high_spectrum, partition_error = self._decompose(
            spectrum, cutoff
        )
        low_encoded, low_channel_loss = self._encode_component(low_spectrum)

        if self.variant == "causal_catch":
            high_encoded = torch.zeros_like(low_encoded)
            channel_loss = low_channel_loss
        else:
            high_encoded, high_channel_loss = self._encode_component(high_spectrum)
            channel_loss = 0.5 * (low_channel_loss + high_channel_loss)

        low_flat = low_encoded.flatten(start_dim=2)
        high_flat = high_encoded.flatten(start_dim=2)
        mean_normalized = self.low_head(low_flat).squeeze(-1)
        mean_normalized = mean_normalized + self.high_head(high_flat).squeeze(-1)

        pooled = torch.cat(
            (
                low_encoded.mean(dim=2),
                high_encoded.mean(dim=2),
                cutoff.unsqueeze(-1),
            ),
            dim=-1,
        )
        scale_normalized = F.softplus(self.scale_head(pooled).squeeze(-1))
        scale_normalized = scale_normalized.clamp(
            min=self.minimum_scale,
            max=self.maximum_scale,
        )

        location = location.squeeze(1)
        history_scale = scale.squeeze(1)
        prediction_mean = location + history_scale * mean_normalized
        prediction_scale = history_scale * scale_normalized

        return {
            "mean": prediction_mean,
            "scale": prediction_scale,
            "cutoff": cutoff,
            "channel_loss": channel_loss,
            "partition_error": partition_error,
        }


def gaussian_nll(
    target: torch.Tensor,
    prediction_mean: torch.Tensor,
    prediction_scale: torch.Tensor,
) -> torch.Tensor:
    residual = (target - prediction_mean) / prediction_scale
    return 0.5 * residual.square() + torch.log(prediction_scale)
