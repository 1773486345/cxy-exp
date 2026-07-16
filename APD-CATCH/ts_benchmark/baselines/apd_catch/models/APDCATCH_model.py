"""Causal state-innovation adaptation of the CATCH frequency-channel backbone."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.channel_mask import channel_mask_generator
from ..layers.cross_channel_Transformer import Trans_C


class APDCATCHModel(nn.Module):
    """Predict a target-blind conditional distribution for the next observation.

    CATCH remains the frequency-patch and channel-relation encoder.  The state
    variants first remove a deterministic causal EMA state, so CATCH explains
    innovation structure rather than the raw level of the series.
    """

    VALID_VARIANTS = {"causal_catch", "state", "state_scale"}

    def __init__(self, configs):
        super().__init__()
        if configs.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(self.VALID_VARIANTS)}, "
                f"got {configs.variant!r}"
            )

        self.variant = configs.variant
        self.seq_len = configs.seq_len
        self.n_vars = configs.c_in
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.state_span = max(
            2,
            min(self.seq_len, int(round(self.seq_len * configs.state_span_ratio))),
        )
        self.register_buffer("reference_location", torch.zeros(self.n_vars))
        self.register_buffer("reference_scale", torch.ones(self.n_vars))

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
        self.mean_head = nn.Sequential(
            nn.LayerNorm(flattened_dim),
            nn.Linear(flattened_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(configs.head_dropout),
            nn.Linear(head_hidden, 1),
        )
        self.scale_head = nn.Sequential(
            nn.LayerNorm(encoded_dim),
            nn.Linear(encoded_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(configs.head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def set_reference_normalization(
        self, location: torch.Tensor, scale: torch.Tensor
    ) -> None:
        """Set fixed training-derived normalization statistics."""
        location = torch.as_tensor(
            location,
            dtype=self.reference_location.dtype,
            device=self.reference_location.device,
        )
        scale = torch.as_tensor(
            scale,
            dtype=self.reference_scale.dtype,
            device=self.reference_scale.device,
        )
        if location.shape != self.reference_location.shape:
            raise ValueError(
                f"location must have shape {tuple(self.reference_location.shape)}, "
                f"got {tuple(location.shape)}"
            )
        if scale.shape != self.reference_scale.shape:
            raise ValueError(
                f"scale must have shape {tuple(self.reference_scale.shape)}, "
                f"got {tuple(scale.shape)}"
            )
        if not torch.isfinite(location).all():
            raise ValueError("reference location must be finite")
        if not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError("reference scale must be finite and strictly positive")
        self.reference_location.copy_(location)
        self.reference_scale.copy_(scale)

    def _causal_ema(self, normalized: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one-step-behind EMA states and the next-step state prediction."""
        alpha = 2.0 / (self.state_span + 1.0)
        running = normalized[:, 0]
        states = []
        for observation in normalized.unbind(dim=1):
            states.append(running)
            running = (1.0 - alpha) * running + alpha * observation
        return torch.stack(states, dim=1), running

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

    def _encode_catch(self, spectrum: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patches = self._frequency_patches(spectrum)
        batch_size, patch_num, n_vars, patch_dim = patches.shape
        flattened = patches.reshape(batch_size * patch_num, n_vars, patch_dim)
        channel_mask = self.mask_generator(flattened)
        encoded, channel_loss = self.frequency_transformer(flattened, channel_mask)
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

        location = self.reference_location.view(1, 1, -1)
        reference_scale = self.reference_scale.view(1, 1, -1)
        normalized = (history - location) / reference_scale
        state_history, next_state = self._causal_ema(normalized)

        if self.variant == "causal_catch":
            representation = normalized
            mean_baseline = torch.zeros_like(next_state)
        else:
            representation = normalized - state_history
            mean_baseline = next_state

        spectrum = torch.fft.rfft(representation.permute(0, 2, 1), dim=-1)
        encoded, channel_loss = self._encode_catch(spectrum)
        encoded_flat = encoded.flatten(start_dim=2)
        mean_normalized = mean_baseline + self.mean_head(encoded_flat).squeeze(-1)
        learned_scale = F.softplus(self.scale_head(encoded.mean(dim=2)).squeeze(-1))
        learned_scale = learned_scale + torch.finfo(learned_scale.dtype).eps

        recent_length = min(self.state_span, self.seq_len)
        innovation_scale = representation[:, -recent_length:].std(
            dim=1, unbiased=False
        )
        if self.variant == "state_scale":
            # Independent innovation variance can only widen the base CATCH scale.
            scale_normalized = torch.sqrt(
                learned_scale.square() + innovation_scale.square()
            )
        else:
            scale_normalized = learned_scale

        prediction_mean = (
            self.reference_location.view(1, -1)
            + self.reference_scale.view(1, -1) * mean_normalized
        )
        prediction_scale = self.reference_scale.view(1, -1) * scale_normalized
        state_mean = (
            self.reference_location.view(1, -1)
            + self.reference_scale.view(1, -1) * mean_baseline
        )
        return {
            "mean": prediction_mean,
            "scale": prediction_scale,
            "state_mean": state_mean,
            "innovation_scale": innovation_scale,
            "channel_loss": channel_loss,
        }


def gaussian_nll(
    target: torch.Tensor,
    prediction_mean: torch.Tensor,
    prediction_scale: torch.Tensor,
) -> torch.Tensor:
    residual = (target - prediction_mean) / prediction_scale
    return 0.5 * residual.square() + torch.log(prediction_scale)
