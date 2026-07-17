"""Multi-scale decomposition CATCH model.

This module reuses CATCH's frequency-channel layers without changing the
original implementation.  The CATCH encoder is split before its reconstruction
head so the two decomposition branches can exchange frequency-channel tokens.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.catch.layers.RevIN import RevIN
from ts_benchmark.baselines.catch.layers.channel_mask import channel_mask_generator
from ts_benchmark.baselines.catch.layers.cross_channel_Transformer import Trans_C
from ts_benchmark.baselines.catch.models.CATCH_model import Flatten_Head


def _legal_odd_kernel(kernel: int, sequence_length: int) -> int:
    """Return the largest valid odd kernel no greater than the sequence."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    maximum = sequence_length if sequence_length % 2 else sequence_length - 1
    return max(1, min(max(1, kernel), maximum))


def moving_average(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Centered channel-wise moving average with exact input shape preservation."""
    if x.ndim != 3:
        raise ValueError(f"expected [batch, time, channel], got {tuple(x.shape)}")
    kernel_size = _legal_odd_kernel(kernel_size, x.shape[1])
    padding = kernel_size // 2
    values = x.transpose(1, 2)
    values = F.pad(values, (padding, padding), mode="replicate")
    return F.avg_pool1d(values, kernel_size=kernel_size, stride=1).transpose(1, 2)


class ScaleGate(nn.Module):
    """Select a moving-average scale independently for every sample and channel."""

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        if x.shape[1] > 1:
            abs_diff_mean = x.diff(dim=1).abs().mean(dim=1)
        else:
            abs_diff_mean = torch.zeros_like(mean)
        features = torch.stack((mean, std, abs_diff_mean), dim=-1)
        return torch.softmax(self.net(features), dim=-1)


def adaptive_multiscale_decompose(
    x: torch.Tensor, patch_size: int, scale_gate: ScaleGate
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    """Create adaptive trend and residual components whose sum is exactly ``x``."""
    sequence_length = x.shape[1]
    requested = (patch_size - 1, 2 * patch_size - 1, 4 * patch_size - 1)
    kernels = tuple(_legal_odd_kernel(kernel, sequence_length) for kernel in requested)
    candidates = torch.stack([moving_average(x, kernel) for kernel in kernels], dim=-1)
    weights = scale_gate(x)
    trend = (candidates * weights.unsqueeze(1)).sum(dim=-1)
    residual = x - trend
    return trend, residual, weights, kernels


class CATCHBranch(nn.Module):
    """One independent CATCH encoder/decoder branch with exposed token features."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.revin_layer = RevIN(
            configs.c_in, affine=configs.affine, subtract_last=configs.subtract_last
        )
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.seq_len = configs.seq_len
        self.horizon = configs.seq_len
        patch_num = int((configs.seq_len - configs.patch_size) / configs.patch_stride + 1)
        self.norm = nn.LayerNorm(self.patch_size)
        self.mask_generator = channel_mask_generator(
            input_size=configs.patch_size, n_vars=configs.c_in
        )
        self.frequency_transformer = Trans_C(
            dim=configs.cf_dim,
            depth=configs.e_layers,
            heads=configs.n_heads,
            mlp_dim=configs.d_ff,
            dim_head=configs.head_dim,
            dropout=configs.dropout,
            patch_dim=configs.patch_size * 2,
            horizon=self.horizon * 2,
            d_model=configs.d_model * 2,
            regular_lambda=configs.regular_lambda,
            temperature=configs.temperature,
        )
        self.head_nf_f = configs.d_model * 2 * patch_num
        self.n_vars = configs.c_in
        self.individual = configs.individual
        self.head_f1 = Flatten_Head(
            self.individual,
            self.n_vars,
            self.head_nf_f,
            configs.seq_len,
            head_dropout=configs.head_dropout,
        )
        self.head_f2 = Flatten_Head(
            self.individual,
            self.n_vars,
            self.head_nf_f,
            configs.seq_len,
            head_dropout=configs.head_dropout,
        )
        # Kept for structural parity with the original CATCH model.
        self.ircom = nn.Linear(self.seq_len * 2, self.seq_len)
        self.rfftlayer = nn.Linear(self.seq_len * 2 - 2, self.seq_len)
        self.final = nn.Linear(self.seq_len * 2, self.seq_len)
        self.get_r = nn.Linear(configs.d_model * 2, configs.d_model * 2)
        self.get_i = nn.Linear(configs.d_model * 2, configs.d_model * 2)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int], torch.Tensor]:
        z = self.revin_layer(x, "norm").permute(0, 2, 1)
        z = torch.fft.fft(z)
        z1 = z.real.unfold(-1, self.patch_size, self.patch_stride)
        z2 = z.imag.unfold(-1, self.patch_size, self.patch_stride)
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)
        batch_size, patch_num, channels, _ = z1.shape
        z1 = z1.reshape(batch_size * patch_num, channels, self.patch_size)
        z2 = z2.reshape(batch_size * patch_num, channels, self.patch_size)
        z_cat = torch.cat((z1, z2), dim=-1)
        channel_mask = self.mask_generator(z_cat)
        tokens, dcloss = self.frequency_transformer(z_cat, channel_mask)
        context = {
            "batch_size": batch_size,
            "patch_num": patch_num,
            "channels": channels,
        }
        return tokens, context, dcloss

    def decode(self, tokens: torch.Tensor, context: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        z1 = self.get_r(tokens)
        z2 = self.get_i(tokens)
        batch_size = context["batch_size"]
        patch_num = context["patch_num"]
        channels = context["channels"]
        z1 = z1.reshape(batch_size, patch_num, channels, z1.shape[-1]).permute(0, 2, 1, 3)
        z2 = z2.reshape(batch_size, patch_num, channels, z2.shape[-1]).permute(0, 2, 1, 3)
        z1 = self.head_f1(z1)
        z2 = self.head_f2(z2)
        complex_z = torch.complex(z1, z2)
        z = torch.fft.ifft(complex_z)
        reconstruction = self.ircom(torch.cat((z.real, z.imag), dim=-1))
        reconstruction = self.revin_layer(reconstruction.permute(0, 2, 1), "denorm")
        return reconstruction, complex_z.permute(0, 2, 1)


class GatedFeatureExchange(nn.Module):
    """Lightweight bidirectional exchange between branch CATCH token tensors."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.trend_message = nn.Linear(feature_dim, feature_dim, bias=False)
        self.residual_message = nn.Linear(feature_dim, feature_dim, bias=False)
        self.trend_gate = nn.Linear(feature_dim * 2, feature_dim)
        self.residual_gate = nn.Linear(feature_dim * 2, feature_dim)

    def forward(
        self, trend_tokens: torch.Tensor, residual_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        trend_context = self.trend_message(residual_tokens)
        residual_context = self.residual_message(trend_tokens)
        trend_gate = torch.sigmoid(self.trend_gate(torch.cat((trend_tokens, trend_context), dim=-1)))
        residual_gate = torch.sigmoid(
            self.residual_gate(torch.cat((residual_tokens, residual_context), dim=-1))
        )
        return (
            trend_tokens + trend_gate * trend_context,
            residual_tokens + residual_gate * residual_context,
        )


class MSDCATCHModel(nn.Module):
    """Adaptive decomposition, independent CATCH branches, and token interaction."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.patch_size = configs.patch_size
        self.scale_gate = ScaleGate(getattr(configs, "scale_gate_hidden", 16))
        self.trend_branch = CATCHBranch(configs)
        self.residual_branch = CATCHBranch(configs)
        self.interaction = GatedFeatureExchange(configs.d_model * 2)

    def decompose(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        return adaptive_multiscale_decompose(x, self.patch_size, self.scale_gate)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        trend, residual, scale_weights, kernels = self.decompose(x)
        trend_tokens, trend_context, trend_dcloss = self.trend_branch.encode(trend)
        residual_tokens, residual_context, residual_dcloss = self.residual_branch.encode(residual)
        trend_tokens, residual_tokens = self.interaction(trend_tokens, residual_tokens)
        trend_hat, trend_complex = self.trend_branch.decode(trend_tokens, trend_context)
        residual_hat, residual_complex = self.residual_branch.decode(
            residual_tokens, residual_context
        )
        x_hat = trend_hat + residual_hat
        return {
            "trend": trend,
            "residual": residual,
            "scale_weights": scale_weights,
            "kernels": kernels,
            "trend_hat": trend_hat,
            "residual_hat": residual_hat,
            "x_hat": x_hat,
            "trend_complex": trend_complex,
            "residual_complex": residual_complex,
            "trend_dcloss": trend_dcloss,
            "residual_dcloss": residual_dcloss,
        }
