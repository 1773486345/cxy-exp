"""Raw-preserving shared-adapter MSD-CATCH model.

The module deliberately reuses CATCH layers while keeping a single CATCH
backbone.  Trend and residual receive separate low-rank adapters rather than
two independent, high-capacity reconstruction heads.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.catch.layers.RevIN import RevIN
from ts_benchmark.baselines.catch.layers.channel_mask import channel_mask_generator
from ts_benchmark.baselines.catch.layers.cross_channel_Transformer import Trans_C
from ts_benchmark.baselines.catch.models.CATCH_model import Flatten_Head
from ts_benchmark.baselines.msd_catch.models.MSDCATCH_model import (
    ScaleGate,
    adaptive_multiscale_decompose,
)


def _zero_linear(linear: nn.Linear, bias: float = 0.0) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.constant_(linear.bias, bias)


class LowRankAdapter(nn.Module):
    """Residual bottleneck adapter with a zero-initialized output projection."""

    def __init__(self, feature_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(feature_dim, rank)
        self.activation = nn.GELU()
        self.up = nn.Linear(rank, feature_dim)
        _zero_linear(self.up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(x)))


class LowRankFeatureExchange(nn.Module):
    """Bidirectional token exchange without channel- or token-square weights."""

    def __init__(self, feature_dim: int, rank: int) -> None:
        super().__init__()
        self.residual_to_trend = LowRankAdapter(feature_dim, rank)
        self.trend_to_residual = LowRankAdapter(feature_dim, rank)
        self.trend_gate = nn.Sequential(
            nn.Linear(feature_dim, rank), nn.GELU(), nn.Linear(rank, feature_dim)
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(feature_dim, rank), nn.GELU(), nn.Linear(rank, feature_dim)
        )
        _zero_linear(self.trend_gate[-1])
        _zero_linear(self.residual_gate[-1])

    def forward(
        self, trend_tokens: torch.Tensor, residual_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        trend_context = self.residual_to_trend(residual_tokens)
        residual_context = self.trend_to_residual(trend_tokens)
        trend_gate = torch.sigmoid(self.trend_gate(residual_tokens))
        residual_gate = torch.sigmoid(self.residual_gate(trend_tokens))
        return (
            trend_tokens + trend_gate * trend_context,
            residual_tokens + residual_gate * residual_context,
        )


class SharedCATCHBackbone(nn.Module):
    """CATCH encoder/decoder whose normalization state is local to each call."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.revin_layer = RevIN(
            configs.c_in, affine=configs.affine, subtract_last=configs.subtract_last
        )
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.seq_len = configs.seq_len
        patch_num = int((configs.seq_len - configs.patch_size) / configs.patch_stride + 1)
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
            horizon=configs.seq_len * 2,
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
        self.ircom = nn.Linear(self.seq_len * 2, self.seq_len)
        self.rfftlayer = nn.Linear(self.seq_len * 2 - 2, self.seq_len)
        self.final = nn.Linear(self.seq_len * 2, self.seq_len)
        self.get_r = nn.Linear(configs.d_model * 2, configs.d_model * 2)
        self.get_i = nn.Linear(configs.d_model * 2, configs.d_model * 2)

    def _normalization_state(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.revin_layer.subtract_last:
            center = x[:, -1:, :].detach()
        else:
            center = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.revin_layer.eps).detach()
        return {"center": center, "stdev": stdev}

    def _normalize(
        self, x: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        z = (x - state["center"]) / state["stdev"]
        if self.revin_layer.affine:
            z = z * self.revin_layer.affine_weight + self.revin_layer.affine_bias
        return z

    def _denormalize(
        self, x: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        z = x
        if self.revin_layer.affine:
            z = z - self.revin_layer.affine_bias
            z = z / (self.revin_layer.affine_weight + self.revin_layer.eps**2)
        return z * state["stdev"] + state["center"]

    def normalize_for_loss(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize(x, self._normalization_state(x))

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, object], torch.Tensor]:
        norm_state = self._normalization_state(x)
        z = self._normalize(x, norm_state).permute(0, 2, 1)
        z = torch.fft.fft(z)
        z1 = z.real.unfold(-1, self.patch_size, self.patch_stride).permute(0, 2, 1, 3)
        z2 = z.imag.unfold(-1, self.patch_size, self.patch_stride).permute(0, 2, 1, 3)
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
            "norm_state": norm_state,
        }
        return tokens, context, dcloss

    def decode(
        self, tokens: torch.Tensor, context: Dict[str, object]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z1 = self.get_r(tokens)
        z2 = self.get_i(tokens)
        batch_size = int(context["batch_size"])
        patch_num = int(context["patch_num"])
        channels = int(context["channels"])
        z1 = z1.reshape(batch_size, patch_num, channels, z1.shape[-1]).permute(0, 2, 1, 3)
        z2 = z2.reshape(batch_size, patch_num, channels, z2.shape[-1]).permute(0, 2, 1, 3)
        z1 = self.head_f1(z1)
        z2 = self.head_f2(z2)
        complex_z = torch.complex(z1, z2)
        z = torch.fft.ifft(complex_z)
        reconstruction = self.ircom(torch.cat((z.real, z.imag), dim=-1))
        reconstruction = self._denormalize(
            reconstruction.permute(0, 2, 1), context["norm_state"]
        )
        return reconstruction, complex_z.permute(0, 2, 1)


class RawStructureAdapter(nn.Module):
    """Restricted raw correction: depthwise temporal filtering plus low-rank channels."""

    def __init__(self, channels: int, rank: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("raw adapter kernel_size must be odd")
        self.depthwise_temporal_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.channel_down = nn.Linear(channels, rank)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(rank, channels)
        _zero_linear(self.output_projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.depthwise_temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        return self.output_projection(self.activation(self.channel_down(z)))


class RawGateNetwork(nn.Module):
    """Per-channel confidence gate bounded to the interval [0, 0.5]."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, rank)
        self.activation = nn.GELU()
        self.output = nn.Linear(rank, 1)
        _zero_linear(self.output, bias=-2.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.output(self.activation(self.hidden(features))))


class RSAMSDCATCHModel(nn.Module):
    """Shared-backbone decomposition CATCH with a gated raw correction path."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.patch_size = configs.patch_size
        self.channels = configs.c_in
        self.feature_dim = configs.d_model * 2
        self.adapter_rank = min(32, max(8, self.feature_dim // 8))
        self.scale_gate = ScaleGate(getattr(configs, "scale_gate_hidden", 16))
        self.shared_catch_backbone = SharedCATCHBackbone(configs)
        self.trend_adapter = LowRankAdapter(self.feature_dim, self.adapter_rank)
        self.residual_adapter = LowRankAdapter(self.feature_dim, self.adapter_rank)
        self.low_rank_exchange = LowRankFeatureExchange(self.feature_dim, self.adapter_rank)
        self.raw_adapter = RawStructureAdapter(self.channels, self.adapter_rank)
        self.raw_gate_network = RawGateNetwork(self.adapter_rank)

    def decompose(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        return adaptive_multiscale_decompose(x, self.patch_size, self.scale_gate)

    def _raw_gate_features(
        self, x: torch.Tensor, trend: torch.Tensor, residual: torch.Tensor, scale_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        entropy = -(scale_weights * torch.log(scale_weights.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / math.log(3.0)
        std = x.std(dim=1, unbiased=False)
        if x.shape[1] > 1:
            abs_diff_mean = x.diff(dim=1).abs().mean(dim=1)
        else:
            abs_diff_mean = torch.zeros_like(std)
        energy_ratio = residual.square().mean(dim=1) / trend.square().mean(dim=1).clamp_min(1e-8)
        return torch.stack((entropy, std, abs_diff_mean, energy_ratio), dim=-1), entropy

    def module_parameter_counts(self) -> Dict[str, int]:
        count = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
        groups = {
            "decomposition_scale_gate": count(self.scale_gate),
            "shared_catch_backbone": count(self.shared_catch_backbone),
            "trend_adapter": count(self.trend_adapter),
            "residual_adapter": count(self.residual_adapter),
            "low_rank_exchange": count(self.low_rank_exchange),
            "raw_adapter": count(self.raw_adapter),
            "raw_gate_network": count(self.raw_gate_network),
        }
        groups["other"] = count(self) - sum(groups.values())
        groups["total"] = count(self)
        return groups

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        trend, residual, scale_weights, kernels = self.decompose(x)
        trend_tokens, trend_context, trend_dcloss = self.shared_catch_backbone.encode(trend)
        residual_tokens, residual_context, residual_dcloss = self.shared_catch_backbone.encode(residual)
        trend_tokens = trend_tokens + self.trend_adapter(trend_tokens)
        residual_tokens = residual_tokens + self.residual_adapter(residual_tokens)
        trend_tokens, residual_tokens = self.low_rank_exchange(trend_tokens, residual_tokens)
        trend_hat, trend_complex = self.shared_catch_backbone.decode(trend_tokens, trend_context)
        residual_hat, residual_complex = self.shared_catch_backbone.decode(
            residual_tokens, residual_context
        )
        decomp_hat = trend_hat + residual_hat
        raw_features, scale_entropy = self._raw_gate_features(x, trend, residual, scale_weights)
        raw_gate = self.raw_gate_network(raw_features).transpose(1, 2).expand(
            -1, x.shape[1], -1
        )
        raw_correction = self.raw_adapter(x)
        x_hat = decomp_hat + raw_gate * raw_correction
        return {
            "trend": trend,
            "residual": residual,
            "scale_weights": scale_weights,
            "scale_entropy": scale_entropy,
            "kernels": kernels,
            "trend_hat": trend_hat,
            "residual_hat": residual_hat,
            "decomp_hat": decomp_hat,
            "raw_correction": raw_correction,
            "raw_gate": raw_gate,
            "x_hat": x_hat,
            "trend_complex": trend_complex,
            "residual_complex": residual_complex,
            "trend_dcloss": trend_dcloss,
            "residual_dcloss": residual_dcloss,
        }
