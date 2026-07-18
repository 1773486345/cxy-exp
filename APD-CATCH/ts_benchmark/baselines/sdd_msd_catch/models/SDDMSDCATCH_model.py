"""Shared-encoder, dual-low-rank-decoder MSD-CATCH model."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from ts_benchmark.baselines.catch.layers.RevIN import RevIN
from ts_benchmark.baselines.catch.layers.channel_mask import channel_mask_generator
from ts_benchmark.baselines.catch.layers.cross_channel_Transformer import Trans_C
from ts_benchmark.baselines.msd_catch.models.MSDCATCH_model import (
    ScaleGate,
    adaptive_multiscale_decompose,
)
from ts_benchmark.baselines.rsa_msd_catch.models.RSAMSDCATCH_model import (
    LowRankAdapter,
    LowRankFeatureExchange,
    RawGateNetwork,
    RawStructureAdapter,
)


class FactorizedLinear(nn.Module):
    """Low-rank replacement for a dense reconstruction projection."""

    def __init__(self, in_features: int, out_features: int, rank: int = 64) -> None:
        super().__init__()
        self.effective_rank = min(rank, in_features, out_features)
        self.down = nn.Linear(in_features, self.effective_rank, bias=False)
        self.up = nn.Linear(self.effective_rank, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class FactorizedFlattenHead(nn.Module):
    """CATCH-compatible flattened reconstruction head using rank-64 maps."""

    def __init__(self, in_features: int, output_length: int, rank: int = 64) -> None:
        super().__init__()
        self.rank = rank
        self.transform1 = FactorizedLinear(in_features, in_features, rank)
        self.transform2 = FactorizedLinear(in_features, in_features, rank)
        self.transform3 = FactorizedLinear(in_features, in_features, rank)
        self.output = FactorizedLinear(in_features, output_length, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.flatten(start_dim=-2)
        z = torch.relu(self.transform1(z)) + z
        z = torch.relu(self.transform2(z)) + z
        z = torch.relu(self.transform3(z)) + z
        return self.output(z)


class SharedCATCHEncoder(nn.Module):
    """Shared FFT patch construction, channel masking, and frequency encoder."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.revin_layer = RevIN(
            configs.c_in, affine=configs.affine, subtract_last=configs.subtract_last
        )
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
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
        self.mask_generator = channel_mask_generator(
            input_size=configs.patch_size, n_vars=configs.c_in
        )

    def _normalization_state(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.revin_layer.subtract_last:
            center = x[:, -1:, :].detach()
        else:
            center = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.revin_layer.eps
        ).detach()
        return {"center": center, "stdev": stdev}

    def _normalize(self, x: torch.Tensor, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = (x - state["center"]) / state["stdev"]
        if self.revin_layer.affine:
            z = z * self.revin_layer.affine_weight + self.revin_layer.affine_bias
        return z

    def denormalize(self, x: torch.Tensor, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = x
        if self.revin_layer.affine:
            z = z - self.revin_layer.affine_bias
            z = z / (self.revin_layer.affine_weight + self.revin_layer.eps**2)
        return z * state["stdev"] + state["center"]

    def normalize_for_loss(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize(x, self._normalization_state(x))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, object], torch.Tensor]:
        norm_state = self._normalization_state(x)
        z = self._normalize(x, norm_state).permute(0, 2, 1)
        z = torch.fft.fft(z)
        z_real = z.real.unfold(-1, self.patch_size, self.patch_stride).permute(0, 2, 1, 3)
        z_imag = z.imag.unfold(-1, self.patch_size, self.patch_stride).permute(0, 2, 1, 3)
        batch_size, patch_num, channels, _ = z_real.shape
        z_real = z_real.reshape(batch_size * patch_num, channels, self.patch_size)
        z_imag = z_imag.reshape(batch_size * patch_num, channels, self.patch_size)
        tokens, dcloss = self.frequency_transformer(
            torch.cat((z_real, z_imag), dim=-1),
            self.mask_generator(torch.cat((z_real, z_imag), dim=-1)),
        )
        return tokens, {
            "batch_size": batch_size,
            "patch_num": patch_num,
            "channels": channels,
            "norm_state": norm_state,
        }, dcloss


class BranchFactorizedDecoder(nn.Module):
    """Branch-specific complex-to-time reconstruction without dense CATCH heads."""

    def __init__(self, configs, rank: int = 64) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.feature_dim = configs.d_model * 2
        self.patch_num = int((configs.seq_len - configs.patch_size) / configs.patch_stride + 1)
        self.flatten_features = self.feature_dim * self.patch_num
        self.decoder_rank = min(rank, self.flatten_features, self.seq_len)
        self.get_r = nn.Linear(self.feature_dim, self.feature_dim)
        self.get_i = nn.Linear(self.feature_dim, self.feature_dim)
        self.real_head = FactorizedFlattenHead(self.flatten_features, self.seq_len, rank)
        self.imag_head = FactorizedFlattenHead(self.flatten_features, self.seq_len, rank)
        self.complex_projection = FactorizedLinear(self.seq_len * 2, self.seq_len, rank)

    def forward(
        self,
        tokens: torch.Tensor,
        context: Dict[str, object],
        encoder: SharedCATCHEncoder,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_real = self.get_r(tokens)
        z_imag = self.get_i(tokens)
        batch_size = int(context["batch_size"])
        patch_num = int(context["patch_num"])
        channels = int(context["channels"])
        z_real = z_real.reshape(batch_size, patch_num, channels, z_real.shape[-1]).permute(0, 2, 1, 3)
        z_imag = z_imag.reshape(batch_size, patch_num, channels, z_imag.shape[-1]).permute(0, 2, 1, 3)
        complex_z = torch.complex(self.real_head(z_real), self.imag_head(z_imag))
        time_z = torch.fft.ifft(complex_z)
        reconstruction = self.complex_projection(torch.cat((time_z.real, time_z.imag), dim=-1))
        reconstruction = encoder.denormalize(reconstruction.permute(0, 2, 1), context["norm_state"])
        return reconstruction, complex_z.permute(0, 2, 1)


class SDDMSDCATCHModel(nn.Module):
    """MSD-CATCH with shared encoder and independent rank-64 branch decoders."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.patch_size = configs.patch_size
        self.channels = configs.c_in
        self.feature_dim = configs.d_model * 2
        self.adapter_rank = min(32, max(8, self.feature_dim // 8))
        self.scale_gate = ScaleGate(getattr(configs, "scale_gate_hidden", 16))
        self.shared_encoder = SharedCATCHEncoder(configs)
        self.low_rank_exchange = LowRankFeatureExchange(self.feature_dim, self.adapter_rank)
        self.trend_adapter = LowRankAdapter(self.feature_dim, self.adapter_rank)
        self.residual_adapter = LowRankAdapter(self.feature_dim, self.adapter_rank)
        self.trend_decoder = BranchFactorizedDecoder(configs)
        self.residual_decoder = BranchFactorizedDecoder(configs)
        # Kept byte-for-byte structurally equivalent to RSA's raw path classes.
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
            "shared_encoder": count(self.shared_encoder),
            "trend_adapter": count(self.trend_adapter),
            "residual_adapter": count(self.residual_adapter),
            "trend_decoder": count(self.trend_decoder),
            "residual_decoder": count(self.residual_decoder),
            "low_rank_exchange": count(self.low_rank_exchange),
            "raw_adapter": count(self.raw_adapter),
            "raw_gate": count(self.raw_gate_network),
            "scale_gate": count(self.scale_gate),
        }
        groups["other"] = count(self) - sum(groups.values())
        groups["total"] = count(self)
        return groups

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        trend, residual, scale_weights, kernels = self.decompose(x)
        trend_tokens, trend_context, trend_dcloss = self.shared_encoder(trend)
        residual_tokens, residual_context, residual_dcloss = self.shared_encoder(residual)
        trend_tokens, residual_tokens = self.low_rank_exchange(trend_tokens, residual_tokens)
        trend_tokens = trend_tokens + self.trend_adapter(trend_tokens)
        residual_tokens = residual_tokens + self.residual_adapter(residual_tokens)
        trend_hat, trend_complex = self.trend_decoder(
            trend_tokens, trend_context, self.shared_encoder
        )
        residual_hat, residual_complex = self.residual_decoder(
            residual_tokens, residual_context, self.shared_encoder
        )
        decomp_hat = trend_hat + residual_hat
        raw_features, scale_entropy = self._raw_gate_features(x, trend, residual, scale_weights)
        raw_gate = self.raw_gate_network(raw_features).transpose(1, 2).expand(-1, x.shape[1], -1)
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
