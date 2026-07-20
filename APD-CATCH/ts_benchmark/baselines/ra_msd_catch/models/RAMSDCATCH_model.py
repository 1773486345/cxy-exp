"""Raw-anchored multi-scale decomposition CATCH model."""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.catch.layers.RevIN import RevIN
from ts_benchmark.baselines.catch.layers.channel_mask import channel_mask_generator
from ts_benchmark.baselines.catch.layers.cross_channel_Transformer import Trans_C
from ts_benchmark.baselines.catch.models.CATCH_model import Flatten_Head


def _legal_odd_kernel(kernel: int, sequence_length: int) -> int:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    maximum = sequence_length if sequence_length % 2 else sequence_length - 1
    return max(1, min(max(1, kernel), maximum))


def _moving_average(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"expected [batch, time, channel], got {tuple(x.shape)}")
    kernel_size = _legal_odd_kernel(kernel_size, x.shape[1])
    padding = kernel_size // 2
    values = F.pad(x.transpose(1, 2), (padding, padding), mode="replicate")
    return F.avg_pool1d(values, kernel_size=kernel_size, stride=1).transpose(1, 2)


class ScaleGate(nn.Module):
    """Verbatim MSD-CATCH three-scale selection gate."""

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
    sequence_length = x.shape[1]
    requested = (patch_size - 1, 2 * patch_size - 1, 4 * patch_size - 1)
    kernels = tuple(_legal_odd_kernel(kernel, sequence_length) for kernel in requested)
    candidates = torch.stack([_moving_average(x, kernel) for kernel in kernels], dim=-1)
    weights = scale_gate(x)
    trend = (candidates * weights.unsqueeze(1)).sum(dim=-1)
    residual = x - trend
    return trend, residual, weights, kernels


class CATCHEncoder(nn.Module):
    """The CATCH path through Trans_C, without its reconstruction head."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.revin_layer = RevIN(
            configs.c_in, affine=configs.affine, subtract_last=configs.subtract_last
        )
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.seq_len = configs.seq_len
        self.horizon = configs.seq_len
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int], torch.Tensor]:
        z = self.revin_layer(x, "norm").permute(0, 2, 1)
        z = torch.fft.fft(z)
        real = z.real.unfold(-1, self.patch_size, self.patch_stride)
        imag = z.imag.unfold(-1, self.patch_size, self.patch_stride)
        real = real.permute(0, 2, 1, 3)
        imag = imag.permute(0, 2, 1, 3)
        batch_size, patch_num, channels, _ = real.shape
        real = real.reshape(batch_size * patch_num, channels, self.patch_size)
        imag = imag.reshape(batch_size * patch_num, channels, self.patch_size)
        patch_tokens = torch.cat((real, imag), dim=-1)
        tokens, dcloss = self.frequency_transformer(
            patch_tokens, self.mask_generator(patch_tokens)
        )
        return tokens, {
            "batch_size": batch_size,
            "patch_num": patch_num,
            "channels": channels,
        }, dcloss


class CATCHReconstructionHead(nn.Module):
    """The original CATCH token-to-time-domain reconstruction head."""

    def __init__(self, configs) -> None:
        super().__init__()
        patch_num = int((configs.seq_len - configs.patch_size) / configs.patch_stride + 1)
        self.seq_len = configs.seq_len
        self.feature_dim = configs.d_model * 2
        self.head_nf_f = self.feature_dim * patch_num
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
        self.ircom = nn.Linear(configs.seq_len * 2, configs.seq_len)
        self.rfftlayer = nn.Linear(configs.seq_len * 2 - 2, configs.seq_len)
        self.final = nn.Linear(configs.seq_len * 2, configs.seq_len)
        self.get_r = nn.Linear(self.feature_dim, self.feature_dim)
        self.get_i = nn.Linear(self.feature_dim, self.feature_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        context: Dict[str, int],
        revin_layer: RevIN,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        real = self.get_r(tokens)
        imag = self.get_i(tokens)
        batch_size = context["batch_size"]
        patch_num = context["patch_num"]
        channels = context["channels"]
        real = real.reshape(batch_size, patch_num, channels, self.feature_dim).permute(0, 2, 1, 3)
        imag = imag.reshape(batch_size, patch_num, channels, self.feature_dim).permute(0, 2, 1, 3)
        real = self.head_f1(real)
        imag = self.head_f2(imag)
        complex_frequency = torch.complex(real, imag)
        time_values = torch.fft.ifft(complex_frequency)
        reconstruction = self.ircom(torch.cat((time_values.real, time_values.imag), dim=-1))
        reconstruction = revin_layer(reconstruction.permute(0, 2, 1), "denorm")
        return reconstruction, complex_frequency.permute(0, 2, 1)


class TokenAdapter(nn.Module):
    """Independent low-rank residual adapter for one auxiliary token view."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        rank = min(64, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, rank),
            nn.GELU(),
            nn.Linear(rank, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class RAMSDCATCHModel(nn.Module):
    """One raw CATCH reconstruction path with two encoder-only auxiliary views."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.patch_size = configs.patch_size
        self.hidden_dim = configs.d_model * 2
        self.scale_gate = ScaleGate(getattr(configs, "scale_gate_hidden", 16))
        self.raw_encoder = CATCHEncoder(configs)
        self.trend_encoder = copy.deepcopy(self.raw_encoder)
        self.residual_encoder = copy.deepcopy(self.raw_encoder)
        self.trend_adapter = TokenAdapter(self.hidden_dim)
        self.residual_adapter = TokenAdapter(self.hidden_dim)
        self.raw_reconstruction_head = CATCHReconstructionHead(configs)
        self.alpha_trend_param = nn.Parameter(torch.zeros(()))
        self.alpha_residual_param = nn.Parameter(torch.zeros(()))

    @property
    def alpha_trend(self) -> torch.Tensor:
        return 0.25 * torch.tanh(self.alpha_trend_param)

    @property
    def alpha_residual(self) -> torch.Tensor:
        return 0.25 * torch.tanh(self.alpha_residual_param)

    def decompose(self, x: torch.Tensor):
        return adaptive_multiscale_decompose(x, self.patch_size, self.scale_gate)

    @torch.no_grad()
    def load_from_catch_model(self, catch_model: nn.Module) -> None:
        """Copy a CATCH model into the raw path and synchronize auxiliary encoders."""
        self.raw_encoder.revin_layer.load_state_dict(catch_model.revin_layer.state_dict())
        self.raw_encoder.norm.load_state_dict(catch_model.norm.state_dict())
        self.raw_encoder.mask_generator.load_state_dict(catch_model.mask_generator.state_dict())
        self.raw_encoder.frequency_transformer.load_state_dict(
            catch_model.frequency_transformer.state_dict()
        )
        for name in ("head_f1", "head_f2", "ircom", "rfftlayer", "final", "get_r", "get_i"):
            getattr(self.raw_reconstruction_head, name).load_state_dict(
                getattr(catch_model, name).state_dict()
            )
        self.trend_encoder.load_state_dict(self.raw_encoder.state_dict())
        self.residual_encoder.load_state_dict(self.raw_encoder.state_dict())

    def module_parameter_counts(self) -> Dict[str, int]:
        count = lambda module: sum(parameter.numel() for parameter in module.parameters())
        groups = {
            "raw_encoder": count(self.raw_encoder),
            "trend_encoder": count(self.trend_encoder),
            "residual_encoder": count(self.residual_encoder),
            "trend_adapter": count(self.trend_adapter),
            "residual_adapter": count(self.residual_adapter),
            "raw_reconstruction_head": count(self.raw_reconstruction_head),
            "scale_gate": count(self.scale_gate),
            "alpha": self.alpha_trend_param.numel() + self.alpha_residual_param.numel(),
        }
        groups["total"] = sum(groups.values())
        return groups

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        trend, residual, scale_weights, kernels = self.decompose(x)
        h_raw, raw_context, raw_dcloss = self.raw_encoder(x)
        h_trend, _, trend_dcloss = self.trend_encoder(trend)
        h_residual, _, residual_dcloss = self.residual_encoder(residual)
        delta_trend = self.trend_adapter(h_trend)
        delta_residual = self.residual_adapter(h_residual)
        h_fused = h_raw + self.alpha_trend * delta_trend + self.alpha_residual * delta_residual
        x_hat, output_complex = self.raw_reconstruction_head(
            h_fused, raw_context, self.raw_encoder.revin_layer
        )
        return {
            "x_hat": x_hat,
            "output_complex": output_complex,
            "raw_dcloss": raw_dcloss,
            "trend_dcloss": trend_dcloss,
            "residual_dcloss": residual_dcloss,
            "trend": trend,
            "residual": residual,
            "scale_weights": scale_weights,
            "kernels": kernels,
            "h_raw": h_raw,
            "h_trend": h_trend,
            "h_residual": h_residual,
            "h_fused": h_fused,
            "delta_trend": delta_trend,
            "delta_residual": delta_residual,
            "alpha_trend": self.alpha_trend,
            "alpha_residual": self.alpha_residual,
        }
