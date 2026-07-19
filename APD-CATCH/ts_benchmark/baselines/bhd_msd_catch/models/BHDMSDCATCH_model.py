"""Blockwise-head shared-encoder decomposition CATCH model."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from ts_benchmark.baselines.sdd_msd_catch.models.SDDMSDCATCH_model import (
    SharedCATCHEncoder,
)


class StableDynamicalContrastiveLoss(nn.Module):
    """BHD-local contrastive loss that remains defined for zero-norm tokens."""

    def __init__(self, temperature: float, k: float) -> None:
        super().__init__()
        self.temperature = temperature
        self.k = k

    def forward(
        self, scores: torch.Tensor, attn_mask: torch.Tensor, norm_matrix: torch.Tensor
    ) -> torch.Tensor:
        denominator = norm_matrix.clamp_min(torch.finfo(norm_matrix.dtype).eps)
        cosine = (scores / denominator).mean(1)
        pos_scores = torch.exp(cosine / self.temperature) * attn_mask
        all_scores = torch.exp(cosine / self.temperature)
        clustering_loss = -torch.log(pos_scores.sum(dim=-1) / all_scores.sum(dim=-1))
        batch_size = scores.shape[0]
        n_vars = scores.shape[-1]
        eye = torch.eye(n_vars, device=attn_mask.device).unsqueeze(0).repeat(batch_size, 1, 1)
        regular_loss = torch.norm(
            eye.reshape(batch_size, -1) - attn_mask.reshape(batch_size, -1), p=1, dim=-1
        ) / (n_vars * (n_vars - 1))
        return (clustering_loss.mean(1) + self.k * regular_loss).mean()


class BlockwiseDecoder(nn.Module):
    """Branch-local patch decoder with overlap-add frequency reconstruction."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.feature_dim = configs.d_model * 2
        self.patch_num = int(
            (self.seq_len - self.patch_size) / self.patch_stride + 1
        )
        decoder_hidden = min(256, max(64, 2 * self.feature_dim))
        self.position_scale = nn.Parameter(torch.ones(self.patch_num, self.feature_dim))
        self.position_bias = nn.Parameter(torch.zeros(self.patch_num, self.feature_dim))
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, 2 * self.patch_size),
        )
        # The projection is applied independently at each reconstructed time step.
        self.output_projection = nn.Linear(2, 1)
        self.last_input_shape: Tuple[int, ...] | None = None

        starts = torch.arange(self.patch_num, dtype=torch.long) * self.patch_stride
        if starts[-1] + self.patch_size < self.seq_len:
            starts[-1] = self.seq_len - self.patch_size
        overlap_count = torch.zeros(self.seq_len)
        for start in starts.tolist():
            overlap_count[start : start + self.patch_size] += 1
        self.register_buffer("patch_starts", starts, persistent=False)
        self.register_buffer(
            "overlap_count", overlap_count.view(1, 1, self.seq_len), persistent=False
        )

    def _as_patch_blocks(
        self, tokens: torch.Tensor, context: Dict[str, object]
    ) -> torch.Tensor:
        batch_size = int(context["batch_size"])
        patch_num = int(context["patch_num"])
        channels = int(context["channels"])
        if patch_num != self.patch_num:
            raise ValueError("block decoder patch count differs from its configured patch count")
        return tokens.reshape(batch_size, patch_num, channels, self.feature_dim).permute(
            0, 2, 1, 3
        )

    def _overlap_add(self, patch_values: torch.Tensor) -> torch.Tensor:
        pieces = []
        for patch_index, start in enumerate(self.patch_starts.tolist()):
            end = start + self.patch_size
            pieces.append(F.pad(patch_values[:, :, patch_index, :], (start, self.seq_len - end)))
        frequency_sum = torch.stack(pieces, dim=0).sum(dim=0)
        return frequency_sum / self.overlap_count.clamp_min(1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        context: Dict[str, object],
        encoder: SharedCATCHEncoder,
    ) -> Dict[str, torch.Tensor]:
        blocks = self._as_patch_blocks(tokens, context)
        self.last_input_shape = tuple(blocks.shape)
        blocks = blocks * self.position_scale.unsqueeze(0).unsqueeze(0)
        blocks = blocks + self.position_bias.unsqueeze(0).unsqueeze(0)
        patch_output = self.proj(blocks)
        patch_real, patch_imag = patch_output.chunk(2, dim=-1)
        frequency_real = self._overlap_add(patch_real)
        frequency_imag = self._overlap_add(patch_imag)
        complex_frequency = torch.complex(frequency_real, frequency_imag)
        time_values = torch.fft.ifft(complex_frequency, dim=-1)
        time_pair = torch.stack((time_values.real, time_values.imag), dim=-1)
        reconstruction = self.output_projection(time_pair).squeeze(-1).permute(0, 2, 1)
        reconstruction = encoder.denormalize(reconstruction, context["norm_state"])
        return {
            "reconstruction": reconstruction,
            "complex_frequency": complex_frequency.permute(0, 2, 1),
            "patch_real": patch_real,
            "patch_imag": patch_imag,
            "overlap_count": self.overlap_count,
        }


class BHDMSDCATCHModel(nn.Module):
    """MSD-CATCH with one shared encoder and independent blockwise decoders."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.patch_size = configs.patch_size
        self.channels = configs.c_in
        self.feature_dim = configs.d_model * 2
        self.adapter_rank = min(32, max(8, self.feature_dim // 8))
        self.scale_gate = ScaleGate(getattr(configs, "scale_gate_hidden", 16))
        self.shared_encoder = SharedCATCHEncoder(configs)
        for layer in self.shared_encoder.frequency_transformer.transformer.layers:
            attention = layer[0].fn
            original_loss = attention.dynamicalContranstiveLoss
            attention.dynamicalContranstiveLoss = StableDynamicalContrastiveLoss(
                temperature=original_loss.temperature,
                k=original_loss.k,
            )
        self.low_rank_exchange = LowRankFeatureExchange(self.feature_dim, self.adapter_rank)
        self.trend_adapter = LowRankAdapter(self.feature_dim, self.adapter_rank)
        self.residual_adapter = LowRankAdapter(self.feature_dim, self.adapter_rank)
        self.trend_block_decoder = BlockwiseDecoder(configs)
        self.residual_block_decoder = BlockwiseDecoder(configs)
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
            "low_rank_exchange": count(self.low_rank_exchange),
            "trend_block_decoder": count(self.trend_block_decoder),
            "residual_block_decoder": count(self.residual_block_decoder),
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
        trend_decoded = self.trend_block_decoder(
            trend_tokens, trend_context, self.shared_encoder
        )
        residual_decoded = self.residual_block_decoder(
            residual_tokens, residual_context, self.shared_encoder
        )
        trend_hat = trend_decoded["reconstruction"]
        residual_hat = residual_decoded["reconstruction"]
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
            "trend_complex": trend_decoded["complex_frequency"],
            "residual_complex": residual_decoded["complex_frequency"],
            "trend_patch_real": trend_decoded["patch_real"],
            "trend_patch_imag": trend_decoded["patch_imag"],
            "residual_patch_real": residual_decoded["patch_real"],
            "residual_patch_imag": residual_decoded["patch_imag"],
            "trend_overlap_count": trend_decoded["overlap_count"],
            "residual_overlap_count": residual_decoded["overlap_count"],
            "trend_dcloss": trend_dcloss,
            "residual_dcloss": residual_dcloss,
        }
