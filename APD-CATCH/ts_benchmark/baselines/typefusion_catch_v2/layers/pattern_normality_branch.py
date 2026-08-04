"""Mask-before-encode temporal pattern normality branch."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .patch_utils import overlap_add, patchify_time


class PatternNormalityBranch(nn.Module):
    def __init__(self, config_or_c_in, patch_size: int = 16, patch_stride: int = 8, branch_dim: int = 128, dropout: float = 0.0, frequency_weight: float = 0.1, seq_len: Optional[int] = None) -> None:
        super().__init__()
        if not isinstance(config_or_c_in, int):
            config = config_or_c_in
            c_in = int(config.c_in)
            patch_size = int(config.patch_size)
            patch_stride = int(config.patch_stride)
            branch_dim = int(config.branch_dim)
            dropout = float(config.dropout)
            frequency_weight = float(config.lambda_pattern_frequency)
            seq_len = int(config.seq_len)
            input_dim = int(config.d_model)
        else:
            c_in = int(config_or_c_in)
            input_dim = branch_dim
            seq_len = int(seq_len or patch_size)
        self.c_in, self.patch_size, self.patch_stride = c_in, patch_size, patch_stride
        self.branch_dim = int(branch_dim)
        self.frequency_weight = float(frequency_weight)
        patch_count = 1 if seq_len <= patch_size else (seq_len - patch_size + patch_stride - 1) // patch_stride + 1
        self.time_patch_projection = nn.Linear(patch_size * c_in, self.branch_dim)
        self.frequency_projection = nn.Linear(input_dim, self.branch_dim)
        self.position = nn.Parameter(torch.randn(1, patch_count, self.branch_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(self.branch_dim, 4, self.branch_dim * 2, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.patch_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.decoder = nn.Linear(self.branch_dim, patch_size * c_in)
        self.evidence_head = nn.Linear(self.branch_dim, 1)

    def _masked_pass(self, x: Tensor, parity: int, frontend: Optional[nn.Module]) -> tuple[Tensor, Tensor, Tensor]:
        patches = patchify_time(x, self.patch_size, self.patch_stride)
        batch, count, size, channels = patches.shape
        tokens = self.time_patch_projection(patches.reshape(batch, count, size * channels))
        if frontend is not None:
            frequency = frontend(x)["h_freq"].mean(dim=2)
            tokens = tokens + self.frequency_projection(frequency)
        tokens = tokens + self.position[:, :count]
        target_mask = (torch.arange(count, device=x.device) % 2) == parity
        tokens = tokens.masked_fill(target_mask.view(1, count, 1), 0.0)
        encoded = self.patch_encoder(tokens)
        predicted = self.decoder(encoded).view(batch, count, size, channels)
        return predicted, target_mask, encoded

    def forward(self, x: Tensor, frontend: Optional[nn.Module] = None) -> Dict[str, Tensor]:
        patches = patchify_time(x, self.patch_size, self.patch_stride)
        even_mask = (torch.arange(patches.size(1), device=x.device) % 2) == 0
        odd_mask = ~even_mask
        even_points = torch.zeros(x.size(1), dtype=torch.bool, device=x.device)
        odd_points = torch.zeros_like(even_points)
        for index in range(patches.size(1)):
            start = index * self.patch_stride
            target = even_points if bool(even_mask[index]) else odd_points
            target[start:min(x.size(1), start + self.patch_size)] = True
        even_input = x.masked_fill(even_points.view(1, -1, 1), 0.0)
        odd_input = x.masked_fill(odd_points.view(1, -1, 1), 0.0)
        pred_even, _, latent_even = self._masked_pass(even_input, 0, frontend)
        pred_odd, _, latent_odd = self._masked_pass(odd_input, 1, frontend)
        selected = torch.where(even_mask.view(1, -1, 1, 1), pred_even, pred_odd)
        original_patches = patchify_time(x, self.patch_size, self.patch_stride)
        point_error = (original_patches - selected).abs().mean(dim=-1, keepdim=True)
        prediction = overlap_add(selected, x.size(1), self.patch_stride)
        raw_error = overlap_add(point_error, x.size(1), self.patch_stride).squeeze(-1)
        local_freq_error = (torch.fft.rfft(original_patches, dim=2).abs() - torch.fft.rfft(selected, dim=2).abs()).abs().mean(dim=(2, 3), keepdim=False)
        # Expand each patch-level scalar over its patch before overlap-add.
        local_freq_error = local_freq_error.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.patch_size, 1)
        local_freq_error = overlap_add(local_freq_error, x.size(1), self.patch_stride).squeeze(-1)
        raw_error = raw_error + self.frequency_weight * local_freq_error
        latent = torch.where(even_mask.view(1, -1, 1), latent_even, latent_odd)
        z = overlap_add(latent.unsqueeze(2).expand(-1, -1, self.patch_size, -1), x.size(1), self.patch_stride)
        evidence_logit = self.evidence_head(z).squeeze(-1) + raw_error
        return {
            "z": z,
            "prediction": prediction,
            "raw_error": raw_error,
            "evidence_logit": evidence_logit,
            "evidence": F.softplus(evidence_logit),
            "task_loss": raw_error.mean(),
            "masked_patch_predictions": selected,
            "masked_patch_mask": torch.stack((even_mask, odd_mask)),
            "masked_frequency_latent": latent,
        }
