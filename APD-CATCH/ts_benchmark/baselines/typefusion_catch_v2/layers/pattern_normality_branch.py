"""Two-pass complementary masked patch completion."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _patchify(x: Tensor, size: int, stride: int) -> Tensor:
    length = x.size(1)
    count = 1 if length <= size else (length - size + stride - 1) // stride + 1
    padded = (count - 1) * stride + size
    if padded > length:
        x = F.pad(x, (0, 0, 0, padded - length))
    return x.unfold(1, size, stride).permute(0, 1, 3, 2).contiguous()


def _overlap_add(patches: Tensor, length: int, stride: int) -> Tensor:
    b, p, k, c = patches.shape
    total = (p - 1) * stride + k
    out = patches.new_zeros(b, total, c)
    count = patches.new_zeros(b, total, c)
    for i in range(p):
        out[:, i * stride:i * stride + k] += patches[:, i]
        count[:, i * stride:i * stride + k] += 1
    return (out / count.clamp_min(1.0))[:, :length]


class PatternNormalityBranch(nn.Module):
    def __init__(self, c_in: int, patch_size: int = 16, patch_stride: int = 8, branch_dim: int = 128, dropout: float = 0.0, frequency_weight: float = 0.1) -> None:
        super().__init__()
        if not isinstance(c_in, int):
            config = c_in
            patch_size = int(getattr(config, "patch_size", patch_size))
            patch_stride = int(getattr(config, "patch_stride", patch_stride))
            branch_dim = int(getattr(config, "branch_dim", branch_dim))
            dropout = float(getattr(config, "dropout", dropout))
            frequency_weight = float(getattr(config, "lambda_pattern_frequency", frequency_weight))
            c_in = int(getattr(config, "c_in"))
        self.c_in, self.patch_size, self.patch_stride = int(c_in), int(patch_size), int(patch_stride)
        self.frequency_weight = float(frequency_weight)
        self.patch_projection = nn.Linear(patch_size * c_in, branch_dim)
        layer = nn.TransformerEncoderLayer(branch_dim, 4, branch_dim * 2, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.decoder = nn.Linear(branch_dim, patch_size * c_in)
        self.evidence_head = nn.Linear(branch_dim, 1)

    def _pass(self, patches: Tensor, masked_parity: int) -> tuple[Tensor, Tensor]:
        b, p, k, c = patches.shape
        tokens = self.patch_projection(patches.reshape(b, p, k * c))
        mask = (torch.arange(p, device=patches.device) % 2) == masked_parity
        visible = tokens.masked_fill(mask.view(1, p, 1), 0.0)
        # No target patch content is present in the encoder input. A learned
        # positional signal is not required for the leakage contract.
        encoded = self.encoder(visible)
        visible_mask = (~mask).view(1, p, 1).to(encoded.dtype)
        context_sum = (encoded * visible_mask).sum(dim=1, keepdim=True)
        visible_count = (~mask).sum().clamp_min(1)
        context = context_sum / visible_count
        predicted = self.decoder(context.expand(-1, p, -1)).reshape(b, p, k, c)
        return predicted, mask

    def forward(self, x: Tensor, anchor_context: Optional[Tensor] = None) -> Dict[str, Tensor]:
        patches = _patchify(x, self.patch_size, self.patch_stride)
        pred_even, even = self._pass(patches, 0)
        pred_odd, odd = self._pass(patches, 1)
        predicted = torch.where(even.view(1, -1, 1, 1), pred_even, pred_odd)
        error = (patches - predicted).abs().mean(dim=-1)
        raw_error = _overlap_add(error.unsqueeze(-1), x.size(1), self.patch_stride).squeeze(-1)
        # Build the branch token from masked completions rather than the raw
        # target patch, preserving the completion normality contract.
        z_patches = self.patch_projection(predicted.reshape(x.size(0), predicted.size(1), -1))
        z = _overlap_add(z_patches.unsqueeze(2).expand(-1, -1, self.patch_size, -1), x.size(1), self.patch_stride)
        evidence_logit = self.evidence_head(z).squeeze(-1) + raw_error
        prediction_time = _overlap_add(predicted, x.size(1), self.patch_stride)
        frequency_loss = (torch.fft.rfft(x, dim=1).abs() - torch.fft.rfft(prediction_time, dim=1).abs()).abs().mean()
        return {
            "z": z,
            "prediction": prediction_time,
            "raw_error": raw_error,
            "evidence_logit": evidence_logit,
            "evidence": F.softplus(evidence_logit),
            "task_loss": error.mean() + self.frequency_weight * frequency_loss,
            "masked_patch_predictions": predicted,
            "masked_patch_mask": torch.stack((even, odd)),
        }
