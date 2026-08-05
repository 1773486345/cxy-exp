"""Masked non-overlapping temporal completion with masked frequency memory."""

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
        self.c_in, self.patch_size = c_in, patch_size
        self.patch_stride = int(patch_stride)  # public CATCH-style frequency stride
        self.time_stride = int(patch_size)  # completion is explicitly non-overlapping
        self.branch_dim = int(branch_dim)
        self.frequency_weight = float(frequency_weight)
        self.time_patch_count = 1 if seq_len <= patch_size else (seq_len - patch_size + patch_size - 1) // patch_size + 1
        self.time_patch_projection = nn.Linear(patch_size * c_in, self.branch_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.branch_dim))
        self.frequency_mask_value = nn.Parameter(torch.zeros(1, 1, c_in))
        self.position = nn.Parameter(torch.randn(1, self.time_patch_count, self.branch_dim) * 0.02)
        self.frequency_projection = nn.Linear(input_dim, self.branch_dim)
        self.frequency_cross_attention = nn.MultiheadAttention(self.branch_dim, 4, dropout=dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(self.branch_dim, 4, self.branch_dim * 2, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.patch_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.decoder = nn.Linear(self.branch_dim, patch_size * c_in)
        self.evidence_head = nn.Linear(self.branch_dim, 1)

    @staticmethod
    def _target_mask(count: int, parity: int, device: torch.device) -> Tensor:
        """Return a mask that always leaves a real visible patch."""
        if count <= 1:
            return torch.zeros(count, dtype=torch.bool, device=device)
        mask = (torch.arange(count, device=device) % 2) == parity
        if bool(mask.all()):
            mask[-1] = False
        if bool((~mask).all()):
            mask[0] = False
        return mask

    def _masked_pass(self, x: Tensor, parity: int, frontend: Optional[nn.Module]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        patches = patchify_time(x, self.patch_size, self.time_stride)
        batch, count, size, channels = patches.shape
        content_tokens = self.time_patch_projection(patches.reshape(batch, count, size * channels))
        target_mask = self._target_mask(count, parity, x.device)
        position = self.position[:, :count]
        # Replace target content first, then add position.  Positional signal
        # therefore remains active on masked target tokens.
        time_tokens = torch.where(target_mask.view(1, count, 1), self.mask_token.expand(batch, count, -1), content_tokens)
        time_tokens = time_tokens + position
        if frontend is not None:
            frequency_output = frontend.encode_frequency(x)
            frequency_memory = self.frequency_projection(frequency_output["h_freq"].mean(dim=2))
            cross_context, _ = self.frequency_cross_attention(time_tokens, frequency_memory, frequency_memory, need_weights=False)
            time_tokens = time_tokens + cross_context
        encoded = self.patch_encoder(time_tokens)
        predicted = self.decoder(encoded).view(batch, count, size, channels)
        return predicted, target_mask, encoded, time_tokens, position.expand(batch, -1, -1)

    def _masked_input(self, x: Tensor, target_mask: Tensor) -> Tensor:
        points = torch.zeros(x.size(1), dtype=torch.bool, device=x.device)
        for index, masked in enumerate(target_mask.tolist()):
            if masked:
                start = index * self.time_stride
                points[start:min(x.size(1), start + self.patch_size)] = True
        return torch.where(points.view(1, -1, 1), self.frequency_mask_value.expand(x.size(0), x.size(1), -1), x)

    def forward(self, x: Tensor, frontend: Optional[nn.Module] = None) -> Dict[str, Tensor]:
        patches = patchify_time(x, self.patch_size, self.time_stride)
        count = patches.size(1)
        even_mask = self._target_mask(count, 0, x.device)
        odd_mask = self._target_mask(count, 1, x.device)
        even_input = self._masked_input(x, even_mask)
        odd_input = self._masked_input(x, odd_mask)
        pred_even, _, latent_even, tokens_even, pos_even = self._masked_pass(even_input, 0, frontend)
        pred_odd, _, latent_odd, tokens_odd, pos_odd = self._masked_pass(odd_input, 1, frontend)
        selected = torch.where(even_mask.view(1, -1, 1, 1), pred_even, pred_odd)
        original_patches = patchify_time(x, self.patch_size, self.time_stride)
        point_error = (original_patches - selected).abs().mean(dim=-1, keepdim=True)
        prediction = overlap_add(selected, x.size(1), self.time_stride)
        raw_error = overlap_add(point_error, x.size(1), self.time_stride).squeeze(-1)
        local_freq_error = (torch.fft.rfft(original_patches, dim=2).abs() - torch.fft.rfft(selected, dim=2).abs()).abs().mean(dim=(2, 3), keepdim=False)
        local_freq_error = local_freq_error.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.patch_size, 1)
        local_freq_error = overlap_add(local_freq_error, x.size(1), self.time_stride).squeeze(-1)
        raw_error = raw_error + self.frequency_weight * local_freq_error
        latent = torch.where(even_mask.view(1, -1, 1), latent_even, latent_odd)
        z = overlap_add(latent.unsqueeze(2).expand(-1, -1, self.patch_size, -1), x.size(1), self.time_stride)
        evidence_logit = self.evidence_head(z).squeeze(-1) + raw_error
        return {
            "z": z,
            "prediction": prediction,
            "raw_error": raw_error,
            "evidence_logit": evidence_logit,
            "evidence": F.softplus(evidence_logit),
            "task_loss": raw_error.mean(),
            "masked_patch_predictions": selected,
            "masked_pass_predictions": torch.stack((pred_even, pred_odd), dim=1),
            "masked_patch_mask": torch.stack((even_mask, odd_mask)),
            "visible_patch_mask": torch.stack((~even_mask, ~odd_mask)),
            "target_position": torch.stack((pos_even, pos_odd), dim=1).detach(),
            "masked_time_tokens": torch.stack((tokens_even, tokens_odd), dim=1),
            "masked_pass_tokens": torch.stack((tokens_even, tokens_odd), dim=1),
            "masked_frequency_used": torch.tensor(True, device=x.device),
        }
