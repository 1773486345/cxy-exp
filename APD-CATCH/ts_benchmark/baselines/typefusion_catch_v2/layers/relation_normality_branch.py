"""Channel-masked conditional relation normality with bounded memory."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class RelationNormalityBranch(nn.Module):
    def __init__(self, config_or_c_in, groups: int = 4, branch_dim: int = 128, dropout: float = 0.0, use_activation_checkpoint: bool = True, max_rows: int = 2048) -> None:
        super().__init__()
        if not isinstance(config_or_c_in, int):
            config = config_or_c_in
            c_in = int(config.c_in)
            groups = int(config.relation_mask_groups)
            branch_dim = int(config.branch_dim)
            dropout = float(config.dropout)
            use_activation_checkpoint = bool(config.use_activation_checkpoint)
            max_rows = int(config.max_relation_attention_rows)
        else:
            c_in = int(config_or_c_in)
        self.c_in = c_in
        self.groups = max(1, min(int(groups), c_in))
        self.branch_dim = int(branch_dim)
        self.max_relation_attention_rows = int(max_rows)
        self.use_activation_checkpoint = bool(use_activation_checkpoint)
        self.value_projection = nn.Linear(1, self.branch_dim)
        self.channel_embedding = nn.Parameter(torch.randn(1, 1, c_in, self.branch_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(self.branch_dim, 4, self.branch_dim * 2, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.channel_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.temporal = nn.Conv1d(self.branch_dim, self.branch_dim, 3, padding=1, groups=self.branch_dim)
        self.token_projection = nn.Linear(self.branch_dim, self.branch_dim)
        self.output_head = nn.Sequential(nn.LayerNorm(self.branch_dim), nn.Linear(self.branch_dim, 1))
        self.evidence_head = nn.Linear(self.branch_dim, 1)

    def _condition_batch_chunk(self, time: int) -> int:
        return max(1, self.max_relation_attention_rows // max(1, time))

    def _condition_forward_selected(self, masked: Tensor, target_channels: Tensor, debug_callback: Optional[Callable[[str], None]] = None) -> tuple[Tensor, Tensor]:
        batch, time, channels = masked.shape
        hidden = self.value_projection(masked.unsqueeze(-1)) + self.channel_embedding
        hidden = self.channel_encoder(hidden.reshape(batch * time, channels, self.branch_dim))
        hidden = hidden.view(batch, time, channels, self.branch_dim)
        temporal = hidden.permute(0, 2, 3, 1).reshape(batch * channels, self.branch_dim, time)
        temporal = self.temporal(temporal)
        temporal = temporal.view(batch, channels, self.branch_dim, time).permute(0, 3, 1, 2)
        temporal = self.token_projection(temporal)
        prediction = self.output_head(temporal).squeeze(-1)
        return prediction.index_select(2, target_channels), temporal.index_select(2, target_channels).sum(dim=2)

    def _condition(self, masked: Tensor, target_channels: Tensor, debug_callback: Optional[Callable[[str], None]]) -> tuple[Tensor, Tensor]:
        checkpoint_enabled = self.use_activation_checkpoint and debug_callback is None and self.training and torch.is_grad_enabled()
        if checkpoint_enabled:
            return checkpoint(lambda value: self._condition_forward_selected(value, target_channels), masked, use_reentrant=False, preserve_rng_state=True)
        return self._condition_forward_selected(masked, target_channels, debug_callback)

    def forward(self, x: Tensor, debug_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Tensor]:
        if x.ndim != 3 or x.size(-1) != self.c_in:
            raise ValueError("x must have shape [B,T,c_in]")
        batch, time, channels = x.shape
        assignments = torch.arange(channels, device=x.device) % self.groups
        predictions = [None] * channels
        hidden_sum = x.new_zeros(batch, time, self.branch_dim)
        chunk_size = self._condition_batch_chunk(time)
        for group in range(self.groups):
            target_channels = (assignments == group).nonzero(as_tuple=False).flatten()
            mask = (assignments == group).view(1, 1, channels)
            prediction_chunks = []
            hidden_chunks = []
            for start in range(0, batch, chunk_size):
                masked_chunk = x[start:start + chunk_size].masked_fill(mask, 0.0)
                pred, hidden = self._condition(masked_chunk, target_channels, debug_callback)
                prediction_chunks.append(pred)
                hidden_chunks.append(hidden)
            group_prediction = torch.cat(prediction_chunks, dim=0)
            hidden_sum = hidden_sum + torch.cat(hidden_chunks, dim=0)
            for local, channel in enumerate(target_channels.tolist()):
                predictions[channel] = group_prediction[:, :, local]
        if any(item is None for item in predictions):
            raise RuntimeError("relation masking failed to select every channel")
        reconstructed = torch.stack([item for item in predictions if item is not None], dim=-1)
        z = hidden_sum / float(channels)
        raw_error = (x - reconstructed).abs().mean(dim=-1)
        evidence_logit = self.evidence_head(z).squeeze(-1) + raw_error
        return {
            "z": z,
            "prediction": reconstructed,
            "raw_error": raw_error,
            "evidence_logit": evidence_logit,
            "evidence": torch.nn.functional.softplus(evidence_logit),
            "task_loss": raw_error.mean(),
            "channel_assignments": assignments,
            "condition_batch_chunk": torch.tensor(chunk_size, device=x.device),
        }
