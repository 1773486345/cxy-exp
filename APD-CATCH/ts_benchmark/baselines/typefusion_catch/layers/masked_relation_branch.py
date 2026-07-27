"""Channel-masked conditional reconstruction for multivariate relations."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.layers.shared_catch_stem import patchify_time


MAX_RELATION_ATTENTION_ROWS = 2048


class MaskedRelationBranch(nn.Module):
    """Reconstructs each channel only in a forward pass where that channel is masked.

    Each deterministic masking condition uses the same channel Transformer.
    Conditions are evaluated group-by-group and batch-chunk-by-batch-chunk so
    the channel-attention batch never scales as ``B * G * T``. Selecting an
    output for a target channel always selects the group where that target was
    zero-masked.
    """

    def __init__(self, config: TypeFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.max_groups = min(config.relation_mask_groups, config.c_in)
        hidden = config.d_model
        self.value_projection = nn.Linear(1, hidden)
        self.channel_embedding = nn.Parameter(torch.randn(1, 1, config.c_in, hidden) * 0.02)
        attention = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.fusion_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_fusion = nn.TransformerEncoder(attention, num_layers=1)
        self.temporal_mixer = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.output_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.token_projection = nn.Linear(hidden, hidden)
        # Implementation-only memory control. It is deliberately not a public
        # benchmark hyperparameter and remains enabled for all formal tasks.
        self.use_activation_checkpoint = True

    def _group_masks(self, device: torch.device, randomize: bool) -> Tuple[Tensor, Tensor]:
        channels = self.config.c_in
        groups = self.max_groups
        if randomize:
            order = torch.randperm(channels, device=device)
        else:
            order = torch.arange(channels, device=device)
        group_index = torch.empty(channels, dtype=torch.long, device=device)
        group_index[order] = torch.arange(channels, device=device) % groups
        masks = torch.zeros(groups, channels, dtype=torch.bool, device=device)
        masks[group_index, torch.arange(channels, device=device)] = True
        return masks, group_index

    @staticmethod
    def _condition_batch_chunk(time: int) -> int:
        """Limit each channel-attention call to a fixed implementation bound."""

        return max(1, MAX_RELATION_ATTENTION_ROWS // time)

    def _condition_forward_selected(
        self,
        masked_input: Tensor,
        target_channels: Tensor,
        debug_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run one condition and retain only its masked-channel outputs.

        The full channel tensor is required while applying the shared channel
        Transformer and temporal mixer, but it never leaves this condition.
        """

        chunk_batch, time, channels = masked_input.shape
        if debug_callback is not None:
            debug_callback("value_projection_start")
        hidden = self.value_projection(masked_input.unsqueeze(-1))
        if debug_callback is not None:
            debug_callback("value_projection_end")
        hidden = hidden + self.channel_embedding
        if debug_callback is not None:
            debug_callback("channel_fusion_start")
        hidden = self.channel_fusion(hidden.reshape(chunk_batch * time, channels, -1))
        if debug_callback is not None:
            debug_callback("channel_fusion_end")
        hidden = hidden.view(chunk_batch, time, channels, -1)

        temporal = hidden.permute(0, 2, 3, 1).reshape(chunk_batch * channels, -1, time)
        if debug_callback is not None:
            debug_callback("temporal_mixer_start")
        temporal = self.temporal_mixer(temporal)
        if debug_callback is not None:
            debug_callback("temporal_mixer_end")
        temporal = temporal.view(chunk_batch, channels, -1, time).permute(0, 3, 1, 2)
        if debug_callback is not None:
            debug_callback("token_projection_start")
        temporal = self.token_projection(temporal)
        if debug_callback is not None:
            debug_callback("token_projection_end")
            debug_callback("output_head_start")
        prediction = self.output_head(temporal).squeeze(-1)
        if debug_callback is not None:
            debug_callback("output_head_end")
            debug_callback("selected_prediction_start")
        selected_prediction = prediction.index_select(dim=2, index=target_channels)
        selected_hidden_sum = temporal.index_select(dim=2, index=target_channels).sum(dim=2)
        if debug_callback is not None:
            debug_callback("selected_prediction_end")
        return selected_prediction, selected_hidden_sum

    def _should_checkpoint_condition(
        self,
        masked_input: Tensor,
        debug_callback: Optional[Callable[[str], None]],
    ) -> bool:
        """Checkpoint only the normal training path, never synchronized debug runs."""

        if (
            not self.use_activation_checkpoint
            or debug_callback is not None
            or not self.training
            or not torch.is_grad_enabled()
        ):
            return False
        return masked_input.requires_grad or any(
            parameter.requires_grad for parameter in self.parameters()
        )

    def _run_condition_selected(
        self,
        masked_input: Tensor,
        target_channels: Tensor,
        debug_callback: Optional[Callable[[str], None]],
    ) -> Tuple[Tensor, Tensor]:
        if not self._should_checkpoint_condition(masked_input, debug_callback):
            return self._condition_forward_selected(
                masked_input,
                target_channels,
                debug_callback=debug_callback,
            )

        # target_channels is a deterministic group mask selected before the
        # checkpoint. It is captured, not regenerated during backward replay.
        return checkpoint(
            lambda condition_input: self._condition_forward_selected(
                condition_input,
                target_channels,
            ),
            masked_input,
            use_reentrant=False,
            preserve_rng_state=True,
        )

    def forward(
        self,
        normalized_input: Tensor,
        randomize_groups: bool,
        debug_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Tensor]:
        batch, time, channels = normalized_input.shape
        masks, group_index = self._group_masks(normalized_input.device, randomize_groups)
        groups = masks.size(0)

        condition_batch_chunk = self._condition_batch_chunk(time)
        selected_channels: List[Optional[Tensor]] = [None] * channels
        selected_hidden_sum: Optional[Tensor] = None
        for group in range(groups):
            group_mask = masks[group]
            target_channels = group_mask.nonzero(as_tuple=False).flatten()
            prediction_chunks: List[Tensor] = []
            hidden_sum_chunks: List[Tensor] = []
            for batch_start in range(0, batch, condition_batch_chunk):
                batch_end = min(batch, batch_start + condition_batch_chunk)
                input_chunk = normalized_input[batch_start:batch_end]
                masked_chunk = input_chunk.masked_fill(group_mask.view(1, 1, channels), 0.0)
                prediction_chunk, hidden_sum_chunk = self._run_condition_selected(
                    masked_chunk,
                    target_channels,
                    debug_callback=debug_callback,
                )
                prediction_chunks.append(prediction_chunk)
                hidden_sum_chunks.append(hidden_sum_chunk)

            group_prediction = torch.cat(prediction_chunks, dim=0)
            group_hidden_sum = torch.cat(hidden_sum_chunks, dim=0)
            for local_index, channel_index in enumerate(target_channels.tolist()):
                selected_channels[channel_index] = group_prediction[:, :, local_index]
            selected_hidden_sum = (
                group_hidden_sum
                if selected_hidden_sum is None
                else selected_hidden_sum + group_hidden_sum
            )

        if selected_hidden_sum is None or any(channel is None for channel in selected_channels):
            raise RuntimeError("relation masking failed to select every channel")
        selected_prediction = torch.stack(
            [channel for channel in selected_channels if channel is not None], dim=2
        )
        selected_hidden_mean = selected_hidden_sum / channels
        z = patchify_time(
            selected_hidden_mean, self.config.patch_size, self.config.patch_stride
        ).mean(dim=2)
        return {
            "z": z,
            "x_hat": selected_prediction,
            "e": (normalized_input - selected_prediction).abs(),
            "channel_masks": masks,
            "group_index": group_index,
        }

    def unfreeze_tail(self) -> None:
        for module in (self.output_head, self.token_projection):
            for parameter in module.parameters():
                parameter.requires_grad = True
