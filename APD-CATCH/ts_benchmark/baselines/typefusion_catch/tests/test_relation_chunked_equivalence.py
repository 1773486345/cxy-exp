"""Chunked relation execution must match the former vectorised computation."""

from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch

import ts_benchmark.baselines.typefusion_catch.layers.masked_relation_branch as relation_module
from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.layers.masked_relation_branch import (
    MAX_RELATION_ATTENTION_ROWS,
    MaskedRelationBranch,
)
from ts_benchmark.baselines.typefusion_catch.layers.shared_catch_stem import patchify_time


def relation_config(seq_len: int = 16, c_in: int = 4, groups: int = 2) -> TypeFusionConfig:
    return TypeFusionConfig(
        seq_len=seq_len,
        patch_size=4,
        patch_stride=2,
        c_in=c_in,
        d_model=32,
        cf_dim=32,
        d_ff=64,
        n_heads=2,
        head_dim=16,
        e_layers=1,
        dropout=0.0,
        temporal_hidden_dim=32,
        temporal_layers=1,
        memory_size=4,
        memory_topk=2,
        branch_dim=32,
        fusion_layers=1,
        fusion_heads=4,
        relation_mask_groups=groups,
    )


def vectorised_reference(
    branch: MaskedRelationBranch, normalized_input: torch.Tensor, randomize_groups: bool
):
    """The pre-chunk implementation retained solely to test semantic identity."""

    batch, time, channels = normalized_input.shape
    masks, group_index = branch._group_masks(normalized_input.device, randomize_groups)
    groups = masks.size(0)
    masked_input = normalized_input[:, None, :, :].expand(batch, groups, time, channels)
    masked_input = masked_input.masked_fill(masks[None, :, None, :], 0.0)
    flat_input = masked_input.reshape(batch * groups, time, channels)
    hidden = branch.value_projection(flat_input.unsqueeze(-1)) + branch.channel_embedding
    hidden = branch.channel_fusion(hidden.reshape(batch * groups * time, channels, -1))
    hidden = hidden.view(batch * groups, time, channels, -1)
    temporal = hidden.permute(0, 2, 3, 1).reshape(batch * groups * channels, -1, time)
    temporal = branch.temporal_mixer(temporal)
    temporal = temporal.view(batch * groups, channels, -1, time).permute(0, 3, 1, 2)
    temporal = branch.token_projection(temporal)
    prediction_all = branch.output_head(temporal).squeeze(-1).view(batch, groups, time, channels)
    hidden_all = temporal.view(batch, groups, time, channels, -1)
    selected_prediction = prediction_all.permute(0, 2, 1, 3).gather(
        2,
        group_index.view(1, 1, 1, channels).expand(batch, time, 1, channels),
    ).squeeze(2)
    selected_hidden = hidden_all.permute(0, 2, 1, 3, 4).gather(
        2,
        group_index.view(1, 1, 1, channels, 1).expand(
            batch, time, 1, channels, hidden_all.size(-1)
        ),
    ).squeeze(2)
    return {
        "z": patchify_time(
            selected_hidden.mean(dim=2), branch.config.patch_size, branch.config.patch_stride
        ).mean(dim=2),
        "x_hat": selected_prediction,
        "e": (normalized_input - selected_prediction).abs(),
        "channel_masks": masks,
        "group_index": group_index,
    }


def relation_loss(output: dict) -> torch.Tensor:
    return output["x_hat"].square().mean() + output["z"].square().mean() + output["e"].mean()


class RelationChunkedEquivalenceTests(unittest.TestCase):
    def test_outputs_input_gradients_and_parameter_gradients_match_vectorised_reference(self) -> None:
        torch.manual_seed(701)
        chunked = MaskedRelationBranch(relation_config())
        reference = copy.deepcopy(chunked)
        chunked.eval()
        reference.eval()
        input_chunked = torch.randn(2, 16, 4, requires_grad=True)
        input_reference = input_chunked.detach().clone().requires_grad_(True)

        output_chunked = chunked(input_chunked, randomize_groups=False)
        output_reference = vectorised_reference(reference, input_reference, randomize_groups=False)
        for field in ("x_hat", "z", "e", "channel_masks", "group_index"):
            torch.testing.assert_close(
                output_chunked[field], output_reference[field], rtol=1e-6, atol=1e-7
            )

        relation_loss(output_chunked).backward()
        relation_loss(output_reference).backward()
        torch.testing.assert_close(input_chunked.grad, input_reference.grad, rtol=1e-6, atol=1e-7)
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in chunked.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertIsNotNone(reference_parameters[name].grad, name)
            torch.testing.assert_close(
                parameter.grad, reference_parameters[name].grad, rtol=1e-6, atol=1e-7
            )

    def test_chunk_boundaries_small_channel_count_and_long_time_rule(self) -> None:
        torch.manual_seed(702)
        config = relation_config()
        branch = MaskedRelationBranch(config).eval()
        x = torch.randn(5, 16, 4)
        observed_rows = []
        hook = branch.channel_fusion.register_forward_pre_hook(
            lambda _module, inputs: observed_rows.append(inputs[0].shape[0])
        )
        try:
            with mock.patch.object(relation_module, "MAX_RELATION_ATTENTION_ROWS", 32):
                chunked = branch(x, randomize_groups=False)
        finally:
            hook.remove()
        reference = vectorised_reference(branch, x, randomize_groups=False)
        self.assertTrue(observed_rows)
        self.assertLessEqual(max(observed_rows), 32)
        for field in ("x_hat", "z", "e", "channel_masks", "group_index"):
            torch.testing.assert_close(chunked[field], reference[field], rtol=1e-6, atol=1e-7)

        small_channels = MaskedRelationBranch(relation_config(c_in=2, groups=4)).eval()
        small_output = small_channels(torch.randn(3, 16, 2), randomize_groups=False)
        self.assertEqual(small_channels.max_groups, 2)
        self.assertEqual(tuple(small_output["channel_masks"].shape), (2, 2))
        self.assertEqual(tuple(small_output["x_hat"].shape), (3, 16, 2))
        self.assertEqual(
            MaskedRelationBranch._condition_batch_chunk(MAX_RELATION_ATTENTION_ROWS + 1), 1
        )


if __name__ == "__main__":
    unittest.main()
