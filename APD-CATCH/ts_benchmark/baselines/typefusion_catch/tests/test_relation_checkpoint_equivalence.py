"""Activation checkpointing must preserve relation-branch semantics exactly."""

from __future__ import annotations

import copy
import inspect
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.layers.masked_relation_branch import (
    MaskedRelationBranch,
)
from ts_benchmark.baselines.typefusion_catch.layers.shared_catch_stem import patchify_time


RTOL = 1e-6
ATOL = 1e-7


def relation_config(dropout: float = 0.0) -> TypeFusionConfig:
    return TypeFusionConfig(
        seq_len=16,
        patch_size=4,
        patch_stride=2,
        c_in=4,
        d_model=32,
        cf_dim=32,
        d_ff=64,
        n_heads=2,
        head_dim=16,
        e_layers=1,
        dropout=dropout,
        temporal_hidden_dim=32,
        temporal_layers=1,
        memory_size=4,
        memory_topk=2,
        branch_dim=32,
        fusion_layers=1,
        fusion_heads=4,
        relation_mask_groups=2,
    )


def vectorised_reference(
    branch: MaskedRelationBranch,
    normalized_input: torch.Tensor,
    randomize_groups: bool,
):
    """The pre-chunk vectorised reference retained only for equivalence tests."""

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


def relation_loss(output: dict, target: torch.Tensor) -> torch.Tensor:
    """The production relation reconstruction objective."""

    return F.smooth_l1_loss(output["x_hat"], target)


def assert_output_close(test: unittest.TestCase, actual: dict, expected: dict) -> None:
    for field in ("x_hat", "z", "e", "channel_masks", "group_index"):
        torch.testing.assert_close(actual[field], expected[field], rtol=RTOL, atol=ATOL)


def assert_gradients_close(
    test: unittest.TestCase,
    actual: MaskedRelationBranch,
    expected: MaskedRelationBranch,
    actual_input: torch.Tensor,
    expected_input: torch.Tensor,
) -> None:
    test.assertIsNotNone(actual_input.grad)
    test.assertIsNotNone(expected_input.grad)
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=RTOL, atol=ATOL)
    expected_parameters = dict(expected.named_parameters())
    for name, parameter in actual.named_parameters():
        test.assertIsNotNone(parameter.grad, name)
        test.assertIsNotNone(expected_parameters[name].grad, name)
        torch.testing.assert_close(
            parameter.grad,
            expected_parameters[name].grad,
            rtol=RTOL,
            atol=ATOL,
        )


class RelationCheckpointEquivalenceTests(unittest.TestCase):
    def test_reference_no_checkpoint_and_checkpoint_match_outputs_losses_and_gradients(self) -> None:
        torch.manual_seed(801)
        template = MaskedRelationBranch(relation_config(dropout=0.0))
        reference = copy.deepcopy(template).train()
        no_checkpoint = copy.deepcopy(template).train()
        checkpointed = copy.deepcopy(template).train()
        no_checkpoint.use_activation_checkpoint = False
        checkpointed.use_activation_checkpoint = True

        source = torch.randn(2, 16, 4)
        reference_input = source.detach().clone().requires_grad_(True)
        no_checkpoint_input = source.detach().clone().requires_grad_(True)
        checkpointed_input = source.detach().clone().requires_grad_(True)

        expected = vectorised_reference(reference, reference_input, randomize_groups=False)
        actual_no_checkpoint = no_checkpoint(no_checkpoint_input, randomize_groups=False)
        actual_checkpointed = checkpointed(checkpointed_input, randomize_groups=False)
        assert_output_close(self, actual_no_checkpoint, expected)
        assert_output_close(self, actual_checkpointed, expected)

        expected_loss = relation_loss(expected, reference_input)
        no_checkpoint_loss = relation_loss(actual_no_checkpoint, no_checkpoint_input)
        checkpointed_loss = relation_loss(actual_checkpointed, checkpointed_input)
        torch.testing.assert_close(no_checkpoint_loss, expected_loss, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(checkpointed_loss, expected_loss, rtol=RTOL, atol=ATOL)

        expected_loss.backward()
        no_checkpoint_loss.backward()
        checkpointed_loss.backward()
        assert_gradients_close(self, no_checkpoint, reference, no_checkpoint_input, reference_input)
        assert_gradients_close(self, checkpointed, reference, checkpointed_input, reference_input)

    def test_dropout_random_groups_and_rng_state_match_checkpoint_toggle(self) -> None:
        torch.manual_seed(802)
        template = MaskedRelationBranch(relation_config(dropout=0.25))
        no_checkpoint = copy.deepcopy(template).train()
        checkpointed = copy.deepcopy(template).train()
        no_checkpoint.use_activation_checkpoint = False
        checkpointed.use_activation_checkpoint = True
        source = torch.randn(2, 16, 4)

        torch.manual_seed(803)
        no_checkpoint_input = source.detach().clone().requires_grad_(True)
        no_checkpoint_output = no_checkpoint(no_checkpoint_input, randomize_groups=True)
        no_checkpoint_rng_after_forward = torch.get_rng_state().clone()
        no_checkpoint_loss = relation_loss(no_checkpoint_output, no_checkpoint_input)
        no_checkpoint_loss.backward()
        no_checkpoint_rng_after_backward = torch.get_rng_state().clone()

        torch.manual_seed(803)
        checkpointed_input = source.detach().clone().requires_grad_(True)
        checkpointed_output = checkpointed(checkpointed_input, randomize_groups=True)
        checkpointed_rng_after_forward = torch.get_rng_state().clone()
        checkpointed_loss = relation_loss(checkpointed_output, checkpointed_input)
        checkpointed_loss.backward()
        checkpointed_rng_after_backward = torch.get_rng_state().clone()

        assert_output_close(self, checkpointed_output, no_checkpoint_output)
        torch.testing.assert_close(checkpointed_loss, no_checkpoint_loss, rtol=RTOL, atol=ATOL)
        assert_gradients_close(
            self,
            checkpointed,
            no_checkpoint,
            checkpointed_input,
            no_checkpoint_input,
        )
        torch.testing.assert_close(checkpointed_rng_after_forward, no_checkpoint_rng_after_forward)
        torch.testing.assert_close(checkpointed_rng_after_backward, no_checkpoint_rng_after_backward)

    def test_swat_condition_returns_selected_tensors_only(self) -> None:
        config = TypeFusionConfig(
            seq_len=2048,
            patch_size=256,
            patch_stride=64,
            c_in=51,
            d_model=128,
            cf_dim=64,
            d_ff=256,
            n_heads=2,
            head_dim=64,
            e_layers=3,
            dropout=0.0,
            temporal_hidden_dim=128,
            temporal_layers=3,
            memory_size=32,
            memory_topk=4,
            branch_dim=128,
            fusion_layers=2,
            fusion_heads=4,
            relation_mask_groups=4,
        )
        branch = MaskedRelationBranch(config).eval()
        # The return contract, not the Transformer cost, is under test here.
        branch.channel_fusion = nn.Identity()
        branch.temporal_mixer = nn.Identity()
        target_channels = torch.arange(0, config.c_in, 4)
        with torch.no_grad():
            prediction, hidden_sum = branch._condition_forward_selected(
                torch.zeros(1, config.seq_len, config.c_in),
                target_channels,
            )
        self.assertEqual(tuple(prediction.shape), (1, 2048, target_channels.numel()))
        self.assertEqual(tuple(hidden_sum.shape), (1, 2048, 128))
        forward_source = inspect.getsource(MaskedRelationBranch.forward)
        self.assertIn("_run_condition_selected", forward_source)
        self.assertNotIn("hidden_chunk", forward_source)


if __name__ == "__main__":
    unittest.main()
