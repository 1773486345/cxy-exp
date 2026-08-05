import json
import re
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from .common import make_model, tiny_config
from ..TypeFusionCATCHV2 import TypeFusionCATCHV2
from ..layers.evolution_normality_branch import EvolutionNormalityBranch
from ..layers.pattern_normality_branch import PatternNormalityBranch
from ..layers.relation_aware_joint_scorer import RelationAwareJointScorer
from ..layers.relation_normality_branch import RelationNormalityBranch
from ..layers.shared_representation_frontend import SharedRepresentationFrontend
from ..layers.state_normality_branch import StateNormalityBranch
from ..losses import _branch_valid_mask, compute_losses


class PatternRepairTests(unittest.TestCase):
    def test_each_pass_has_visible_patch_and_position_survives_mask(self):
        config = tiny_config()
        branch = PatternNormalityBranch(config)
        output = branch(torch.randn(2, config.seq_len, config.c_in))
        visible = output["visible_patch_mask"]
        self.assertTrue(torch.all(visible.sum(dim=1) >= 1))
        positions = output["target_position"]
        for pass_index in range(2):
            target = output["masked_patch_mask"][pass_index]
            self.assertGreater(float(positions[:, pass_index][:, target].abs().sum()), 0.0)

    def test_frontend_masked_target_has_no_content_leakage(self):
        config = tiny_config()
        frontend = SharedRepresentationFrontend(config)
        branch = PatternNormalityBranch(config)
        x = torch.randn(2, config.seq_len, config.c_in)
        first = branch(x, frontend=frontend)
        changed = x.clone()
        changed[:, :config.patch_size] += 100.0
        second = branch(changed, frontend=frontend)
        # The even pass masks the first non-overlapping time patch.
        self.assertTrue(torch.allclose(first["masked_pass_predictions"][:, 0], second["masked_pass_predictions"][:, 0], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(first["masked_pass_tokens"][:, 0], second["masked_pass_tokens"][:, 0], atol=1e-6, rtol=1e-6))

    def test_frequency_length_is_not_index_aligned_to_time_length(self):
        config = tiny_config()
        branch = PatternNormalityBranch(config)

        class DifferentLengthFrontend:
            def encode_frequency(self, x):
                return {"h_freq": torch.randn(x.size(0), 3, x.size(2), config.d_model)}

        output = branch(torch.randn(2, config.seq_len, config.c_in), frontend=DifferentLengthFrontend())
        self.assertEqual(output["prediction"].shape[:2], (2, config.seq_len))

    def test_frequency_debug_flag_reflects_frontend(self):
        config = tiny_config()
        branch = PatternNormalityBranch(config)
        x = torch.randn(2, config.seq_len, config.c_in)
        self.assertFalse(bool(branch(x, frontend=None)["masked_frequency_used"]))
        self.assertTrue(bool(branch(x, frontend=SharedRepresentationFrontend(config))["masked_frequency_used"]))


class InterventionAndValidationTests(unittest.TestCase):
    def test_donor_never_self_for_batch_two_and_three(self):
        generator = make_model().intervention_generator
        for batch in (2, 3):
            x = torch.randn(batch, 16, 4)
            result = generator.generate(x, validation=True, sample_indices=list(range(batch)))
            self.assertTrue(torch.all(result["donor_indices"] != torch.arange(batch)))
            self.assertTrue(torch.equal(result["donor_indices"], result["debug_donor_indices"]))

    def test_validation_windows_and_interventions_repeat_exactly(self):
        frame = pd.DataFrame(np.arange(40 * 2, dtype=np.float32).reshape(40, 2))
        adapter = TypeFusionCATCHV2(seq_len=8, patch_size=4, patch_stride=2, batch_size=64, c_in=2)
        _, loader = adapter._prepare_loaders(frame)

        # The adapter owns no model before fit; use a generator with the same seed
        # for this loader-level determinism check.
        from ..layers.type_interventions import TypeInterventionGenerator
        generator = TypeInterventionGenerator(adapter.config.seed)

        def collect_with(generator):
            rows = []
            offset = 0
            for batch, _ in loader:
                intervention = generator.generate(batch, validation=True, sample_indices=range(offset, offset + batch.size(0)))
                rows.append(tuple(intervention[key].clone() for key in ("corrupted_x", "type_targets", "type_masks", "donor_indices", "debug_intervals", "debug_channels", "debug_strengths")))
                offset += batch.size(0)
            return rows

        first = collect_with(generator)
        second = collect_with(TypeInterventionGenerator(adapter.config.seed))
        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            for left_value, right_value in zip(left, right):
                self.assertTrue(torch.equal(left_value, right_value))


class CompactWeakViewTests(unittest.TestCase):
    @staticmethod
    def _forced_intervention(model, x, weak_indices):
        """Create deterministic intervention metadata with selected weak rows."""
        batch, time, channels = x.shape
        intervention = model.intervention_generator.generate(
            x.detach(), validation=True, sample_indices=list(range(batch))
        )
        weak_indices = torch.as_tensor(weak_indices, dtype=torch.long)
        scenario = torch.zeros(batch, dtype=torch.long)
        scenario[weak_indices] = 3
        intervention["scenario_kind"] = scenario
        intervention["corrupted_x"] = x.detach().clone()
        intervention["weak_view_i"] = x.detach().clone()
        intervention["weak_view_j"] = x.detach().clone()
        intervention["type_targets"].zero_()
        intervention["type_targets"][weak_indices, 0] = 1.0
        intervention["type_targets"][weak_indices, 1] = 1.0
        intervention["type_masks"].zero_()
        intervention["weak_mask_i"].zero_()
        intervention["weak_mask_j"].zero_()
        for index in weak_indices.tolist():
            interval = slice(1, min(time, 4))
            intervention["corrupted_x"][index, interval, 0] += 2.0
            intervention["weak_view_i"][index, interval, 0] += 0.75
            intervention["weak_view_j"][index, interval, 1 % channels] -= 0.75
            intervention["type_masks"][index, 0, interval, 0] = True
            intervention["type_masks"][index, 1, interval, 0] = True
            intervention["weak_mask_i"][index, interval] = True
            intervention["weak_mask_j"][index, interval] = True
        intervention["union_mask"] = intervention["type_masks"].any(dim=1).any(dim=-1)
        return intervention

    def test_one_weak_compound_uses_compact_views(self):
        model = make_model()
        model.eval()
        x = torch.randn(8, model.config.seq_len, model.config.c_in)
        intervention = self._forced_intervention(model, x, [3])
        calls = []
        original = model._run_views

        def counted(view):
            calls.append(view.size(0))
            return original(view)

        model._run_views = counted
        output = model(x, intervention=intervention, compute_loss=False)
        self.assertEqual(calls[-2:], [1, 1])
        weak = output["intervention_output"]["weak_views"]
        self.assertEqual(tuple(weak["sample_indices"].tolist()), (3,))
        self.assertEqual(weak["logit_i"].size(0), 1)
        self.assertEqual(weak["logit_j"].size(0), 1)

    def test_no_weak_compound_skips_views_and_synergy_is_zero(self):
        model = make_model()
        model.eval()
        x = torch.randn(8, model.config.seq_len, model.config.c_in)
        intervention = self._forced_intervention(model, x, [])
        calls = []
        original = model._run_views

        def counted(view):
            calls.append(view.size(0))
            return original(view)

        model._run_views = counted
        output = model(x, intervention=intervention, compute_loss=True)
        self.assertEqual(calls, [8, 8])
        self.assertNotIn("weak_views", output["intervention_output"])
        self.assertTrue(torch.isfinite(output["losses"]["synergy"]))
        self.assertEqual(float(output["losses"]["synergy"]), 0.0)

    def test_multiple_weak_samples_have_compact_indices_and_masks(self):
        model = make_model()
        model.eval()
        x = torch.randn(8, model.config.seq_len, model.config.c_in)
        intervention = self._forced_intervention(model, x, [1, 4, 7])
        output = model(x, intervention=intervention, compute_loss=False)
        weak = output["intervention_output"]["weak_views"]
        self.assertTrue(torch.equal(weak["sample_indices"], torch.tensor([1, 4, 7])))
        for key in ("logit_i", "logit_j", "compound_logit", "mask_i", "mask_j", "union_mask"):
            self.assertEqual(weak[key].size(0), 3)

    def test_compact_synergy_matches_direct_per_sample_reference(self):
        config = tiny_config(
            lambda_task=0.0, lambda_evidence=0.0, lambda_responsibility=0.0,
            lambda_score=0.0, lambda_score_rank=0.0, lambda_clean_score=0.0,
            lambda_synergy=1.0,
        )
        clean = {
            "branches": {name: {"task_loss": torch.zeros(()), "evidence_logit": torch.zeros(2, 4)} for name in ("state", "evolution", "pattern", "relation")},
            "joint_logit": torch.zeros(2, 4), "joint_score": torch.zeros(2, 4),
        }
        mask_i = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=torch.bool)
        mask_j = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 0]], dtype=torch.bool)
        union = mask_i | mask_j
        weak_views = {
            "logit_i": torch.tensor([[0.2, 0.4, 0.0, 0.0], [0.1, 0.3, 0.5, 0.0]], requires_grad=True),
            "logit_j": torch.tensor([[0.3, 0.1, 0.6, 0.0], [0.4, 0.2, 0.2, 0.0]], requires_grad=True),
            "compound_logit": torch.tensor([[0.5, 0.6, 0.2, 0.0], [0.5, 0.7, 0.4, 0.0]], requires_grad=True),
            "mask_i": mask_i, "mask_j": mask_j, "union_mask": union,
            "sample_indices": torch.tensor([2, 6]),
        }
        intervention = {
            "branches": {name: {"evidence_logit": torch.zeros(2, 4)} for name in ("state", "evolution", "pattern", "relation")},
            "joint_logit": torch.zeros(2, 4),
            "type_targets": torch.zeros(2, 4), "type_masks": torch.zeros(2, 4, 4, 1),
            "union_mask": union, "weak_views": weak_views,
        }
        actual = compute_losses(clean, intervention, config)["synergy"]
        score_i = (torch.nn.functional.softplus(weak_views["logit_i"]) * mask_i).sum(dim=1) / mask_i.sum(dim=1)
        score_j = (torch.nn.functional.softplus(weak_views["logit_j"]) * mask_j).sum(dim=1) / mask_j.sum(dim=1)
        score_c = (torch.nn.functional.softplus(weak_views["compound_logit"]) * union).sum(dim=1) / union.sum(dim=1)
        expected = torch.relu(config.synergy_margin - score_c + torch.maximum(score_i, score_j)).mean()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-7))
        actual.backward()
        for key in ("logit_i", "logit_j", "compound_logit"):
            self.assertTrue(torch.isfinite(weak_views[key].grad).all())

    def test_compact_and_full_reference_have_equal_loss_and_gradients(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        config = tiny_config(batch_size=8)
        model_compact = make_model(batch_size=8)
        model_reference = make_model(batch_size=8)
        model_reference.load_state_dict(model_compact.state_dict())
        model_compact.eval()
        model_reference.eval()
        base = torch.randn(8, config.seq_len, config.c_in)
        x_compact = base.clone().requires_grad_(True)
        x_reference = base.clone().requires_grad_(True)
        intervention_compact = self._forced_intervention(model_compact, x_compact, [3])
        intervention_reference = self._forced_intervention(model_reference, x_reference, [3])

        compact_output = model_compact(x_compact, intervention=intervention_compact, compute_loss=True)
        compact_loss = compact_output["losses"]["total"]
        compact_loss.backward()

        clean = model_reference._run_views(x_reference)
        corrupted = model_reference._run_views(intervention_reference["corrupted_x"])
        full_i = model_reference._run_views(intervention_reference["weak_view_i"])
        full_j = model_reference._run_views(intervention_reference["weak_view_j"])
        index = torch.tensor([3], dtype=torch.long)
        reference_intervention = {
            **corrupted,
            "type_targets": intervention_reference["type_targets"],
            "type_masks": intervention_reference["type_masks"],
            "union_mask": intervention_reference["union_mask"],
            "scenario_kind": intervention_reference["scenario_kind"],
            "weak_views": {
                "logit_i": full_i["joint_logit"].index_select(0, index),
                "logit_j": full_j["joint_logit"].index_select(0, index),
                "compound_logit": corrupted["joint_logit"].index_select(0, index),
                "mask_i": intervention_reference["weak_mask_i"].index_select(0, index),
                "mask_j": intervention_reference["weak_mask_j"].index_select(0, index),
                "union_mask": intervention_reference["union_mask"].index_select(0, index),
                "sample_indices": index,
            },
        }
        reference_loss = compute_losses(clean, reference_intervention, config)["total"]
        reference_loss.backward()

        self.assertTrue(torch.allclose(compact_loss, reference_loss, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(x_compact.grad, x_reference.grad, atol=1e-5, rtol=1e-5))
        for left_module, right_module in (
            (model_compact.pattern_branch, model_reference.pattern_branch),
            (model_compact.relation_branch, model_reference.relation_branch),
            (model_compact.joint_scorer, model_reference.joint_scorer),
        ):
            left_grads = [parameter.grad for parameter in left_module.parameters() if parameter.grad is not None]
            right_grads = [parameter.grad for parameter in right_module.parameters() if parameter.grad is not None]
            self.assertTrue(left_grads and right_grads)
            self.assertEqual(len(left_grads), len(right_grads))
            for left_grad, right_grad in zip(left_grads, right_grads):
                self.assertTrue(torch.allclose(left_grad, right_grad, atol=1e-5, rtol=1e-5))


class StateLossEvolutionTests(unittest.TestCase):
    def test_state_topk_changes_assignment(self):
        config = tiny_config(state_topk=1)
        sparse = StateNormalityBranch(config)
        dense_config = tiny_config(state_topk=4)
        dense = StateNormalityBranch(dense_config)
        dense.load_state_dict(sparse.state_dict())
        hidden = torch.randn(2, config.seq_len, config.d_model)
        sparse_output = sparse(hidden)
        dense_output = dense(hidden)
        self.assertTrue(torch.all((sparse_output["prototype_assignment"] > 0).sum(dim=-1) == 1))
        self.assertTrue(torch.all((dense_output["prototype_assignment"] > 0).sum(dim=-1) == 4))
        self.assertFalse(torch.allclose(sparse_output["raw_error"], dense_output["raw_error"]))

    def test_dense_usage_assignment_regularizes_unselected_prototypes(self):
        config = tiny_config(state_topk=1)
        branch = StateNormalityBranch(config)
        with torch.no_grad():
            branch.state_projection[0].weight.zero_()
            branch.state_projection[0].bias.zero_()
            branch.prototypes.copy_(torch.arange(config.state_memory_size, dtype=torch.float32).view(-1, 1).expand(-1, config.branch_dim) * 0.5)
        output = branch(torch.zeros(2, config.seq_len, config.d_model))
        usage_assignment = output["prototype_usage_assignment"]
        self.assertTrue(torch.all(usage_assignment > 0))
        self.assertTrue(torch.allclose(usage_assignment.sum(dim=-1), torch.ones_like(usage_assignment.sum(dim=-1))))
        formal_assignment = output["prototype_assignment"]
        selected = formal_assignment.sum(dim=(0, 1)) > 0
        self.assertTrue(torch.any(~selected))
        branch.zero_grad(set_to_none=True)
        output["prototype_usage_loss"].backward()
        unselected_grad = branch.prototypes.grad[~selected]
        self.assertTrue(torch.isfinite(unselected_grad).all())
        self.assertGreater(float(unselected_grad.abs().sum()), 0.0)

    def test_branch_valid_fallback_b2_t7_clean_and_intervention(self):
        config = tiny_config(lambda_task=0.0, lambda_evidence=1.0, lambda_responsibility=0.0, lambda_score=0.0, lambda_score_rank=0.0, lambda_clean_score=0.0, lambda_synergy=0.0)
        batch, time = 2, 7
        branches = {name: {"task_loss": torch.zeros(()), "evidence_logit": torch.zeros(batch, time)} for name in ("state", "evolution", "pattern", "relation")}
        evolution_valid = torch.ones(batch, time, dtype=torch.bool)
        evolution_valid[:, 0] = False
        branches["evolution"]["valid_mask"] = evolution_valid
        fallback = _branch_valid_mask(branches, torch.zeros(batch, time))
        self.assertEqual(tuple(fallback.shape), (batch, time, 4))
        clean = {"branches": branches, "joint_logit": torch.zeros(batch, time), "joint_score": torch.zeros(batch, time)}
        clean_losses = compute_losses(clean, None, config)
        self.assertTrue(torch.isfinite(clean_losses["clean_evidence"]))
        intervention_branches = {name: dict(value) for name, value in branches.items()}
        type_masks = torch.zeros(batch, 4, time, 1, dtype=torch.bool)
        type_masks[:, 0, :, 0] = True
        intervention = {
            "branches": intervention_branches,
            "joint_logit": torch.zeros(batch, time),
            "type_targets": torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.float32),
            "type_masks": type_masks,
            "union_mask": type_masks[:, 0, :, 0],
        }
        intervention_losses = compute_losses(clean, intervention, config)
        self.assertTrue(torch.isfinite(intervention_losses["target_positive"]))
        self.assertTrue(torch.equal(fallback[:, 0, 1], torch.zeros(batch, dtype=torch.bool)))

    def _loss_inputs(self, config, batch=2, time=4):
        zero = torch.zeros(())
        branches = {name: {"task_loss": zero, "evidence_logit": torch.zeros(batch, time)} for name in ("state", "evolution", "pattern", "relation")}
        clean = {"branches": branches, "joint_logit": torch.zeros(batch, time), "joint_score": torch.zeros(batch, time)}
        int_branches = {name: {"evidence_logit": torch.ones(batch, time) * (0.0 if name == "state" else 1.0)} for name in branches}
        type_masks = torch.zeros(batch, 4, time, 1, dtype=torch.bool)
        type_masks[:, 0, :, 0] = True
        intervention = {
            "branches": int_branches,
            "joint_logit": torch.zeros(batch, time),
            "type_targets": torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)[:batch],
            "type_masks": type_masks,
            "union_mask": type_masks[:, 0, :, 0],
            "scenario_kind": torch.zeros(batch, dtype=torch.long),
        }
        return clean, intervention

    def test_responsibility_is_per_sample_and_noncompetitive(self):
        config = tiny_config(lambda_task=0.0, lambda_evidence=0.0, lambda_responsibility=1.0, lambda_score=0.0, lambda_score_rank=0.0, lambda_clean_score=0.0, lambda_synergy=0.0)
        clean, one_valid = self._loss_inputs(config)
        one = compute_losses(clean, one_valid, config)["responsibility"]
        clean_two, two_valid = self._loss_inputs(config)
        two_valid["type_targets"][1, 0] = 1.0
        two_valid["type_masks"][1, 0, :, 0] = True
        two = compute_losses(clean_two, two_valid, config)["responsibility"]
        self.assertTrue(torch.allclose(one, two, atol=1e-6, rtol=1e-6))

    def test_synergy_is_per_weak_sample(self):
        config = tiny_config(lambda_task=0.0, lambda_evidence=0.0, lambda_responsibility=0.0, lambda_score=0.0, lambda_score_rank=0.0, lambda_clean_score=0.0, lambda_synergy=1.0)
        clean, intervention = self._loss_inputs(config)
        intervention["scenario_kind"][0] = 3
        masks = torch.ones(2, 4, dtype=torch.bool)
        intervention["weak_views"] = {
            "logit_i": torch.ones(2, 4), "logit_j": torch.ones(2, 4), "compound_logit": torch.zeros(2, 4),
            "mask_i": masks, "mask_j": masks, "union_mask": masks,
        }
        first = compute_losses(clean, intervention, config)["synergy"]
        intervention["scenario_kind"][1] = 3
        second = compute_losses(clean, intervention, config)["synergy"]
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))

    def test_evolution_t0_is_invalid_and_sufficient_excludes_it(self):
        config = tiny_config()
        branch = EvolutionNormalityBranch(config)
        output = branch(torch.randn(2, config.seq_len, config.c_in))
        self.assertFalse(bool(output["valid_mask"][:, 0].any()))
        self.assertTrue(torch.equal(output["z"][:, 0], torch.zeros_like(output["z"][:, 0])))
        scorer = RelationAwareJointScorer(config)
        valid = torch.ones(2, config.seq_len, 4, dtype=torch.bool)
        valid[:, 0, 1] = False
        scored = scorer(torch.randn(2, config.seq_len, 4, config.joint_dim), torch.randn(2, config.seq_len, 4), valid)
        self.assertTrue(torch.equal(scored["sufficient_branch_logits"][:, 0, 1], torch.full((2,), torch.finfo(torch.float32).min)))

    def test_invalid_evolution_is_masked_from_all_scorer_paths(self):
        config = tiny_config()
        scorer = RelationAwareJointScorer(config)
        tokens = torch.randn(1, config.seq_len, 4, config.joint_dim)
        logits = torch.randn(1, config.seq_len, 4)
        valid = torch.ones(1, config.seq_len, 4, dtype=torch.bool)
        valid[:, 0, 1] = False
        reference = scorer(tokens, logits, valid)
        changed_tokens = tokens.clone()
        changed_logits = logits.clone()
        changed_tokens[:, 0, 1] += 1000.0
        changed_logits[:, 0, 1] += 1000.0
        changed = scorer(changed_tokens, changed_logits, valid)
        self.assertTrue(torch.allclose(reference["joint_logit"][:, 0], changed["joint_logit"][:, 0], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.all(~reference["relation_token_valid_mask"][:, 0, (4, 7, 8)]))
        self.assertTrue(torch.equal(reference["branch_valid_mask"], valid))
        self.assertEqual(tuple(reference["relation_token_valid_mask"].shape), (1, config.seq_len, 10))
        changed_valid = scorer(tokens, logits, valid)
        changed_valid_tokens = tokens.clone()
        changed_valid_tokens[:, 1, 1] += 100.0
        changed_valid_result = scorer(changed_valid_tokens, logits, valid)
        self.assertFalse(torch.allclose(changed_valid["joint_logit"][:, 1], changed_valid_result["joint_logit"][:, 1]))

    def test_clean_evidence_ignores_invalid_evolution_t0(self):
        config = tiny_config(lambda_task=0.0, lambda_evidence=1.0, lambda_responsibility=0.0, lambda_score=0.0, lambda_score_rank=0.0, lambda_clean_score=0.0, lambda_synergy=0.0)
        time = config.seq_len
        branches = {name: {"task_loss": torch.zeros(()), "evidence_logit": torch.zeros(1, time)} for name in ("state", "evolution", "pattern", "relation")}
        branches["evolution"]["valid_mask"] = torch.ones(1, time, dtype=torch.bool)
        branches["evolution"]["valid_mask"][:, 0] = False
        clean = {"branches": branches, "joint_logit": torch.zeros(1, time), "joint_score": torch.zeros(1, time), "branch_valid_mask": torch.stack((torch.ones(1, time, dtype=torch.bool), branches["evolution"]["valid_mask"], torch.ones(1, time, dtype=torch.bool), torch.ones(1, time, dtype=torch.bool)), dim=2)}
        first = compute_losses(clean, None, config)["clean_evidence"]
        branches["evolution"]["evidence_logit"][:, 0] = 10000.0
        second = compute_losses(clean, None, config)["clean_evidence"]
        self.assertTrue(torch.equal(first, second))

    def test_intervention_evidence_ignores_invalid_evolution_t0(self):
        config = tiny_config(lambda_task=0.0, lambda_evidence=1.0, lambda_responsibility=0.0, lambda_score=0.0, lambda_score_rank=0.0, lambda_clean_score=0.0, lambda_synergy=0.0)
        clean, intervention = self._loss_inputs(config)
        time = clean["joint_logit"].size(1)
        valid_evolution = torch.ones(2, time, dtype=torch.bool)
        valid_evolution[:, 0] = False
        branch_valid = torch.ones(2, time, 4, dtype=torch.bool)
        branch_valid[:, :, 1] = valid_evolution
        intervention["branch_valid_mask"] = branch_valid
        intervention["type_targets"][:] = 0
        intervention["type_targets"][0, 1] = 1
        intervention["type_masks"][:] = False
        intervention["type_masks"][0, 1, :, 0] = True
        intervention["union_mask"][:] = True
        first = compute_losses(clean, intervention, config)["target_positive"]
        intervention["branches"]["evolution"]["evidence_logit"][0, 0] = 10000.0
        second = compute_losses(clean, intervention, config)["target_positive"]
        self.assertTrue(torch.equal(first, second))

    def test_relation_checkpoint_on_off_forward_and_gradient_equivalent(self):
        on = RelationNormalityBranch(tiny_config(use_activation_checkpoint=True))
        off = RelationNormalityBranch(tiny_config(use_activation_checkpoint=False))
        off.load_state_dict(on.state_dict())
        on.train()
        off.train()
        left = torch.randn(2, 16, 4, requires_grad=True)
        right = left.detach().clone().requires_grad_(True)
        left_out = on(left)
        right_out = off(right)
        self.assertTrue(torch.allclose(left_out["prediction"], right_out["prediction"], atol=1e-6, rtol=1e-6))
        left_out["task_loss"].backward()
        right_out["task_loss"].backward()
        self.assertTrue(torch.allclose(on.value_projection.weight.grad, off.value_projection.weight.grad, atol=1e-5, rtol=1e-5))

    def test_temporal_mixer_responds_to_adjacent_token(self):
        config = tiny_config()
        scorer = RelationAwareJointScorer(config)
        tokens = torch.randn(1, config.seq_len, 4, config.joint_dim)
        logits = torch.randn(1, config.seq_len, 4)
        original = scorer(tokens, logits)["joint_logit"]
        changed = tokens.clone()
        changed[:, 1] += 50.0
        updated = scorer(changed, logits)["joint_logit"]
        self.assertFalse(torch.allclose(original[:, 0], updated[:, 0]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_relation_scorer_math_fallback_is_finite(self):
        config = tiny_config()
        scorer = RelationAwareJointScorer(config).cuda().train()
        tokens = torch.randn(2, config.seq_len, 4, config.joint_dim, device="cuda", requires_grad=True)
        logits = torch.randn(2, config.seq_len, 4, device="cuda", requires_grad=True)
        valid = torch.ones(2, config.seq_len, 4, dtype=torch.bool, device="cuda")
        output = scorer(tokens, logits, valid)
        loss = output["joint_score"].mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in scorer.parameters() if p.requires_grad))


class AdapterAndScriptTests(unittest.TestCase):
    def test_small_dataframe_fit_and_score(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(rng.normal(size=(40, 2)).astype(np.float32))
        adapter = TypeFusionCATCHV2(
            seq_len=8, patch_size=4, patch_stride=2, batch_size=64, num_epochs=1, patience=1,
            c_in=2, d_model=8, cf_dim=4, d_ff=8, e_layers=1, n_heads=2, branch_dim=8,
            temporal_layers=1, joint_dim=8, joint_layers=1, joint_heads=2,
            relation_mask_groups=2, state_memory_size=8, state_topk=2, dropout=0.0,
            use_activation_checkpoint=False,
        )
        adapter.detect_fit(frame)
        scores, points = adapter.detect_score(frame.iloc[-16:])
        self.assertGreater(len(scores), 0)
        self.assertEqual(scores.shape, points.shape)
        self.assertTrue(np.isfinite(scores).all())

    def test_batch_placeholder_and_gecco_seq_len(self):
        root = Path(__file__).resolve().parents[4]
        scripts = sorted((root / "scripts/multivariate_detection/detect_score").glob("*/TypeFusionCATCHV2.sh"))
        self.assertEqual(len(scripts), 23)
        for script in scripts:
            text = script.read_text()
            self.assertIn('"batch_size":__BATCH_SIZE__', text)
            self.assertIn('MODEL_HYPER_PARAMS="${MODEL_HYPER_PARAMS/__BATCH_SIZE__/${BATCH_SIZE}}"', text)
            match = re.search(r"MODEL_HYPER_PARAMS='([^']+)'", text)
            self.assertIsNotNone(match)
            config = json.loads(match.group(1).replace("__BATCH_SIZE__", "7"))
            self.assertEqual(config["batch_size"], 7)
        gecco = (root / "scripts/multivariate_detection/detect_score/GECCO_script/TypeFusionCATCHV2.sh").read_text()
        self.assertIn('"seq_len":192', gecco)
