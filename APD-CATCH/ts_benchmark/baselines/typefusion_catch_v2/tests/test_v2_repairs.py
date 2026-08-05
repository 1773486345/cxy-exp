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
from ..losses import compute_losses


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
        scored = scorer(torch.randn(2, config.seq_len, 4, config.joint_dim), torch.randn(2, config.seq_len, 4))
        self.assertTrue(torch.equal(scored["sufficient_branch_logits"][:, 0, 1], torch.full((2,), torch.finfo(torch.float32).min)))

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
