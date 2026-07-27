"""Import and forward-contract tests."""

import importlib
import unittest

import torch

from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel
from ts_benchmark.baselines.typefusion_catch.tests.common import tiny_config


class TypeFusionForwardTests(unittest.TestCase):
    def test_original_catch_and_typefusion_import(self) -> None:
        original = importlib.import_module("ts_benchmark.baselines.catch")
        typefusion = importlib.import_module("ts_benchmark.baselines.typefusion_catch")
        self.assertTrue(hasattr(original, "CATCH"))
        self.assertTrue(hasattr(typefusion, "TypeFusionCATCH"))

    def test_forward_shapes_and_diagnostics(self) -> None:
        torch.manual_seed(7)
        config = tiny_config()
        model = TypeFusionCATCHModel(config).train()
        x = torch.randn(2, config.seq_len, config.c_in, dtype=torch.float32)
        output = model(x)
        self.assertEqual(output["x_hat_joint"].shape, x.shape)
        self.assertEqual(output["total_score"].shape, x.shape)
        self.assertTrue(torch.isfinite(output["total_score"]).all())
        self.assertEqual(output["q"].shape, (2, config.num_patches, 4, config.branch_dim))
        self.assertEqual(output["q_normal"].shape, (2, config.num_patches, 4, config.branch_dim))
        self.assertEqual(output["branch_conflict_map"].shape, (2, config.num_patches, 4))
        for branch_name in TypeFusionCATCHModel.branch_names:
            branch = output["branches"][branch_name]
            self.assertEqual(branch["z"].shape, (2, config.num_patches, config.d_model))
            self.assertEqual(branch["x_hat"].shape, x.shape)
            self.assertEqual(branch["e"].shape, x.shape)
        branches = [
            model.state_branch,
            model.evolution_branch,
            model.pattern_branch,
            model.relation_branch,
        ]
        self.assertEqual(len({id(branch) for branch in branches}), 4)
        parameter_sets = [{id(parameter) for parameter in branch.parameters()} for branch in branches]
        for left in range(len(parameter_sets)):
            for right in range(left + 1, len(parameter_sets)):
                self.assertFalse(parameter_sets[left] & parameter_sets[right])

    def test_train_and_eval_modes_run(self) -> None:
        config = tiny_config()
        model = TypeFusionCATCHModel(config)
        x = torch.randn(2, config.seq_len, config.c_in)
        model.train()
        train_output = model(x)
        self.assertTrue(torch.isfinite(train_output["losses"]["total"]))
        model.eval()
        with torch.no_grad():
            eval_output = model(x)
        self.assertTrue(torch.isfinite(eval_output["losses"]["total"]))

    def test_fusion_stage_keeps_frozen_branches_in_eval_mode(self) -> None:
        model = TypeFusionCATCHModel(tiny_config("fusion_train")).train()
        self.assertFalse(model.shared_stem.training)
        self.assertFalse(model.state_branch.training)
        self.assertFalse(model.evolution_branch.training)
        self.assertTrue(model.branch_fusion.training)
        self.assertTrue(model.joint_decoder.training)


if __name__ == "__main__":
    unittest.main()
