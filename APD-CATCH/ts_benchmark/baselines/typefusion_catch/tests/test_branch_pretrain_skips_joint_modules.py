"""Stage 1 must not execute frozen joint-path modules."""

import copy
import unittest

import torch
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.typefusion_catch.TypeFusionCATCH import TypeFusionCATCH
from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel
from ts_benchmark.baselines.typefusion_catch.tests.common import (
    small_normal_frame,
    tiny_config,
    tiny_fit_kwargs,
)


class BranchPretrainJointSkipTests(unittest.TestCase):
    def test_stage_one_skips_joint_modules_and_other_stages_use_them(self) -> None:
        torch.manual_seed(61)
        config = tiny_config("branch_pretrain")
        model = TypeFusionCATCHModel(config)
        x = torch.randn(2, config.seq_len, config.c_in)
        calls = {"adapter": 0, "fusion": 0, "decoder": 0}
        hooks = []
        for adapter in model.evidence_adapters.values():
            hooks.append(adapter.register_forward_hook(lambda *_: calls.__setitem__("adapter", calls["adapter"] + 1)))
        hooks.append(model.branch_fusion.register_forward_hook(lambda *_: calls.__setitem__("fusion", calls["fusion"] + 1)))
        hooks.append(model.joint_decoder.register_forward_hook(lambda *_: calls.__setitem__("decoder", calls["decoder"] + 1)))
        try:
            model.eval()
            fast = model(x)
            self.assertIsNone(fast["x_hat_joint"])
            self.assertIsNone(fast["total_score"])
            self.assertEqual(calls, {"adapter": 0, "fusion": 0, "decoder": 0})

            full = model(x, compute_joint=True)
            self.assertEqual(calls["adapter"], 4)
            self.assertEqual(calls["fusion"], 1)
            self.assertEqual(calls["decoder"], 1)
            for loss_name in ("state", "evolution", "pattern_time", "relation", "pattern_freq"):
                torch.testing.assert_close(fast["losses"][loss_name], full["losses"][loss_name], rtol=0.0, atol=0.0)

            model.train()
            model.zero_grad(set_to_none=True)
            calls.update(adapter=0, fusion=0, decoder=0)
            fast_train = model(x)
            fast_train["losses"]["total"].backward()
            self.assertEqual(calls, {"adapter": 0, "fusion": 0, "decoder": 0})
            self.assertTrue(all(parameter.grad is None for parameter in model.branch_fusion.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in model.joint_decoder.parameters()))
            required = [
                model.shared_stem.channel_fusion.output_projection.weight,
                model.state_branch.patch_decoder[-1].weight,
                model.evolution_branch.prediction_head.weight,
                model.pattern_branch.channel_decoder[-1].weight,
                model.relation_branch.output_head[-1].weight,
            ]
            for parameter in required:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

            for stage in ("fusion_train", "joint_finetune"):
                model.set_training_stage(stage)
                model.train()
                calls.update(adapter=0, fusion=0, decoder=0)
                model(x)
                self.assertEqual(calls["adapter"], 4)
                self.assertGreaterEqual(calls["fusion"], 1)
                self.assertEqual(calls["decoder"], 1)
        finally:
            for hook in hooks:
                hook.remove()

    def test_detect_score_always_uses_the_joint_path(self) -> None:
        frame = small_normal_frame()
        adapter = TypeFusionCATCH(**tiny_fit_kwargs())
        adapter.detect_hyper_param_tune(frame)
        adapter.model = TypeFusionCATCHModel(adapter.config)
        adapter.model.set_training_stage("joint_finetune")
        adapter.best_state = copy.deepcopy(adapter.model.state_dict())
        adapter.scaler = StandardScaler().fit(frame.values)
        calls = {"adapter": 0, "fusion": 0, "decoder": 0}
        hooks = []
        for evidence_adapter in adapter.model.evidence_adapters.values():
            hooks.append(evidence_adapter.register_forward_hook(lambda *_: calls.__setitem__("adapter", calls["adapter"] + 1)))
        hooks.append(adapter.model.branch_fusion.register_forward_hook(lambda *_: calls.__setitem__("fusion", calls["fusion"] + 1)))
        hooks.append(adapter.model.joint_decoder.register_forward_hook(lambda *_: calls.__setitem__("decoder", calls["decoder"] + 1)))
        try:
            adapter.detect_score(frame)
            self.assertGreater(calls["adapter"], 0)
            self.assertGreater(calls["fusion"], 0)
            self.assertGreater(calls["decoder"], 0)
        finally:
            for hook in hooks:
                hook.remove()


if __name__ == "__main__":
    unittest.main()
