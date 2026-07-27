"""Formal seed and small CPU three-stage reproducibility checks."""

from __future__ import annotations

from pathlib import Path
from unittest import mock
import unittest

import numpy as np
import pandas as pd
import torch

from ts_benchmark.baselines.typefusion_catch.TypeFusionCATCH import TypeFusionCATCH
from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.tests.common import tiny_config
from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel


_STAGES = ("branch_pretrain", "fusion_train", "joint_finetune")


def _fixed_frame() -> pd.DataFrame:
    """Use an independent fixed data seed, separate from the model seed."""

    generator = np.random.RandomState(99173)
    values = generator.normal(loc=0.0, scale=1.0, size=(32, 3)).astype(np.float32)
    return pd.DataFrame(values, columns=["c0", "c1", "c2"])


def _formal_fit_kwargs(seed: int) -> dict:
    # Keep TypeFusionConfig's formal dropout default instead of overriding it.
    return {
        "seq_len": 4,
        "patch_size": 2,
        "patch_stride": 1,
        "d_model": 8,
        "cf_dim": 8,
        "d_ff": 16,
        "n_heads": 2,
        "head_dim": 4,
        "e_layers": 1,
        "temporal_hidden_dim": 8,
        "temporal_layers": 1,
        "memory_size": 2,
        "memory_topk": 1,
        "branch_dim": 8,
        "fusion_layers": 1,
        "fusion_heads": 2,
        "relation_mask_groups": 2,
        "pattern_mask_ratio": 0.25,
        "batch_size": 32,
        "patience": 10,
        "fit_mode": "three_stage",
        "training_budget_mode": "equal_total_steps",
        "catch_train_epochs": 3,
        "seed": seed,
    }


class FormalSeedConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        torch.set_num_threads(self._original_num_threads)

    @staticmethod
    def _cpu_adapter(seed: int) -> TypeFusionCATCH:
        adapter = TypeFusionCATCH(**_formal_fit_kwargs(seed))
        adapter.device = torch.device("cpu")
        return adapter

    @staticmethod
    def _initialised_model(seed: int) -> TypeFusionCATCHModel:
        config = tiny_config()
        config.seed = seed
        TypeFusionCATCH._seed_everything(seed)
        return TypeFusionCATCHModel(config)

    def test_default_config_and_formal_material_use_2021(self) -> None:
        self.assertEqual(TypeFusionConfig().seed, 2021)
        repository = Path(__file__).resolve().parents[4]
        formal_files = (
            repository / "ts_benchmark/baselines/typefusion_catch/config.py",
            repository / "ts_benchmark/baselines/typefusion_catch/TypeFusionCATCH.py",
            repository / "ts_benchmark/baselines/typefusion_catch/smoke.py",
            repository / "TYPEFUSION_CATCH_MODEL.md",
            repository / "TYPEFUSION_CATCH_SMOKE_COMMANDS.md",
            repository / "TYPEFUSION_CATCH_EXPERIMENT_COMMANDS.md",
        )
        for path in formal_files:
            self.assertNotRegex(path.read_text(encoding="utf-8"), r"(?i)seed\s*[:=]\s*42\b")

    def test_same_seed_initialization_is_bitwise_equal(self) -> None:
        model_a = self._initialised_model(2021)
        model_b = self._initialised_model(2021)
        state_a = model_a.state_dict()
        state_b = model_b.state_dict()
        self.assertEqual(tuple(state_a), tuple(state_b))
        for name in state_a:
            self.assertTrue(torch.equal(state_a[name], state_b[name]), name)

    def test_different_seed_initialization_changes_a_trainable_tensor(self) -> None:
        model_a = self._initialised_model(2021)
        model_b = self._initialised_model(2022)
        parameters_a = dict(model_a.named_parameters())
        parameters_b = dict(model_b.named_parameters())
        self.assertTrue(
            any(not torch.equal(parameters_a[name], parameters_b[name]) for name in parameters_a)
        )

    def test_same_seed_reproduces_complete_cpu_three_stage_fit(self) -> None:
        frame = _fixed_frame()
        first = self._cpu_adapter(2021)
        second = self._cpu_adapter(2021)
        first.detect_fit(frame)
        second.detect_fit(frame)

        for name in first.best_state:
            self.assertTrue(torch.equal(first.best_state[name], second.best_state[name]), name)
        first_score, _ = first.detect_score(frame)
        second_score, _ = second.detect_score(frame)
        np.testing.assert_allclose(first_score, second_score, rtol=0.0, atol=0.0)
        self.assertEqual(first.stage_optimizer_steps, second.stage_optimizer_steps)
        self.assertEqual(first.stage_validation_losses, second.stage_validation_losses)
        for stage in _STAGES:
            for name in first.stage_best_states[stage]:
                self.assertTrue(
                    torch.equal(
                        first.stage_best_states[stage][name], second.stage_best_states[stage][name]
                    ),
                    f"{stage}: {name}",
                )
        np.testing.assert_allclose(first.scaler.mean_, second.scaler.mean_, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(first.scaler.scale_, second.scaler.scale_, rtol=0.0, atol=0.0)

    def test_different_seed_changes_complete_cpu_three_stage_fit(self) -> None:
        frame = _fixed_frame()
        first = self._cpu_adapter(2021)
        second = self._cpu_adapter(2022)
        first.detect_fit(frame)
        second.detect_fit(frame)
        self.assertTrue(
            any(not torch.equal(first.best_state[name], second.best_state[name]) for name in first.best_state)
        )

    def test_explicit_seed_override_is_used_by_detect_fit(self) -> None:
        adapter = self._cpu_adapter(2022)
        used_seeds = []
        original = TypeFusionCATCH._seed_everything

        def record_and_seed(seed: int) -> None:
            used_seeds.append(seed)
            original(seed)

        with mock.patch.object(TypeFusionCATCH, "_seed_everything", side_effect=record_and_seed):
            adapter.detect_fit(_fixed_frame())
        self.assertEqual(adapter.config.seed, 2022)
        self.assertEqual(used_seeds, [2022])


if __name__ == "__main__":
    unittest.main()
