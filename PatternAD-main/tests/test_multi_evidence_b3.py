import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.multi_evidence.generate_b2a_gc import _load_json, generate_suite
from scripts.multi_evidence.run_b3_relation_conditioned import (
    DEFAULT_CONFIG as B3_DEFAULT_CONFIG,
    run_experiment,
)
from ts_benchmark.baselines.MultiEvidenceRepair.MultiEvidenceRepair import terminal_windows
from ts_benchmark.baselines.MultiEvidenceRepair.MultiEvidenceRepair import (
    MultiEvidenceRepair,
)
from ts_benchmark.baselines.MultiEvidenceRepair.MultiTargetRelationConditionedEvidenceRepair import (
    MultiTargetRelationConditionedEvidenceRepair,
)


def _small_config():
    config = copy.deepcopy(_load_json(B3_DEFAULT_CONFIG))
    config["model"].update(
        {
            "batch_size": 32,
            "epochs": 1,
            "patience": 1,
        }
    )
    return config


class MultiEvidenceB3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = _small_config()
        cls.suite = generate_suite(cls.config)

    def test_relation_cross_is_terminal_blind_but_history_conditioned(self):
        windows = terminal_windows(self.suite["train"]["values"][:160], 8)
        model = MultiTargetRelationConditionedEvidenceRepair(
            dimensions=6,
            target_indices=(0, 3),
            temporal_d_model=8,
            cross_d_model=6,
            cross_head_d_model=5,
            epochs=1,
            patience=1,
            batch_size=16,
            device="cpu",
        ).fit(windows[:80], windows[80:110], windows[110:], seed=41)
        report = model.evidence_isolation_report(windows[:1])
        self.assertTrue(report["all_branch_parameter_sets_disjoint"])
        self.assertEqual(len(report["branch_parameter_counts"]), 4)
        for target_report in report["per_target"].values():
            self.assertEqual(target_report["temporal_driver_delta"], 0.0)
            self.assertEqual(target_report["temporal_terminal_target_delta"], 0.0)
            self.assertEqual(target_report["cross_terminal_target_delta"], 0.0)
            self.assertEqual(
                target_report["relation_history_terminal_target_input_delta"], 0.0
            )
            self.assertGreater(
                target_report["relation_history_target_history_input_delta"], 0.0
            )

    def test_cross_capacity_is_below_b2a_gc_control(self):
        windows = terminal_windows(self.suite["train"]["values"][:160], 8)
        model = MultiTargetRelationConditionedEvidenceRepair(
            dimensions=6,
            target_indices=(0,),
            temporal_d_model=32,
            cross_d_model=22,
            cross_head_d_model=20,
            epochs=1,
            patience=1,
            batch_size=16,
            device="cpu",
        ).fit(windows[:80], windows[80:110], windows[110:], seed=43)
        count = model.fit_metadata_["targets"]["0"]["fit"]["parameter_counts"]["cross"]
        self.assertEqual(count, 4815)
        self.assertLessEqual(count, 4833)

    def test_temporal_initialization_matches_b2a_gc_same_seed(self):
        seed = 4_001
        baseline = MultiEvidenceRepair(
            dimensions=6,
            target_index=2,
            d_model=32,
            epochs=1,
            patience=1,
            batch_size=16,
            device="cpu",
        )
        torch.manual_seed(seed)
        baseline.net = baseline.net.__class__(6, 2, 32)
        torch.manual_seed(seed)
        relation = MultiTargetRelationConditionedEvidenceRepair(
            dimensions=6,
            target_indices=(2,),
            temporal_d_model=32,
            cross_d_model=22,
            cross_head_d_model=20,
            epochs=1,
            patience=1,
            batch_size=16,
            device="cpu",
        )._build_model(2, seed)
        baseline_state = baseline.net.temporal_gru.state_dict()
        relation_state = relation.net.temporal_gru.state_dict()
        self.assertEqual(set(baseline_state), set(relation_state))
        for name in baseline_state:
            torch.testing.assert_close(baseline_state[name], relation_state[name])
        baseline_head = baseline.net.temporal_head.state_dict()
        relation_head = relation.net.temporal_head.state_dict()
        for name in baseline_head:
            torch.testing.assert_close(baseline_head[name], relation_head[name])

    def test_frozen_temporal_checkpoint_remains_identical_after_cross_fit(self):
        windows = terminal_windows(self.suite["train"]["values"][:180], 8)
        baseline = MultiEvidenceRepair(
            dimensions=6,
            target_index=2,
            d_model=8,
            epochs=1,
            patience=1,
            batch_size=16,
            device="cpu",
        )
        torch.manual_seed(4_001)
        baseline.net = baseline.net.__class__(6, 2, 8)
        source_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in baseline.net.state_dict().items()
        }
        model = MultiTargetRelationConditionedEvidenceRepair(
            dimensions=6,
            target_indices=(2,),
            temporal_d_model=8,
            cross_d_model=6,
            cross_head_d_model=5,
            epochs=2,
            patience=2,
            batch_size=16,
            device="cpu",
        ).fit(
            windows[:90],
            windows[90:130],
            windows[130:],
            seed=41,
            frozen_temporal_states={"2": source_state},
        )
        fitted = model.models[2].net.state_dict()
        for name, tensor in source_state.items():
            if name.startswith(("temporal_gru.", "temporal_head.")):
                torch.testing.assert_close(fitted[name], tensor)
        fit = model.fit_metadata_["targets"]["2"]["fit"]
        self.assertTrue(fit["temporal_frozen"])
        self.assertEqual(
            fit["selection_metric"], "cross_validation_loss_with_frozen_temporal"
        )

    def test_b3_runner_writes_frozen_temporal_provenance_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control_dir = root / "control"
            shutil.copytree(
                Path(__file__).resolve().parents[1]
                / "result/multi_evidence/b3a_baseline_seed4401_gpu",
                control_dir,
            )
            output_dir = Path(temporary) / "b3"
            result = run_experiment(
                self.config,
                output_dir,
                device="cpu",
                seed=int(self.config["seed"]),
                frozen_control_dir=control_dir,
            )
            self.assertIn(result["status"], {"passed", "failed_gates"})
            evaluation = json.loads(
                (output_dir / "b3a_evaluation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                evaluation["phase"], "B3a-RelationConditioned-FrozenTemporal"
            )
            self.assertTrue(evaluation["no_score_fusion"])
            self.assertTrue(evaluation["no_cross_target_aggregation"])
            self.assertTrue(
                evaluation["gates"]["counterfactual_input_ties"][
                    "coherent_target_spike_cross_prediction_max_abs_difference"
                ]
                <= 1e-7
            )
            self.assertIn("cross_branch_capacity_control", evaluation["gates"])
            frozen = evaluation["gates"]["frozen_temporal_checkpoint_and_replay"]
            self.assertTrue(frozen["pass"])
            self.assertEqual(frozen["replay_mode"], "cross_device_diagnostic")
            self.assertEqual(
                frozen["frozen_temporal_state_sha256"],
                frozen["final_temporal_state_sha256"],
            )
            self.assertTrue((output_dir / "background_scores.npz").is_file())
            self.assertTrue((output_dir / "multi_target_model_state.pt").is_file())
            self.assertTrue((output_dir / "episode_scores.csv").is_file())

    def test_b3_runner_rejects_changed_frozen_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control_dir = root / "control"
            shutil.copytree(
                Path(__file__).resolve().parents[1]
                / "result/multi_evidence/b3a_baseline_seed4401_gpu",
                control_dir,
            )
            with (control_dir / "synthetic_suite/episodes.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "episodes_sha256 mismatch"):
                run_experiment(
                    self.config,
                    root / "b3",
                    device="cpu",
                    seed=int(self.config["seed"]),
                    frozen_control_dir=control_dir,
                )


if __name__ == "__main__":
    unittest.main()
