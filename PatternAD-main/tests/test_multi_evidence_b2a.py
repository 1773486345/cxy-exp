import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.multi_evidence.generate_b2a_holdout import (
    DEFAULT_CONFIG,
    DRIFT_CONTROL_ROLE,
    PAIR_ROLE_ORDER,
    _load_json,
    generate_suite,
)
from scripts.multi_evidence.multi_target_calibration import (
    MultiTargetEvidenceReliabilityCalibration,
)
from scripts.multi_evidence.run_b2a_transfer import run_experiment
from ts_benchmark.baselines.MultiEvidenceRepair.MultiEvidenceRepair import (
    terminal_windows,
)
from ts_benchmark.baselines.MultiEvidenceRepair.MultiTargetEvidenceRepair import (
    MultiTargetEvidenceRepair,
)


def _small_config():
    config = copy.deepcopy(_load_json(DEFAULT_CONFIG))
    config.update(
        {
            "burn_in": 32,
            "history_length": 8,
        }
    )
    config["normal_process"]["relation_period"] = 64
    config["episodes"].update(
        {"pairs_per_phase": 1, "drift_controls_per_phase": 1}
    )
    config["split"].update(
        {
            "outer_calibration_fraction": 0.2,
            "validation_fraction": 0.12,
            "reference_fraction": 0.2,
        }
    )
    config["model"].update(
        {"d_model": 6, "batch_size": 32, "epochs": 1, "patience": 1}
    )
    config["evaluation"].update(
        {
            "paired_order_min": 1,
            "target_spike_exceedance_min": 1,
            "coherent_exceedance_max": 4,
            "drift_control_exceedance_max": 4,
        }
    )
    return config


class MultiEvidenceB2aTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = _small_config()
        cls.suite = generate_suite(cls.config)

    def test_generator_is_deterministic_and_rotates_all_targets(self):
        repeated = generate_suite(self.config)
        np.testing.assert_array_equal(
            self.suite["train"]["values"], repeated["train"]["values"]
        )
        self.assertEqual(self.suite["target_indices"], tuple(range(6)))
        pairs = {}
        drift_controls = []
        for episode in self.suite["episodes"]:
            if episode["is_pair"]:
                pairs.setdefault((episode["target_index"], episode["pair_id"]), {})[
                    episode["role"]
                ] = episode
            else:
                drift_controls.append(episode)
        self.assertEqual(len(pairs), 6 * 4)
        self.assertEqual(len(drift_controls), 6 * 4)
        self.assertTrue(
            all(control["role"] == DRIFT_CONTROL_ROLE for control in drift_controls)
        )
        for (target, _), roles in pairs.items():
            self.assertEqual(set(roles), set(PAIR_ROLE_ORDER))
            coherent = roles["coherent_control"]["values"]
            unsupported = roles["unsupported_target_break"]["values"]
            omission = roles["target_omission_break"]["values"]
            drivers = [index for index in range(6) if index != target]
            np.testing.assert_array_equal(coherent[:, target], unsupported[:, target])
            np.testing.assert_array_equal(coherent[:, drivers], omission[:, drivers])

    def test_multitarget_models_are_disjoint_and_return_ordered_matrices(self):
        values = self.suite["train"]["values"][:160]
        windows = terminal_windows(values, 8)
        model = MultiTargetEvidenceRepair(
            dimensions=6,
            target_indices=(0, 2, 5),
            d_model=4,
            epochs=1,
            patience=1,
            batch_size=16,
            device="cpu",
        ).fit(windows[:80], windows[80:110], windows[110:], seed=31)
        scores = model.score_windows(windows[:5], include_tails=False)
        self.assertEqual(scores["target"].shape, (5, 3))
        np.testing.assert_allclose(scores["target"][:, 0], windows[:5, -1, 0])
        np.testing.assert_allclose(scores["target"][:, 1], windows[:5, -1, 2])
        np.testing.assert_allclose(scores["target"][:, 2], windows[:5, -1, 5])
        report = model.parameter_isolation_report()
        self.assertTrue(report["all_branch_parameter_sets_disjoint"])
        self.assertEqual(len(report["branch_parameter_counts"]), 6)

    def test_target_calibration_is_independent_without_flattening(self):
        rng = np.random.default_rng(53)
        windows = rng.normal(size=(240, 9, 6)).astype(np.float32)
        raw = {
            component: rng.uniform(0.0, 1.0, size=(240, 2))
            for component in ("temporal_residual", "cross_residual", "disagreement")
        }
        first = MultiTargetEvidenceReliabilityCalibration(
            dimensions=6,
            target_indices=(0, 4),
            target_fpr=0.05,
            reliability_strata=3,
            min_reference_per_stratum=8,
        ).fit(windows[:80], windows[80:160], {k: v[80:160] for k, v in raw.items()}, windows[160:], {k: v[160:] for k, v in raw.items()})
        changed = {name: values.copy() for name, values in raw.items()}
        for values in changed.values():
            values[80:160, 0] += 100.0
            values[160:, 0] += 100.0
        second = MultiTargetEvidenceReliabilityCalibration(
            dimensions=6,
            target_indices=(0, 4),
            target_fpr=0.05,
            reliability_strata=3,
            min_reference_per_stratum=8,
        ).fit(windows[:80], windows[80:160], {k: v[80:160] for k, v in changed.items()}, windows[160:], {k: v[160:] for k, v in changed.items()})
        first_target_four = first.metadata()["calibration_by_target"]["4"]
        second_target_four = second.metadata()["calibration_by_target"]["4"]
        self.assertEqual(first_target_four, second_target_four)
        transformed = first.transform(windows[160:], {k: v[160:] for k, v in raw.items()})
        self.assertEqual(transformed["cross_residual_tail"].shape, (80, 2))
        self.assertTrue(first.metadata()["no_target_flattening"])

    def test_runner_writes_targetwise_artifacts_without_a_scalar_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "b2a"
            result = run_experiment(
                self.config, output_dir, device="cpu", seed=int(self.config["seed"])
            )
            self.assertIn(result["status"], {"passed", "failed_gates"})
            evaluation = json.loads(
                (output_dir / "b2a_evaluation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(evaluation["no_score_fusion"])
            self.assertTrue(evaluation["no_cross_target_aggregation"])
            self.assertFalse(evaluation["provenance"]["test_scores_used_for_thresholds"])
            self.assertFalse(evaluation["provenance"]["test_labels_used"])
            self.assertTrue((output_dir / "background_scores.npz").is_file())
            self.assertTrue((output_dir / "multi_target_model_state.pt").is_file())
            self.assertTrue((output_dir / "episode_scores.csv").is_file())


if __name__ == "__main__":
    unittest.main()
