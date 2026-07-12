import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.multi_evidence.generate_b0_synthetic import (
    DEFAULT_CONFIG,
    ROLE_ORDER,
    _load_json,
    generate_suite,
)
from scripts.multi_evidence.run_b0 import run_experiment, split_normal_train
from scripts.multi_evidence.reliability_calibration import (
    EvidenceReliabilityCalibration,
)
from ts_benchmark.baselines.MultiEvidenceRepair.MultiEvidenceRepair import (
    EmpiricalUpperTail,
    EvidenceRepairNet,
    MultiEvidenceRepair,
    terminal_windows,
)


def _small_config():
    config = copy.deepcopy(_load_json(DEFAULT_CONFIG))
    config.update(
        {
            "burn_in": 24,
            "train_length": 360,
            "test_length": 240,
            "history_length": 8,
        }
    )
    config["normal_process"].update(
        {"regime_segment_length": 30, "regime_noise_scales": [0.8, 1.2]}
    )
    config["episodes"].update({"episodes_per_regime": 2, "shock_magnitude": 0.8})
    config["split"].update(
        {
            "outer_calibration_fraction": 0.2,
            "validation_fraction": 0.12,
            "reference_fraction": 0.12,
        }
    )
    config["model"].update(
        {"d_model": 8, "batch_size": 32, "epochs": 2, "patience": 2}
    )
    config["evaluation"].update(
        {
            "paired_order_min": 1,
            "coherent_exceedance_max": 4,
            "target_spike_exceedance_min": 1,
        }
    )
    return config


class MultiEvidenceB0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = _small_config()
        cls.suite = generate_suite(cls.config)

    def test_generator_is_deterministic_and_counterfactual_contracts_hold(self):
        repeated = generate_suite(self.config)
        np.testing.assert_array_equal(
            self.suite["train_values"], repeated["train_values"]
        )
        np.testing.assert_array_equal(
            self.suite["background_values"], repeated["background_values"]
        )
        target = int(self.config["episodes"]["target_channel"])
        by_pair = {}
        for episode in self.suite["episodes"]:
            by_pair.setdefault(episode["pair_id"], {})[episode["role"]] = episode
        self.assertEqual(len(by_pair), 4)
        for pair in by_pair.values():
            self.assertEqual(set(pair), set(ROLE_ORDER))
            coherent = pair["coherent_control"]["values"]
            unsupported = pair["unsupported_target_break"]["values"]
            omission = pair["target_omission_break"]["values"]
            np.testing.assert_array_equal(coherent[:, target], unsupported[:, target])
            drivers = [index for index in range(coherent.shape[-1]) if index != target]
            np.testing.assert_array_equal(
                coherent[:, drivers], omission[:, drivers]
            )
            self.assertGreater(
                np.max(np.abs(coherent[:, drivers] - unsupported[:, drivers])), 0.0
            )
            self.assertGreater(
                np.max(np.abs(coherent[:, target] - omission[:, target])), 0.0
            )

    def test_network_information_paths_are_strictly_isolated(self):
        torch.manual_seed(7)
        net = EvidenceRepairNet(dimensions=4, target_index=0, d_model=6)
        net.eval()
        windows = torch.randn(3, 9, 4)
        drivers_changed = windows.clone()
        drivers_changed[:, :, 1:] += 9.0
        terminal_changed = windows.clone()
        terminal_changed[:, -1, 0] += 9.0
        target_changed = windows.clone()
        target_changed[:, :, 0] += 9.0
        with torch.no_grad():
            base = net(windows)
            with_drivers = net(drivers_changed)
            with_terminal = net(terminal_changed)
            with_target = net(target_changed)
        torch.testing.assert_close(base["mu_temporal"], with_drivers["mu_temporal"])
        torch.testing.assert_close(base["mu_temporal"], with_terminal["mu_temporal"])
        torch.testing.assert_close(base["mu_cross"], with_target["mu_cross"])
        parameter_sets = net.branch_parameter_ids()
        self.assertFalse(parameter_sets["temporal"] & parameter_sets["cross"])

    def test_tail_is_monotone_and_model_exposes_individual_components(self):
        tail = EmpiricalUpperTail().fit(np.asarray([0.0, 0.1, 0.3, 0.7]))
        transformed = tail.transform(np.asarray([0.0, 0.2, 0.8]))
        self.assertLess(transformed[0], transformed[1])
        self.assertLess(transformed[1], transformed[2])
        windows = terminal_windows(self.suite["train_values"][:120], 8)
        model = MultiEvidenceRepair(
            dimensions=4,
            target_index=0,
            d_model=6,
            epochs=2,
            patience=2,
            batch_size=16,
            device="cpu",
        ).fit(windows[:60], windows[60:80], windows[80:], seed=11)
        scores = model.score_windows(windows[:5])
        self.assertEqual(
            set(MultiEvidenceRepair.SCORE_COMPONENTS),
            {"temporal_residual", "cross_residual", "disagreement"},
        )
        for name in (
            "temporal_residual",
            "cross_residual",
            "disagreement",
            "temporal_residual_tail",
            "cross_residual_tail",
            "disagreement_tail",
        ):
            self.assertEqual(scores[name].shape, (5,))
            self.assertTrue(np.isfinite(scores[name]).all())

    def test_split_has_explicit_history_gaps(self):
        segments = split_normal_train(360, 8, self.config["split"])
        self.assertEqual(segments["optimization"]["end"] + 8, segments["validation"]["start"])
        self.assertEqual(segments["validation"]["end"] + 8, segments["reference"]["start"])
        self.assertEqual(segments["reference"]["end"] + 8, segments["outer_calibration"]["start"])

    def test_reliability_routing_uses_only_allowed_evidence_inputs(self):
        rng = np.random.default_rng(19)
        windows = rng.normal(size=(180, 9, 4)).astype(np.float32)
        raw_scores = {
            "temporal_residual": rng.uniform(0.0, 1.0, size=180),
            "cross_residual": rng.uniform(0.0, 1.0, size=180),
            "disagreement": rng.uniform(0.0, 1.0, size=180),
        }
        calibration = EvidenceReliabilityCalibration(
            dimensions=4,
            target_index=0,
            target_fpr=0.05,
            mode="input_energy_stratified",
            reliability_strata=3,
            min_reference_per_stratum=8,
        ).fit(
            windows[:80], windows[80:140],
            {name: values[80:140] for name, values in raw_scores.items()},
            windows[140:], {name: values[140:] for name, values in raw_scores.items()},
        )
        base = windows[:1].copy()
        drivers_changed = base.copy()
        drivers_changed[:, :, 1:] += np.linspace(0.0, 3.0, 9)[None, :, None]
        terminal_changed = base.copy()
        terminal_changed[:, -1, 0] += 3.0
        target_changed = base.copy()
        target_changed[:, :, 0] += np.linspace(0.0, 3.0, 9)[None, :]
        base_features = calibration.features(base)
        driver_features = calibration.features(drivers_changed)
        terminal_features = calibration.features(terminal_changed)
        target_features = calibration.features(target_changed)
        np.testing.assert_allclose(
            base_features["temporal_residual"],
            driver_features["temporal_residual"],
        )
        np.testing.assert_allclose(
            base_features["temporal_residual"],
            terminal_features["temporal_residual"],
        )
        np.testing.assert_allclose(
            base_features["cross_residual"],
            target_features["cross_residual"],
        )
        scored = calibration.transform(windows[140:], {
            name: values[140:] for name, values in raw_scores.items()
        })
        for component in ("temporal_residual", "cross_residual", "disagreement"):
            self.assertIn(f"{component}_tail", scored)
            self.assertIn(f"{component}_reliability_stratum", scored)
        metadata = calibration.metadata()
        self.assertFalse(metadata["target_terminal_used_by_features"])
        self.assertFalse(metadata["oracle_regime_used"])

    def test_runner_writes_complete_provenance_without_test_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary) / "b0"
            result = run_experiment(
                self.config, result_dir, device="cpu", seed=int(self.config["seed"])
            )
            self.assertIn(result["status"], {"passed", "failed_gates"})
            evaluation = json.loads(
                (result_dir / "b0_evaluation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(evaluation["no_score_fusion"])
            self.assertFalse(
                evaluation["provenance"]["test_scores_used_for_thresholds"]
            )
            self.assertTrue((result_dir / "episode_scores.csv").is_file())
            self.assertTrue((result_dir / "model_state.pt").is_file())
            self.assertTrue((result_dir / "synthetic_suite" / "paired_episodes.npz").is_file())


if __name__ == "__main__":
    unittest.main()
