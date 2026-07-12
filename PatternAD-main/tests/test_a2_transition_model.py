import copy
import unittest

import numpy as np
import torch

from scripts.a2.generate_transition_contract import DEFAULT_CONFIG, _load_json, generate_suite
from scripts.a2.run_transition_compatibility import _normal_continuation_windows
from ts_benchmark.baselines.A2TransitionCompatibility import (
    A2ContrastiveCompatibility,
    A2TransitionCodeCompatibility,
    A2LandmarkCompatibility,
    A2TransitionCompatibility,
    ContrastiveCompatibilityNet,
    TransitionCodeNet,
    TrajectoryCompatibilityNet,
)


def _small_config():
    config = copy.deepcopy(_load_json(DEFAULT_CONFIG))
    config.update(
        {
            "burn_in": 24,
            "train_length": 320,
            "background_length": 240,
            "history_length": 8,
            "horizon_length": 10,
        }
    )
    config["normal_splits"].update(
        {
            "optimization_length": 80,
            "validation_length": 40,
            "reference_length": 80,
            "outer_calibration_length": 69,
            "guard_length": 17,
        }
    )
    config["normal_process"]["regime_segment_length"] = 18
    config["episodes"].update(
        {
            "pairs_per_regime": 2,
            "normal_transition_sources_per_regime": {
                "optimization": 2,
                "validation": 2,
                "reference": 2,
                "outer_calibration": 2,
            },
            "cue_length": 4,
            "normal_transition_onsets": [2, 5],
            "incompatible_transition_onsets": [5, 2],
            "transition_ramp_length": 3,
        }
    )
    return config


def _bank(suite, name):
    return np.stack([episode["values"] for episode in suite["normal_transition_banks"][name]])


class A2TransitionModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _small_config()
        cls.suite = generate_suite(cls.config)
        cls.model = A2TransitionCompatibility(
            dimensions=cls.config["dimensions"],
            history_length=cls.config["history_length"],
            horizon_length=cls.config["horizon_length"],
            hidden_size=8,
            mixture_components=3,
            learning_rate=3e-3,
            epochs=5,
            patience=2,
            batch_size=8,
            outer_alpha=0.1,
            reliability_bin_count=2,
            device="cpu",
        ).fit(
            _bank(cls.suite, "optimization"),
            _bank(cls.suite, "validation"),
            _bank(cls.suite, "reference"),
            _bank(cls.suite, "outer_calibration"),
            seed=6101,
        )

    def test_score_pipeline_returns_full_trajectory_compatibility_outputs(self):
        windows = _bank(self.suite, "validation")
        scores = self.model.score_windows(windows)
        self.assertEqual(
            set(scores),
            {"trajectory_nll", "compatibility_tail", "reliability_bin", "outer_threshold", "outer_exceedance"},
        )
        self.assertEqual(scores["trajectory_nll"].shape, (len(windows),))
        self.assertTrue(np.isfinite(scores["trajectory_nll"]).all())
        self.assertTrue(np.isfinite(scores["compatibility_tail"]).all())
        self.assertTrue(np.isin(scores["outer_exceedance"], [0, 1]).all())
        self.assertTrue(np.isin(scores["reliability_bin"], [0, 1]).all())
        self.assertTrue(np.isfinite(scores["outer_threshold"]).all())

    def test_event_pre_state_does_not_depend_on_candidate_future(self):
        windows = _bank(self.suite, "reference")[:3]
        altered = windows.copy()
        altered[:, self.config["history_length"] :] *= -7.0
        np.testing.assert_array_equal(
            self.model.event_pre_state(windows), self.model.event_pre_state(altered)
        )

    def test_mixture_mean_has_the_full_future_shape(self):
        windows = _bank(self.suite, "outer_calibration")[:2]
        predicted = self.model.predict_mean_trajectory(windows)
        self.assertEqual(
            predicted.shape,
            (2, self.config["horizon_length"], self.config["dimensions"]),
        )
        self.assertTrue(np.isfinite(predicted).all())

    def test_continuation_windows_stay_inside_their_frozen_split(self):
        history = self.config["history_length"]
        horizon = self.config["horizon_length"]
        windows = _normal_continuation_windows(self.suite, "reference", history, horizon)
        split_start, split_end = self.suite["normal_split_ranges"]["reference"]
        expected_count = split_end - split_start - history - horizon + 1
        self.assertEqual(len(windows), expected_count)
        np.testing.assert_array_equal(
            windows[0],
            self.suite["train_values"][split_start : split_start + history + horizon],
        )
        np.testing.assert_array_equal(
            windows[-1],
            self.suite["train_values"][split_end - history - horizon : split_end],
        )

    def test_unconditional_ablation_has_no_event_pre_state_dependence(self):
        net = TrajectoryCompatibilityNet(
            dimensions=self.config["dimensions"],
            horizon_length=self.config["horizon_length"],
            hidden_size=8,
            mixture_components=3,
            condition_on_event_pre=False,
        )
        first = torch.zeros(2, self.config["history_length"], self.config["dimensions"])
        second = torch.ones_like(first)
        with torch.no_grad():
            np.testing.assert_array_equal(
                net.encode_event_pre(first).numpy(), net.encode_event_pre(second).numpy()
            )

    def test_contrastive_future_encoder_uses_internal_trajectory_increments(self):
        net = ContrastiveCompatibilityNet(
            dimensions=self.config["dimensions"],
            horizon_length=self.config["horizon_length"],
            hidden_size=8,
        )
        future = torch.randn(2, self.config["horizon_length"], self.config["dimensions"])
        shifted = future + 7.0
        with torch.no_grad():
            np.testing.assert_allclose(
                net.encode_candidate_future(future).numpy(),
                net.encode_candidate_future(shifted).numpy(),
                atol=1e-6,
            )

    def test_contrastive_model_scores_energy_and_preserves_event_pre_isolation(self):
        model = A2ContrastiveCompatibility(
            dimensions=self.config["dimensions"],
            history_length=self.config["history_length"],
            horizon_length=self.config["horizon_length"],
            hidden_size=8,
            learning_rate=3e-3,
            epochs=5,
            patience=2,
            batch_size=8,
            outer_alpha=0.1,
            reliability_bin_count=2,
            contrastive_temperature=0.2,
            forecast_weight=0.25,
            device="cpu",
        ).fit(
            _bank(self.suite, "optimization"),
            _bank(self.suite, "validation"),
            _bank(self.suite, "reference"),
            _bank(self.suite, "outer_calibration"),
            seed=6301,
        )
        windows = _bank(self.suite, "reference")[:3]
        altered = windows.copy()
        altered[:, self.config["history_length"] :] *= -5.0
        scores = model.score_windows(windows)
        self.assertIn("contrastive_energy", scores)
        self.assertTrue(np.isfinite(scores["contrastive_energy"]).all())
        first_state = model.event_pre_state(windows)
        second_state = model.event_pre_state(altered)
        self.assertIsInstance(first_state, np.ndarray)
        self.assertTrue(np.isfinite(first_state).all())
        np.testing.assert_array_equal(first_state, second_state)

    def test_transition_code_encoder_uses_internal_trajectory_increments(self):
        net = TransitionCodeNet(
            dimensions=self.config["dimensions"],
            horizon_length=self.config["horizon_length"],
            hidden_size=8,
            codebook_size=3,
        )
        future = torch.randn(2, self.config["horizon_length"], self.config["dimensions"])
        shifted = future + 7.0
        with torch.no_grad():
            np.testing.assert_allclose(
                net.encode_future_increments(future).numpy(),
                net.encode_future_increments(shifted).numpy(),
                atol=1e-6,
            )

    def test_transition_code_model_scores_global_code_support(self):
        model = A2TransitionCodeCompatibility(
            dimensions=self.config["dimensions"],
            history_length=self.config["history_length"],
            horizon_length=self.config["horizon_length"],
            hidden_size=8,
            learning_rate=3e-3,
            epochs=5,
            patience=2,
            batch_size=8,
            outer_alpha=0.1,
            reliability_bin_count=1,
            codebook_size=3,
            minimum_code_occupancy=1,
            device="cpu",
        ).fit(
            _bank(self.suite, "optimization"),
            _bank(self.suite, "validation"),
            _bank(self.suite, "reference"),
            _bank(self.suite, "outer_calibration"),
            seed=6401,
        )
        windows = _bank(self.suite, "reference")[:3]
        altered = windows.copy()
        altered[:, self.config["history_length"] :] *= -5.0
        scores = model.score_windows(windows)
        self.assertIn("transition_code_surprisal", scores)
        self.assertTrue(np.isfinite(scores["transition_code_surprisal"]).all())
        self.assertTrue(np.array_equal(scores["reliability_bin"], np.zeros(3, dtype=np.int64)))
        self.assertEqual(len(model.fit_metadata_["optimization_code_usage"]), 3)
        self.assertIn("transition_code_coverage", model.additional_gates())
        np.testing.assert_array_equal(
            model.event_pre_state(windows), model.event_pre_state(altered)
        )

    def test_landmark_representation_uses_internal_future_changes(self):
        model = A2LandmarkCompatibility(
            dimensions=self.config["dimensions"],
            history_length=self.config["history_length"],
            horizon_length=self.config["horizon_length"],
            neighbor_count=4,
            state_increment_length=4,
            device="cpu",
        )
        future = np.random.default_rng(6402).normal(
            size=(2, self.config["horizon_length"], self.config["dimensions"])
        ).astype(np.float32)
        landmarks, directions = model._future_landmarks(future)
        shifted_landmarks, shifted_directions = model._future_landmarks(future + 9.0)
        np.testing.assert_array_equal(landmarks, shifted_landmarks)
        np.testing.assert_allclose(directions, shifted_directions, atol=1e-6)

    def test_landmark_model_scores_reference_support_and_preserves_isolation(self):
        model = A2LandmarkCompatibility(
            dimensions=self.config["dimensions"],
            history_length=self.config["history_length"],
            horizon_length=self.config["horizon_length"],
            neighbor_count=4,
            state_increment_length=4,
            outer_alpha=0.1,
            reliability_bin_count=1,
            device="cpu",
        ).fit(
            _bank(self.suite, "optimization"),
            _bank(self.suite, "validation"),
            _bank(self.suite, "reference"),
            _bank(self.suite, "outer_calibration"),
            seed=6403,
        )
        windows = _bank(self.suite, "reference")[:3]
        altered = windows.copy()
        altered[:, self.config["history_length"] :] += 17.0
        scores = model.score_windows(windows)
        self.assertIn("landmark_direction_surprisal", scores)
        self.assertTrue(np.isfinite(scores["landmark_direction_surprisal"]).all())
        self.assertTrue(np.array_equal(scores["reliability_bin"], np.zeros(3, dtype=np.int64)))
        self.assertEqual(
            len(model.fit_metadata_["reference_landmark_counts"]),
            self.config["horizon_length"] - 1,
        )
        np.testing.assert_array_equal(
            model.event_pre_state(windows), model.event_pre_state(altered)
        )


if __name__ == "__main__":
    unittest.main()
