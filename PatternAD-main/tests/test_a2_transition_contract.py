import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.a2.generate_transition_contract import (
    DEFAULT_CONFIG,
    ROLE_ORDER,
    _load_json,
    generate_suite,
    write_suite,
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
    config["normal_process"].update(
        {
            "regime_segment_length": 32,
            "normal_transition_max_profile_increment": 0.2,
        }
    )
    config["episodes"].update({"pairs_per_regime": 2, "abrupt_step_index": 2})
    return config


class A2TransitionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _small_config()
        cls.suite = generate_suite(cls.config)

    def test_generation_is_deterministic_and_has_every_role(self):
        repeated = generate_suite(self.config)
        np.testing.assert_array_equal(self.suite["train_values"], repeated["train_values"])
        np.testing.assert_array_equal(
            self.suite["background_values"], repeated["background_values"]
        )
        self.assertEqual(len(self.suite["contracts"]), 4)
        by_source = {}
        for episode in self.suite["episodes"]:
            by_source.setdefault(episode["source_id"], {})[episode["role"]] = episode
        self.assertEqual(len(by_source), 4)
        self.assertTrue(all(set(roles) == set(ROLE_ORDER) for roles in by_source.values()))

    def test_primary_pair_ties_event_pre_and_endpoint_but_not_trajectory(self):
        history = int(self.config["history_length"])
        by_source = {}
        for episode in self.suite["episodes"]:
            by_source.setdefault(episode["source_id"], {})[episode["role"]] = episode
        for roles in by_source.values():
            gradual = roles["normal_gradual_transition"]["values"]
            abrupt = roles["incompatible_abrupt_transition"]["values"]
            np.testing.assert_array_equal(gradual[:history], abrupt[:history])
            np.testing.assert_array_equal(gradual[-1], abrupt[-1])
            self.assertGreater(np.max(np.abs(gradual[history:] - abrupt[history:])), 0.0)

    def test_coordination_pair_ties_target_trajectory_but_breaks_driver_support(self):
        history = int(self.config["history_length"])
        target = int(self.config["episodes"]["target_channel"])
        drivers = [index for index in range(int(self.config["dimensions"])) if index != target]
        by_source = {}
        for episode in self.suite["episodes"]:
            by_source.setdefault(episode["source_id"], {})[episode["role"]] = episode
        for roles in by_source.values():
            normal = roles["normal_coordinated_transition"]["values"]
            unsupported = roles["unsupported_transition"]["values"]
            np.testing.assert_array_equal(normal[:history], unsupported[:history])
            np.testing.assert_array_equal(normal[:, target], unsupported[:, target])
            self.assertGreater(
                np.max(np.abs(normal[history:, drivers] - unsupported[history:, drivers])),
                0.0,
            )
        for contract in self.suite["contracts"]:
            self.assertLessEqual(
                contract["normal_profile_max_increment"],
                contract["normal_transition_max_profile_increment"],
            )
            self.assertGreater(
                contract["abrupt_profile_max_increment"],
                contract["normal_transition_max_profile_increment"],
            )
            self.assertEqual(contract["primary_endpoint_max_abs_difference"], 0.0)
            self.assertEqual(contract["coordination_target_trajectory_max_abs_difference"], 0.0)
            self.assertGreater(
                contract["coordination_driver_trajectory_max_abs_difference"], 0.0
            )
            self.assertGreaterEqual(
                contract["coordination_error_target_std"],
                contract["minimum_coordination_error_target_std"],
            )

    def test_generator_does_not_depend_on_future_model_or_calibration_choices(self):
        changed = copy.deepcopy(self.config)
        changed["model"] = {"family": "future_a2_candidate", "width": 999}
        changed["calibration"] = {"threshold": "future_outer_normal_only"}
        repeated = generate_suite(changed)
        np.testing.assert_array_equal(
            self.suite["background_values"], repeated["background_values"]
        )
        self.assertEqual(self.suite["contracts"], repeated["contracts"])
        for first, second in zip(self.suite["episodes"], repeated["episodes"]):
            np.testing.assert_array_equal(first["values"], second["values"])

    def test_contract_fails_closed_when_coordination_gap_is_impossible(self):
        impossible = copy.deepcopy(self.config)
        impossible["episodes"]["minimum_coordination_error_target_std"] = 1e6
        with self.assertRaisesRegex(ValueError, "coordination-error contract"):
            generate_suite(impossible)

    def test_persisted_contract_has_no_model_scores_or_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "suite"
            metadata_path = write_suite(self.config, self.suite, output_dir)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertFalse(metadata["contains_model_scores"])
            self.assertFalse(metadata["contains_calibration_thresholds"])
            self.assertEqual(metadata["episode_count"], 20)
            self.assertTrue((output_dir / "normal_streams.npz").is_file())
            self.assertTrue((output_dir / "episodes.npz").is_file())


if __name__ == "__main__":
    unittest.main()
