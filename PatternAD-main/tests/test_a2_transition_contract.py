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
from scripts.a2.audit_transition_contract import audit_suite


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
    config["normal_process"].update(
        {
            "regime_segment_length": 18,
        }
    )
    config["episodes"].update(
        {
            "pairs_per_regime": 2,
            "normal_transition_sources_per_regime": {
                "optimization": 2,
                "validation": 2,
                "reference": 4,
                "outer_calibration": 2,
            },
            "cue_length": 4,
            "normal_transition_onsets": [2, 5],
            "incompatible_transition_onsets": [5, 2],
            "transition_ramp_length": 3,
        }
    )
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
        self.assertEqual(
            self.suite["normal_split_ranges"],
            {
                "optimization": (0, 80),
                "validation": (97, 137),
                "reference": (154, 234),
                "outer_calibration": (251, 320),
            },
        )
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
            scheduled = roles["normal_scheduled_transition"]
            incompatible = roles["incompatible_timing_transition"]
            np.testing.assert_array_equal(
                scheduled["values"][:history], incompatible["values"][:history]
            )
            np.testing.assert_array_equal(
                scheduled["values"][-1], incompatible["values"][-1]
            )
            self.assertGreater(
                np.max(
                    np.abs(
                        scheduled["values"][history:] - incompatible["values"][history:]
                    )
                ),
                0.0,
            )
            self.assertEqual(scheduled["expected_onset"], scheduled["observed_onset"])
            self.assertNotEqual(
                incompatible["expected_onset"], incompatible["observed_onset"]
            )
            scheduled_increment = np.diff(scheduled["values"][history:], axis=0)
            incompatible_increment = np.diff(incompatible["values"][history:], axis=0)
            for first, second in (
                (np.max(np.abs(scheduled_increment)), np.max(np.abs(incompatible_increment))),
                (np.sum(np.abs(scheduled_increment)), np.sum(np.abs(incompatible_increment))),
                (np.sum(np.square(scheduled_increment)), np.sum(np.square(incompatible_increment))),
            ):
                self.assertAlmostEqual(float(first), float(second), places=7)

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
            self.assertEqual(contract["primary_endpoint_max_abs_difference"], 0.0)
            self.assertEqual(contract["primary_increment_max_abs_difference"], 0.0)
            self.assertEqual(contract["primary_increment_l1_abs_difference"], 0.0)
            self.assertEqual(contract["primary_increment_l2_abs_difference"], 0.0)
            self.assertEqual(contract["coordination_target_trajectory_max_abs_difference"], 0.0)
            self.assertGreater(
                contract["coordination_driver_trajectory_max_abs_difference"], 0.0
            )
            self.assertGreaterEqual(
                contract["coordination_error_target_std"],
                contract["minimum_coordination_error_target_std"],
            )

    def test_no_event_control_is_a_direct_normal_background_window(self):
        history = int(self.config["history_length"])
        horizon = int(self.config["horizon_length"])
        for episode in self.suite["episodes"]:
            if episode["role"] != "no_event_normal_control":
                continue
            source_start = int(episode["source_start"])
            np.testing.assert_array_equal(
                episode["values"],
                self.suite["background_values"][
                    source_start - history : source_start + horizon
                ],
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

    def test_contract_fails_closed_when_timing_mapping_loses_its_counterfactual(self):
        invalid = copy.deepcopy(self.config)
        invalid["episodes"]["incompatible_transition_onsets"] = [2, 5]
        with self.assertRaisesRegex(ValueError, "Incompatible timing"):
            generate_suite(invalid)

    def test_audit_certifies_conditional_timing_and_shortcut_controls(self):
        audit = audit_suite(self.config, self.suite)
        self.assertTrue(audit["passed"], audit["violations"])
        self.assertEqual(
            audit["primary_normal_onset_marginal"],
            audit["primary_incompatible_onset_marginal"],
        )
        self.assertEqual(audit["primary_cue_only_majority_accuracy"], 0.5)
        self.assertEqual(audit["primary_onset_only_majority_accuracy"], 0.5)
        self.assertEqual(audit["primary_cue_onset_majority_accuracy"], 1.0)
        self.assertEqual(audit["event_pre_cue_observability_accuracy"], 1.0)
        self.assertGreater(audit["event_pre_cue_observability_margin"], 0.0)
        self.assertEqual(audit["normal_transition_reference_count"], 24)
        self.assertEqual(
            audit["normal_transition_bank_counts"],
            {"optimization": 12, "validation": 12, "reference": 24, "outer_calibration": 12},
        )
        self.assertEqual(audit["normal_split_ranges"]["reference"], [154, 234])

    def test_audit_fails_when_reference_crosses_its_time_disjoint_split(self):
        altered = copy.deepcopy(self.suite)
        altered["normal_transition_banks"]["reference"][0]["source_start"] = 8
        audit = audit_suite(self.config, altered)
        self.assertFalse(audit["passed"])
        self.assertIn(
            "normal transition window crosses its time-disjoint split",
            audit["violations"],
        )

    def test_audit_fails_when_event_pre_cue_is_erased(self):
        altered = copy.deepcopy(self.suite)
        cue_channel = int(self.config["episodes"]["cue_channel"])
        cue_length = int(self.config["episodes"]["cue_length"])
        history = int(self.config["history_length"])
        for episode in altered["episodes"]:
            if episode["role"] in {
                "normal_scheduled_transition",
                "incompatible_timing_transition",
            }:
                episode["values"][history - 1, cue_channel] = episode["values"][
                    history - cue_length, cue_channel
                ]
        audit = audit_suite(self.config, altered)
        self.assertFalse(audit["passed"])
        self.assertIn(
            "event-pre cue is not observable with its predeclared raw-state rule",
            audit["violations"],
        )

    def test_v2_anchored_cue_is_exact_for_development_and_confirmation_seeds(self):
        v2_config = _load_json(DEFAULT_CONFIG.parent / "transition_contract_v2.json")
        history = int(v2_config["history_length"])
        cue_channel = int(v2_config["episodes"]["cue_channel"])
        cue_length = int(v2_config["episodes"]["cue_length"])
        cue_amplitudes = v2_config["episodes"]["cue_amplitudes"]
        for seed in (5110, 5111, 5112, 5113, 5114):
            config = copy.deepcopy(v2_config)
            config["seed"] = seed
            suite = generate_suite(config)
            audit = audit_suite(config, suite)
            self.assertTrue(audit["passed"], audit["violations"])
            self.assertEqual(audit["event_pre_cue_encoding"], "anchored_overwrite_v1")
            self.assertLessEqual(
                audit["event_pre_cue_maximum_amplitude_error"],
                audit["event_pre_cue_amplitude_tolerance"],
            )
            for episode in suite["episodes"]:
                if episode["role"] != "normal_scheduled_transition":
                    continue
                cue_mode = int(episode["cue_mode"])
                cue_values = episode["values"][:history, cue_channel]
                self.assertAlmostEqual(
                    float(cue_values[-1] - cue_values[-cue_length]),
                    float(cue_amplitudes[cue_mode]),
                    places=6,
                )

    def test_persisted_contract_has_no_model_scores_or_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "suite"
            metadata_path = write_suite(self.config, self.suite, output_dir)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertFalse(metadata["contains_model_scores"])
            self.assertFalse(metadata["contains_calibration_thresholds"])
            self.assertEqual(metadata["episode_count"], 20)
            self.assertEqual(metadata["normal_transition_reference_count"], 24)
            self.assertEqual(
                metadata["normal_transition_bank_counts"],
                {"optimization": 12, "validation": 12, "reference": 24, "outer_calibration": 12},
            )
            self.assertTrue((output_dir / "normal_streams.npz").is_file())
            self.assertTrue((output_dir / "episodes.npz").is_file())
            self.assertTrue((output_dir / "normal_transition_references.npz").is_file())
            self.assertTrue((output_dir / "normal_transition_banks.npz").is_file())
            streams = np.load(output_dir / "normal_streams.npz")
            np.testing.assert_array_equal(
                streams["normal_split_starts"], np.asarray([0, 97, 154, 251])
            )
            np.testing.assert_array_equal(
                streams["normal_split_ends"], np.asarray([80, 137, 234, 320])
            )


if __name__ == "__main__":
    unittest.main()
