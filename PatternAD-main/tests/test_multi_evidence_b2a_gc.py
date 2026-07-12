import copy
import unittest

import numpy as np

from scripts.multi_evidence.generate_b2a_gc import (
    DEFAULT_CONFIG,
    _legacy_structural_cross_support,
    _load_json,
    _structural_cross_support,
    generate_suite,
)


def _small_config():
    config = copy.deepcopy(_load_json(DEFAULT_CONFIG))
    config.update(
        {
            "burn_in": 32,
            "history_length": 8,
            "train_length": 1000,
            "test_length": 384,
        }
    )
    config["normal_process"]["relation_period"] = 64
    config["episodes"].update({"pairs_per_phase": 1, "drift_controls_per_phase": 1})
    config["counterfactual_contract"].update(
        {
            "maximum_relation_value_delta": 0.20,
            "minimum_structural_cross_gap_target_std": 0.25,
            "minimum_terminal_target_gap_target_std": 0.25,
            "source_candidates_per_chronological_block": 8,
        }
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


class MultiEvidenceB2aGcTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _small_config()
        cls.suite = generate_suite(cls.config)

    def test_structural_cross_support_excludes_target_self_lag(self):
        values = np.zeros((4, 3), dtype=np.float64)
        values[1] = np.asarray([100.0, 4.0, -2.0])
        factors = np.zeros((4, 2), dtype=np.float64)
        factors[2] = np.asarray([3.0, -1.0])
        relation = np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float64)
        process = {
            "lag_base": [[2.0, 0.5, -0.25], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "lag_drift": [[3.0, 0.1, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "loading_base": [[1.0, -2.0], [0.0, 0.0], [0.0, 0.0]],
            "loading_drift": [[0.5, 0.25], [0.0, 0.0], [0.0, 0.0]],
        }
        expected = (0.5 + 0.5 * 0.1) * 4.0 + (-0.25 + 0.5 * 0.2) * -2.0
        expected += np.dot(np.asarray([1.0, -2.0]) + 0.5 * np.asarray([0.5, 0.25]), factors[2])
        observed = _structural_cross_support(values, factors, relation, process, 0, 2)
        self.assertAlmostEqual(observed, expected)
        self.assertAlmostEqual(
            observed,
            _legacy_structural_cross_support(values, factors, relation, process, 0, 2),
        )

    def test_suite_is_deterministic_and_contracts_are_terminally_auditable(self):
        repeated = generate_suite(self.config)
        np.testing.assert_array_equal(self.suite["train"]["values"], repeated["train"]["values"])
        self.assertEqual(len(self.suite["contracts"]), 6 * 4)
        contract_config = self.config["counterfactual_contract"]
        pair_roles = {}
        for episode in self.suite["episodes"]:
            if episode["is_pair"]:
                pair_roles.setdefault((episode["target_index"], episode["pair_id"]), {})[
                    episode["role"]
                ] = episode
        self.assertEqual(len(pair_roles), 6 * 4)
        for record in self.suite["contracts"]:
            self.assertGreater(record["source_donor_terminal_distance"], self.config["history_length"])
            self.assertLessEqual(
                record["relation_value_abs_difference"],
                contract_config["maximum_relation_value_delta"] + 1e-12,
            )
            self.assertGreaterEqual(
                record["structural_cross_gap_target_std"],
                contract_config["minimum_structural_cross_gap_target_std"],
            )
            self.assertGreaterEqual(
                record["terminal_target_gap_target_std"],
                contract_config["minimum_terminal_target_gap_target_std"],
            )
            roles = pair_roles[(record["target_index"], record["pair_id"])]
            target = record["target_index"]
            drivers = [index for index in range(6) if index != target]
            coherent = roles["coherent_control"]["values"]
            unsupported = roles["unsupported_target_break"]["values"]
            omission = roles["target_omission_break"]["values"]
            np.testing.assert_array_equal(coherent[:, target], unsupported[:, target])
            np.testing.assert_array_equal(coherent[:, drivers], omission[:, drivers])
            self.assertGreater(
                record["coherent_unsupported_driver_terminal_max_abs_difference"], 0.0
            )
            self.assertGreater(
                record["coherent_omission_target_terminal_abs_difference"], 0.0
            )

    def test_generator_is_independent_of_model_and_calibration_settings(self):
        changed = copy.deepcopy(self.config)
        changed["model"].update({"d_model": 12, "epochs": 7, "batch_size": 8})
        changed["calibration"].update({"min_reference_per_stratum": 51})
        changed["evaluation"]["target_fpr"] = 0.03
        repeated = generate_suite(changed)
        np.testing.assert_array_equal(
            self.suite["background"]["values"], repeated["background"]["values"]
        )
        self.assertEqual(self.suite["contracts"], repeated["contracts"])
        for original, alternative in zip(self.suite["episodes"], repeated["episodes"]):
            self.assertEqual(original["source_terminal_index"], alternative["source_terminal_index"])
            self.assertEqual(original["donor_terminal_index"], alternative["donor_terminal_index"])
            np.testing.assert_array_equal(original["values"], alternative["values"])

    def test_generator_fails_closed_when_terminal_contract_is_impossible(self):
        impossible = copy.deepcopy(self.config)
        impossible["counterfactual_contract"]["minimum_structural_cross_gap_target_std"] = 1e6
        with self.assertRaisesRegex(ValueError, "valid source/donor pair"):
            generate_suite(impossible)

if __name__ == "__main__":
    unittest.main()
