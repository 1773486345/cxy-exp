"""Regression coverage for A3-v2's independent-background protocol only."""

import copy
import unittest
from pathlib import Path

import numpy as np

from scripts.a3.audit_independent_background_contract import (
    audit_independent_background_contract,
    one_sided_wilson_upper,
)
from scripts.a3.generate_independent_background_contract import (
    generate_independent_background_suite,
)
from scripts.a3.generate_trigger_response_contract import DEFAULT_CONFIG, _load_json


PROTOCOL_CONFIG = Path("config/a3/independent_background_calibration_v2.json")


def _small_base_contract():
    base = copy.deepcopy(_load_json(DEFAULT_CONFIG))
    base.update(
        {"burn_in": 16, "train_length": 320, "background_length": 160, "history_length": 8, "horizon_length": 6}
    )
    base["normal_splits"].update(
        {
            "optimization_length": 96,
            "validation_length": 48,
            "reference_length": 96,
            "outer_calibration_length": 41,
            "guard_length": 13,
        }
    )
    base["normal_process"]["regime_segment_length"] = 16
    base["episodes"].update(
        {
            "pairs_per_mode": 2,
            "normal_transition_sources_per_regime": {
                "optimization": 4,
                "validation": 4,
                "reference": 4,
                "outer_calibration": 4,
            },
            "cue_length": 6,
            "response_onsets": [1, 3],
            "response_ramp_length": 2,
        }
    )
    return base


class A3IndependentBackgroundContractTest(unittest.TestCase):
    def test_protocol_is_balanced_reproducible_and_auditable(self):
        base = _small_base_contract()
        protocol = copy.deepcopy(_load_json(PROTOCOL_CONFIG))
        protocol["background"].update({"burn_in": 16, "blocks_per_regime": 32})
        protocol["evaluation"].update({"minimum_total_blocks": 64, "require_per_regime_bound": True})
        first = generate_independent_background_suite(base, protocol)
        second = generate_independent_background_suite(base, protocol)
        np.testing.assert_array_equal(first["windows"], second["windows"])
        self.assertEqual(first["windows"].shape, (64, 14, base["dimensions"]))
        audit = audit_independent_background_contract(base, protocol)
        self.assertTrue(audit["passed"], audit["violations"])
        self.assertEqual(audit["metrics"]["regime_counts"], [32, 32])
        self.assertEqual(audit["metrics"]["unique_spawn_key_count"], 64)
        self.assertEqual(audit["metrics"]["fixed_trigger_false_acceptances"], 0)

    def test_wilson_rule_requires_observed_headroom_below_the_target(self):
        upper_at_target = one_sided_wilson_upper(205, 2048, 0.95)
        self.assertGreater(upper_at_target, 0.10)
        audit = audit_independent_background_contract(
            _small_base_contract(),
            {**copy.deepcopy(_load_json(PROTOCOL_CONFIG)), "background": {"seed": 7301, "burn_in": 16, "blocks_per_regime": 32, "fixed_regime_per_block": True}, "evaluation": {"operating_fpr": 0.10, "confidence_level": 0.95, "interval": "wilson_one_sided", "minimum_total_blocks": 64, "require_per_regime_bound": True}},
        )
        maximum = audit["metrics"]["maximum_accepted_exceedances"]
        self.assertLess(maximum / 64.0, 0.10)
        self.assertLess(audit["metrics"]["per_regime_maximum_accepted_empirical_fpr"], 0.10)


if __name__ == "__main__":
    unittest.main()
