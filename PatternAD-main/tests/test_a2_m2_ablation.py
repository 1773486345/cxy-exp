import copy
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a2.analyze_m2_ablation import analyze_ablation


def _summary(conditioned: bool, all_gates: bool, primary_passed: bool):
    return {
        "experiment_id": (
            "a2_contrastive_energy_m2_v1"
            if conditioned
            else "a2_contrastive_energy_unconditional_m2_v1"
        ),
        "condition_on_event_pre": conditioned,
        "raw_score_name": "event_pre_future_contrastive_energy",
        "contract_config_hash": "shared-contract",
        "experiment_config": {
            "schema_version": 1,
            "experiment_id": "conditioned" if conditioned else "unconditional",
            "seed": 6301,
            "device": "cpu",
            "model": {
                "hidden_size": 32,
                "condition_on_event_pre": conditioned,
                "learning_rate": 0.003,
            },
            "calibration": {"outer_alpha": 0.05},
        },
        "all_gates_passed": all_gates,
        "gates": {
            "primary_ordering": {
                "passed": primary_passed,
                "positive_pairs": 16 if primary_passed else 8,
                "pair_count": 16,
                "median_tail_margin": 1.0 if primary_passed else -0.1,
            }
        },
    }


class A2M2AblationTest(unittest.TestCase):
    def test_requires_conditioned_pass_and_unconditional_primary_failure(self):
        result = analyze_ablation(_summary(True, True, True), _summary(False, False, False))
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(result["conditioned"]["primary_positive_pairs"], 16)
        self.assertEqual(result["unconditional"]["primary_positive_pairs"], 8)

    def test_rejects_unconditional_primary_success(self):
        result = analyze_ablation(_summary(True, True, True), _summary(False, True, True))
        self.assertFalse(result["gate"]["passed"])

    def test_rejects_more_than_condition_setting_changes(self):
        unconditional = _summary(False, False, False)
        unconditional = copy.deepcopy(unconditional)
        unconditional["experiment_config"]["model"]["hidden_size"] = 64
        with self.assertRaisesRegex(ValueError, "may differ only"):
            analyze_ablation(_summary(True, True, True), unconditional)


if __name__ == "__main__":
    unittest.main()
