import json
import tempfile
import unittest
from pathlib import Path

from scripts.a2.run_m4_confirmation import _development_contract, _require_passing_ablation
from scripts.a2.run_transition_compatibility import _canonical_hash


class A2M4ConfirmationTest(unittest.TestCase):
    def test_ablation_is_checked_against_its_distinct_development_contract_seed(self):
        base_contract = {"suite_id": "a2_transition_contract_v2", "seed": 5110}
        confirmation = {"development_contract_seed": 5120}
        development_contract = _development_contract(base_contract, confirmation)
        self.assertEqual(development_contract["seed"], 5120)
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "ablation.json"
            artifact.write_text(
                json.dumps(
                    {
                        "gate": {"passed": True},
                        "contract_config_hash": _canonical_hash(development_contract),
                    }
                ),
                encoding="utf-8",
            )
            _require_passing_ablation(artifact, development_contract)
            with self.assertRaisesRegex(RuntimeError, "different contract"):
                _require_passing_ablation(artifact, base_contract)

    def test_confirmation_requires_a_development_contract_seed(self):
        with self.assertRaisesRegex(ValueError, "development_contract_seed"):
            _development_contract({"seed": 5110}, {})


if __name__ == "__main__":
    unittest.main()
