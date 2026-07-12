"""Regression coverage for A3-N1's frozen past-free control contract."""

import unittest
from pathlib import Path

from scripts.a3.analyze_background_nulling_ablation import analyze_ablation
from scripts.a3.generate_trigger_response_contract import _load_json
from scripts.a3.run_background_nulling_route_graph import _validate_experiment_config


CONTRACT = Path("config/a3/trigger_response_route_identifiability_v2.json")
PREFLIGHT = Path("config/a3/background_nulling_n1_v1.json")
BACKGROUND = Path("config/a3/independent_background_route_v2.json")
DEVELOPMENT = Path("config/a3/background_nulling_n1_development_v1.json")
CONTROL = Path("config/a3/background_nulling_n1_past_free_control_v1.json")


class A3BackgroundNullingAblationTest(unittest.TestCase):
    def test_control_config_only_removes_event_pre_and_requires_primary_failure(self):
        contract = _load_json(CONTRACT)
        preflight = _load_json(PREFLIGHT)
        background = _load_json(BACKGROUND)
        development = _load_json(DEVELOPMENT)
        control = _load_json(CONTROL)
        _validate_experiment_config(contract, preflight, development)
        _validate_experiment_config(contract, preflight, control)
        common = {
            "contract_config_hash": "contract",
            "background_protocol_hash": "background",
            "preflight_config_hash": "preflight",
        }
        development_summary = {
            **common,
            "experiment_id": development["experiment_id"],
            "experiment_config": development,
            "all_gates_passed": True,
            "gates": {"primary_misrouted_null_route": {"passed": True, "positive_pairs": 16}},
        }
        control_summary = {
            **common,
            "experiment_id": control["experiment_id"],
            "experiment_config": control,
            "gates": {"primary_misrouted_null_route": {"passed": False, "positive_pairs": 7}},
        }
        result = analyze_ablation(development_summary, control_summary)
        self.assertTrue(result["passed"], result["violations"])
        self.assertTrue(result["metrics"]["control_failed_primary"])


if __name__ == "__main__":
    unittest.main()
