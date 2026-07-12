"""Regression coverage for N1's normal-only raw preflight."""

import unittest
from pathlib import Path

from scripts.a3.audit_background_nulling_preflight import audit_background_nulling_preflight
from scripts.a3.generate_trigger_response_contract import _load_json


CONTRACT = Path("config/a3/trigger_response_route_identifiability_v2.json")
BACKGROUND_PROTOCOL = Path("config/a3/independent_background_route_v2.json")
PREFLIGHT = Path("config/a3/background_nulling_n1_v1.json")


class A3BackgroundNullingPreflightTest(unittest.TestCase):
    def test_normal_only_factor_keeps_the_declared_route(self):
        result = audit_background_nulling_preflight(
            _load_json(CONTRACT), _load_json(BACKGROUND_PROTOCOL), _load_json(PREFLIGHT)
        )
        self.assertTrue(result["passed"], result["violations"])
        self.assertEqual(result["metrics"]["fit_source"], "normal_optimization_increments_only")
        self.assertGreaterEqual(
            result["metrics"]["factor_background_alignment"],
            result["metrics"]["minimum_factor_alignment"],
        )
        self.assertGreaterEqual(
            result["metrics"]["route_retention_after_projection"],
            result["metrics"]["minimum_route_retention"],
        )


if __name__ == "__main__":
    unittest.main()
