"""Regression coverage for the A3-v2 route-identifiability contract only."""

import unittest
from pathlib import Path

from scripts.a3.audit_route_identifiability_contract import (
    audit_route_identifiability_contract,
    response_background_alignment,
)
from scripts.a3.generate_trigger_response_contract import _load_json


ROUTE_CONTRACT = Path("config/a3/trigger_response_route_identifiability_v2.json")
BACKGROUND_PROTOCOL = Path("config/a3/independent_background_route_v2.json")


class A3RouteIdentifiabilityContractTest(unittest.TestCase):
    def test_successor_route_is_orthogonal_and_preserves_raw_contract(self):
        contract = _load_json(ROUTE_CONTRACT)
        protocol = _load_json(BACKGROUND_PROTOCOL)
        alignment = response_background_alignment(contract)
        self.assertLessEqual(alignment["absolute_cosine"], 0.05)
        audit = audit_route_identifiability_contract(contract, protocol)
        self.assertTrue(audit["passed"], audit["violations"])
        self.assertEqual(audit["metrics"]["raw_contract"]["primary_raw_relation_count"], 16)
        self.assertEqual(audit["metrics"]["independent_background"]["independent_block_count"], 2048)
        self.assertEqual(audit["metrics"]["independent_background"]["fixed_trigger_false_acceptances"], 0)


if __name__ == "__main__":
    unittest.main()
