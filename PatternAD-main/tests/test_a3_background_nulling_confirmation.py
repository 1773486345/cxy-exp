"""Regression coverage for the frozen A3-N1 confirmation sequence."""

import copy
import unittest
from pathlib import Path

from scripts.a3.analyze_background_nulling_confirmation import analyze_confirmation
from scripts.a3.n1_confirmation import (
    DEFAULT_CONFIRMATION_CONFIG,
    canonical_hash,
    load_confirmation_plan,
    prepared_confirmation_pair,
)


class A3BackgroundNullingConfirmationTest(unittest.TestCase):
    def _passing_summaries(self):
        plan, _, background, preflight, _ = load_confirmation_plan(DEFAULT_CONFIRMATION_CONFIG)
        summaries = []
        for pair_index, _ in enumerate(plan["pairs"]):
            _, contract, _, _, experiment, _ = prepared_confirmation_pair(pair_index)
            summaries.append(
                {
                    "experiment_id": "a3_n1_background_nulling_route_graph_development_v1",
                    "seed": experiment["seed"],
                    "device": "cuda",
                    "contract_config_hash": canonical_hash(contract),
                    "background_protocol_hash": canonical_hash(background),
                    "preflight_config_hash": canonical_hash(preflight),
                    "experiment_config": experiment,
                    "all_gates_passed": True,
                    "gates": {},
                }
            )
        return summaries

    def test_plan_registers_four_unique_nondevelopment_cuda_pairs(self):
        plan, contract, _, _, experiment = load_confirmation_plan(DEFAULT_CONFIRMATION_CONFIG)
        self.assertEqual(plan["required_complete_passes"], 4)
        self.assertEqual(len(plan["pairs"]), 4)
        self.assertNotIn(contract["seed"], {pair["contract_seed"] for pair in plan["pairs"]})
        self.assertNotIn(experiment["seed"], {pair["model_seed"] for pair in plan["pairs"]})
        self.assertEqual(prepared_confirmation_pair(0)[4]["device"], "cuda")

    def test_analyzer_requires_all_four_exact_complete_passes(self):
        summaries = self._passing_summaries()
        passing = analyze_confirmation(summaries)
        self.assertTrue(passing["passed"], passing["violations"])
        self.assertEqual(passing["complete_passes"], 4)
        failed_summaries = copy.deepcopy(summaries)
        failed_summaries[2]["all_gates_passed"] = False
        failed = analyze_confirmation(failed_summaries)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["complete_passes"], 3)


if __name__ == "__main__":
    unittest.main()
