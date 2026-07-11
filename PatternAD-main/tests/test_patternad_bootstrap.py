import csv
import tempfile
import unittest
from pathlib import Path

from scripts.patternad.bootstrap_factorial import (
    _bootstrap_summaries,
    _gate_diagnostics,
    _paired_from_entity_rows,
    main,
)


def _paired_rows():
    rows = []
    layout = {
        "F1": {"e1": [1.0, 1.0]},
        "F2": {"e2": [3.0, 3.0], "e3": [3.0, 3.0]},
    }
    for family, entities in layout.items():
        for entity, values in entities.items():
            for seed, value in enumerate(values, start=2021):
                rows.append(
                    {
                        "run_name": "run",
                        "group": "motivation",
                        "family": family,
                        "entity": entity,
                        "seed": seed,
                        "comparison": "candidate_vs_baseline",
                        "lhs": "A11",
                        "rhs": "A00",
                        "auc_pr": value,
                        "VUS_PR": value / 10.0,
                    }
                )
    return rows


def _p4_rows():
    rows = []
    entities = {"HAI21": ("HAI21_part2", "HAI21_part3"), "SMD": ("SMD_1", "SMD_2")}
    for family, family_entities in entities.items():
        for entity in family_entities:
            for seed in range(2021, 2026):
                rows.append(
                    {
                        "run_name": "locked",
                        "group": "confirmation",
                        "family": family,
                        "entity": entity,
                        "seed": seed,
                        "comparison": "full_vs_baseline",
                        "lhs": "A11",
                        "rhs": "A00",
                        "auc_pr": 0.02,
                        "VUS_PR": 0.01,
                    }
                )
    return rows


class PatternADBootstrapTest(unittest.TestCase):
    def test_hierarchy_is_family_weighted_and_seed_deterministic(self):
        rows = _paired_rows()
        diagnostics = [
            {
                "run_name": "run",
                "group": "motivation",
                "comparison": "candidate_vs_baseline",
                "n_dropped_identities": 0,
            }
        ]
        first = _bootstrap_summaries(
            rows, ["auc_pr"], 300, 7, 0.95, diagnostics, "fixed"
        )
        second = _bootstrap_summaries(
            rows, ["auc_pr"], 300, 7, 0.95, diagnostics, "fixed"
        )

        self.assertEqual(first, second)
        self.assertEqual(first[0]["mean"], 2.0)
        self.assertEqual(first[0]["n_families"], 2)
        self.assertEqual(first[0]["n_entities"], 3)
        self.assertEqual(first[0]["reliability"], "standard")

    def test_entity_input_rejects_incomplete_pairs_by_default(self):
        rows = [
            {
                "run_name": "run",
                "group": "motivation",
                "family": "F1",
                "entity": "e1",
                "seed": "2021",
                "variant": "A00",
                "auc_pr": "0.1",
            },
            {
                "run_name": "run",
                "group": "motivation",
                "family": "F1",
                "entity": "e1",
                "seed": "2021",
                "variant": "A11",
                "auc_pr": "0.2",
            },
            {
                "run_name": "run",
                "group": "motivation",
                "family": "F2",
                "entity": "e2",
                "seed": "2021",
                "variant": "A00",
                "auc_pr": "0.3",
            },
        ]
        comparison = [{"name": "full", "lhs": "A11", "rhs": "A00"}]
        with self.assertRaisesRegex(ValueError, "incomplete entity/seed pairs"):
            _paired_from_entity_rows(
                rows, comparison, ["auc_pr"], "error", True
            )

    def test_small_sample_is_computed_but_marked_limited(self):
        row = {
            "run_name": "run",
            "group": "smoke",
            "family": "Weather",
            "entity": "Weather",
            "seed": 2021,
            "comparison": "full",
            "lhs": "A11",
            "rhs": "A00",
            "auc_pr": 0.01,
        }
        summary = _bootstrap_summaries(
            [row], ["auc_pr"], 50, 3, 0.95, []
        )[0]

        self.assertEqual(summary["mean"], 0.01)
        self.assertEqual(summary["ci_lower"], 0.01)
        self.assertEqual(summary["ci_upper"], 0.01)
        self.assertEqual(summary["reliability"], "limited")
        self.assertIn("fewer_than_3_families", summary["warnings"])
        self.assertIn("fewer_than_2_seeds", summary["warnings"])

    def test_p4_gates_are_criterion_rows_not_an_overall_go_decision(self):
        rows = _p4_rows()
        summaries = _bootstrap_summaries(
            rows, ["auc_pr", "VUS_PR"], 100, 11, 0.95, [], "fixed"
        )
        gates = _gate_diagnostics(
            rows,
            summaries,
            "p4",
            "full_vs_baseline",
            ["HAI21", "SMD"],
            "error",
            100,
        )

        self.assertNotIn("overall", {row["criterion"] for row in gates})
        ci_gate = next(row for row in gates if row["criterion"] == "auc_pr_ci_lower_positive")
        self.assertEqual(ci_gate["status"], "pass")
        fpr_gate = next(
            row
            for row in gates
            if row["criterion"] == "fixed_threshold_false_alarm_calibration"
        )
        self.assertEqual(fpr_gate["status"], "not_evaluated")

    def test_gate_is_fail_closed_for_two_bootstraps_or_drop_policy(self):
        rows = _p4_rows()
        summaries = _bootstrap_summaries(
            rows, ["auc_pr", "VUS_PR"], 2, 11, 0.95, [], "fixed"
        )
        too_few = _gate_diagnostics(
            rows,
            summaries,
            "p4",
            "full_vs_baseline",
            ["HAI21", "SMD"],
            "error",
            10000,
        )
        self.assertEqual(too_few[0]["status"], "insufficient_data")
        self.assertIn("n_bootstrap_below", too_few[0]["detail"])
        self.assertFalse(any(row["status"] == "pass" for row in too_few))

        dropped = _gate_diagnostics(
            rows,
            summaries,
            "p4",
            "full_vs_baseline",
            ["HAI21", "SMD"],
            "drop",
            2,
        )
        self.assertEqual(dropped[0]["status"], "insufficient_data")
        self.assertIn("missing_policy_must_be_error", dropped[0]["detail"])
        self.assertFalse(any(row["status"] == "pass" for row in dropped))

    def test_cli_accepts_paired_summary_and_writes_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "paired_entity_seed_delta.csv"
            fields = [
                "run_name",
                "group",
                "family",
                "entity",
                "seed",
                "comparison",
                "lhs",
                "rhs",
                "auc_pr",
            ]
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=fields, extrasaction="ignore"
                )
                writer.writeheader()
                for row in _paired_rows():
                    writer.writerow(row)

            output_dir = root / "bootstrap"
            result = main(
                [
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--metrics",
                    "auc_pr",
                    "--n-bootstrap",
                    "50",
                    "--seed",
                    "19",
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "hierarchical_bootstrap.csv").is_file())
            self.assertTrue((output_dir / "input_diagnostics.csv").is_file())
            metadata = (output_dir / "bootstrap_metadata.json").read_text(
                encoding="utf-8"
            )
            self.assertIn('"selection_policy"', metadata)
            self.assertIn('"paired_entity_seed_delta"', metadata)

            with self.assertRaisesRegex(ValueError, "Formal gate diagnostics require"):
                main(
                    [
                        "--input",
                        str(input_path),
                        "--output-dir",
                        str(root / "rejected_gate"),
                        "--metrics",
                        "auc_pr",
                        "--n-bootstrap",
                        "2",
                        "--gate-profile",
                        "p2",
                        "--primary-comparison",
                        "candidate_vs_baseline",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
