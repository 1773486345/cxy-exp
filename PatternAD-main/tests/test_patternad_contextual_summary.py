import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.patternad.summarize_contextual_factorial import (
    MECHANISMS,
    METRICS,
    _canonical_hash,
    _file_sha256,
    summarize,
)


MAIN_VARIANTS = ("A00", "A10", "A01", "A11")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _variant_metrics(variant):
    values = {
        "A00": {
            "average_precision": [0.20, 0.30, 0.20, 0.10],
            "regime_fpr_gap": [0.40, 0.30, 0.20, 0.10],
            "same_ordering": [False, False, False],
            "slow_ordering": [True, False],
        },
        "A10": {
            "average_precision": [0.25, 0.32, 0.22, 0.11],
            "regime_fpr_gap": [0.35, 0.25, 0.20, 0.10],
            "same_ordering": [True, False, False],
            "slow_ordering": [True, False],
        },
        "A01": {
            "average_precision": [0.30, 0.35, 0.25, 0.12],
            "regime_fpr_gap": [0.30, 0.25, 0.15, 0.10],
            "same_ordering": [True, False, False],
            "slow_ordering": [True, False],
        },
        "A11": {
            "average_precision": [0.40, 0.50, 0.35, 0.15],
            "regime_fpr_gap": [0.20, 0.15, 0.10, 0.05],
            "same_ordering": [True, True, True],
            "slow_ordering": [True, False],
        },
    }
    return values[variant]


def _evaluation(variant, model_seed, synthetic_hash, result_dir):
    values = _variant_metrics(variant)
    mechanisms = []
    orderings = []
    for mechanism, average_precision, fpr_gap in zip(
        MECHANISMS,
        values["average_precision"],
        values["regime_fpr_gap"],
    ):
        if mechanism == "same_deviation_different_context":
            correctness = values["same_ordering"]
        elif mechanism == "slow_drift_vs_abrupt_shift":
            correctness = values["slow_ordering"]
        else:
            correctness = []
        ordering_rate = sum(correctness) / len(correctness) if correctness else None
        mechanisms.append(
            {
                "mechanism": mechanism,
                "average_precision": average_precision,
                "ap_over_prevalence": average_precision - 0.10,
                "regime_fpr_gap": fpr_gap,
                "matched_ordering_rate": ordering_rate,
                "score_file": str(result_dir / "scores" / f"{mechanism}.npz"),
                "score_sha256": _file_sha256(
                    result_dir / "scores" / f"{mechanism}.npz"
                ),
            }
        )
        orderings.extend(
            {"mechanism": mechanism, "correct": correct} for correct in correctness
        )
    all_correct = [row["correct"] for row in orderings]
    return {
        "schema_version": 1,
        "method": f"{variant}_seed_{model_seed}",
        "config_hash": synthetic_hash,
        "macro_average_precision": sum(values["average_precision"]) / 4,
        "matched_ordering_rate": sum(all_correct) / len(all_correct),
        "maximum_regime_fpr_gap": max(values["regime_fpr_gap"]),
        "mechanisms": mechanisms,
        "matched_orderings": orderings,
    }


def _build_run(
    root,
    generator_seeds=(3101,),
    model_seeds=(2021,),
    variants=MAIN_VARIANTS,
    phase="development",
):
    root.mkdir(parents=True)
    synthetic = {
        "schema_version": 1,
        "suite_id": "test_contextual",
        "seed": 314159,
        "seed_groups": {
            "development": list(range(3101, 3111)),
            "confirmation": list(range(3111, 3121)),
        },
    }
    factorial = {
        "schema_version": 1,
        "development_seeds": [2021, 2022, 2023],
        "shared_hyperparameters": {"common": 1},
        "variants": {
            variant: {"hyperparameters": {"variant_name": variant}}
            for variant in MAIN_VARIANTS
        },
        "comparisons": [
            {"name": "full_vs_baseline", "lhs": "A11", "rhs": "A00"},
            {"name": "context_at_D0", "lhs": "A10", "rhs": "A00"},
            {"name": "context_at_D1", "lhs": "A11", "rhs": "A01"},
            {"name": "distribution_at_C0", "lhs": "A01", "rhs": "A00"},
            {"name": "distribution_at_C1", "lhs": "A11", "rhs": "A10"},
            {"name": "unused_B", "lhs": "B11", "rhs": "B00"},
        ],
    }
    synthetic_path = root / "frozen_synthetic.json"
    factorial_path = root / "frozen_factorial.json"
    _write_json(synthetic_path, synthetic)
    _write_json(factorial_path, factorial)
    synthetic_hash = _canonical_hash(synthetic)
    factorial_hash = _canonical_hash(factorial)
    expected = []
    for variant in variants:
        for generator_seed in generator_seeds:
            resolved = dict(synthetic)
            resolved["seed"] = generator_seed
            resolved_hash = _canonical_hash(resolved)
            for model_seed in model_seeds:
                hyperparameters = dict(factorial["shared_hyperparameters"])
                hyperparameters.update(
                    factorial["variants"][variant]["hyperparameters"]
                )
                hyperparameters["train_mask_seed"] = model_seed
                relative = (
                    Path(variant)
                    / f"generator_seed_{generator_seed}"
                    / f"model_seed_{model_seed}"
                )
                expected.append(
                    {
                        "variant": variant,
                        "generator_seed": generator_seed,
                        "model_seed": model_seed,
                        "result_dir": relative.as_posix(),
                        "config_hash": (
                            f"config-{variant}-{generator_seed}-{model_seed}"
                        ),
                        "synthetic_config_hash": resolved_hash,
                        "hyperparameters_hash": _canonical_hash(hyperparameters),
                    }
                )
    complete_phase_grid = bool(
        phase == "development"
        and list(generator_seeds) == synthetic["seed_groups"]["development"]
        and list(model_seeds) == factorial["development_seeds"]
        and list(variants) == list(MAIN_VARIANTS)
    )
    plan_core = {
        "schema_version": 1,
        "phase": phase,
        "complete_phase_grid": complete_phase_grid,
        "generator_seeds": list(generator_seeds),
        "model_seeds": list(model_seeds),
        "variants": list(variants),
        "synthetic_config_path": str(synthetic_path),
        "synthetic_config_hash": synthetic_hash,
        "factorial_manifest_path": str(factorial_path),
        "factorial_manifest_hash": factorial_hash,
        "expected_identities": expected,
    }
    plan = {
        **plan_core,
        "plan_hash": _canonical_hash(plan_core),
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    _write_json(root / "run_plan.json", plan)
    for identity in expected:
        result_dir = root / identity["result_dir"]
        for mechanism in MECHANISMS:
            score_path = result_dir / "scores" / f"{mechanism}.npz"
            score_path.parent.mkdir(parents=True, exist_ok=True)
            score_path.write_bytes(
                f"{identity['variant']}:{identity['generator_seed']}:"
                f"{identity['model_seed']}:{mechanism}".encode("ascii")
            )
        _write_json(
            result_dir / "identity_metadata.json",
            {
                "status": "completed",
                "plan_hash": plan["plan_hash"],
                "config_hash": identity["config_hash"],
                "factorial_manifest_hash": factorial_hash,
                "synthetic_config_hash": identity["synthetic_config_hash"],
                "variant": identity["variant"],
                "generator_seed": identity["generator_seed"],
                "model_seed": identity["model_seed"],
            },
        )
        _write_json(
            result_dir / "contextual_evaluation.json",
            _evaluation(
                identity["variant"],
                identity["model_seed"],
                identity["synthetic_config_hash"],
                result_dir,
            ),
        )
        hyperparameters = dict(factorial["shared_hyperparameters"])
        hyperparameters.update(
            factorial["variants"][identity["variant"]]["hyperparameters"]
        )
        hyperparameters["train_mask_seed"] = identity["model_seed"]
        _write_json(
            result_dir / "scores" / "score_run_metadata.json",
            {
                "variant": identity["variant"],
                "seed": identity["model_seed"],
                "synthetic_config_hash": identity["synthetic_config_hash"],
                "factorial_manifest_hash": factorial_hash,
                "hyperparameters": hyperparameters,
            },
        )
    return plan


class ContextualFactorialSummaryTest(unittest.TestCase):
    def test_complete_development_grid_writes_deterministic_summary_and_passes_gates(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            run_root = temporary / "run"
            _build_run(
                run_root,
                generator_seeds=tuple(range(3101, 3111)),
                model_seeds=(2021, 2022, 2023),
            )
            first_output = temporary / "first"
            second_output = temporary / "second"
            metadata = summarize(run_root, 300, 2021, first_output)
            summarize(run_root, 300, 2021, second_output)

            expected_files = {
                "identity_metrics.csv",
                "paired_deltas.csv",
                "paired_bootstrap.csv",
                "gate_diagnostics.csv",
                "summary_metadata.json",
            }
            self.assertEqual(
                {path.name for path in first_output.iterdir()}, expected_files
            )
            self.assertEqual(metadata["n_identities"], 120)
            self.assertEqual(metadata["n_paired_rows"], 150)
            self.assertTrue(metadata["grid_diagnostics"]["gate_eligible"])
            bootstrap = _read_csv(first_output / "paired_bootstrap.csv")
            self.assertEqual(len(bootstrap), 5 * len(METRICS))
            self.assertEqual(
                (first_output / "paired_bootstrap.csv").read_bytes(),
                (second_output / "paired_bootstrap.csv").read_bytes(),
            )
            self.assertEqual(
                {row["sampling"] for row in bootstrap},
                {"independent_generator_and_model_crossed"},
            )

            gates = {
                row["criterion"]: row
                for row in _read_csv(first_output / "gate_diagnostics.csv")
            }
            self.assertEqual(gates["gate_preconditions"]["status"], "eligible")
            self.assertEqual(gates["matched_ordering_improvement"]["status"], "pass")
            self.assertEqual(
                gates["maximum_regime_fpr_gap_relative_reduction"]["status"],
                "pass",
            )
            self.assertAlmostEqual(
                float(
                    gates["maximum_regime_fpr_gap_relative_reduction"]["observed_value"]
                ),
                0.5,
            )
            self.assertEqual(
                gates["dependency_break_average_precision"]["status"], "pass"
            )
            self.assertEqual(gates["ordinary_large_spike"]["status"], "not_evaluated")

    def test_ten_generators_with_one_model_seed_has_insufficient_formal_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            _build_run(
                root,
                generator_seeds=tuple(range(3101, 3111)),
                model_seeds=(2021,),
            )
            output = Path(temporary) / "summary"
            metadata = summarize(root, 40, 7, output)
            self.assertFalse(metadata["grid_diagnostics"]["gate_eligible"])
            self.assertFalse(
                metadata["grid_diagnostics"]["complete_development_model_seed_group"]
            )
            bootstrap = _read_csv(output / "paired_bootstrap.csv")
            self.assertEqual({row["sampling"] for row in bootstrap}, {"generator_only"})
            gates = {
                row["criterion"]: row
                for row in _read_csv(output / "gate_diagnostics.csv")
            }
            for criterion in (
                "matched_ordering_improvement",
                "maximum_regime_fpr_gap_relative_reduction",
                "dependency_break_average_precision",
            ):
                self.assertEqual(gates[criterion]["status"], "insufficient_data")
            self.assertEqual(gates["ordinary_large_spike"]["status"], "not_evaluated")

    def test_missing_or_unexpected_identity_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            missing_root = temporary / "missing"
            plan = _build_run(missing_root)
            missing_path = (
                missing_root
                / plan["expected_identities"][0]["result_dir"]
                / "identity_metadata.json"
            )
            missing_path.unlink()
            with self.assertRaisesRegex(ValueError, "selective summary"):
                summarize(missing_root, 10, 1)

            unexpected_root = temporary / "unexpected"
            _build_run(unexpected_root)
            _write_json(
                unexpected_root / "stray" / "identity_metadata.json",
                {"status": "completed"},
            )
            with self.assertRaisesRegex(ValueError, "unexpected identity metadata"):
                summarize(unexpected_root, 10, 1)

    def test_score_provenance_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            plan = _build_run(root)
            identity = plan["expected_identities"][0]
            score_metadata_path = (
                root / identity["result_dir"] / "scores" / "score_run_metadata.json"
            )
            score_metadata = json.loads(score_metadata_path.read_text(encoding="utf-8"))
            score_metadata["factorial_manifest_hash"] = "tampered"
            _write_json(score_metadata_path, score_metadata)
            with self.assertRaisesRegex(ValueError, "Factorial manifest hash mismatch"):
                summarize(root, 10, 1)

    def test_score_file_and_hyperparameter_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            score_root = temporary / "score"
            score_plan = _build_run(score_root)
            score_identity = score_plan["expected_identities"][0]
            score_path = (
                score_root
                / score_identity["result_dir"]
                / "scores"
                / "dependency_break.npz"
            )
            score_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Score SHA-256 mismatch"):
                summarize(score_root, 10, 1)

            hyperparameter_root = temporary / "hyperparameters"
            hyperparameter_plan = _build_run(hyperparameter_root)
            hyperparameter_identity = hyperparameter_plan["expected_identities"][0]
            metadata_path = (
                hyperparameter_root
                / hyperparameter_identity["result_dir"]
                / "scores"
                / "score_run_metadata.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["hyperparameters"]["common"] = 2
            _write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ValueError, "Hyperparameters hash mismatch"):
                summarize(hyperparameter_root, 10, 1)

    def test_run_plan_core_tampering_is_rejected_before_identity_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            _build_run(root)
            plan_path = root / "run_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["expected_identities"].pop()
            _write_json(plan_path, plan)
            with self.assertRaisesRegex(ValueError, "plan_hash does not match"):
                summarize(root, 10, 1)


if __name__ == "__main__":
    unittest.main()
