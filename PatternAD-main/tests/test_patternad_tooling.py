import csv
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.patternad.run_factorial_ablation import (
    _build_command,
    _completed_attempt,
    _next_attempt_dir,
    _validate_artifact,
    _validate_model_diagnostics,
)
from scripts.patternad.summarize_factorial import (
    _flatten_run_diagnostics,
    _verify_complete_plan,
)


class PatternADToolingTest(unittest.TestCase):
    def test_auxiliary_diagnostics_are_disabled_in_all_formal_cells(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (repo_root / "config/patternad/factorial_ablation.json").read_text(
                encoding="utf-8"
            )
        )
        shared = manifest["shared_hyperparameters"]
        observed = {}
        causal_level = {}
        causal_delta = {}
        for variant, definition in manifest["variants"].items():
            merged = {**shared, **definition["hyperparameters"]}
            observed[variant] = merged["reconstruction_transition_loss_weight"]
            causal_level[variant] = merged[
                "reconstruction_causal_innovation_loss_weight"
            ]
            causal_delta[variant] = merged[
                "reconstruction_causal_delta_innovation_loss_weight"
            ]

        self.assertEqual(observed["A00"], 0.0)
        self.assertEqual(observed["A10"], 0.0)
        self.assertEqual(observed["B00"], 0.0)
        self.assertEqual(observed["A01"], 0.0)
        self.assertEqual(observed["A11"], 0.0)
        self.assertEqual(observed["B11"], 0.0)
        self.assertTrue(all(value == 0.0 for value in causal_level.values()))
        self.assertTrue(all(value == 0.0 for value in causal_delta.values()))
        self.assertFalse(shared["use_causal_innovation_diagnostics"])
        self.assertFalse(shared["use_causal_delta_innovation_diagnostics"])
        self.assertEqual(
            manifest["variants"]["A01"]["hyperparameters"]["pattern_score_mode"],
            "contextual_tail_probability",
        )
        self.assertEqual(
            manifest["variants"]["A11"]["hyperparameters"]["pattern_score_mode"],
            "contextual_tail_probability",
        )

    @staticmethod
    def _diagnostics(distribution):
        scale = None
        if distribution != "mse":
            scale = {
                "count": 10,
                "finite_count": 10,
                "nonfinite_count": 0,
                "min": 0.5,
                "max": 1.5,
                "mean": 1.0,
                "std": 0.2,
                "lower_bound": 0.001,
                "upper_bound": 100.0,
                "lower_bound_count": 0,
                "upper_bound_count": 0,
                "lower_bound_fraction": 0.0,
                "upper_bound_fraction": 0.0,
            }
        calls = []
        for index, phase in enumerate(("calibration", "test")):
            calls.append(
                {
                    "call_index": index,
                    "phase": phase,
                    "input_length": 10,
                    "batch_count": 1,
                    "window_count": 5,
                    "elapsed_seconds": 0.1,
                    "score": {
                        "count": 10,
                        "finite_count": 10,
                        "nonfinite_count": 0,
                        "min": 0.1,
                        "max": 1.0,
                        "mean": 0.4,
                    },
                    "scale": scale,
                }
            )
        return {
            "schema_version": 1,
            "model": "PatternAD",
            "distribution": distribution,
            "score_mode": "raw" if distribution == "mse" else "nll",
            "training": {
                "fit_seconds": 1.0,
                "training_seconds": 0.8,
                "scorer_fit_seconds": 0.01,
                "epochs_requested": 2,
                "epochs_completed": 2,
                "best_epoch": 2,
                "best_validation_loss": 0.3,
                "stopped_early": False,
                "parameter_count": 100,
                "optimization_train_points": 50,
                "validation_points": 10,
                "epoch_history": [
                    {
                        "epoch": 1,
                        "train_loss": 0.6,
                        "validation_loss": 0.5,
                        "learning_rate": 0.001,
                        "elapsed_seconds": 0.4,
                    },
                    {
                        "epoch": 2,
                        "train_loss": 0.4,
                        "validation_loss": 0.3,
                        "learning_rate": 0.0005,
                        "elapsed_seconds": 0.4,
                    },
                ],
            },
            "score_calls": calls,
        }

    def test_runner_validates_mse_and_gaussian_diagnostics(self):
        mse = self._diagnostics("mse")
        gaussian = self._diagnostics("gaussian")
        self.assertEqual(
            _validate_model_diagnostics(
                mse, {"reconstruction_distribution": "mse"}, 0.01
            )["distribution"],
            "mse",
        )
        self.assertEqual(
            _validate_model_diagnostics(
                gaussian,
                {
                    "reconstruction_distribution": "gaussian",
                    "pattern_score_mode": "nll",
                },
                0.01,
            )["distribution"],
            "gaussian",
        )
        gaussian["score_calls"][0]["scale"]["upper_bound_count"] = 1
        gaussian["score_calls"][0]["scale"]["upper_bound_fraction"] = 0.1
        with self.assertRaisesRegex(RuntimeError, "exceeds the frozen limit"):
            _validate_model_diagnostics(
                gaussian,
                {
                    "reconstruction_distribution": "gaussian",
                    "pattern_score_mode": "nll",
                },
                0.01,
            )

    def test_contextual_tail_requires_disjoint_reference_provenance(self):
        diagnostics = self._diagnostics("gaussian")
        diagnostics["score_mode"] = "contextual_tail_probability"
        diagnostics["training"].update(
            {
                "scorer_reference_points": 10,
                "fit_partition": {
                    "reference_source": "disjoint_temporal_normal_holdout",
                    "optimization_points": 80,
                    "validation_points": 10,
                    "reference_points": 10,
                    "inter_partition_gap_points": 23,
                    "validation_fraction": 0.1,
                    "reference_fraction": 0.1,
                },
            }
        )
        diagnostics["score_calibration"] = {
            "reference_source": "disjoint_temporal_normal_holdout",
            "reference_points": 10,
            "global_count": 10,
            "bin_counts": [5, 5],
            "minimum_bin_size": 128,
            "shrinkage": 128.0,
        }
        hyperparameters = {
            "seq_len": 24,
            "reconstruction_distribution": "gaussian",
            "pattern_score_mode": "contextual_tail_probability",
            "reconstruction_validation_fraction": 0.1,
            "pattern_score_reference_fraction": 0.1,
            "pattern_score_contextual_calibration_min_bin_size": 128,
            "pattern_score_contextual_calibration_shrinkage": 128.0,
        }
        self.assertEqual(
            _validate_model_diagnostics(diagnostics, hyperparameters, 0.01)[
                "score_calibration"
            ]["global_count"],
            10,
        )
        diagnostics.pop("score_calibration")
        with self.assertRaisesRegex(RuntimeError, "missing ECDF provenance"):
            _validate_model_diagnostics(diagnostics, hyperparameters, 0.01)

    def test_detailed_artifact_round_trips_model_diagnostics(self):
        diagnostics = self._diagnostics("gaussian")
        row = {
            "strategy_args": json.dumps(
                {
                    "evaluation_protocol": "train_calibration",
                    "calibration_fraction": 0.2,
                    "seed": 2021,
                }
            ),
            "typical_anomaly_ratio": "1.0",
            "auc_pr": "0.5",
            "model_diagnostics": json.dumps(diagnostics, allow_nan=False),
            "log_info": "",
        }
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        payload = stream.getvalue().encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            archive_path = attempt / "result.csv.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                member = tarfile.TarInfo("result.csv")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            artifact, loaded = _validate_artifact(
                attempt,
                {
                    "evaluation_protocol": "train_calibration",
                    "calibration_fraction": 0.2,
                    "anomaly_ratios": [1.0],
                    "max_scale_boundary_fraction": 0.01,
                },
                ["auc_pr"],
                2021,
                {
                    "reconstruction_distribution": "gaussian",
                    "pattern_score_mode": "nll",
                },
            )
            self.assertEqual(artifact, archive_path)
            self.assertEqual(loaded["score_calls"][1]["phase"], "test")

    def test_summary_flattens_machine_readable_run_diagnostics(self):
        flattened = _flatten_run_diagnostics(
            {
                "runner_wall_seconds": 2.0,
                "benchmark_log": "benchmark.log",
                "model_diagnostics": self._diagnostics("gaussian"),
            }
        )
        self.assertEqual(flattened["epochs_completed"], 2)
        self.assertEqual(flattened["test_scale_mean"], 1.0)
        self.assertEqual(flattened["benchmark_log"], "benchmark.log")

    def test_resume_rejects_completed_attempt_without_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            seed_dir = Path(directory)
            attempt = seed_dir / "attempt_001"
            attempt.mkdir()
            (attempt / "result.csv.tar.gz").write_bytes(b"placeholder")
            (attempt / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "config_hash": "frozen",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "lacks validated"):
                _completed_attempt(seed_dir, "frozen")

    def test_runner_allocates_the_next_attempt_on_python38(self):
        with tempfile.TemporaryDirectory() as directory:
            seed_dir = Path(directory)
            (seed_dir / "attempt_001").mkdir()
            (seed_dir / "attempt_invalid").mkdir()
            (seed_dir / "unrelated_999").mkdir()

            self.assertEqual(
                _next_attempt_dir(seed_dir), seed_dir / "attempt_002"
            )

    def test_runner_preserves_physical_gpu_ids_and_calibration_fraction(self):
        manifest = {
            "benchmark": {
                "script": "scripts/run_benchmark.py",
                "config_path": "strict.json",
                "model_name": "PatternAD.PatternAD",
                "evaluation_protocol": "train_calibration",
                "anomaly_ratios": [1.0],
                "calibration_fraction": 0.2,
                "eval_backend": "sequential",
                "num_workers": 1,
                "num_cpus": 3,
                "timeout_seconds": 60,
                "aggregate_type": "mean",
            }
        }
        command = _build_command(
            "python",
            manifest,
            {"data_name": "data.csv", "text_name": "text.csv"},
            {"seq_len": 24},
            2021,
            Path("result"),
            [2, 5],
        )

        gpu_index = command.index("--gpus")
        self.assertEqual(command[gpu_index + 1 :], ["2", "5"])
        strategy_args = json.loads(command[command.index("--strategy-args") + 1])
        self.assertEqual(strategy_args["calibration_fraction"], 0.2)

    def test_summary_rejects_a_missing_planned_identity(self):
        plan = {
            "plan_hash": "plan",
            "expected_identities": [
                {
                    "dataset_id": "Weather",
                    "variant": "A00",
                    "seed": 2021,
                    "config_hash": "a",
                },
                {
                    "dataset_id": "Weather",
                    "variant": "A11",
                    "seed": 2021,
                    "config_hash": "b",
                },
            ],
        }
        rows = [
            {
                "dataset_id": "Weather",
                "variant": "A00",
                "seed": 2021,
                "config_hash": "a",
                "plan_hash": "plan",
            }
        ]
        with self.assertRaisesRegex(ValueError, "selective summary"):
            _verify_complete_plan(rows, plan)


if __name__ == "__main__":
    unittest.main()
