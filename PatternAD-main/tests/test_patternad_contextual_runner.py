import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.patternad.run_contextual_factorial import (
    DEFAULT_FACTORIAL_MANIFEST,
    DEFAULT_SYNTHETIC_CONFIG,
    _artifact_status,
    _canonical_hash,
    _file_sha256,
    _load_json,
    _run_logged,
    _select_grid,
    _validate_completed_identity,
    main,
)


class PatternADContextualRunnerTest(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = {
            "seed_group": "development",
            "generator_seeds": None,
            "model_seeds": None,
            "variant": None,
            "allow_locked": False,
            "run_name": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_default_development_grid_is_crossed_and_complete(self):
        synthetic = _load_json(DEFAULT_SYNTHETIC_CONFIG)
        factorial = _load_json(DEFAULT_FACTORIAL_MANIFEST)
        generators, models, variants, complete = _select_grid(
            self._args(), synthetic, factorial
        )
        self.assertEqual(generators, list(range(3101, 3111)))
        self.assertEqual(models, [2021, 2022, 2023])
        self.assertEqual(variants, ["A00", "A10", "A01", "A11"])
        self.assertTrue(complete)

    def test_confirmation_requires_explicit_complete_acknowledgement(self):
        synthetic = _load_json(DEFAULT_SYNTHETIC_CONFIG)
        factorial = _load_json(DEFAULT_FACTORIAL_MANIFEST)
        with self.assertRaisesRegex(ValueError, "Locked confirmation requires"):
            _select_grid(
                self._args(seed_group="confirmation"), synthetic, factorial
            )

        generators = list(range(3111, 3121))
        models = [2021, 2022, 2023, 2024, 2025]
        selected = _select_grid(
            self._args(
                seed_group="confirmation",
                generator_seeds=generators,
                model_seeds=models,
                variant=["A00", "A10", "A01", "A11"],
                allow_locked=True,
                run_name="locked_p1",
            ),
            synthetic,
            factorial,
        )
        self.assertEqual(selected, (generators, models, ["A00", "A10", "A01", "A11"], True))

    def test_artifact_validation_checks_config_and_generator_provenance(self):
        config = _load_json(DEFAULT_SYNTHETIC_CONFIG)
        config["seed"] = 3102
        with tempfile.TemporaryDirectory() as temporary:
            artifact_dir = Path(temporary)
            required = (
                "same_deviation_different_context",
                "slow_drift_vs_abrupt_shift",
                "dependency_break",
                "context_ood",
            )
            (artifact_dir / "resolved_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            for mechanism in required:
                (artifact_dir / f"{mechanism}.npz").touch()
                (artifact_dir / f"{mechanism}.metadata.json").write_text(
                    "{}", encoding="utf-8"
                )
            (artifact_dir / "suite_manifest.json").write_text(
                json.dumps(
                    {
                        "config_hash": _canonical_hash(config),
                        "source_hashes": {"generator": "expected"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                _artifact_status(artifact_dir, config, "expected"),
                (True, "complete"),
            )
            ready, reason = _artifact_status(artifact_dir, config, "changed")
            self.assertFalse(ready)
            self.assertIn("different generator", reason)

    def test_completed_identity_is_fail_closed_on_provenance(self):
        identity = {
            "variant": "A11",
            "generator_seed": 3101,
            "model_seed": 2021,
            "config_hash": "identity-hash",
            "synthetic_config_hash": "synthetic-hash",
            "hyperparameters_hash": _canonical_hash({"train_mask_seed": 2021}),
        }
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary)
            (result_dir / "scores").mkdir()
            (result_dir / "identity_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "plan_hash": "plan-hash",
                        "config_hash": "identity-hash",
                    }
                ),
                encoding="utf-8",
            )
            mechanisms = []
            for mechanism in (
                "same_deviation_different_context",
                "slow_drift_vs_abrupt_shift",
                "dependency_break",
                "context_ood",
            ):
                mechanism_path = result_dir / "scores" / f"{mechanism}.npz"
                mechanism_path.write_bytes(mechanism.encode("ascii"))
                mechanisms.append(
                    {
                        "mechanism": mechanism,
                        "score_sha256": _file_sha256(mechanism_path),
                    }
                )
            (result_dir / "contextual_evaluation.json").write_text(
                json.dumps(
                    {
                        "method": "A11_seed_2021",
                        "config_hash": "synthetic-hash",
                        "mechanisms": mechanisms,
                    }
                ),
                encoding="utf-8",
            )
            score_metadata = {
                "variant": "A11",
                "seed": 2021,
                "synthetic_config_hash": "synthetic-hash",
                "factorial_manifest_hash": "factorial-hash",
                "hyperparameters": {"train_mask_seed": 2021},
            }
            score_path = result_dir / "scores/score_run_metadata.json"
            score_path.write_text(json.dumps(score_metadata), encoding="utf-8")
            metadata_path = result_dir / "identity_metadata.json"
            identity_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            identity_metadata.update(
                {
                    "variant": "A11",
                    "generator_seed": 3101,
                    "model_seed": 2021,
                    "output_hashes": {
                        "contextual_evaluation.json": _file_sha256(
                            result_dir / "contextual_evaluation.json"
                        )
                    },
                }
            )
            metadata_path.write_text(json.dumps(identity_metadata), encoding="utf-8")
            self.assertTrue(
                _validate_completed_identity(
                    result_dir, identity, "plan-hash", "factorial-hash"
                )
            )
            score_metadata["seed"] = 2022
            score_path.write_text(json.dumps(score_metadata), encoding="utf-8")
            self.assertFalse(
                _validate_completed_identity(
                    result_dir, identity, "plan-hash", "factorial-hash"
                )
            )

    def test_dry_run_does_not_create_artifacts_or_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "results"
            artifact_root = root / "artifacts"
            with contextlib.redirect_stdout(io.StringIO()):
                return_code = main(
                    [
                        "--run-name",
                        "tiny_dry_run",
                        "--generator-seeds",
                        "3102",
                        "--model-seeds",
                        "2021",
                        "--variant",
                        "A11",
                        "--artifact-root",
                        str(artifact_root),
                        "--output-root",
                        str(output_root),
                        "--dry-run",
                    ]
                )
            self.assertEqual(return_code, 0)
            self.assertFalse(output_root.exists())
            self.assertFalse(artifact_root.exists())

    def test_logged_timeout_reaps_the_child_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_path = root / "child.pid"
            command = [
                sys.executable,
                "-c",
                (
                    "import os,time,pathlib; "
                    f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
                    "time.sleep(60)"
                ),
            ]
            with self.assertRaises(subprocess.TimeoutExpired):
                _run_logged(command, root / "child.log", os.environ, 0.5)
            pid = int(pid_path.read_text(encoding="utf-8"))
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
