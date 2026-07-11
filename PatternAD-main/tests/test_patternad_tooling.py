import json
import tempfile
import unittest
from pathlib import Path

from scripts.patternad.run_factorial_ablation import _build_command, _next_attempt_dir
from scripts.patternad.summarize_factorial import _verify_complete_plan


class PatternADToolingTest(unittest.TestCase):
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
