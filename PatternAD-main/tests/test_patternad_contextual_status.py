import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.patternad.status_contextual_factorial import (
    collect_status,
    render_status,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ContextualFactorialStatusTest(unittest.TestCase):
    def test_reports_completed_running_and_missing_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities = []
            for model_seed in (2021, 2022, 2023):
                relative = f"A00/generator_seed_3101/model_seed_{model_seed}"
                identities.append(
                    {
                        "variant": "A00",
                        "generator_seed": 3101,
                        "model_seed": model_seed,
                        "result_dir": relative,
                    }
                )
            _write_json(
                root / "run_plan.json",
                {
                    "run_name": "test_run",
                    "phase": "development",
                    "plan_hash": "hash",
                    "expected_identities": identities,
                },
            )
            _write_json(
                root / identities[0]["result_dir"] / "identity_metadata.json",
                {
                    "status": "completed",
                    "started_at": "2026-07-11T16:00:00+00:00",
                    "completed_at": "2026-07-11T16:00:20+00:00",
                    "runner_wall_seconds": 20.0,
                },
            )
            running_dir = root / identities[1]["result_dir"]
            _write_json(
                running_dir / "identity_metadata.json",
                {
                    "status": "running",
                    "started_at": "2026-07-11T16:00:30+00:00",
                },
            )
            (running_dir / "run.log").write_text(
                "epoch 1\nepoch 2\n", encoding="utf-8"
            )
            snapshot = collect_status(
                root,
                tail_lines=1,
                now=datetime(2026, 7, 11, 16, 1, tzinfo=timezone.utc),
            )
            self.assertEqual(snapshot["completed"], 1)
            self.assertEqual(snapshot["counts"], {"completed": 1, "running": 1, "missing": 1})
            self.assertEqual(snapshot["running"][0]["log_tail"], ["epoch 2"])
            rendered = render_status(snapshot)
            self.assertIn("Progress: 1/3 (33.3%)", rendered)
            self.assertIn("Current [2/3]: A00 generator=3101 model=2022", rendered)
            self.assertIn("missing=1", rendered)


if __name__ == "__main__":
    unittest.main()
