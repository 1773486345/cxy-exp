import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.patternad.evaluate_contextual_mechanisms import (
    conformal_upper_threshold,
    evaluate_scores,
)
from scripts.patternad.generate_contextual_synthetic import (
    DEFAULT_CONFIG,
    MECHANISM_ORDER,
    _load_json,
    generate_suite,
    write_suite,
)
from ts_benchmark.data.utils import read_data


def _small_config():
    config = copy.deepcopy(_load_json(DEFAULT_CONFIG))
    config.update(
        {
            "burn_in": 32,
            "train_length": 256,
            "test_length": 256,
            "regime_segment_length": 32,
        }
    )
    mechanisms = config["mechanisms"]
    mechanisms["same_deviation_different_context"].update(
        {"event_length": 4, "event_offset": 14, "pair_count": 3}
    )
    mechanisms["slow_drift_vs_abrupt_shift"].update(
        {
            "gradual_length": 16,
            "abrupt_length": 4,
            "event_offset": 8,
            "pair_count": 2,
        }
    )
    mechanisms["dependency_break"].update(
        {
            "event_length": 12,
            "event_offset": 8,
            "event_count": 3,
            "circular_offsets": [3, 7],
        }
    )
    mechanisms["context_ood"].update(
        {
            "event_length": 16,
            "event_offset": 8,
            "event_count": 2,
            "recovery_length": 2,
        }
    )
    config["evaluation"].update(
        {
            "calibration_fraction": 0.25,
            "calibration_gap": 7,
            "score_window_length": 8,
            "target_fpr": 0.05,
        }
    )
    return config


class PatternADSyntheticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _small_config()
        cls.artifacts = generate_suite(cls.config)

    def test_generation_is_deterministic_and_train_is_clean(self):
        repeated = generate_suite(self.config)
        self.assertEqual(set(self.artifacts), set(MECHANISM_ORDER))
        train_length = self.config["train_length"]
        for mechanism in MECHANISM_ORDER:
            first = self.artifacts[mechanism]
            second = repeated[mechanism]
            np.testing.assert_array_equal(first["values"], second["values"])
            np.testing.assert_array_equal(first["labels"], second["labels"])
            self.assertFalse(first["labels"][:train_length].any())
            self.assertTrue(first["labels"][train_length:].any())
            guard = self.config["evaluation"]["score_window_length"] - 1
            for event in first["events"]:
                start = int(event.get("injection_start", event["start"]))
                end = int(event.get("injection_end", event["end"]))
                guarded = first["fpr_eligible"][
                    max(train_length, start - guard) : min(
                        len(first["fpr_eligible"]), end + guard
                    )
                ]
                self.assertFalse(guarded.any())

    def test_mechanism_contracts_are_real_and_auditable(self):
        same = self.artifacts["same_deviation_different_context"]
        deviation = np.asarray(
            self.config["mechanisms"]["same_deviation_different_context"][
                "deviation"
            ],
            dtype=np.float32,
        )
        for event in same["events"]:
            observed = (
                same["values"][event["start"] : event["end"]]
                - same["clean_values"][event["start"] : event["end"]]
            )
            np.testing.assert_allclose(
                observed, np.broadcast_to(deviation, observed.shape), atol=1e-6
            )
        event_regimes = {
            int(same["regime"][event["start"]]) for event in same["events"]
        }
        self.assertEqual(event_regimes, {0, 1})
        pair_labels = {}
        for event in same["events"]:
            pair_labels.setdefault(event["pair_id"], []).append(event["label"])
        self.assertTrue(
            all(sorted(labels) == [False, True] for labels in pair_labels.values())
        )

        broken = self.artifacts["dependency_break"]
        channels = self.config["mechanisms"]["dependency_break"][
            "shifted_channels"
        ]
        changed = False
        for event in broken["events"]:
            event_slice = slice(event["start"], event["end"])
            for channel in channels:
                before = broken["clean_values"][event_slice, channel]
                after = broken["values"][event_slice, channel]
                np.testing.assert_allclose(np.sort(before), np.sort(after), atol=0)
                changed = changed or not np.array_equal(before, after)
        self.assertTrue(changed)

        drift = self.artifacts["slow_drift_vs_abrupt_shift"]
        self.assertTrue(any(not event["label"] for event in drift["events"]))
        shift = np.asarray(
            self.config["mechanisms"]["slow_drift_vs_abrupt_shift"][
                "endpoint_shift"
            ],
            dtype=np.float32,
        )
        for ordering in drift["orderings"]:
            events = {event["event_id"]: event for event in drift["events"]}
            higher = events[ordering["higher_event"]]
            lower = events[ordering["lower_event"]]
            higher_deviation = (
                drift["values"][higher["start"] : higher["end"]]
                - drift["clean_values"][higher["start"] : higher["end"]]
            )
            lower_deviation = (
                drift["values"][lower["start"] : lower["end"]]
                - drift["clean_values"][lower["start"] : lower["end"]]
            )
            np.testing.assert_allclose(
                higher_deviation, np.broadcast_to(shift, higher_deviation.shape)
            )
            np.testing.assert_allclose(
                higher_deviation, lower_deviation, atol=1e-6
            )
        ood = self.artifacts["context_ood"]
        self.assertTrue(np.any(ood["regime"] == 2))

    def test_benchmark_fixture_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "suite"
            manifest_path = write_suite(
                self.config, self.artifacts, output_dir, register_benchmark=False
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["registered_with_benchmark"])
            self.assertEqual(len(manifest["entries"]), 4)
            self.assertEqual(
                json.loads((output_dir / "resolved_config.json").read_text()),
                self.config,
            )
            entry = manifest["entries"][0]
            frame = read_data(output_dir / "benchmark" / entry["benchmark_data_name"])
            self.assertEqual(frame.shape, (512, 6))
            self.assertEqual(frame.columns[-1], "label")

    def test_threshold_is_independent_of_test_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            artifact_dir = temporary / "suite"
            score_dir_a = temporary / "scores_a"
            score_dir_b = temporary / "scores_b"
            score_dir_a.mkdir()
            score_dir_b.mkdir()
            write_suite(
                self.config, self.artifacts, artifact_dir, register_benchmark=False
            )
            split = self.config["train_length"]
            for mechanism, artifact in self.artifacts.items():
                score = artifact["oracle_context_score"].copy()
                np.savez_compressed(score_dir_a / f"{mechanism}.npz", score=score)
                changed_test = score.copy()
                changed_test[split:] += np.linspace(100.0, 200.0, len(score) - split)
                np.savez_compressed(
                    score_dir_b / f"{mechanism}.npz", score=changed_test
                )
            first, rows_a, _ = evaluate_scores(
                self.config, artifact_dir, score_dir_a, "score", "first"
            )
            second, rows_b, _ = evaluate_scores(
                self.config, artifact_dir, score_dir_b, "score", "second"
            )
            np.testing.assert_allclose(
                [row["calibration_threshold"] for row in rows_a],
                [row["calibration_threshold"] for row in rows_b],
            )
            self.assertFalse(
                first["threshold_provenance"]["test_scores_used_for_threshold"]
            )
            self.assertFalse(
                second["threshold_provenance"]["test_scores_used_for_threshold"]
            )
            self.assertIn("scope", first["gates"])
            self.assertTrue(
                first["gates"]["raw_magnitude_negative_control"]["pass"]
            )
            self.assertIn(
                "dependency_break",
                first["gates"]["mechanism_ap_over_prevalence"],
            )
            self.assertEqual(
                first["mechanism_ap_diagnostics"]["context_ood"]["interpretation"],
                "negative_control_for_conditional_only_score",
            )
            dependency = next(
                row
                for row in rows_a
                if row["mechanism"] == "dependency_break"
            )
            self.assertGreater(
                dependency["average_precision"], dependency["test_prevalence"]
            )

    def test_conformal_cutoff_is_conservative_for_tiny_calibration(self):
        self.assertTrue(
            np.isinf(conformal_upper_threshold(np.arange(9, dtype=float), 0.01))
        )


if __name__ == "__main__":
    unittest.main()
