from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.decomp_gate_v0_data import (
    ANOMALY_TYPES,
    TEST_LENGTH,
    TRAIN_CORE_LENGTH,
    TRAIN_LENGTH,
    TRAIN_SEEDS,
    VALIDATION_LENGTH,
    fixed_generator_parameters,
    generate_test_baseline,
    generate_training_series,
    inject_anomaly,
    precompute_anomaly_events,
    split_training_validation,
)
from scripts.analysis.run_decomp_gate_v0 import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    _gate_decision,
    bootstrap_fusion_delta,
    create_run_directory,
)
from scripts.analysis.finalize_decomp_gate_v0 import finalize
from scripts.analysis.decomp_gate_v0_runtime import atomic_write_json


class DecompGateV0DataTest(unittest.TestCase):
    def test_three_training_seeds_are_reproducible(self):
        self.assertEqual(len(TRAIN_SEEDS), 3)
        for seed in TRAIN_SEEDS:
            first = generate_training_series(seed)
            second = generate_training_series(seed)
            np.testing.assert_array_equal(first.frame.to_numpy(), second.frame.to_numpy())
            self.assertEqual(first.baseline_hash, second.baseline_hash)

    def test_training_validation_are_normal_and_pre_registered_lengths(self):
        train = generate_training_series(TRAIN_SEEDS[0])
        core, validation = split_training_validation(train)
        self.assertEqual(len(train.frame), TRAIN_LENGTH)
        self.assertEqual(len(core), TRAIN_CORE_LENGTH)
        self.assertEqual(len(validation), VALIDATION_LENGTH)
        self.assertFalse(np.isnan(core.to_numpy()).any())
        self.assertFalse(np.isnan(validation.to_numpy()).any())

    def test_anomaly_labels_match_only_pre_registered_event_support(self):
        seed = TRAIN_SEEDS[0]
        training = generate_training_series(seed)
        baseline = generate_test_baseline(seed)
        events = precompute_anomaly_events(seed)
        train_std = training.frame.to_numpy().std(axis=0)
        for anomaly_type in ANOMALY_TYPES:
            injected, labels = inject_anomaly(baseline, train_std, anomaly_type, events, seed)
            self.assertEqual(len(injected), TEST_LENGTH)
            self.assertEqual(len(labels), TEST_LENGTH)
            self.assertGreaterEqual(int(labels.sum()), 10)
            self.assertEqual(labels.dtype, np.int64)
            self.assertTrue(set(np.unique(labels)).issubset({0, 1}))
            self.assertTrue((labels == 1).any())
            if anomaly_type == "spike":
                expected = np.asarray(events["spike"]["positions"])
            else:
                event = events[anomaly_type]
                expected = np.arange(event["start"], event["start"] + event["length"])
            np.testing.assert_array_equal(np.flatnonzero(labels), expected)
            # No unlabeled timestamp may be altered by an injection.
            changed = np.any(injected.to_numpy() != baseline.frame.to_numpy(), axis=1)
            self.assertFalse(np.any(changed & (labels == 0)))

    def test_all_anomaly_types_share_one_normal_test_baseline(self):
        seed = TRAIN_SEEDS[1]
        baseline = generate_test_baseline(seed)
        events = precompute_anomaly_events(seed)
        train_std = generate_training_series(seed).frame.to_numpy().std(axis=0)
        hashes = []
        for anomaly_type in ANOMALY_TYPES:
            inject_anomaly(baseline, train_std, anomaly_type, events, seed)
            hashes.append(baseline.baseline_hash)
        self.assertEqual(len(set(hashes)), 1)

    def test_event_locations_are_deterministic_legal_and_spikes_are_spaced(self):
        events = precompute_anomaly_events(TRAIN_SEEDS[2])
        self.assertEqual(events, precompute_anomaly_events(TRAIN_SEEDS[2]))
        for name in ("level_shift", "slope_change", "variance_increase", "periodic_amplitude", "periodic_phase"):
            start = events[name]["start"]
            length = events[name]["length"]
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(start + length, TEST_LENGTH)
        positions = np.asarray(events["spike"]["positions"])
        self.assertEqual(len(positions), 12)
        self.assertTrue(np.all(np.diff(positions) >= 48))

    def test_bootstrap_uses_seed_category_units_and_run_ids_never_overwrite(self):
        deltas = np.linspace(-0.1, 0.1, 18)
        result = bootstrap_fusion_delta(deltas, seed=20260717, samples=100)
        self.assertEqual(result["unit_count"], 18)
        self.assertEqual(result["resamples"], 100)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_run_directory(root, "fixed-run")
            with self.assertRaises(FileExistsError):
                create_run_directory(root, "fixed-run")

    def test_generator_parameters_are_fixed_not_metric_derived(self):
        parameters = fixed_generator_parameters()
        self.assertEqual(parameters["shared_noise_std"], 0.05)
        self.assertEqual(parameters["independent_noise_std"], 0.03)
        self.assertEqual(parameters["anomaly_specs"]["level_shift"]["multiplier"], 2.5)
        self.assertEqual(parameters["anomaly_specs"]["spike"]["count"], 12)
        self.assertEqual(parameters["anomaly_specs"]["periodic_phase"]["phase_shift"], np.pi / 2)

    def test_gate_empty_or_incomplete_is_not_evaluable(self):
        empty = _gate_decision(pd.DataFrame(), None)
        self.assertEqual(empty["decision"], "GATE_NOT_EVALUABLE")
        self.assertTrue(all(value == "NOT_EVALUABLE" for value in empty["conditions"].values()))
        rows = self._complete_branch_rows()[:-1]
        incomplete = _gate_decision(pd.DataFrame(rows), None)
        self.assertEqual(incomplete["decision"], "GATE_NOT_EVALUABLE")

    def test_complete_units_require_complete_bootstrap_before_gate_status(self):
        branch = pd.DataFrame(self._complete_branch_rows())
        not_bootstrapped = _gate_decision(branch, None)
        self.assertEqual(not_bootstrapped["decision"], "GATE_NOT_EVALUABLE")
        bootstrap = {
            "unit_count": 18,
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_SAMPLES,
            "one_sided_95_lower_bound": -0.02,
        }
        complete = _gate_decision(branch, bootstrap)
        self.assertIn(complete["decision"], {"GATE_PASSED", "GATE_FAILED"})

    def test_finalizer_rejects_incomplete_shards_without_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = root / "study"
            (study / "shards").mkdir(parents=True)
            atomic_write_json(study / "study_manifest.json", {"study_id": "study"})
            result = finalize(
                "study",
                {20260717: "a", 20260718: "b", 20260719: "c"},
                root,
            )
            self.assertEqual(result["decision"], "GATE_NOT_EVALUABLE")
            self.assertFalse((study / "bootstrap.json").exists())
            with (study / "gate_decision.json").open() as handle:
                self.assertEqual(json.load(handle)["decision"], "GATE_NOT_EVALUABLE")

    @staticmethod
    def _complete_branch_rows():
        return [
            {
                "seed": seed,
                "anomaly_type": anomaly_type,
                "anomaly_points": 12,
                "slow_fast_anomaly_spearman": 0.0,
                "slow_auc_pr": 0.8,
                "fast_auc_pr": 0.8,
                "original_top_k_out_component_top_k_in": 1,
            }
            for seed in TRAIN_SEEDS
            for anomaly_type in ANOMALY_TYPES
        ]


if __name__ == "__main__":
    unittest.main()
