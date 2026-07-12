"""Minimal regression coverage for the current Direction A3 model path."""

import copy
import unittest
from pathlib import Path

import numpy as np

from scripts.a3.audit_counterfactual_effect_graph import audit_counterfactual_effect_graph_inputs
from scripts.a3.audit_observable_graph_grammar import audit_graph_grammar_inputs
from scripts.a3.audit_trigger_response_contract import audit_suite
from scripts.a3.generate_trigger_response_contract import DEFAULT_CONFIG, _load_json, generate_suite
from ts_benchmark.baselines.A3TriggerResponse import (
    A3CounterfactualEffectGraphGrammar,
    A3ObservableGraphGrammar,
    extract_trigger_states,
    response_graph_tokens,
)


G2_CONFIG = Path("config/a3/observable_graph_grammar_g2_v1.json")
G3_CONFIG = Path("config/a3/counterfactual_effect_graph_g3_v1.json")


def _small_contract():
    config = copy.deepcopy(_load_json(DEFAULT_CONFIG))
    config.update(
        {
            "burn_in": 16,
            "train_length": 320,
            "background_length": 160,
            "history_length": 8,
            "horizon_length": 6,
        }
    )
    config["normal_splits"].update(
        {
            "optimization_length": 96,
            "validation_length": 48,
            "reference_length": 96,
            "outer_calibration_length": 41,
            "guard_length": 13,
        }
    )
    config["normal_process"]["regime_segment_length"] = 16
    config["episodes"].update(
        {
            "pairs_per_mode": 2,
            "normal_transition_sources_per_regime": {
                "optimization": 4,
                "validation": 4,
                "reference": 4,
                "outer_calibration": 4,
            },
            "cue_length": 6,
            "response_onsets": [1, 3],
            "response_ramp_length": 2,
        }
    )
    return config


def _normal_event_bank(suite, split):
    return np.stack([entry["values"] for entry in suite["normal_event_banks"][split]])


def _ordinary_normal_windows(suite, split, history, horizon):
    start, end = suite["normal_split_ranges"][split]
    values = np.asarray(suite["train_values"], dtype=np.float32)
    starts = np.arange(start, end - history - horizon + 1, dtype=np.int64)
    return np.stack([values[index : index + history + horizon] for index in starts])


def _normal_windows(suite, split, history, horizon):
    return np.concatenate(
        (_ordinary_normal_windows(suite, split, history, horizon), _normal_event_bank(suite, split)),
        axis=0,
    )


class A3ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = _small_contract()
        cls.experiment = copy.deepcopy(_load_json(G2_CONFIG))
        cls.experiment["device"] = "cpu"
        cls.experiment["model"].update(
            {"hidden_size": 8, "epochs": 5, "patience": 2, "batch_size": 8}
        )
        cls.suite = generate_suite(cls.contract)

    def test_raw_contract_and_fixed_graph_state_are_valid(self):
        contract_audit = audit_suite(self.contract, self.suite)
        graph_audit = audit_graph_grammar_inputs(self.contract, self.experiment, self.suite)
        self.assertTrue(contract_audit["passed"], contract_audit["violations"])
        self.assertTrue(graph_audit["passed"], graph_audit["violations"])
        self.assertEqual(graph_audit["metrics"]["trigger_only_primary_accuracy"], 0.5)
        self.assertEqual(graph_audit["metrics"]["response_mode_only_primary_accuracy"], 0.5)

    def test_observable_tokens_and_past_only_joint_grammar_pipeline(self):
        model = A3ObservableGraphGrammar(
            dimensions=self.contract["dimensions"],
            history_length=self.contract["history_length"],
            horizon_length=self.contract["horizon_length"],
            token_energy_threshold=self.contract["episodes"]["token_energy_threshold"],
            cue_length=self.experiment["trigger_extractor"]["cue_length"],
            minimum_trigger_amplitude=self.experiment["trigger_extractor"]["minimum_amplitude"],
            trigger_linear_tolerance=self.experiment["trigger_extractor"]["linear_tolerance"],
            hidden_size=8,
            learning_rate=3e-3,
            epochs=5,
            patience=2,
            batch_size=8,
            outer_alpha=0.1,
            device="cpu",
        ).fit(
            _normal_event_bank(self.suite, "optimization"),
            _normal_event_bank(self.suite, "validation"),
            _normal_event_bank(self.suite, "reference"),
            _normal_event_bank(self.suite, "outer_calibration"),
            seed=8201,
        )
        windows = _normal_event_bank(self.suite, "reference")[:3]
        history = self.contract["history_length"]
        trigger_states = extract_trigger_states(
            windows[:, :history], cue_length=6, minimum_amplitude=1.0, linear_tolerance=5e-6
        )
        tokens = response_graph_tokens(
            windows[:, history:], self.contract["episodes"]["token_energy_threshold"]
        )
        self.assertEqual(trigger_states.shape, (3, 2))
        self.assertEqual(tokens.shape, (3, self.contract["dimensions"]))
        altered = windows.copy()
        altered[:, history:] *= -4.0
        np.testing.assert_array_equal(
            model.event_pre_state(windows), model.event_pre_state(altered)
        )
        scores = model.score_windows(windows)
        self.assertTrue(np.isfinite(scores["joint_graph_tail"]).all())
        self.assertEqual(scores["node_surprisal"].shape, (3, self.contract["dimensions"]))

    def test_counterfactual_effect_graph_uses_only_event_pre_for_its_baseline(self):
        experiment = copy.deepcopy(_load_json(G3_CONFIG))
        experiment["device"] = "cpu"
        experiment["model"].update({"hidden_size": 8, "epochs": 5, "patience": 2, "batch_size": 8})
        audit = audit_counterfactual_effect_graph_inputs(self.contract, experiment, self.suite)
        self.assertTrue(audit["passed"], audit["violations"])
        history = self.contract["history_length"]
        horizon = self.contract["horizon_length"]
        model = A3CounterfactualEffectGraphGrammar(
            dimensions=self.contract["dimensions"],
            history_length=history,
            horizon_length=horizon,
            effect_token_energy_threshold=experiment["effect_extractor"]["token_energy_threshold"],
            cue_length=experiment["trigger_extractor"]["cue_length"],
            minimum_trigger_amplitude=experiment["trigger_extractor"]["minimum_amplitude"],
            trigger_linear_tolerance=experiment["trigger_extractor"]["linear_tolerance"],
            ridge_penalty=experiment["counterfactual"]["ridge_penalty"],
            hidden_size=8,
            learning_rate=3e-3,
            epochs=5,
            patience=2,
            batch_size=8,
            outer_alpha=0.1,
            device="cpu",
        ).fit(
            _ordinary_normal_windows(self.suite, "optimization", history, horizon),
            _normal_event_bank(self.suite, "optimization"),
            _normal_windows(self.suite, "optimization", history, horizon),
            _normal_windows(self.suite, "validation", history, horizon),
            _normal_windows(self.suite, "reference", history, horizon),
            _normal_windows(self.suite, "outer_calibration", history, horizon),
            seed=8301,
        )
        windows = _normal_event_bank(self.suite, "reference")[:3]
        altered = windows.copy()
        altered[:, history:] *= -4.0
        np.testing.assert_allclose(
            model.counterfactual_baseline(windows),
            model.counterfactual_baseline(altered),
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_array_equal(model.event_pre_state(windows), model.event_pre_state(altered))
        scores = model.score_windows(windows)
        self.assertTrue(np.isfinite(scores["effect_graph_tail"]).all())
        self.assertEqual(scores["node_surprisal"].shape, (3, self.contract["dimensions"]))


if __name__ == "__main__":
    unittest.main()
