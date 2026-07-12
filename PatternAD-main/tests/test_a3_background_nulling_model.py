"""Regression coverage for A3-N1's all-channel background-nulling model."""

import copy
import unittest
from pathlib import Path

import numpy as np

from scripts.a3.generate_trigger_response_contract import _load_json, generate_suite
from scripts.a3.run_background_nulling_route_graph import (
    _normal_optimization_values,
    _normal_windows,
)
from ts_benchmark.baselines.A3TriggerResponse import A3BackgroundNullingRouteGraph


CONTRACT = Path("config/a3/trigger_response_route_identifiability_v2.json")
EXPERIMENT = Path("config/a3/background_nulling_n1_development_v1.json")


def _small_route_contract():
    config = copy.deepcopy(_load_json(CONTRACT))
    config.update({"burn_in": 16, "train_length": 320, "background_length": 160, "history_length": 8, "horizon_length": 6})
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


class A3BackgroundNullingRouteGraphTest(unittest.TestCase):
    def test_all_channel_factor_and_past_only_trigger_pipeline(self):
        contract = _small_route_contract()
        experiment = _load_json(EXPERIMENT)
        suite = generate_suite(contract)
        history = contract["history_length"]
        horizon = contract["horizon_length"]
        model = A3BackgroundNullingRouteGraph(
            dimensions=contract["dimensions"],
            history_length=history,
            horizon_length=horizon,
            token_energy_threshold=experiment["token_extractor"]["token_energy_threshold"],
            cue_length=experiment["trigger_extractor"]["cue_length"],
            minimum_trigger_amplitude=experiment["trigger_extractor"]["minimum_amplitude"],
            trigger_linear_tolerance=experiment["trigger_extractor"]["linear_tolerance"],
            hidden_size=8,
            learning_rate=3e-3,
            epochs=5,
            patience=2,
            batch_size=8,
            outer_alpha=0.05,
            device="cpu",
        ).fit(
            _normal_optimization_values(suite),
            _normal_windows(suite, "optimization", history, horizon),
            _normal_windows(suite, "validation", history, horizon),
            _normal_windows(suite, "reference", history, horizon),
            _normal_windows(suite, "outer_calibration", history, horizon),
            seed=8401,
        )
        windows = _normal_windows(suite, "reference", history, horizon)[:3]
        altered = windows.copy()
        altered[:, history:] *= -4.0
        self.assertEqual(model.background_factor().shape, (contract["dimensions"],))
        self.assertEqual(model.projected_future(windows).shape, (3, horizon, contract["dimensions"]))
        np.testing.assert_array_equal(model.event_pre_state(windows), model.event_pre_state(altered))
        scores = model.score_windows(windows)
        self.assertTrue(np.isfinite(scores["null_route_tail"]).all())
        self.assertEqual(scores["node_surprisal"].shape, (3, contract["dimensions"]))


if __name__ == "__main__":
    unittest.main()
