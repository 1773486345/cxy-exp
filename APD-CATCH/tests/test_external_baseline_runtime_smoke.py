"""Opt-in runtime smoke tests for the frozen external Baseline scripts.

Set ``APD_EXTERNAL_BASELINE_SMOKE_ENV=catch_env`` or ``tods_legacy`` before
running this file.  The tests use only in-memory regular two-channel sequences,
never create a benchmark archive, and reduce only the smoke-test epoch count.
"""

from __future__ import annotations

import gc
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_SCRIPT_DIR = ROOT / "scripts" / "data_preparation" / "external_validation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXTERNAL_SCRIPT_DIR))

from external_baseline_assets import BASELINE_SPECS  # noqa: E402
from ts_benchmark.models.model_loader import get_model_info  # noqa: E402


SMOKE_ENV = os.environ.get("APD_EXTERNAL_BASELINE_SMOKE_ENV")
SMOKE_MODELS = {
    name.strip() for name in os.environ.get("APD_EXTERNAL_BASELINE_SMOKE_MODELS", "").split(",") if name.strip()
}
VALID_ENVIRONMENTS = {"catch_env", "tods_legacy"}


def smoke_hyper_params(spec: dict) -> dict:
    """Keep the frozen template apart from test-only runtime bounds."""
    params = dict(spec["model_hyper_params"])
    if "num_epochs" in params:
        params["num_epochs"] = 1
    elif spec["paper_name"] == "TFAD":
        params["num_epochs"] = 1
    if "batch_size" in params and spec["paper_name"] != "DualTF":
        params["batch_size"] = min(params["batch_size"], 2)
    return params


def smoke_length(spec: dict) -> int:
    if spec["paper_name"] == "TFAD":
        return 1000
    if spec["paper_name"] == "DCdetector":
        return 600
    if spec["paper_name"] in {"ModernTCN", "iTransformer", "TimesNet", "PatchTST", "DLinear", "NLinear"}:
        return 500
    return 300


def smoke_series(length: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2020-01-01", periods=length, freq="min")
    phase = np.linspace(0.0, 12.0, length, dtype=np.float64)
    train = pd.DataFrame({"sensor_a": np.sin(phase), "sensor_b": np.cos(phase)}, index=index)
    test = pd.DataFrame({"sensor_a": np.sin(phase + 0.1), "sensor_b": np.cos(phase + 0.1)}, index=index)
    labels = pd.DataFrame({"label": np.zeros(length, dtype=np.int64)}, index=index)
    return train, test, labels


def build_model(spec: dict, params: dict):
    config = {"model_name": spec["framework_model_name"]}
    if spec["adapter"]:
        config["adapter"] = spec["adapter"]
    info = get_model_info(config)
    return info["model_factory"](**params) if isinstance(info, dict) else info(**params)


@unittest.skipUnless(SMOKE_ENV in VALID_ENVIRONMENTS, "set APD_EXTERNAL_BASELINE_SMOKE_ENV to run opt-in runtime smoke tests")
class TestExternalBaselineRuntimeSmoke(unittest.TestCase):
    def test_detect_fit_and_detect_score(self):
        specs = [
            spec for spec in BASELINE_SPECS
            if spec["environment_name"] == SMOKE_ENV
            and (not SMOKE_MODELS or spec["paper_name"] in SMOKE_MODELS)
        ]
        self.assertTrue(specs)
        for spec in specs:
            with self.subTest(model=spec["paper_name"], environment=SMOKE_ENV):
                params = smoke_hyper_params(spec)
                print(f"[smoke] {spec['paper_name']}: construct", flush=True)
                model = build_model(spec, params)
                if spec["paper_name"] == "DualTF":
                    self.assertEqual(model.config.batch_size, 8)
                if spec["paper_name"] == "TFAD":
                    self.assertNotIn("batch_size", params)
                    self.assertEqual(model.config.batch_size, 256)
                if spec["paper_name"] == "AutoEncoder":
                    self.assertNotIn("batch_size", params)

                train, test, labels = smoke_series(smoke_length(spec))
                print(f"[smoke] {spec['paper_name']}: fit", flush=True)
                model.detect_fit(train, labels)
                print(f"[smoke] {spec['paper_name']}: score", flush=True)
                score = model.detect_score(test)
                if isinstance(score, tuple):
                    score = score[0]
                score = np.asarray(score).reshape(-1)
                self.assertGreater(score.size, 0)
                self.assertTrue(np.isfinite(score).all())
                print(f"[smoke] {spec['paper_name']}: passed", flush=True)

                del model, train, test, labels, score
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass


if __name__ == "__main__":
    unittest.main()
