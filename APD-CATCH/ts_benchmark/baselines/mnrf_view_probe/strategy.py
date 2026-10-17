"""Index-aware strategy for the independent MNRF-ViewProbe experiment.

The legacy benchmark strategy expects a dense score vector and therefore may
shift endpoint scores to the beginning of a test sequence.  This adapter is
deliberately separate: it keeps the probe's frozen endpoint coordinates and
returns both local and original coordinates without padding or interpolation.
"""

import time

import numpy as np
import pandas as pd


class IndexedViewProbeStrategy:
    """Run a probe on explicit train/test frames with auditable coordinates."""

    def __init__(self, model):
        self.model = model

    def run(self, train_frame, test_frame, test_labels=None):
        train = train_frame if isinstance(train_frame, pd.DataFrame) else pd.DataFrame(train_frame)
        test = test_frame if isinstance(test_frame, pd.DataFrame) else pd.DataFrame(test_frame)
        if len(train) == 0 or len(test) == 0:
            raise ValueError("train and test frames must both be non-empty")
        if tuple(map(str, train.columns)) != tuple(map(str, test.columns)):
            raise ValueError("train and test frame columns must match")
        combined = pd.concat([train, test], axis=0, ignore_index=True)
        labels = None if test_labels is None else np.asarray(test_labels).reshape(-1)
        if labels is not None and len(labels) != len(test):
            raise ValueError("test_labels must have one value per test row")
        fit_start = time.perf_counter()
        self.model.fit(combined, len(train), labels=labels)
        fit_elapsed = time.perf_counter() - fit_start
        metadata = self.model.score_with_metadata()
        endpoints = np.asarray(metadata["time_indices"])
        local = np.asarray(metadata["local_indices"])
        expected = len(train) + local
        if not np.array_equal(endpoints, expected):
            raise ValueError("probe endpoint coordinates are not aligned to the test frame")
        result = {name: metadata[name] for name in metadata}
        result["fit_time_seconds"] = float(fit_elapsed)
        metrics = None if getattr(self.model, "output_dir", None) is None else self.model.output_dir / "results" / "metrics.json"
        if metrics is not None and metrics.is_file():
            import json
            saved_metrics = json.loads(metrics.read_text())
            # fit() records head fitting separately from test scoring; use
            # those protocol timings instead of charging scoring to training.
            result["fit_time_seconds"] = float(saved_metrics.get("fit_time_seconds", fit_elapsed))
            result["inference_time_seconds"] = float(saved_metrics.get("inference_time_seconds", 0.0))
        result["coordinate_protocol"] = "explicit_test_endpoints"
        return result

