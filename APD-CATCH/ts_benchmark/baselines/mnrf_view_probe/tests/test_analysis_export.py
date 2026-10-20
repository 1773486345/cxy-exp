import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts.export_mnrf_view_probe_analysis import export
from scripts.validate_mnrf_view_probe import validate
from ts_benchmark.baselines.mnrf_view_probe import MNRFViewProbe


def _snapshot(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def test_full_source_analysis_export_is_read_only_and_bounded(tmp_path):
    rng = np.random.default_rng(20261013)
    n, train = 520, 430
    frame = pd.DataFrame(rng.normal(size=(n, 2)).astype(np.float32), columns=("x", "y"),
                         index=pd.date_range("2026-01-01", periods=n, freq="min"))
    frame["label"] = np.r_[np.zeros(train, dtype=np.int64), np.arange(n - train) % 2]
    run = tmp_path / "full"
    MNRFViewProbe(seq_len=32, pattern_len=32, batch_size=32, stage1_epochs=1,
                  stage2_epochs=1, patience=1, device="cpu", output_dir=str(run),
                  mode="full_gmm", gmm_components=1, gmm_fit_max_windows=8,
                  artifact_policy="full", task_name="SYN.csv").fit(
                      frame, train, frame["label"].to_numpy()[train:])
    assert validate(run)["status"] == "complete"
    before = _snapshot(run)
    analysis = tmp_path / "analysis"
    result = export(run, analysis, already_validated=True)
    assert result["status"] == "complete"
    assert json.loads((analysis / "analysis_complete.json").read_text())["status"] == "complete"
    assert json.loads((analysis / "metric_recheck.json").read_text())["status"] == "complete"
    assert _snapshot(run) == before
    with pytest.raises(FileExistsError):
        export(run, analysis, already_validated=True)
