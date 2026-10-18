import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.run_mnrf_catch_common import (
    _WindowDataset, _filter_test_endpoints, _validate_window_boundaries,
)


def test_catch_windows_are_lazy_and_batch_bounded():
    values = np.arange(400 * 2, dtype=np.float32).reshape(400, 2)
    endpoints = np.arange(31, 100, dtype=np.int64)
    dataset = _WindowDataset(values, endpoints, 32)
    assert len(dataset) == len(endpoints)
    assert not hasattr(dataset, "windows")
    loader = DataLoader(dataset, batch_size=7, shuffle=False)
    batches = list(loader)
    assert max(batch[0].shape[0] for batch in batches) == 7
    np.testing.assert_array_equal(batches[0][0][0].numpy(), values[:32])
    np.testing.assert_array_equal(batches[-1][0][-1].numpy(), values[endpoints[-1] - 31:endpoints[-1] + 1])


def test_disjoint_windows_are_not_rejected_by_union_gaps():
    # Only the interval helper's valid partition form is tested here: windows
    # [0,4) and [8,12) are disjoint despite the gap in their row union.
    from scripts.run_mnrf_catch_common import _intervals_overlap
    assert not _intervals_overlap(np.array([[0, 4]]), np.array([[8, 12]]))


def test_gap_fixture_has_28_cold_start_safe_test_windows():
    n, seq_len, test_start = 520, 32, 430
    times = pd.date_range("2026-01-01", periods=n, freq="min")
    times = times.to_series().reset_index(drop=True)
    times.iloc[480:] += pd.Timedelta(minutes=60)
    frame = pd.DataFrame({"timestamp": times, "x": np.zeros(n), "y": np.zeros(n)})
    train = np.arange(seq_len - 1, 200, dtype=np.int64)
    validation = np.arange(231, 300, dtype=np.int64)
    calibration = np.arange(331, test_start, dtype=np.int64)
    test = _filter_test_endpoints(frame, np.arange(test_start + seq_len - 1, n), seq_len, train)
    np.testing.assert_array_equal(test, np.r_[np.arange(461, 480), np.arange(511, 520)])
    intervals = _validate_window_boundaries(
        frame, {"train": train, "validation": validation,
                "calibration": calibration, "test": test}, seq_len, test_start,
    )
    assert len(test) == 28
    assert not np.any((test >= 480) & (test <= 510))
    assert not np.any((intervals["test"][:, 0] < 480) & (intervals["test"][:, 1] > 480))


def test_entity_session_boundaries_and_entity_clock_resets():
    frame = pd.DataFrame({
        "timestamp": list(pd.date_range("2026-01-01", periods=40, freq="min"))
                     + list(pd.date_range("2026-01-01", periods=40, freq="min")),
        "entity_id": ["a"] * 40 + ["b"] * 40,
        "session_id": ["s1"] * 80,
        "x": np.zeros(80), "y": np.zeros(80),
    })
    endpoints = {"train": np.array([3, 43]), "validation": np.array([11, 51]),
                 "calibration": np.array([19, 59]), "test": np.array([27, 67])}
    # Each entity has its own clock. No window crosses the entity boundary.
    _validate_window_boundaries(frame, endpoints, 4, 24)

    session_frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=40, freq="min"),
        "entity_id": ["a"] * 40,
        "session_id": ["before"] * 21 + ["after"] * 19,
        "x": np.zeros(40), "y": np.zeros(40),
    })
    with pytest.raises(ValueError, match="session_id"):
        _validate_window_boundaries(
            session_frame, {"train": np.array([15]), "validation": np.array([23]),
                            "calibration": np.array([31]), "test": np.array([39])}, 4, 30,
        )
