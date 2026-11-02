import numpy as np
import pandas as pd
from pathlib import Path

from scripts.compare_msl_boundary_score_objects import (
    _metric_row, common_endpoint_coordinates, derive_score_views,
)
from scripts.mnrf_msl_sequence_boundaries import (
    EXPECTED_FEATURE_SHA256, FEATURE_COLUMNS, _frame_feature_hash, annotate_frame,
)
from ts_benchmark.baselines.mnrf_lite_v2.training import EndWindowDataset
from ts_benchmark.baselines.mnrf_view_probe import MNRFViewProbe


def _synthetic_mapping():
    channels = ["A", "B"]
    return {
        "schema": "msl-raw-mapping-v1",
        "channels": channels,
        "channel_row_counts": {name: {"train": 50, "test": 30} for name in channels},
        "train_rows": 100,
        "test_rows": 60,
        "columns": 55,
        "local_feature_names": FEATURE_COLUMNS,
        "expected_feature_sha256": EXPECTED_FEATURE_SHA256,
    }


def test_sequence_mapping_is_non_feature_and_preserves_all_rows():
    rng = np.random.default_rng(19)
    n, train = 160, 100
    original = pd.DataFrame(rng.normal(size=(n, 55)).astype(np.float32),
                            columns=FEATURE_COLUMNS,
                            index=pd.date_range("2026-01-01", periods=n, freq="min"))
    original["label"] = np.zeros(n, dtype=np.int8)
    mapping = _synthetic_mapping()
    mapping["expected_feature_sha256"] = _frame_feature_hash(original, FEATURE_COLUMNS)
    mapped = annotate_frame(original, train, mapping, require_registry=False)
    assert len(mapped) == len(original)
    assert list(mapped.columns[:-2]) == FEATURE_COLUMNS
    assert np.array_equal(mapped[FEATURE_COLUMNS].to_numpy(), original[FEATURE_COLUMNS].to_numpy())
    assert np.array_equal(mapped["label"].to_numpy(), original["label"].to_numpy())
    assert mapped["entity_id"].value_counts().to_dict() == {
        "train:A": 50, "train:B": 50, "test:A": 30, "test:B": 30,
    }


def test_real_probe_protocol_never_builds_windows_across_sequence_boundaries():
    rng = np.random.default_rng(7)
    n, train = 160, 100
    frame = pd.DataFrame(rng.normal(size=(n, 55)).astype(np.float32),
                         columns=FEATURE_COLUMNS,
                         index=pd.date_range("2026-01-01", periods=n, freq="min"))
    frame["entity_id"] = (["train:A"] * 50 + ["train:B"] * 50
                          + ["test:A"] * 30 + ["test:B"] * 30)
    probe = MNRFViewProbe(seq_len=8, pattern_len=8, batch_size=8,
                          device="cpu", mode="full_gmm", task_name="MSL.csv")
    values, test_endpoints, _ = probe._protocol(frame, train)
    assert values.shape == (n, 55)
    assert len(probe.segments) == 4
    assert len(probe.ranges["segment_ranges"]["train"]) == 2
    assert len(probe.ranges["segment_ranges"]["validation"]) == 1
    assert len(probe.ranges["segment_ranges"]["calibration"]) == 1
    assert len(test_endpoints) == 46
    for name in ("train", "validation", "calibration"):
        dataset = EndWindowDataset(values, *probe.ranges[name], 8,
                                   probe.ranges["segment_ranges"][name])
        for endpoint in dataset.endpoints:
            assert len(set(frame.entity_id.iloc[endpoint - 7:endpoint + 1])) == 1
    for endpoint in test_endpoints:
        assert len(set(frame.entity_id.iloc[endpoint - 7:endpoint + 1])) == 1


def test_same_nll_derives_full_and_target_scores_with_independent_thresholds():
    rng = np.random.default_rng(33)
    validation = rng.normal(size=(80, 55)).astype(np.float32)
    calibration_nll = rng.normal(size=(100, 55)).astype(np.float32)
    test = rng.normal(size=(25, 55)).astype(np.float32)
    center = np.median(validation, axis=0).astype(np.float32)
    scale = np.maximum(1.4826 * np.median(np.abs(validation - center), axis=0), 0.001).astype(np.float32)
    calibration_z = (calibration_nll - center) / scale
    calibration = {
        "center": center, "scale": scale,
        "threshold": np.asarray(np.quantile(np.max(calibration_z, axis=1), 0.995)),
        "quantile": np.asarray(0.995), "score_mad_floor": 0.001,
    }
    views = derive_score_views(validation, calibration_nll, test, calibration)
    test_z = (test - center) / scale
    assert np.allclose(views["all_55_max"]["scores"], np.max(test_z, axis=1))
    assert np.allclose(views["col_0_only"]["scores"], test_z[:, 0])
    assert views["all_55_max"]["threshold"] == float(calibration["threshold"])
    assert views["col_0_only"]["threshold"] == float(np.quantile(calibration_z[:, 0], 0.995))
    assert views["all_55_max"]["threshold"] != views["col_0_only"]["threshold"]


def test_common_endpoint_intersection_is_exact_and_sorted():
    old = np.array([192, 193, 194, 195, 196], dtype=np.int64)
    new = np.array([194, 195, 196, 220], dtype=np.int64)
    assert np.array_equal(common_endpoint_coordinates(old, new), [194, 195, 196])
    try:
        common_endpoint_coordinates(np.array([2.0, 3.0]), new)
    except ValueError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("non-integer endpoint coordinates must be rejected")


def test_affiliation_metrics_keep_unscored_gaps_and_sequence_edges_separate():
    full_labels = np.array([0, 1, 1, 1, 0, 0, 1, 1], dtype=np.int8)
    local_coords = np.array([0, 1, 3, 4, 5, 6, 7], dtype=np.int64)
    common_labels = full_labels[local_coords]
    scores = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    boundaries = np.array(["A"] * 4 + ["B"] * 4, dtype=object)
    row = _metric_row("MSL", "col_0_only", Path("synthetic-run"),
                      np.arange(len(local_coords)), common_labels, scores, 0.5,
                      local_coords, full_labels, boundaries)
    assert row["affiliation_total_segments"] == 3
    assert row["common_valid_points"] == len(local_coords)
