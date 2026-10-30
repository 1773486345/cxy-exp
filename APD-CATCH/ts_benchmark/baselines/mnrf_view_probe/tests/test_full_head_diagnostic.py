import numpy as np
import pytest

from scripts.run_mnrf_full_head_diagnostic import (
    _aggregate, _check_equal, _diagnostic_output_paths, _effect_deltas,
)
from scripts.analyze_mnrf_full_loss import _distribution, _full_axis_context, _recall_at_fpr


def test_aggregate_standardizes_variables_before_max_or_mean_and_calibrates_separately():
    center = np.asarray([1.0, 10.0], dtype=np.float32)
    scale = np.asarray([2.0, 5.0], dtype=np.float32)
    calibration = np.asarray([[1.0, 15.0], [5.0, 10.0], [3.0, 20.0], [9.0, 10.0]], dtype=np.float32)
    test = np.asarray([[5.0, 10.0], [1.0, 25.0]], dtype=np.float32)
    max_score, max_threshold, max_pred = _aggregate(calibration, test, center, scale, "max", 0.75)
    mean_score, mean_threshold, mean_pred = _aggregate(calibration, test, center, scale, "mean", 0.75)
    np.testing.assert_allclose(max_score, [2.0, 3.0])
    np.testing.assert_allclose(mean_score, [1.0, 1.5])
    assert max_threshold != mean_threshold
    assert np.array_equal(max_pred, max_score > max_threshold)
    assert np.array_equal(mean_pred, mean_score > mean_threshold)


def test_single_variable_max_and_mean_are_identical():
    rng = np.random.RandomState(2021)
    calibration = rng.normal(size=(64, 1)).astype(np.float32)
    test = rng.normal(size=(17, 1)).astype(np.float32)
    center = np.asarray([0.1], dtype=np.float32)
    scale = np.asarray([1.0], dtype=np.float32)
    max_result = _aggregate(calibration, test, center, scale, "max")
    mean_result = _aggregate(calibration, test, center, scale, "mean")
    np.testing.assert_array_equal(max_result[0], mean_result[0])
    assert max_result[1] == mean_result[1]
    np.testing.assert_array_equal(max_result[2], mean_result[2])


@pytest.mark.parametrize("cal,test,center,scale", [
    (np.ones((2, 2)), np.ones((2, 2)), np.ones(1), np.ones(1)),
    (np.asarray([[np.nan, 1.0]]), np.ones((1, 2)), np.zeros(2), np.ones(2)),
    (np.ones((2, 2)), np.ones((2, 2)), np.zeros(2), np.asarray([1.0, 0.0])),
])
def test_invalid_full_head_inputs_are_rejected(cal, test, center, scale):
    with pytest.raises(ValueError):
        _aggregate(cal, test, center, scale, "max")


def test_saved_metric_replay_uses_strict_existing_tolerance():
    _check_equal(0.5, 0.50000005, "nearby")
    with pytest.raises(ValueError, match="differs"):
        _check_equal(0.5, 0.50001, "bad")


def test_effects_keep_valid_full_aggregation_when_diagonal_fit_failed():
    rows = {
        ("MSL", "full", "max"): {"average_precision": 0.1590, "roc_auc": 0.6024},
        ("MSL", "full", "mean"): {"average_precision": 0.1621, "roc_auc": 0.6124},
        ("MSL", "diag", "max"): None,
        ("MSL", "diag", "mean"): None,
    }
    effects = _effect_deltas(rows, "MSL")
    assert effects["aggregation_ap"] == "+0.00310/n/a"
    assert effects["aggregation_auc"] == "+0.01000/n/a"
    assert effects["covariance_ap"] == "n/a/n/a"
    assert effects["interaction_ap"] == "n/a"


def test_explicit_reuse_batch_uses_existing_score_tree_without_allocating_new_score_tree():
    name = "full-head-diagnostic-20261030T062555Z-3226097"
    analysis, scores, reuse_only = _diagnostic_output_paths(name, timestamp="fixed", pid=7)
    assert scores.name == name
    assert scores.is_dir()
    assert analysis.name.startswith("full-head-diagnostic-reuse-3226097-fixed-")
    assert not analysis.exists()
    assert reuse_only
    with pytest.raises((ValueError, FileNotFoundError)):
        _diagnostic_output_paths("../other", timestamp="fixed", pid=7)


def test_recall_at_fixed_fpr_is_posthoc_and_returns_undefined_for_single_class():
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.2, 0.3, 0.25, 0.5, 0.9])
    # At zero empirical FPR, thresholds can still recover the two highest
    # anomaly scores (0.5 and 0.9) above the largest normal score (0.3).
    assert _recall_at_fpr(labels, scores, 0.0) == pytest.approx(2.0 / 3.0)
    assert _recall_at_fpr(labels, scores, 0.5) >= _recall_at_fpr(labels, scores, 0.0)
    assert _recall_at_fpr(np.zeros(3, dtype=np.int8), np.arange(3), 0.1) is None


def test_calibration_context_respects_frozen_segment_boundary():
    labels = np.asarray([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)
    global_indices = np.arange(10, dtype=np.int64)
    endpoints = np.asarray([3, 5, 8], dtype=np.int64)
    context = _full_axis_context(labels, global_indices, [[0, 5], [5, 10]], endpoints, 4)
    np.testing.assert_array_equal(context[:, 1], [0, 0, 0])
    assert context[1, 2] == 1


def test_distribution_reports_empty_population_as_undefined():
    assert _distribution(np.asarray([]))["undefined_reason"] == "empty_or_nonfinite"
    assert _distribution(np.asarray([1.0, 2.0]))["q0.5"] == 1.5
