import numpy as np
import json
import importlib.util
from pathlib import Path

from ts_benchmark.baselines.mnrf_view_probe.evaluation import (
    affiliation_f_segment_macro,
    common_index_intersection,
    validate_common_splits,
)


def test_common_intersection_is_coordinate_based_and_hashed():
    common, details = common_index_intersection([1, 3, 5], [0, 3, 4, 5])
    assert np.array_equal(common, np.array([3, 5]))
    assert details["intersection_count"] == 2
    assert details["excluded_counts"] == [1, 2]
    assert details["hash"]


def test_affiliation_keeps_invalid_gaps_and_entities_separate():
    actual = np.zeros(12, dtype=np.int64)
    actual[[1, 2, 8]] = 1
    predicted = np.zeros_like(actual)
    predicted[[1, 8]] = 1
    valid = np.ones(12, dtype=bool)
    valid[5:7] = False
    entities = np.array(["a"] * 6 + ["b"] * 6)
    result = affiliation_f_segment_macro(actual, predicted, valid, entities)
    assert result["total_segment_count"] == 2
    assert result["normal_false_positive_points"] == 0
    assert result["segments"][0]["start"] == 0
    assert result["segments"][1]["start"] == 7


def test_common_splits_are_explicit_and_disjoint():
    result = validate_common_splits([0, 1], [3], [5, 6], total_length=8)
    assert result["train"]["count"] == 2


def test_nonfinite_affiliation_values_are_null_and_undefined_is_distinct():
    actual = np.array([0, 1, 1, 0], dtype=np.int64)
    predicted = np.zeros_like(actual)
    result = affiliation_f_segment_macro(actual, predicted)
    record = result["segments"][0]
    assert record["defined"] is False
    assert record["f"] is None
    assert record["undefined_reason"] == "non_finite_precision_or_recall"
    json.dumps(result, allow_nan=False)


def test_common_split_rejects_float_coordinates():
    try:
        validate_common_splits(np.array([0.0, 1.0]), np.array([2]), np.array([3]), 4)
    except ValueError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("fractional split coordinates were accepted")


def test_unregistered_score_pair_is_reference_not_catch_common(tmp_path):
    script = Path(__file__).resolve().parents[4] / "scripts" / "evaluate_mnrf_view_probe_common.py"
    spec = importlib.util.spec_from_file_location("mnrf_common_eval", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probe = tmp_path / "probe.npz"
    catch = tmp_path / "catch.npz"
    labels = tmp_path / "labels.npy"
    np.savez(probe, scores=np.array([0.1, 0.9]), time_indices=np.array([1, 2]), threshold=np.array(0.5))
    np.savez(catch, scores=np.array([0.2, 0.8]), time_indices=np.array([1, 2]), threshold=np.array(0.5))
    np.save(labels, np.array([0, 0, 1], dtype=np.int64))
    result = module.evaluate(str(probe), str(catch), str(labels))
    assert result["protocol"] == "index_intersection_reference"
    json.dumps(result, allow_nan=False)
