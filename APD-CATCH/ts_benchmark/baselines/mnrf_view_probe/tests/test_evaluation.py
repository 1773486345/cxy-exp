import numpy as np

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
