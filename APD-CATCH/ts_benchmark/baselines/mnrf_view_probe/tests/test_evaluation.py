import numpy as np
import json
import importlib.util
from pathlib import Path

from ts_benchmark.baselines.mnrf_view_probe.evaluation import (
    affiliation_f_segment_macro,
    common_index_intersection,
    strict_integer_indices,
    validate_common_splits,
)
from ts_benchmark.evaluation.metrics.classification_metrics_label import (
    affiliation_f, affiliation_precision, affiliation_recall,
)
from scripts.summarize_mnrf_branch_ablation import _affiliation_for_bundle, affiliation_only
import scripts.summarize_mnrf_branch_ablation as branch_summary
import ts_benchmark.baselines.mnrf_view_probe.evaluation as probe_evaluation


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


def _affiliation_bundle(actual, predicted, valid=None, global_indices=None, segment_ids=None):
    actual = np.asarray(actual, dtype=np.int8)
    valid = np.ones(len(actual), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    local = np.flatnonzero(valid).astype(np.int64)
    global_indices = (np.arange(len(actual), dtype=np.int64) if global_indices is None
                      else np.asarray(global_indices, dtype=np.int64))
    segment_ids = (np.zeros(len(actual), dtype=np.int64) if segment_ids is None
                   else np.asarray(segment_ids, dtype=np.int64))
    return ({"valid_mask": valid, "local_indices": local,
             "predictions": np.asarray(predicted, dtype=np.int8)[local]},
            actual, global_indices, segment_ids)


def test_affiliation_single_continuous_interval_matches_original_core():
    actual = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int8)
    predicted = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    bundle, labels, global_ids, segments = _affiliation_bundle(actual, predicted)
    original_predictions = bundle["predictions"].copy()
    original_threshold = np.asarray(0.75)
    bundle["threshold"] = original_threshold.copy()
    result = _affiliation_for_bundle(bundle, labels, global_ids, segments)
    assert result["calculation_scope"] == "full_single_continuous_valid_interval"
    assert result["original_algorithm_crosscheck"] == "passed"
    assert result["affiliation_precision"] == affiliation_precision(actual, predicted)
    assert result["affiliation_recall"] == affiliation_recall(actual, predicted)
    assert result["affiliation_f"] == affiliation_f(actual, predicted)
    assert np.array_equal(bundle["predictions"], original_predictions)
    assert float(bundle["threshold"]) == 0.75


def test_affiliation_does_not_join_invalid_gap_coordinate_gap_or_entity_boundary():
    actual = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=np.int8)
    predicted = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 0], dtype=np.int8)
    valid = np.ones(len(actual), dtype=bool)
    valid[3] = False
    global_ids = np.array([0, 1, 2, 3, 8, 9, 10, 11, 12, 13])
    segments = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    bundle, labels, global_ids, segments = _affiliation_bundle(
        actual, predicted, valid, global_ids, segments)
    result = _affiliation_for_bundle(bundle, labels, global_ids, segments)
    assert result["calculation_scope"] == "valid_continuous_intervals"
    assert result["interval_count"] == 3
    assert [item["global_start"] for item in result["intervals"]] == [0, 8, 11]
    assert [item["segment_id"] for item in result["intervals"]] == [0, 1, 2]
    assert result["original_algorithm_crosscheck"] == "not_applicable_multiple_or_partial_intervals"


def test_affiliation_distinguishes_no_prediction_zero_recall_from_undefined_f():
    actual = np.array([0, 1, 1, 1, 0], dtype=np.int8)
    predicted = np.zeros(len(actual), dtype=np.int8)
    bundle, labels, global_ids, segments = _affiliation_bundle(actual, predicted)
    result = _affiliation_for_bundle(bundle, labels, global_ids, segments)
    assert result["affiliation_precision"] is None
    assert result["affiliation_recall"] == 0.0
    assert result["affiliation_f"] is None
    assert "precision_undefined_no_predicted_event" in result["undefined_reasons"]
    no_anomaly = np.zeros(5, dtype=np.int8)
    bundle, labels, global_ids, segments = _affiliation_bundle(no_anomaly, predicted)
    undefined = _affiliation_for_bundle(bundle, labels, global_ids, segments)
    assert undefined["affiliation_precision"] is None
    assert undefined["affiliation_recall"] is None
    assert undefined["affiliation_f"] is None
    assert "no_ground_truth_event" in undefined["undefined_reasons"]


def test_affiliation_zero_f_is_distinct_from_undefined(monkeypatch):
    zero_result = {"precision": 0.0, "recall": 0.0}
    monkeypatch.setattr(branch_summary, "pr_from_events", lambda *args: zero_result)
    monkeypatch.setattr(probe_evaluation, "pr_from_events", lambda *args: zero_result)
    actual = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int8)
    predicted = np.array([0, 1, 0, 0, 0, 0, 1, 0], dtype=np.int8)
    valid = np.array([True, True, True, True, False, True, True, True])
    segments = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    bundle, labels, global_ids, segments = _affiliation_bundle(
        actual, predicted, valid, segment_ids=segments)
    result = _affiliation_for_bundle(bundle, labels, global_ids, segments)
    assert result["affiliation_precision"] == 0.0
    assert result["affiliation_recall"] == 0.0
    assert result["affiliation_f"] == 0.0
    assert result["defined_interval_count"] == 2


def test_affiliation_only_retains_original_failed_cell_without_fabricating_values(tmp_path):
    plan = {"rows": [{"dataset": "CICIDS", "combination": "WITHOUT_E", "head_seed": 2021,
                      "status": "failed", "reason": "historical non-convergence"}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    rows = affiliation_only(plan_path)
    assert len(rows) == 1
    assert rows[0]["metric_status"] == "original_failed"
    assert rows[0]["affiliation_f"] is None
    assert rows[0]["undefined_or_failure_reason"] == "historical non-convergence"


def test_endpoint_validation_rejects_fractional_boolean_and_overflow_before_cast():
    for value in (np.array([1.9, 2.9]), np.array([True, False]),
                  np.array([[1, 2]], dtype=np.int64),
                  np.array([2**63], dtype=np.uint64)):
        try:
            strict_integer_indices(value, "endpoint")
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid endpoint accepted: {value!r}")


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
