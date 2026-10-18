"""Read-only evaluation helpers for probe artifacts and common protocols."""

import hashlib

import numpy as np

from ts_benchmark.evaluation.metrics.affiliation.generics import convert_vector_to_events
from ts_benchmark.evaluation.metrics.affiliation.metrics import pr_from_events


def strict_integer_indices(value, name, *, upper_bound=None, require_sorted=False):
    """Validate endpoint/split coordinates before any dtype conversion."""
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    info = np.iinfo(array.dtype)
    if np.any(array < 0) or np.any(array > info.max) or (
            array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} contains invalid negative or overflowing coordinates")
    if upper_bound is not None and np.any(array >= int(upper_bound)):
        raise ValueError(f"{name} contains out-of-range coordinates")
    converted = np.asarray(array, dtype=np.int64)
    if len(np.unique(converted)) != len(converted):
        raise ValueError(f"{name} contains duplicate coordinates")
    if require_sorted and len(converted) > 1 and np.any(np.diff(converted) <= 0):
        raise ValueError(f"{name} must be sorted and strictly increasing")
    return converted


def calibrated_quantile(values, quantile):
    """Reproduce a threshold using the calibration score array's stored dtype."""
    scores = np.asarray(values)
    if scores.ndim != 1 or scores.dtype.kind not in "fiu" or len(scores) == 0:
        raise ValueError("calibration scores must be a non-empty one-dimensional numeric array")
    if not np.isfinite(scores).all() or not 0.0 < float(quantile) < 1.0:
        raise ValueError("calibration scores or quantile are invalid")
    return float(np.quantile(scores, float(quantile), method="linear"))


def _hash_array(value):
    array = np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def common_index_intersection(*index_sets):
    """Return sorted shared integer coordinates and coverage diagnostics."""
    if not index_sets:
        return np.empty(0, dtype=np.int64), {"hash": _hash_array(np.empty(0, dtype=np.int64))}
    normalized = []
    for position, item in enumerate(index_sets):
        array = strict_integer_indices(item, f"index set {position}")
        normalized.append(array)
    common = normalized[0]
    for item in normalized[1:]:
        common = np.intersect1d(common, item, assume_unique=False)
    common = np.unique(common)
    return common, {
        "hash": _hash_array(common),
        "input_counts": [int(len(item)) for item in normalized],
        "intersection_count": int(len(common)),
        "excluded_counts": [int(len(item) - len(common)) for item in normalized],
    }


def validate_common_splits(train_indices, validation_indices, calibration_indices, total_length):
    """Validate predeclared train/validation/calibration coordinates.

    The common evaluator accepts these coordinates as an audit record; it
    never creates a second 80/20 split or uses labels to alter them.
    """
    arrays = []
    for name, item in zip(("train", "validation", "calibration"),
                          (train_indices, validation_indices, calibration_indices)):
        arrays.append(strict_integer_indices(item, f"{name} split", upper_bound=total_length,
                                             require_sorted=True))
    names = ("train", "validation", "calibration")
    seen = set()
    result = {}
    for name, values in zip(names, arrays):
        overlap = seen.intersection(map(int, values))
        if overlap:
            raise ValueError(f"common splits overlap at coordinates: {sorted(overlap)[:5]}")
        seen.update(map(int, values))
        result[name] = {"count": int(len(values)), "sha256": _hash_array(values)}
    return result


def _valid_segments(valid_mask, boundary_ids=None):
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    boundaries = None if boundary_ids is None else np.asarray(boundary_ids).reshape(-1)
    if boundaries is not None and len(boundaries) != len(valid):
        raise ValueError("boundary_ids and valid_mask must have equal length")
    segments = []
    start = None
    for index, is_valid in enumerate(valid):
        same_entity = index == 0 or boundaries is None or boundaries[index] == boundaries[index - 1]
        if is_valid and same_entity:
            if start is None:
                start = index
        else:
            if start is not None:
                segments.append((start, index))
                start = None
            if is_valid:
                start = index
    if start is not None:
        segments.append((start, len(valid)))
    return segments


def affiliation_f_segment_macro(actual, predicted, valid_mask=None, boundary_ids=None):
    """Compute segment-wise affiliation metrics without joining invalid gaps.

    Undefined segments are retained in the diagnostic list and excluded from
    the macro mean. No undefined case is coerced to zero or one.
    """
    actual = np.asarray(actual).reshape(-1).astype(np.int64)
    predicted = np.asarray(predicted).reshape(-1).astype(np.int64)
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have equal length")
    valid = np.ones(len(actual), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
    if len(valid) != len(actual):
        raise ValueError("valid_mask must match actual length")
    records = []
    defined = []
    for left, right in _valid_segments(valid, boundary_ids):
        label = actual[left:right]
        prediction = predicted[left:right]
        events_gt = convert_vector_to_events(label)
        events_pred = convert_vector_to_events(prediction)
        record = {"start": int(left), "end": int(right),
                  "ground_truth_event_count": int(len(events_gt)),
                  "predicted_event_count": int(len(events_pred)),
                  "precision": None, "recall": None, "f": None,
                  "defined": False, "undefined_reason": None}
        if not events_gt:
            record["undefined_reason"] = "no_ground_truth_event"
        else:
            try:
                result = pr_from_events(events_pred, events_gt, (0, right - left))
                precision = float(result["precision"])
                recall = float(result["recall"])
                record["precision"] = precision if np.isfinite(precision) else None
                record["recall"] = recall if np.isfinite(recall) else None
                if not np.isfinite(precision) or not np.isfinite(recall):
                    record["undefined_reason"] = "non_finite_precision_or_recall"
                elif precision + recall == 0:
                    # A finite zero/zero result is a defined zero, distinct
                    # from a metric that is mathematically undefined.
                    record.update({"f": 0.0, "defined": True,
                                   "undefined_reason": None})
                    defined.append(0.0)
                else:
                    value = 2 * precision * recall / (precision + recall)
                    if not np.isfinite(value):
                        record["undefined_reason"] = "non_finite_f_score"
                    else:
                        record.update({"f": float(value), "defined": True,
                                       "undefined_reason": None})
                        defined.append(float(value))
            except (ValueError, ZeroDivisionError) as error:
                record["undefined_reason"] = str(error)
        records.append(record)
    return {
        "affiliation_f_segment_macro": None if not defined else float(np.mean(defined)),
        "defined_segment_count": len(defined), "total_segment_count": len(records),
        "segments": records,
        "normal_false_positive_points": int(np.sum((actual == 0) & (predicted > 0) & valid)),
    }


def aligned_comparison(actual, score, indices, threshold, valid_mask=None, boundary_ids=None):
    """Build an auditable aligned score/prediction payload for two methods."""
    actual = np.asarray(actual).reshape(-1)
    score = np.asarray(score).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(score) != len(indices) or np.any(indices < 0) or np.any(indices >= len(actual)):
        raise ValueError("score indices are not valid coordinates in the actual sequence")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("score indices contain duplicates")
    if not np.isfinite(score).all():
        raise ValueError("score contains NaN or Inf")
    prediction = (score > float(threshold)).astype(np.int64)
    local_valid = np.ones(len(indices), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if len(local_valid) != len(indices):
        raise ValueError("score valid mask length mismatch")
    return {"indices": indices, "scores": score, "predictions": prediction,
            "actual": actual[indices], "valid_mask": local_valid,
            "coverage": float(np.sum(local_valid) / max(len(actual), 1)),
            "excluded_anomaly_points": int(np.sum((actual > 0)[~np.isin(np.arange(len(actual)), indices)]))}
