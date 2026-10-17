#!/usr/bin/env python
"""Evaluate a probe and an existing CATCH score file on common coordinates.

This is a read-only comparison adapter.  It never calls a CATCH trainer and
does not change the original CATCH scoring protocol.  Both methods must
provide their own endpoint coordinates; only their explicit intersection is
used for the common report.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ts_benchmark.baselines.mnrf_view_probe.evaluation import (
    _hash_array,
    affiliation_f_segment_macro,
    common_index_intersection,
    validate_common_splits,
)


def _read_npz(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {"scores", "time_indices"}
        if not required.issubset(payload.files):
            raise ValueError(f"{path} must contain scores and time_indices")
        result = {name: np.asarray(payload[name]) for name in payload.files}
    indices = np.asarray(result["time_indices"], dtype=np.int64).reshape(-1)
    scores = np.asarray(result["scores"], dtype=np.float64).reshape(-1)
    if len(indices) != len(scores) or len(np.unique(indices)) != len(indices):
        raise ValueError(f"{path} has invalid or duplicate endpoint coordinates")
    if np.any(indices < 0) or not np.isfinite(scores).all():
        raise ValueError(f"{path} has invalid endpoint or score values")
    result["time_indices"] = indices
    result["scores"] = scores
    return result


def _metric(actual, scores):
    actual = np.asarray(actual, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if np.unique(actual).size < 2:
        return {"auc_roc": None, "auc_roc_reason": "common coordinates contain one class",
                "auc_pr": float(average_precision_score(actual, scores))}
    return {"auc_roc": float(roc_auc_score(actual, scores)),
            "auc_pr": float(average_precision_score(actual, scores))}


def evaluate(probe_path, catch_path, labels_path, probe_threshold=None,
             catch_threshold=None, boundary_path=None, split_paths=None):
    probe = _read_npz(probe_path)
    catch = _read_npz(catch_path)
    labels = np.asarray(np.load(labels_path, allow_pickle=False)).reshape(-1).astype(np.int64)
    boundary_ids = None if boundary_path is None else np.asarray(np.load(boundary_path, allow_pickle=False)).reshape(-1)
    if boundary_ids is not None and len(boundary_ids) != len(labels):
        raise ValueError("boundary ids must have the same length as labels")
    common, intersection = common_index_intersection(probe["time_indices"], catch["time_indices"])
    if np.any(common >= len(labels)):
        raise ValueError("common score coordinates exceed the label sequence")
    probe_lookup = {int(index): pos for pos, index in enumerate(probe["time_indices"])}
    catch_lookup = {int(index): pos for pos, index in enumerate(catch["time_indices"])}
    probe_scores = np.asarray([probe["scores"][probe_lookup[int(index)]] for index in common])
    catch_scores = np.asarray([catch["scores"][catch_lookup[int(index)]] for index in common])
    actual = labels[common]
    if probe_threshold is None:
        probe_threshold = float(np.asarray(probe.get("threshold", np.nan)))
    if catch_threshold is None:
        catch_threshold = float(np.asarray(catch.get("threshold", np.nan)))
    if not np.isfinite(probe_threshold) or not np.isfinite(catch_threshold):
        raise ValueError("both methods need finite, independently calibrated thresholds")
    # Build full-coordinate predictions so invalid gaps/entities are retained
    # as gaps in the affiliation calculation rather than concatenated.
    probe_prediction = np.zeros(len(labels), dtype=np.int64)
    catch_prediction = np.zeros(len(labels), dtype=np.int64)
    valid = np.zeros(len(labels), dtype=bool)
    valid[common] = True
    probe_prediction[common] = (probe_scores > probe_threshold).astype(np.int64)
    catch_prediction[common] = (catch_scores > catch_threshold).astype(np.int64)
    split_record = None
    if split_paths is not None:
        split_arrays = [np.asarray(np.load(item, allow_pickle=False), dtype=np.int64) for item in split_paths]
        split_record = validate_common_splits(*split_arrays, len(labels))
    return {
        "protocol": "CATCH-common",
        "probe_source": str(Path(probe_path).resolve()),
        "catch_source": str(Path(catch_path).resolve()),
        "labels_source": str(Path(labels_path).resolve()),
        "common_index_sha256": _hash_array(common),
        "common_count": int(len(common)),
        "probe_count": int(len(probe["time_indices"])),
        "catch_count": int(len(catch["time_indices"])),
        "coverage": float(len(common) / max(len(labels), 1)),
        "excluded_anomaly_points": int(np.sum((labels > 0) & ~valid)),
        "probe": {"threshold": float(probe_threshold), **_metric(actual, probe_scores),
                  "affiliation": affiliation_f_segment_macro(labels, probe_prediction, valid, boundary_ids)},
        "catch": {"threshold": float(catch_threshold), **_metric(actual, catch_scores),
                  "affiliation": affiliation_f_segment_macro(labels, catch_prediction, valid, boundary_ids)},
        "intersection": intersection,
        "common_splits": split_record,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, help="probe results/aligned_scores.npz")
    parser.add_argument("--catch", required=True, help="existing CATCH score npz")
    parser.add_argument("--labels", required=True, help="read-only full-coordinate labels .npy")
    parser.add_argument("--probe-threshold", type=float, default=None)
    parser.add_argument("--catch-threshold", type=float, default=None)
    parser.add_argument("--boundary-ids", default="")
    parser.add_argument("--train-indices", default="")
    parser.add_argument("--validation-indices", default="")
    parser.add_argument("--calibration-indices", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    split_args = (args.train_indices, args.validation_indices, args.calibration_indices)
    split_paths = split_args if all(split_args) else None
    if any(split_args) and split_paths is None:
        raise SystemExit("train, validation and calibration indices must be supplied together")
    result = evaluate(args.probe, args.catch, args.labels,
                      args.probe_threshold, args.catch_threshold,
                      args.boundary_ids or None, split_paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output),
                      "common_count": result["common_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
