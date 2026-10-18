#!/usr/bin/env python
"""Register, but do not fabricate, a CATCH-common execution manifest.

This helper only accepts artifacts produced by an explicit common-split
training/score adapter.  It never trains CATCH and it refuses score files
without immutable training and calibration evidence.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.mnrf_view_probe.evaluation import calibrated_quantile, validate_common_splits
from ts_benchmark.baselines.mnrf_view_probe.catch_common import load_verified_catch_checkpoint
from ts_benchmark.baselines.mnrf_view_probe.common_contract import CONTRACT_SCHEMA, missing_contract_fields


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(path):
    values = np.asarray(np.load(path, allow_pickle=False))
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _array_metadata(path):
    values = np.asarray(np.load(path, allow_pickle=False))
    return {"dtype": values.dtype.str, "shape": list(values.shape), "count": int(values.size)}


def _npz_coordinates(path):
    with np.load(path, allow_pickle=False) as payload:
        if not {"scores", "time_indices"}.issubset(payload.files):
            raise ValueError(f"{path} lacks scores/time_indices")
        indices = np.asarray(payload["time_indices"])
        scores = np.asarray(payload["scores"])
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or indices.dtype.kind == "b":
        raise ValueError(f"{path} has invalid endpoint coordinates")
    if indices.dtype.kind == "u" and np.any(indices > np.iinfo(np.int64).max):
        raise ValueError(f"{path} endpoint coordinates overflow int64")
    indices = np.asarray(indices, dtype=np.int64)
    if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise ValueError(f"{path} has invalid or duplicate endpoint coordinates")
    if scores.ndim != 1 or len(scores) != len(indices) or not np.issubdtype(scores.dtype, np.number) or not np.isfinite(scores).all():
        raise ValueError(f"{path} has invalid scores")
    return indices


def _load_execution_evidence(path, kind, task_name, split_paths):
    """Require a real completed artifact, not an identity-only placeholder."""
    path = Path(path)
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"{kind} evidence is not valid JSON: {error}") from error
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError(f"{kind} evidence must be a non-empty JSON object")
    if evidence.get("status") != "complete":
        raise ValueError(f"{kind} evidence is not complete")
    if evidence.get("protocol") not in {"CATCH-common", "CATCH-common-v1"}:
        raise ValueError(f"{kind} evidence does not declare CATCH-common protocol")
    if task_name and evidence.get("task_name") != task_name:
        raise ValueError(f"{kind} evidence task_name differs")
    required = ("run_identity", "data_identity", "split_identity")
    missing = [key for key in required if not evidence.get(key)]
    if missing:
        raise ValueError(f"{kind} evidence is missing required execution fields: {missing}")
    checkpoint = evidence.get("checkpoint_path")
    if checkpoint is None and isinstance(evidence.get("checkpoint"), dict):
        checkpoint = evidence["checkpoint"].get("path")
    if not checkpoint or not Path(checkpoint).is_file():
        raise ValueError(f"{kind} evidence checkpoint is missing")
    declared_checkpoint = evidence.get("checkpoint_sha256")
    if not declared_checkpoint:
        raise ValueError(f"{kind} evidence is missing checkpoint_sha256")
    if declared_checkpoint != _sha256(checkpoint):
        raise ValueError(f"{kind} evidence checkpoint hash mismatch")
    try:
        checkpoint_payload = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"{kind} evidence checkpoint cannot be loaded: {error}") from error
    if not isinstance(checkpoint_payload, dict) or checkpoint_payload.get("protocol") != "CATCH-common":
        raise ValueError(f"{kind} evidence checkpoint is not a CATCH-common checkpoint")
    load_verified_catch_checkpoint(checkpoint_payload, evidence)
    if kind == "calibration":
        threshold = evidence.get("threshold")
        if threshold is None or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
            raise ValueError("calibration evidence must contain a finite threshold")
        calibration_scores = evidence.get("calibration_scores_path")
        if not calibration_scores or not Path(calibration_scores).is_file():
            raise ValueError("calibration evidence scores are missing")
        if evidence.get("calibration_scores_sha256") and evidence["calibration_scores_sha256"] != _sha256(calibration_scores):
            raise ValueError("calibration evidence scores hash mismatch")
        with np.load(calibration_scores, allow_pickle=False) as scores_payload:
            if "scores" not in scores_payload.files:
                raise ValueError("calibration evidence scores have no scores array")
            calibration_values = np.asarray(scores_payload["scores"])
        quantile = float(evidence.get("calibration_quantile", 0.995))
        if not 0.0 < quantile < 1.0 or not np.isfinite(calibration_values).all():
            raise ValueError("calibration evidence quantile or scores are invalid")
        if not np.isclose(float(threshold), calibrated_quantile(calibration_values, quantile), rtol=0, atol=1e-12):
            raise ValueError("calibration threshold does not reproduce from calibration scores")
    contract_path = evidence.get("common_contract_path")
    contract_hash = evidence.get("common_contract_sha256")
    if not contract_path or not contract_hash or not Path(contract_path).is_file():
        raise ValueError(f"{kind} evidence is missing the frozen common contract")
    if _sha256(contract_path) != contract_hash:
        raise ValueError(f"{kind} evidence common contract hash mismatch")
    try:
        contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"{kind} evidence common contract is invalid: {error}") from error
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("status") != "complete":
        raise ValueError(f"{kind} evidence common contract is incomplete or unsupported")
    missing_contract = missing_contract_fields(contract)
    if missing_contract:
        raise ValueError(f"{kind} evidence common contract is missing fields: {missing_contract}")
    for split_name, split_path in zip(("train", "validation", "calibration"), split_paths):
        key = f"{split_name}_indices_sha256"
        if key not in evidence:
            raise ValueError(f"{kind} evidence is missing {key}")
        if evidence[key] != _array_sha256(split_path):
            raise ValueError(f"{kind} evidence {split_name} split hash mismatch")
    return evidence


def _verify_production_score_binding(evidence, catch_path, labels):
    score_path = evidence.get("test_scores_path")
    declared_hash = evidence.get("test_scores_sha256")
    if not score_path or not declared_hash:
        raise ValueError("CATCH production score binding is missing")
    if Path(score_path).resolve() != Path(catch_path).resolve():
        raise ValueError("CATCH production score path differs from evidence")
    if _sha256(catch_path) != declared_hash:
        raise ValueError("CATCH production score hash differs from evidence")
    with np.load(catch_path, allow_pickle=False) as payload:
        for name in ("scores", "time_indices", "predictions", "actual", "threshold"):
            if name not in payload.files:
                raise ValueError(f"CATCH production score artifact lacks {name}")
        indices = np.asarray(payload["time_indices"])
        scores = np.asarray(payload["scores"])
        predictions = np.asarray(payload["predictions"])
        actual = np.asarray(payload["actual"])
        threshold = float(np.asarray(payload["threshold"]))
    if indices.ndim != 1 or scores.ndim != 1 or predictions.ndim != 1 or actual.ndim != 1:
        raise ValueError("CATCH production score arrays must be one-dimensional")
    if not np.isfinite(scores).all() or len(indices) != len(scores) or len(predictions) != len(scores) or len(actual) != len(scores):
        raise ValueError("CATCH production score arrays have invalid lengths or values")
    if evidence.get("test_indices_sha256") != _array_sha256_from_values(indices):
        raise ValueError("CATCH production endpoint identity differs")
    if int(evidence.get("test_count", -1)) != len(indices):
        raise ValueError("CATCH production endpoint count differs")
    if not np.isclose(threshold, float(evidence.get("test_threshold", np.nan)), rtol=0, atol=1e-12):
        raise ValueError("CATCH production threshold differs from evidence")
    expected_predictions = (scores.astype(np.float64) > threshold).astype(np.int64)
    if not np.array_equal(predictions.astype(np.int64), expected_predictions):
        raise ValueError("CATCH production predictions do not match threshold")
    if evidence.get("test_predictions_sha256") != _array_sha256_from_values(predictions):
        raise ValueError("CATCH production prediction identity differs")
    if evidence.get("test_actual_sha256") != _array_sha256_from_values(actual):
        raise ValueError("CATCH production label identity differs")
    if np.any(indices < 0) or np.any(indices >= len(labels)) or not np.array_equal(actual.astype(np.int64), labels[indices]):
        raise ValueError("CATCH production labels do not match the registered full-coordinate labels")


def _array_sha256_from_values(values):
    array = np.asarray(values)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def register(args):
    files = [args.probe, args.catch, args.labels, args.train_indices,
             args.validation_indices, args.calibration_indices,
             args.training_evidence, args.calibration_evidence]
    missing = [item for item in files if not Path(item).is_file()]
    if missing:
        raise ValueError("CATCH-common evidence is missing: " + ", ".join(missing))
    labels = np.asarray(np.load(args.labels, allow_pickle=False))
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("labels must be one-dimensional integer values")
    split_values = [np.asarray(np.load(item, allow_pickle=False)) for item in
                    (args.train_indices, args.validation_indices, args.calibration_indices)]
    split_record = validate_common_splits(*split_values, len(labels))
    _npz_coordinates(args.probe)
    _npz_coordinates(args.catch)
    training_evidence = _load_execution_evidence(args.training_evidence, "training", args.task_name,
                                                  (args.train_indices, args.validation_indices, args.calibration_indices))
    calibration_evidence = _load_execution_evidence(args.calibration_evidence, "calibration", args.task_name,
                                                     (args.train_indices, args.validation_indices, args.calibration_indices))
    _verify_production_score_binding(calibration_evidence, args.catch, labels.astype(np.int64))
    with np.load(args.catch, allow_pickle=False) as catch_payload:
        catch_threshold = catch_payload.get("threshold")
    if catch_threshold is None or not np.isfinite(float(np.asarray(catch_threshold))):
        raise ValueError("CATCH score artifact must contain a finite calibrated threshold")
    if not np.isclose(float(np.asarray(catch_threshold)), float(calibration_evidence["threshold"]), rtol=0, atol=1e-12):
        raise ValueError("CATCH score threshold does not match its calibration evidence")
    for field in ("protocol", "task_name", "run_identity", "data_identity", "split_identity",
                  "time_identity", "time_kind", "test_start", "seq_len", "variable_identity",
                  "boundary_columns", "score_protocol", "common_contract_path", "common_contract_sha256",
                  "checkpoint_path", "checkpoint_sha256", "normalization_path", "normalization_sha256",
                  "train_fit_rows_path", "train_fit_rows_sha256", "test_scores_path",
                  "test_scores_sha256", "test_indices_sha256", "test_predictions_sha256",
                  "test_actual_sha256", "test_count", "test_threshold"):
        if training_evidence.get(field) != calibration_evidence.get(field):
            raise ValueError(f"training/calibration evidence {field} mismatch")
    manifest = {
        "schema": "mnrf-view-probe-catch-common-v1",
        "protocol": "CATCH-common",
        "status": "complete",
        "task_name": args.task_name,
        "probe_score_sha256": _sha256(args.probe),
        "catch_score_sha256": _sha256(args.catch),
        "labels_sha256": _sha256(args.labels),
        "train_score_identity": {"path": str(Path(args.training_evidence).resolve()),
                                  "file_sha256": _sha256(args.training_evidence),
                                  "evidence": training_evidence},
        "calibration_identity": {"path": str(Path(args.calibration_evidence).resolve()),
                                  "file_sha256": _sha256(args.calibration_evidence),
                                  "evidence": calibration_evidence},
        "splits": {
            name: {"path": str(Path(path).resolve()),
                   "file_sha256": _sha256(path),
                   "indices_sha256": split_record[name]["sha256"],
                   **_array_metadata(path),
                   "count": split_record[name]["count"]}
            for name, path in zip(("train", "validation", "calibration"),
                                  (args.train_indices, args.validation_indices, args.calibration_indices))
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--catch", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--train-indices", required=True)
    parser.add_argument("--validation-indices", required=True)
    parser.add_argument("--calibration-indices", required=True)
    parser.add_argument("--training-evidence", required=True)
    parser.add_argument("--calibration-evidence", required=True)
    parser.add_argument("--task-name", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        register(args)
    except Exception as error:
        print(json.dumps({"status": "invalid", "reason": str(error)}, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps({"status": "complete", "output": str(Path(args.output).resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()

