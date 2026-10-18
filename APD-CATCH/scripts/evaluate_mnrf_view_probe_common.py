#!/usr/bin/env python
"""Evaluate a probe and an existing CATCH score file on common coordinates.

This is a read-only comparison adapter.  It never calls a CATCH trainer and
does not change the original CATCH scoring protocol.  Both methods must
provide their own endpoint coordinates; only their explicit intersection is
used for the common report.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.mnrf_view_probe.evaluation import (
    _hash_array,
    affiliation_f_segment_macro,
    calibrated_quantile,
    common_index_intersection,
    strict_integer_indices,
    validate_common_splits,
)
from ts_benchmark.baselines.mnrf_view_probe.catch_common import load_verified_catch_checkpoint


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _common_evidence(manifest_path, probe_path, catch_path, labels_path,
                     split_paths):
    """Validate execution evidence for the strict CATCH-common protocol.

    Split files alone only describe coordinates.  The manifest must bind the
    actual score/label files and each split hash to a completed run produced
    with those coordinates.
    """
    if not manifest_path:
        return False, "no CATCH-common execution manifest was supplied", None
    path = Path(manifest_path)
    if not path.is_file():
        return False, "CATCH-common execution manifest is missing", None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "CATCH-common" or manifest.get("status") != "complete":
        return False, "execution manifest is not a completed CATCH-common run", None
    expected_files = {
        "probe_score_sha256": probe_path,
        "catch_score_sha256": catch_path,
        "labels_sha256": labels_path,
    }
    for field, file_path in expected_files.items():
        declared = manifest.get(field)
        if not declared or _sha256_file(file_path) != declared:
            return False, f"CATCH-common evidence hash mismatch: {field}", None
    if not split_paths or len(split_paths) != 3:
        return False, "CATCH-common evidence must bind train/validation/calibration splits", None
    declared_splits = manifest.get("splits")
    if not isinstance(declared_splits, dict):
        return False, "CATCH-common evidence has no split records", None
    for name, split_path in zip(("train", "validation", "calibration"), split_paths):
        record = declared_splits.get(name)
        if not isinstance(record, dict) or not record.get("file_sha256") or not record.get("indices_sha256"):
            return False, f"CATCH-common evidence has no {name} hash", None
        if Path(record.get("path", "")).resolve() != Path(split_path).resolve():
            return False, f"CATCH-common evidence {name} path mismatch", None
        if _sha256_file(split_path) != record["file_sha256"]:
            return False, f"CATCH-common evidence {name} hash mismatch", None
        values = np.load(split_path, allow_pickle=False)
        if _hash_array(values) != record["indices_sha256"]:
            return False, f"CATCH-common evidence {name} coordinate hash mismatch", None
        if record.get("dtype") != values.dtype.str or record.get("shape") != list(values.shape):
            return False, f"CATCH-common evidence {name} dtype/shape mismatch", None
        if int(record.get("count", -1)) != int(values.size):
            return False, f"CATCH-common evidence {name} count mismatch", None
    train_identity = manifest.get("train_score_identity")
    calibration_identity = manifest.get("calibration_identity")
    if not isinstance(train_identity, dict) or not isinstance(calibration_identity, dict):
        return False, "CATCH-common training/calibration identities are missing", None
    loaded_evidence = {}
    for name, identity in (("training", train_identity), ("calibration", calibration_identity)):
        evidence_path = identity.get("path")
        if not evidence_path or not identity.get("file_sha256") or not Path(evidence_path).is_file():
            return False, f"CATCH-common {name} evidence file is missing", None
        if _sha256_file(evidence_path) != identity["file_sha256"]:
            return False, f"CATCH-common {name} evidence file hash mismatch", None
        try:
            evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
        except Exception:
            return False, f"CATCH-common {name} evidence is not valid JSON", None
        if (not isinstance(evidence, dict) or evidence.get("status") != "complete"
                or evidence.get("protocol") not in {"CATCH-common", "CATCH-common-v1"}):
            return False, f"CATCH-common {name} evidence is incomplete", None
        loaded_evidence[name] = evidence
        if manifest.get("task_name") and evidence.get("task_name") != manifest.get("task_name"):
            return False, f"CATCH-common {name} evidence task differs", None
        if not evidence.get("run_identity") or not evidence.get("data_identity") or not evidence.get("split_identity"):
            return False, f"CATCH-common {name} evidence lacks execution identity", None
        checkpoint = evidence.get("checkpoint_path")
        if checkpoint is None and isinstance(evidence.get("checkpoint"), dict):
            checkpoint = evidence["checkpoint"].get("path")
        if not checkpoint or not Path(checkpoint).is_file():
            return False, f"CATCH-common {name} checkpoint is missing", None
        if evidence.get("checkpoint_sha256") != _sha256_file(checkpoint):
            return False, f"CATCH-common {name} checkpoint hash mismatch", None
        try:
            import torch
            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception:
            return False, f"CATCH-common {name} checkpoint cannot be loaded", None
        if not isinstance(checkpoint_payload, dict) or checkpoint_payload.get("protocol") != "CATCH-common":
            return False, f"CATCH-common {name} checkpoint is not bound to the protocol", None
        try:
            load_verified_catch_checkpoint(checkpoint_payload, evidence)
        except Exception as error:
            return False, f"CATCH-common {name} checkpoint/model configuration is invalid: {error}", None
        if identity.get("evidence") != evidence:
            return False, f"CATCH-common {name} embedded evidence differs from its verified file", None
        if name == "calibration":
            threshold = evidence.get("threshold")
            scores_path = evidence.get("calibration_scores_path")
            if threshold is None or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
                return False, "CATCH-common calibration threshold is missing or non-finite", None
            if not scores_path or not Path(scores_path).is_file():
                return False, "CATCH-common calibration scores are missing", None
            if evidence.get("calibration_scores_sha256") != _sha256_file(scores_path):
                return False, "CATCH-common calibration scores hash mismatch", None
            with np.load(scores_path, allow_pickle=False) as score_payload:
                if not {"scores", "time_indices"}.issubset(score_payload.files):
                    return False, "CATCH-common calibration scores lack scores/endpoints", None
                values = np.asarray(score_payload["scores"])
                calibration_indices = strict_integer_indices(score_payload["time_indices"],
                                                             "CATCH calibration endpoints")
            quantile = float(evidence.get("calibration_quantile", 0.995))
            if not 0.0 < quantile < 1.0 or not np.isfinite(values).all():
                return False, "CATCH-common calibration scores or quantile are invalid", None
            if not np.isclose(float(threshold), calibrated_quantile(values, quantile), rtol=0, atol=1e-12):
                return False, "CATCH-common calibration threshold does not reproduce", None
            if not np.array_equal(calibration_indices, np.load(split_paths[2], allow_pickle=False)):
                return False, "CATCH-common calibration scores endpoints differ from frozen split", None
    training = loaded_evidence["training"]
    calibration = loaded_evidence["calibration"]
    for field in ("protocol", "task_name", "run_identity", "data_identity", "time_identity",
                  "variable_identity", "boundary_columns", "score_protocol", "split_identity",
                  "checkpoint_path", "checkpoint_sha256", "normalization_path", "normalization_sha256",
                  "train_fit_rows_path", "train_fit_rows_sha256"):
        if training.get(field) != calibration.get(field):
            return False, f"CATCH-common training/calibration {field} mismatch", None
    normalization = calibration.get("normalization_path")
    if not normalization or not Path(normalization).is_file():
        return False, "CATCH-common normalization artifact is missing", None
    if calibration.get("normalization_sha256") != _sha256_file(normalization):
        return False, "CATCH-common normalization hash mismatch", None
    fit_rows = calibration.get("train_fit_rows_path")
    if (not fit_rows or not Path(fit_rows).is_file()
            or calibration.get("train_fit_rows_sha256") != _sha256_file(fit_rows)):
        return False, "CATCH-common training-row binding is missing or changed", None
    # Downstream threshold decisions consume the exact objects whose external
    # files and embedded copies were verified above.
    manifest["_verified_training_evidence"] = loaded_evidence["training"]
    manifest["_verified_calibration_evidence"] = loaded_evidence["calibration"]
    return True, None, manifest


def _read_npz(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {"scores", "time_indices"}
        if not required.issubset(payload.files):
            raise ValueError(f"{path} must contain scores and time_indices")
        result = {name: np.asarray(payload[name]) for name in payload.files}
    indices = strict_integer_indices(result["time_indices"], f"{path} time_indices")
    score_values = np.asarray(result["scores"])
    if score_values.ndim != 1:
        raise ValueError(f"{path} scores must be one-dimensional")
    scores = np.asarray(score_values, dtype=np.float64)
    if len(indices) != len(scores) or len(np.unique(indices)) != len(indices):
        raise ValueError(f"{path} has invalid or duplicate endpoint coordinates")
    if np.any(indices < 0) or not np.isfinite(scores).all():
        raise ValueError(f"{path} has invalid endpoint or score values")
    result["time_indices"] = indices
    result["scores"] = scores
    return result


def _validate_probe_execution(probe_path, probe, catch, common_evidence, split_paths, labels):
    """Bind the score NPZ to a completed, loadable Probe run in strict mode."""
    from ts_benchmark.baselines.mnrf_view_probe.probe import load_probe
    common_evidence = (common_evidence.get("_verified_calibration_evidence", {})
                       if isinstance(common_evidence, dict) else {})

    run_root = Path(probe_path).resolve().parent.parent
    manifest_path = run_root / "probe_manifest.json"
    status_path = run_root / "run_status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise ValueError("strict CATCH-common probe artifact is missing its run manifest/status")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete" or manifest.get("artifact_type") != "probe_head":
        raise ValueError("strict CATCH-common probe run is not a completed probe head")
    if manifest.get("aligned_scores_sha256") != _sha256_file(probe_path):
        raise ValueError("strict CATCH-common probe score hash is not bound to its manifest")
    try:
        restored = load_probe(run_root, device="cpu").score_cached_test()
    except Exception as error:
        raise ValueError(f"strict CATCH-common probe artifact cannot be restored: {error}") from error
    if not np.array_equal(np.asarray(restored["time_indices"], dtype=np.int64), probe["time_indices"]):
        raise ValueError("strict CATCH-common probe endpoints differ from restored artifact")
    if not np.allclose(np.asarray(restored["scores"], dtype=np.float64), probe["scores"], rtol=0, atol=1e-12):
        raise ValueError("strict CATCH-common probe scores differ from restored artifact")
    declared_threshold = float(np.asarray(probe.get("threshold", np.nan)))
    restored_threshold = float(restored["threshold"])
    if not np.isfinite(declared_threshold) or not np.isclose(declared_threshold, restored_threshold, rtol=0, atol=1e-12):
        raise ValueError("strict CATCH-common probe threshold is not its calibrated threshold")
    source = manifest.get("source_identity") or {}
    if not source.get("task_name") or source.get("task_name") != common_evidence.get("task_name"):
        raise ValueError("strict CATCH-common probe task identity differs")
    if common_evidence.get("data_identity") and source.get("feature_data_sha256"):
        if common_evidence.get("data_identity") != source.get("feature_data_sha256"):
            raise ValueError("strict CATCH-common probe data identity differs")
    if common_evidence.get("time_identity") != source.get("time_index", {}).get("values_sha256"):
        raise ValueError("strict CATCH-common probe absolute time identity differs")
    if common_evidence.get("time_kind") != source.get("time_index", {}).get("kind"):
        raise ValueError("strict CATCH-common time-coordinate semantics differ")
    expected_boundary = source.get("boundary_values_sha256")
    if common_evidence.get("boundary_values_identity") != expected_boundary:
        raise ValueError("strict CATCH-common probe entity/session identity differs")
    # Window lengths remain model-specific under the common raw-data protocol;
    # endpoint intersection is explicit and does not rewrite either method's
    # context rule.
    expected_variables = source.get("variable_names", [])
    expected_variable_identity = hashlib.sha256(json.dumps(expected_variables).encode("utf-8")).hexdigest()
    if common_evidence.get("variable_identity") != expected_variable_identity:
        raise ValueError("strict CATCH-common variable order differs from the Probe source")
    ranges = source.get("ranges", {})
    if int(common_evidence.get("test_start", -1)) != int(source.get("train_length", -2)):
        raise ValueError("strict CATCH-common official train/test boundary differs")
    training_rows_path = common_evidence.get("train_fit_rows_path")
    if not training_rows_path or not Path(training_rows_path).is_file():
        raise ValueError("strict CATCH-common raw training rows are missing")
    raw_training_rows = strict_integer_indices(np.load(training_rows_path, allow_pickle=False),
                                               "CATCH scaler fit rows")
    raw_start, raw_stop = ranges.get("train", [0, 0])
    expected_raw_rows = np.arange(int(raw_start), int(raw_stop), dtype=np.int64)
    if not np.array_equal(raw_training_rows, expected_raw_rows):
        raise ValueError("strict CATCH-common scaler fit rows differ from Probe raw training rows")
    if split_paths and isinstance(ranges, dict):
        for name, split_path in zip(("train", "validation", "calibration"), split_paths):
            values = strict_integer_indices(np.load(split_path, allow_pickle=False), name)
            start, end = ranges.get(name, [0, 0])
            if len(values) and (np.any(values < int(start)) or np.any(values >= int(end))):
                raise ValueError(f"strict CATCH-common {name} split is outside the probe split identity")
    actual = probe.get("actual")
    if actual is not None:
        actual = np.asarray(actual, dtype=np.int64).reshape(-1)
        if len(actual) != len(probe["time_indices"]) or not np.array_equal(actual, labels[probe["time_indices"]]):
            raise ValueError("strict CATCH-common probe actual labels are not bound to common labels")
    catch_actual = catch.get("actual")
    if catch_actual is None or not np.array_equal(np.asarray(catch_actual, dtype=np.int64), labels[catch["time_indices"]]):
        raise ValueError("strict CATCH-common CATCH labels are not bound to common labels")


def _metric(actual, scores):
    actual = np.asarray(actual)
    if actual.ndim != 1 or actual.dtype.kind not in "iu":
        raise ValueError("common labels must be a one-dimensional integer array")
    actual = np.asarray(actual, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("common scores must be finite one-dimensional values")
    if len(actual) == 0:
        return {"auc_roc": None, "auc_roc_reason": "common coordinates are empty",
                "auc_pr": None, "auc_pr_reason": "common coordinates are empty"}
    average_precision = float(average_precision_score(actual, scores))
    if not np.isfinite(average_precision):
        average_precision = None
    if np.unique(actual).size < 2:
        return {"auc_roc": None, "auc_roc_reason": "common coordinates contain one class",
                "auc_pr": average_precision,
                "auc_pr_reason": None if average_precision is not None else "non-finite average precision"}
    auc_roc = float(roc_auc_score(actual, scores))
    return {"auc_roc": auc_roc if np.isfinite(auc_roc) else None,
            "auc_pr": average_precision}


def evaluate(probe_path, catch_path, labels_path, probe_threshold=None,
             catch_threshold=None, boundary_path=None, split_paths=None,
             common_manifest=None, strict_common=None):
    probe = _read_npz(probe_path)
    catch = _read_npz(catch_path)
    labels_raw = np.asarray(np.load(labels_path, allow_pickle=False))
    if labels_raw.ndim != 1 or labels_raw.dtype.kind not in "iu":
        raise ValueError("labels must be a one-dimensional integer array")
    labels = labels_raw.astype(np.int64)
    for name, values in (("probe", probe["time_indices"]), ("catch", catch["time_indices"])):
        strict_integer_indices(values, f"{name} time_indices", upper_bound=len(labels))
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
        split_arrays = [np.load(item, allow_pickle=False) for item in split_paths]
        split_record = validate_common_splits(*split_arrays, len(labels))
    evidence_ok, evidence_reason, evidence = _common_evidence(
        common_manifest, probe_path, catch_path, labels_path, split_paths)
    if strict_common is None:
        strict_common = bool(common_manifest)
    if strict_common and not evidence_ok:
        raise ValueError(f"strict CATCH-common evidence rejected: {evidence_reason}")
    if strict_common:
        _validate_probe_execution(probe_path, probe, catch, evidence, split_paths, labels)
        if probe_threshold is not None and not np.isclose(float(probe_threshold),
                                                          float(probe["threshold"]), rtol=0, atol=1e-12):
            raise ValueError("strict CATCH-common probe threshold override does not match its calibration")
    if evidence_ok:
        calibration_evidence = evidence.get("_verified_calibration_evidence", {})
        calibrated_threshold = calibration_evidence.get("threshold")
        if calibrated_threshold is None or not math.isfinite(float(calibrated_threshold)):
            raise ValueError("strict CATCH-common evidence has no finite calibrated threshold")
        if not np.isclose(float(catch_threshold), float(calibrated_threshold), rtol=0, atol=1e-12):
            raise ValueError("strict CATCH-common catch threshold does not match calibration evidence")
        declared_score_threshold = catch.get("threshold")
        if declared_score_threshold is not None:
            declared_score_threshold = float(np.asarray(declared_score_threshold))
            if not np.isclose(declared_score_threshold, float(calibrated_threshold), rtol=0, atol=1e-12):
                raise ValueError("strict CATCH-common score threshold does not match calibration evidence")
    protocol = "CATCH-common" if evidence_ok else "index_intersection_reference"
    return {
        "protocol": protocol,
        "protocol_reason": None if evidence_ok else evidence_reason,
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
        "common_evidence": evidence,
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
    parser.add_argument("--common-manifest", default="",
                        help="completed execution evidence for strict CATCH-common")
    parser.add_argument("--reference-mode", action="store_true",
                        help="allow invalid/missing evidence as index_intersection_reference")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    split_args = (args.train_indices, args.validation_indices, args.calibration_indices)
    split_paths = split_args if all(split_args) else None
    if any(split_args) and split_paths is None:
        raise SystemExit("train, validation and calibration indices must be supplied together")
    result = evaluate(args.probe, args.catch, args.labels,
                      args.probe_threshold, args.catch_threshold,
                      args.boundary_ids or None, split_paths,
                      args.common_manifest or None,
                      strict_common=bool(args.common_manifest) and not args.reference_mode)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output),
                      "protocol": result["protocol"],
                      "common_count": result["common_count"]}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
