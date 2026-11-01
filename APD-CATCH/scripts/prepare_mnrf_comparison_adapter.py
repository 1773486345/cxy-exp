#!/usr/bin/env python
"""Prepare and validate the MNRF/native comparison adapter.

The default modes are read-only inventory and plan generation.  ``score`` is
an explicit deployment operation: it restores a completed MNRF GMM artifact,
scores the already saved test feature cache, and writes a new valid-endpoint
bundle without fitting, encoding, or recalibrating.  It never invents scores
for the cold-start prefix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("PSM", "MSL", "GECCO", "Genesis", "NYC", "CalIt2", "Creditcard", "CICIDS")
COMMON_BATCH = ROOT / "result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun"
DEFAULT_OUTPUT = ROOT / "result/analysis/MNRF-Comparison-Adapter-20261102"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_indices(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    values = values.astype(np.int64, copy=False)
    if len(values) and (np.any(values < 0) or np.any(np.diff(values) <= 0)):
        raise ValueError(f"{name} must be non-negative, sorted and unique")
    return values


def _inspect_score(path: Path, expected_indices=None):
    record = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        record["status"] = "missing"
        return record
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {"scores", "predictions", "time_indices", "threshold"}
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError("missing fields: " + ", ".join(missing))
            scores = np.asarray(payload["scores"])
            predictions = np.asarray(payload["predictions"])
            indices = _strict_indices(payload["time_indices"], "time_indices")
            threshold = float(np.asarray(payload["threshold"]))
        if scores.ndim != 1 or predictions.ndim != 1 or len(scores) != len(indices) or len(predictions) != len(indices):
            raise ValueError("scores, predictions and time_indices have incompatible shapes")
        if not np.isfinite(scores).all() or not np.isfinite(threshold):
            raise ValueError("scores or threshold are non-finite")
        if not np.array_equal(predictions.astype(np.int8), (scores > threshold).astype(np.int8)):
            raise ValueError("predictions do not implement strict score > threshold")
        if expected_indices is not None and not np.array_equal(indices, expected_indices):
            raise ValueError("test coordinates differ from the frozen common contract")
        record.update({"status": "complete", "count": int(len(scores)),
                       "score_dtype": str(scores.dtype), "threshold": threshold,
                       "indices_first": int(indices[0]) if len(indices) else None,
                       "indices_last": int(indices[-1]) if len(indices) else None,
                       "sha256": _sha256(path)})
    except Exception as error:
        record.update({"status": "invalid", "reason": str(error)})
    return record


def _load_task_status():
    path = COMMON_BATCH / "task_status.json"
    if not path.is_file():
        raise FileNotFoundError(f"common task_status.json is missing: {path}")
    return path, _json(path)


def inventory(output_dir: Path):
    status_path, batch_status = _load_task_status()
    rows = []
    for task in TASKS:
        task_status_path = COMMON_BATCH / task / "task_status.json"
        if not task_status_path.is_file():
            rows.append({"dataset": task, "status_path": str(task_status_path), "status": "missing"})
            continue
        task_status = _json(task_status_path)
        contract_dir = COMMON_BATCH / task / "common_contract"
        test_path = contract_dir / "catch_test_indices.npy"
        expected = None
        if test_path.is_file():
            expected = _strict_indices(np.load(test_path, allow_pickle=False), f"{task} contract test")
        for method in ("CATCH-common", "IsolationForest", "PCA"):
            method_record = task_status.get("methods", {}).get(method, {})
            output = Path(method_record.get("output", ""))
            score_path = output / "results" / "aligned_scores.npz"
            checked = _inspect_score(score_path, expected)
            if method == "CATCH-common":
                evidence = output / "results" / "common_manifest.json"
            else:
                evidence = output / "evidence" / "baseline.json"
            evidence_status = "present" if evidence.is_file() else "missing"
            evidence_check = {"status": evidence_status, "path": str(evidence)}
            if evidence.is_file() and checked.get("status") == "complete":
                try:
                    evidence_payload = _json(evidence)
                    if method == "CATCH-common":
                        if evidence_payload.get("protocol") != "CATCH-common" or evidence_payload.get("status") != "complete":
                            raise ValueError("common_manifest is not a complete CATCH-common record")
                        if evidence_payload.get("catch_score_sha256") != checked.get("sha256"):
                            raise ValueError("common_manifest score hash differs")
                    else:
                        if evidence_payload.get("status") != "complete" or evidence_payload.get("method") != method:
                            raise ValueError("baseline evidence is not a complete method record")
                        if Path(evidence_payload.get("score_path", "")).resolve() != score_path.resolve():
                            raise ValueError("baseline evidence score path differs")
                        if evidence_payload.get("score_sha256") != checked.get("sha256"):
                            raise ValueError("baseline evidence score hash differs")
                    evidence_check.update({"status": "verified", "sha256": _sha256(evidence)})
                except Exception as error:
                    evidence_check.update({"status": "invalid", "reason": str(error)})
            contract_path = contract_dir / "common_contract.json"
            contract_check = {"path": str(contract_path), "status": "missing"}
            if contract_path.is_file() and expected is not None:
                try:
                    contract = _json(contract_path)
                    test_record = contract.get("catch_test", {})
                    test_record_path = Path(test_record.get("path", ""))
                    if not test_record_path.is_file() or not np.array_equal(
                            _strict_indices(np.load(test_record_path, allow_pickle=False), f"{task} contract test"), expected):
                        raise ValueError("contract catch_test coordinates differ")
                    if test_record.get("sha256") != _sha256(test_record_path):
                        raise ValueError("contract catch_test hash differs")
                    contract_times_path = contract_dir / "time_values_ns.npy"
                    contract_labels_path = contract_dir / "labels.npy"
                    if not contract_times_path.is_file() or not contract_labels_path.is_file():
                        raise ValueError("contract time/label arrays are missing")
                    contract_times = np.asarray(np.load(contract_times_path, allow_pickle=False))
                    contract_labels = np.asarray(np.load(contract_labels_path, allow_pickle=False))
                    with np.load(score_path, allow_pickle=False) as scored:
                        if "time_values_ns" not in scored.files or "actual" not in scored.files:
                            raise ValueError("score artifact lacks bound time_values_ns/actual")
                        if not np.array_equal(np.asarray(scored["time_values_ns"]), contract_times[expected]):
                            raise ValueError("score time coordinates differ from contract")
                        if not np.array_equal(np.asarray(scored["actual"]), contract_labels[expected]):
                            raise ValueError("score labels differ from contract")
                    contract_check.update({"status": "verified", "sha256": _sha256(contract_path),
                                          "test_count": int(expected.size)})
                except Exception as error:
                    contract_check.update({"status": "invalid", "reason": str(error)})
            rows.append({"dataset": task, "method": method,
                         "task_status_path": str(task_status_path),
                         "recorded_status": method_record.get("status"),
                         "recorded_output": str(output),
                         "checked_score": checked,
                         "contract_path": str(contract_dir / "common_contract.json"),
                         "test_indices_path": str(test_path),
                         "evidence_path": str(evidence),
                         "evidence_status": evidence_status,
                         "evidence_check": evidence_check,
                         "contract_check": contract_check})
    payload = {"schema": "mnrf-comparison-adapter-inventory-v1",
               "batch_status_path": str(status_path), "batch_status": batch_status,
               "checked": rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "common_output_inventory.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _source_plan(task: str, source: Path):
    manifest_path = source / "probe_manifest.json"
    inputs_path = source / "analysis_inputs.npz"
    aligned_path = source / "results" / "aligned_scores.npz"
    if not manifest_path.is_file() or not inputs_path.is_file() or not aligned_path.is_file():
        raise ValueError(f"{task}: source is missing manifest, analysis_inputs or aligned_scores")
    manifest = _json(manifest_path)
    identity = manifest.get("source_identity") or {}
    train_length = int(identity.get("train_length", identity.get("input_length", 0)))
    seq_len = int(identity.get("seq_len", manifest.get("config", {}).get("seq_len", 0)))
    with np.load(inputs_path, allow_pickle=False) as values:
        test_labels = np.asarray(values["test_labels"])
        test_global = _strict_indices(values["test_global_indices"], f"{task} test_global_indices")
        test_segments = np.asarray(values["test_segment_ids"])
        all_global = _strict_indices(values["all_global_indices"], f"{task} all_global_indices")
        all_segments = np.asarray(values["all_segment_ids"])
        all_times = np.asarray(values["all_time_values_ns"])
    with np.load(aligned_path, allow_pickle=False) as values:
        valid_mask = np.asarray(values["valid_mask"], dtype=bool)
        local_indices = _strict_indices(values["local_indices"], f"{task} local_indices")
        aligned_scores = np.asarray(values["aligned_scores"])
    if (len(test_labels) != len(test_global) or len(test_segments) != len(test_global)
            or len(valid_mask) != len(test_global) or len(all_segments) != len(all_global)
            or len(all_times) != len(all_global)):
        raise ValueError(f"{task}: complete test axis and valid mask lengths differ")
    if not np.array_equal(test_global, all_global[train_length:]):
        raise ValueError(f"{task}: test global axis is not the complete-axis slice")
    if not np.array_equal(test_segments, all_segments[train_length:]):
        raise ValueError(f"{task}: test segment IDs are not the complete-axis slice")
    valid_local = np.flatnonzero(valid_mask).astype(np.int64)
    if not np.array_equal(valid_local, local_indices):
        raise ValueError(f"{task}: saved valid mask and local indices differ")
    if len(valid_local) and not np.isfinite(aligned_scores[valid_local]).all():
        raise ValueError(f"{task}: valid scores contain non-finite values")
    if np.any(np.isfinite(aligned_scores[~valid_mask])):
        raise ValueError(f"{task}: invalid coordinates contain finite scores")
    def _window_check(local):
        global_position = train_length + int(local)
        start = global_position - seq_len + 1
        if start < 0 or global_position >= len(all_global):
            return False, "endpoint window falls outside complete axis"
        if all_global[global_position] - all_global[start] != seq_len - 1:
            return False, "global row coordinates contain a gap in an endpoint window"
        if np.any(all_segments[start:global_position + 1] != all_segments[global_position]):
            return False, "endpoint window crosses a frozen segment boundary"
        diffs = np.diff(all_times[start:global_position + 1])
        if len(diffs) and (not np.isfinite(diffs).all() or np.any(diffs <= 0)):
            return False, "endpoint window has invalid or non-increasing time values"
        return True, "within-segment history only"
    boundary_safe = True
    boundary_reason = "within-segment history only"
    for local in valid_local:
        safe, reason = _window_check(local)
        if not safe:
            boundary_safe, boundary_reason = False, reason
            break
    first_valid = int(valid_local[0]) if len(valid_local) else None
    last_valid = int(valid_local[-1]) if len(valid_local) else None
    prefix_coordinates = np.arange(int(first_valid or 0), dtype=np.int64)
    prefix_safe = True
    prefix_reason = "within-frozen-segment context is available"
    for local in prefix_coordinates:
        safe, reason = _window_check(local)
        if not safe:
            prefix_safe, prefix_reason = False, reason
            break
    return {
        "dataset": task, "source_path": str(source),
        "source_run_id": source.name, "source_manifest_sha256": _sha256(manifest_path),
        "source_status": _json(source / "run_status.json").get("status") if (source / "run_status.json").is_file() else None,
        "seq_len": seq_len, "official_train_length": train_length,
        "test_length": int(len(test_global)), "valid_points": int(len(valid_local)),
        "invalid_prefix_points": int(first_valid or 0),
        "invalid_suffix_points": int(len(test_global) - 1 - last_valid) if last_valid is not None else int(len(test_global)),
        "valid_first_local": first_valid, "valid_last_local": last_valid,
        "history_policy": "within-frozen-segment-only" if boundary_safe else "blocked-boundary-check",
        "history_check": {"allowed_for_saved_endpoints": boundary_safe, "reason": boundary_reason,
                           "entity_boundary_columns": identity.get("boundary_columns", []),
                           "entity_identity_evidence": ("explicit-boundary-columns"
                                                         if identity.get("boundary_columns")
                                                         else "no-explicit-entity-columns-recorded")},
        "information_policy": {"window": f"[{seq_len}-1 past rows, current endpoint]",
                                "future_rows": 0,
                                "missing_history_mask_supported": False,
                                "missing_history_mask_reason": "frozen encoder receives normalized window values only; no mask channel is recorded",
                                "source_temporal_rule": identity.get("temporal_rule", {}),
                                "permission": "only rows within the frozen source segment and saved endpoint cache"},
        "cold_start_extension": {"status": "ready-frozen-forward" if prefix_safe else "blocked-boundary",
                                  "coordinates": prefix_coordinates.tolist(),
                                  "reason": ("source has no saved prefix features; explicit frozen-encoder forward is required"
                                             if prefix_safe else prefix_reason)},
        "tail_extension": {"status": "not-needed" if last_valid == len(test_global) - 1 else "blocked",
                            "coordinates": list(range(last_valid + 1, len(test_global))) if last_valid is not None else []},
        "required_training_side_artifacts": [
            "heads/gmm_models.pkl", "heads/gmm_status.json", "heads/gmm_fit_endpoints.npy",
            "heads/feature_statistics.npz", "heads/calibration.npz",
            "heads/validation_nll.npy", "heads/calibration_nll.npy",
            "cache/test_features.npy", "cache/test_window_end_indices.npy",
        ],
        "score_artifact": str(aligned_path),
        "axis_artifact": str(inputs_path),
    }


def plan(output_dir: Path, source_root: Path):
    rows = []
    for task in TASKS:
        task_status_path = COMMON_BATCH / task / "task_status.json"
        if not task_status_path.is_file():
            raise FileNotFoundError(task_status_path)
        source_path = Path(_json(task_status_path)["source_path"])
        if not source_path.is_absolute():
            source_path = (ROOT / source_path).resolve()
        rows.append(_source_plan(task, source_path))
    payload = {"schema": "mnrf-comparison-adapter-plan-v1", "mode": "valid-endpoint-read-only",
               "tasks": rows, "source_root": str(source_root),
               "no_training_forward_fit_or_recalibration": True}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mnrf_evaluation_plan.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def score_cached(task: str, source: Path, output: Path):
    """Explicitly score the frozen test cache without fitting/calibrating."""
    from ts_benchmark.baselines.mnrf_view_probe.probe import load_probe
    if output.exists():
        raise FileExistsError(f"refusing to overwrite adapter output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{task}-adapter-", dir=str(output.parent)))
    probe = load_probe(source, device="cpu")
    try:
        result = probe.score_cached_test()
        source_npz = np.load(source / "results" / "aligned_scores.npz", allow_pickle=False)
        source_scores = np.asarray(source_npz["scores"])
        source_indices = _strict_indices(source_npz["time_indices"], "source time_indices")
        if not np.array_equal(source_indices, result["time_indices"]) or not np.array_equal(source_scores, result["scores"]):
            raise RuntimeError(f"{task}: frozen cache re-score differs from existing source scores; stopped")
        with np.load(source / "results" / "aligned_scores.npz", allow_pickle=False) as old:
            payload = {name: np.asarray(old[name]) for name in old.files}
        (temporary / "results").mkdir()
        np.savez_compressed(temporary / "results" / "aligned_scores.npz", **payload)
        manifest = {"schema": "mnrf-comparison-adapter-score-v1", "task": task,
                    "source_path": str(source.resolve()), "source_run_id": source.name,
                    "score_mode": "frozen-test-cache-gmm-no-fit-no-calibration",
                    "source_aligned_scores_sha256": _sha256(source / "results" / "aligned_scores.npz"),
                    "output_aligned_scores_sha256": _sha256(temporary / "results" / "aligned_scores.npz"),
                    "valid_endpoint_only": True,
                    "cold_start_policy": "not-scored; no saved features for prefix",
                    "cold_start_coordinates": np.flatnonzero(~np.asarray(payload["valid_mask"], dtype=bool)).tolist(),
                    "valid_endpoint_count": int(np.asarray(payload["valid_mask"], dtype=bool).sum()),
                    "source_valid_mask_sha256": hashlib.sha256(
                        np.ascontiguousarray(np.asarray(payload["valid_mask"], dtype=bool)).tobytes()).hexdigest(),
                    "evaluation_rule": "ROC-AUC/AP on valid endpoints; alarm prediction is saved strict score > threshold",
                    "training_side_artifacts_reused": [
                        "heads/gmm_models.pkl", "heads/feature_statistics.npz",
                        "heads/validation_nll.npy", "heads/calibration_nll.npy",
                        "heads/calibration.npz", "heads/fixed_noise.npy",
                    ]}
        (temporary / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, output)
        manifest["output_dir"] = str(output.resolve())
        return manifest
    except Exception:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def evaluate(bundle: Path, source: Path, output: Path):
    """Evaluate a bundle only after proving its identity with the frozen source.

    A same-length score file is not sufficient: every coordinate, time value,
    label, mask, threshold and source score bytes must be bound to the source.
    """
    from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
    manifest_path = bundle / "adapter_manifest.json"
    source_manifest_path = source / "probe_manifest.json"
    status_path = source / "run_status.json"
    source_scores_path = source / "results" / "aligned_scores.npz"
    calibration_path = source / "heads" / "calibration.npz"
    for required in (manifest_path, source_manifest_path, status_path,
                     source / "analysis_inputs.npz", source_scores_path,
                     calibration_path, bundle / "results" / "aligned_scores.npz"):
        if not required.is_file():
            raise ValueError(f"required binding artifact is missing: {required}")
    adapter_manifest = _json(manifest_path)
    source_manifest = _json(source_manifest_path)
    source_status = _json(status_path)
    if adapter_manifest.get("schema") != "mnrf-comparison-adapter-score-v1":
        raise ValueError("unsupported adapter manifest schema")
    if adapter_manifest.get("source_path") != str(source.resolve()):
        raise ValueError("adapter source_path is not the requested source")
    if adapter_manifest.get("source_run_id") != source.name:
        raise ValueError("adapter source_run_id does not match source directory")
    if adapter_manifest.get("task") != source_manifest.get("config", {}).get("task_name", source_manifest.get("source_identity", {}).get("task")):
        # Some historical manifests use TASK.csv and some use TASK.  Compare
        # the normalized stem while still rejecting a different dataset.
        declared = str(adapter_manifest.get("task", ""))
        source_task = str(source_manifest.get("config", {}).get("task_name", ""))
        if Path(declared).stem != Path(source_task).stem:
            raise ValueError("adapter task does not match source identity")
    if source_status.get("status") != "complete":
        raise ValueError(f"source run is not complete: {source_status.get('status')}")
    if source_manifest.get("artifact_type") != "probe_head":
        raise ValueError("source is not a completed probe head")
    if adapter_manifest.get("source_aligned_scores_sha256") != _sha256(source_scores_path):
        raise ValueError("source aligned score bytes changed after adapter creation")

    def _load(path):
        with np.load(path, allow_pickle=False) as values:
            return {name: np.asarray(values[name]) for name in values.files}

    source_payload = _load(source_scores_path)
    bundle_payload = _load(bundle / "results" / "aligned_scores.npz")
    # Native ViewProbe uses short field names; adapter bundles may use the
    # explicit aligned aliases.  Normalize names only in memory, never rewrite
    # either artifact.
    for payload in (source_payload, bundle_payload):
        if "aligned_scores" in payload and "scores" not in payload:
            payload["scores"] = payload["aligned_scores"]
        if "aligned_predictions" in payload and "predictions" not in payload:
            payload["predictions"] = payload["aligned_predictions"]
    required_fields = {"scores", "predictions", "time_indices", "threshold", "valid_mask"}
    if not required_fields.issubset(source_payload) or not required_fields.issubset(bundle_payload):
        raise ValueError("source and adapter score artifacts lack required binding fields")
    source_indices = _strict_indices(source_payload["time_indices"], "source time_indices")
    bundle_indices = _strict_indices(bundle_payload["time_indices"], "adapter time_indices")
    if not np.array_equal(source_indices, bundle_indices):
        raise ValueError("adapter global score coordinates differ from source")
    for field in ("scores", "predictions", "valid_mask", "threshold"):
        if field == "threshold":
            if not np.isclose(float(source_payload[field]), float(bundle_payload[field]), rtol=0, atol=1e-12):
                raise ValueError("adapter threshold differs from source")
        elif not np.array_equal(source_payload[field], bundle_payload[field], equal_nan=(field == "scores")):
            raise ValueError(f"adapter {field} differs from source")
    for field in ("time_values_ns", "actual", "local_indices", "aligned_scores", "aligned_predictions"):
        if field in source_payload:
            if field not in bundle_payload:
                raise ValueError(f"adapter is missing source-bound field: {field}")
            if not np.array_equal(source_payload[field], bundle_payload[field], equal_nan=(field == "aligned_scores")):
                raise ValueError(f"adapter {field} differs from source")
    if adapter_manifest.get("output_aligned_scores_sha256") != _sha256(bundle / "results" / "aligned_scores.npz"):
        raise ValueError("adapter output score hash is stale")
    source_mask = np.asarray(source_payload["valid_mask"], dtype=bool)
    declared_mask_hash = adapter_manifest.get("source_valid_mask_sha256")
    actual_mask_hash = hashlib.sha256(np.ascontiguousarray(source_mask).tobytes()).hexdigest()
    if declared_mask_hash != actual_mask_hash:
        raise ValueError("adapter source valid-mask binding is stale")
    with np.load(source / "analysis_inputs.npz", allow_pickle=False) as inputs:
        required_inputs = {"test_labels", "test_global_indices", "test_time_values_ns"}
        if not required_inputs.issubset(inputs.files):
            raise ValueError("source analysis inputs lack test labels/global coordinates/time")
        labels = np.asarray(inputs["test_labels"])
        test_global = _strict_indices(inputs["test_global_indices"], "source test_global_indices")
        test_times = np.asarray(inputs["test_time_values_ns"])
        test_segment_ids = np.asarray(inputs["test_segment_ids"]) if "test_segment_ids" in inputs.files else None
    if "local_indices" in source_payload:
        local_indices = np.asarray(source_payload["local_indices"])
    else:
        local_indices = np.searchsorted(test_global, source_indices)
    local_indices = _strict_indices(local_indices, "source local_indices") if len(local_indices) else local_indices.astype(np.int64)
    if len(local_indices) != len(source_indices) or np.any(local_indices >= len(test_global)):
        raise ValueError("source local_indices are not a valid test-axis mapping")
    if not np.array_equal(source_indices, test_global[local_indices]):
        raise ValueError("source global coordinates do not map to analysis test axis")
    if "time_values_ns" not in source_payload or not np.array_equal(source_payload["time_values_ns"], test_times[local_indices]):
        raise ValueError("source time coordinates do not map to analysis test axis")
    if "actual" in source_payload and not np.array_equal(source_payload["actual"], labels[local_indices]):
        raise ValueError("source labels do not map to analysis test axis")
    if labels.ndim != 1 or len(labels) != len(test_global) or len(test_times) != len(test_global):
        raise ValueError("source test axis fields have incompatible lengths")
    mask = np.asarray(bundle_payload["valid_mask"], dtype=bool)
    scores = np.asarray(bundle_payload["scores"])
    predictions = np.asarray(bundle_payload["predictions"])
    threshold = float(np.asarray(bundle_payload["threshold"]))
    aligned_scores = np.asarray(bundle_payload.get("aligned_scores", scores))
    aligned_predictions = np.asarray(bundle_payload.get("aligned_predictions", predictions))
    if len(aligned_scores) != len(mask) or len(aligned_predictions) != len(mask):
        raise ValueError("aligned score arrays and valid mask have incompatible lengths")
    if np.any(~mask & np.isfinite(aligned_scores)) or np.any(mask & ~np.isfinite(aligned_scores)):
        raise ValueError("valid-mask and finite aligned score states disagree")
    if len(predictions) == len(mask) and len(scores) == len(mask):
        if not np.array_equal(aligned_predictions[mask], (aligned_scores[mask] > threshold).astype(aligned_predictions.dtype)):
            raise ValueError("adapter aligned predictions do not implement strict score > threshold")
        scores = scores[mask]
        predictions = predictions[mask]
    elif len(predictions) == len(scores):
        if not np.array_equal(predictions, (scores > threshold).astype(predictions.dtype)):
            raise ValueError("adapter predictions do not implement strict score > threshold")
        if not np.array_equal(aligned_predictions[mask], predictions.astype(aligned_predictions.dtype)):
            raise ValueError("aligned and valid-endpoint predictions differ")
    elif len(predictions) == len(mask):
        if not np.array_equal(aligned_predictions[mask], (aligned_scores[mask] > threshold).astype(aligned_predictions.dtype)):
            raise ValueError("adapter aligned predictions do not implement strict score > threshold")
        scores = scores[mask]
        predictions = predictions[mask]
    else:
        raise ValueError("adapter predictions length is not bound to valid or full axis")
    with np.load(calibration_path, allow_pickle=False) as calibration:
        calibration_threshold = float(np.asarray(calibration["threshold"]))
    if not np.isclose(threshold, calibration_threshold, rtol=0, atol=1e-12):
        raise ValueError("adapter threshold differs from frozen calibration threshold")
    y = labels[local_indices][mask] if len(local_indices) == len(mask) else labels[local_indices]
    s = scores
    result = {"valid_points": int(len(scores)), "total_test_points": int(len(labels)),
              "coverage": float(mask.sum() / max(len(labels), 1)), "threshold": threshold,
              "roc_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, s)) if np.unique(y).size == 2 else None,
              "average_precision": float(average_precision_score(y, s)) if np.unique(y).size else None,
              "precision": float(precision_score(y, predictions, zero_division=0)) if len(y) else None,
              "recall": float(recall_score(y, predictions, zero_division=0)) if len(y) else None,
              "f1": float(f1_score(y, predictions, zero_division=0)) if len(y) else None,
              "alarm_points": int(predictions.sum()),
              "invalid_points": int((~mask).sum()), "protocol": "MNRF-valid-endpoint-q995",
              "source_run_id": source.name, "source_manifest_sha256": _sha256(source_manifest_path),
              "source_time_binding_checked": True, "source_label_binding_checked": True,
              "source_segment_binding_checked": test_segment_ids is not None}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _common_metrics(method: str, score_path: Path, contract_dir: Path, labels: np.ndarray,
                    expected_indices: np.ndarray):
    """Validate a non-CATCH common artifact against the same frozen contract."""
    from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    evidence = _json(score_path.parent.parent / "evidence" / "baseline.json")
    if evidence.get("status") != "complete" or evidence.get("method") != method:
        raise ValueError(f"{method}: execution evidence is not complete")
    if Path(evidence.get("score_path", "")).resolve() != score_path.resolve():
        raise ValueError(f"{method}: evidence score path mismatch")
    if evidence.get("score_sha256") != _sha256(score_path):
        raise ValueError(f"{method}: score hash mismatch")
    contract = _json(contract_dir / "common_contract.json")
    test_file = Path(contract["catch_test"]["path"])
    if contract["catch_test"].get("sha256") != _sha256(test_file):
        raise ValueError(f"{method}: contract test hash mismatch")
    if not np.array_equal(_strict_indices(np.load(test_file, allow_pickle=False), "contract test"), expected_indices):
        raise ValueError(f"{method}: contract coordinates differ")
    payload = _load_npz(score_path)
    for key in ("scores", "predictions", "time_indices", "threshold", "actual", "time_values_ns"):
        if key not in payload:
            raise ValueError(f"{method}: score artifact lacks {key}")
    indices = _strict_indices(payload["time_indices"], f"{method} time_indices")
    if not np.array_equal(indices, expected_indices):
        raise ValueError(f"{method}: score coordinates differ from common contract")
    contract_times = np.asarray(np.load(contract_dir / "time_values_ns.npy", allow_pickle=False))
    contract_labels = np.asarray(np.load(contract_dir / "labels.npy", allow_pickle=False))
    if not np.array_equal(payload["time_values_ns"], contract_times[indices]):
        raise ValueError(f"{method}: time coordinates differ from contract")
    if not np.array_equal(payload["actual"], contract_labels[indices]):
        raise ValueError(f"{method}: labels differ from contract")
    scores = np.asarray(payload["scores"])
    predictions = np.asarray(payload["predictions"])
    threshold = float(np.asarray(payload["threshold"]))
    if len(scores) != len(indices) or not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError(f"{method}: invalid finite score state")
    if not np.array_equal(predictions.astype(np.int8), (scores > threshold).astype(np.int8)):
        raise ValueError(f"{method}: predictions do not implement strict score > threshold")
    actual = contract_labels[indices].astype(np.int64, copy=False)
    boundary = None
    full_valid = np.zeros(len(contract_labels), dtype=bool)
    full_valid[indices] = True
    full_pred = np.zeros(len(contract_labels), dtype=np.int64)
    full_pred[indices] = predictions.astype(np.int64)
    affiliation = _affiliation_fallback(contract_labels, full_pred, full_valid, contract)
    return {"method": method, "threshold": threshold, "valid_points": int(len(indices)),
            "total_points": int(len(contract_labels)),
            "coverage": float(len(indices) / max(len(contract_labels), 1)),
            "auc_roc": float(roc_auc_score(actual, scores)) if np.unique(actual).size == 2 else None,
            "auc_pr": float(average_precision_score(actual, scores)),
            "precision": float(precision_score(actual, predictions, zero_division=0)),
            "recall": float(recall_score(actual, predictions, zero_division=0)),
            "f1": float(f1_score(actual, predictions, zero_division=0)),
            "fpr": float(((predictions == 1) & (actual == 0)).sum() / max((actual == 0).sum(), 1)),
            "affiliation": affiliation, "protocol": "common-protocol", "boundary_ids": boundary}


def _load_npz(path):
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _affiliation_fallback(actual, predictions, valid, contract):
    # Keep the shared segment-macro implementation; absent boundary IDs are
    # explicitly recorded as unknown rather than inventing entity joins.
    from ts_benchmark.baselines.mnrf_view_probe.evaluation import affiliation_f_segment_macro
    return affiliation_f_segment_macro(actual=np.asarray(actual), predicted=np.asarray(predictions),
                                       valid_mask=np.asarray(valid), boundary_ids=None)


def common_compare(output_dir: Path):
    """Read-only eight-task comparison of MNRF, CATCH-common, IF and PCA."""
    import sys
    from sklearn.metrics import f1_score, precision_score, recall_score
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.evaluate_mnrf_view_probe_common import evaluate as strict_evaluate
    rows = []
    for task in TASKS:
        task_status = _json(COMMON_BATCH / task / "task_status.json")
        source = Path(task_status["source_path"]).resolve()
        contract_dir = COMMON_BATCH / task / "common_contract"
        expected = _strict_indices(np.load(contract_dir / "catch_test_indices.npy", allow_pickle=False), f"{task} common test")
        labels_path = contract_dir / "labels.npy"
        splits = tuple(str(contract_dir / f"{name}_indices.npy") for name in ("catch_train", "catch_validation", "catch_calibration"))
        catch_output = Path(task_status["methods"]["CATCH-common"]["output"])
        catch_path = catch_output / "results" / "aligned_scores.npz"
        common_manifest = catch_output / "results" / "common_manifest.json"
        probe_path = source / "results" / "aligned_scores.npz"
        strict = strict_evaluate(str(probe_path), str(catch_path), str(labels_path),
                                 split_paths=splits, common_manifest=str(common_manifest), strict_common=True)
        labels = np.asarray(np.load(labels_path, allow_pickle=False))
        for method_name, key in (("MNRF-Full-GMM", "probe"), ("CATCH-common", "catch")):
            metric = strict[key]
            affiliation = metric.get("affiliation", {})
            score_payload = _load_npz(probe_path if key == "probe" else catch_path)
            score_values = np.asarray(score_payload["scores"], dtype=np.float64)
            score_predictions = (score_values > float(metric["threshold"])).astype(np.int64)
            actual = labels[expected]
            rows.append({"dataset": task, "method": method_name, "protocol": "common-protocol",
                         "source_run_id": source.name, "valid_points": strict["common_count"],
                         "total_points": len(labels), "coverage": strict["coverage"],
                         "threshold": metric.get("threshold"), "auc_roc": metric.get("auc_roc"),
                         "auc_pr": metric.get("auc_pr"),
                         "precision": float(precision_score(actual, score_predictions, zero_division=0)),
                         "recall": float(recall_score(actual, score_predictions, zero_division=0)),
                         "f1": float(f1_score(actual, score_predictions, zero_division=0)),
                         "fpr": float(((score_predictions == 1) & (actual == 0)).sum() / max((actual == 0).sum(), 1)),
                         "affiliation_f": affiliation.get("affiliation_f_segment_macro", affiliation.get("affiliation_f")),
                         "affiliation_defined_segments": affiliation.get("defined_segment_count"),
                         "status": "verified"})
        for method in ("IsolationForest", "PCA"):
            output = Path(task_status["methods"][method]["output"])
            metric = _common_metrics(method, output / "results" / "aligned_scores.npz",
                                     contract_dir, labels, expected)
            metric.update({"dataset": task, "source_run_id": source.name, "status": "verified"})
            metric["affiliation_f"] = metric.get("affiliation", {}).get("affiliation_f_segment_macro",
                                                                            metric.get("affiliation", {}).get("affiliation_f"))
            metric["affiliation_defined_segments"] = metric.get("affiliation", {}).get("defined_segment_count")
            rows.append(metric)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "common_comparison.json"
    json_path.write_text(json.dumps({"schema": "mnrf-common-comparison-v1",
                                     "protocol": "common-protocol",
                                     "rows": rows,
                                     "source_note": "read-only reuse; no new scoring/fitting/calibration"},
                                    indent=2, sort_keys=True), encoding="utf-8")
    csv_path = output_dir / "common_comparison.csv"
    fields = ["dataset", "method", "protocol", "source_run_id", "valid_points", "total_points",
              "coverage", "threshold", "auc_roc", "auc_pr", "precision", "recall", "f1", "fpr",
              "affiliation_f", "affiliation_defined_segments", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return {"rows": len(rows), "datasets": len(TASKS), "path": str(csv_path), "json": str(json_path)}


def score_cold_start(task: str, source: Path, output: Path):
    """Generate only the missing prefix with the frozen encoder and GMM.

    This is deliberately an explicit later operation.  It loads the validated
    checkpoint, input statistics, GMM and fixed noise; no training or fitting
    is performed.  Windows are admitted only when their global coordinates,
    timestamps and frozen segment IDs are contiguous.  The current command is
    never run by inventory/plan/common-compare.
    """
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cold-start output: {output}")
    from ts_benchmark.baselines.mnrf_view_probe.probe import load_probe
    import torch
    source = source.resolve()
    task_record = _source_plan(task, source)
    if task_record["cold_start_extension"]["status"] != "ready-frozen-forward":
        raise ValueError(f"{task}: cold-start prefix is not legally scoreable: {task_record['cold_start_extension']['reason']}")
    with np.load(source / "analysis_inputs.npz", allow_pickle=False) as inputs:
        test_global = _strict_indices(inputs["test_global_indices"], f"{task} test_global_indices")
        test_segments = np.asarray(inputs["test_segment_ids"])
        all_global = _strict_indices(inputs["all_global_indices"], f"{task} all_global_indices")
        all_segments = np.asarray(inputs["all_segment_ids"])
        all_times = np.asarray(inputs["all_time_values_ns"])
        if len(test_segments) != len(test_global) or not np.array_equal(test_segments, all_segments[int(task_record["official_train_length"]):]):
            raise ValueError(f"{task}: test segment IDs are not the frozen full-axis slice")
    with np.load(source / "cache/normalized_values.npy", mmap_mode="r", allow_pickle=False) as normalized:
        if normalized.ndim != 2 or normalized.shape[0] != len(all_global) or normalized.dtype != np.float32:
            raise ValueError(f"{task}: normalized source cache is not a complete float32 row axis")
        prefix = np.asarray(task_record["cold_start_extension"]["coordinates"], dtype=np.int64)
        if len(prefix) == 0:
            raise ValueError(f"{task}: no prefix coordinates require cold-start generation")
        seq_len = int(task_record["seq_len"])
        positions = []
        for local in prefix:
            endpoint = int(test_global[local])
            start = endpoint - seq_len + 1
            if start < 0 or endpoint >= len(all_global):
                raise ValueError(f"{task}: prefix endpoint {int(local)} has no complete history window")
            if not np.array_equal(all_global[start:endpoint + 1], np.arange(all_global[start], endpoint + 1, dtype=np.int64)):
                raise ValueError(f"{task}: prefix endpoint {int(local)} crosses a global coordinate gap")
            if np.any(all_segments[start:endpoint + 1] != all_segments[endpoint]):
                raise ValueError(f"{task}: prefix endpoint {int(local)} crosses a frozen segment boundary")
            times = np.diff(all_times[start:endpoint + 1])
            if len(times) and (not np.isfinite(times).all() or np.any(times <= 0)):
                raise ValueError(f"{task}: prefix endpoint {int(local)} has invalid time continuity")
            positions.append(endpoint)

        probe = load_probe(source, device="cpu")
        # Reproduce the already-saved endpoint score before combining a prefix.
        # This is an in-memory cache/GMM check, not a new fit or calibration.
        probe._log_event = lambda *args, **kwargs: None
        with np.load(source / "results/aligned_scores.npz", allow_pickle=False) as old:
            old_scores = np.asarray(old["scores"])
            restored = probe.score_cached_test()
            if not np.array_equal(np.asarray(restored["time_indices"]), np.asarray(old["time_indices"])) or not np.allclose(restored["scores"], old_scores, rtol=0, atol=1e-12):
                raise ValueError(f"{task}: existing valid endpoint score cannot be reproduced; cold-start stopped")

        checkpoint = source / "checkpoints" / "best_encoder.pt"
        source_manifest = _json(source / "probe_manifest.json")
        declared_checkpoint = source_manifest.get("checkpoint_sha256", {}).get("best_encoder.pt")
        if not checkpoint.is_file() or not declared_checkpoint or _sha256(checkpoint) != declared_checkpoint:
            raise ValueError(f"{task}: frozen best encoder checkpoint is missing or changed")
        config = dict(source_manifest.get("config", {}))
        for key in ("resolved_layout", "selected_indices", "selected_dim"):
            config.pop(key, None)
        config["output_dir"] = str(source)
        config["device"] = "cpu"
        encoder_probe = probe.__class__.from_kwargs(**config)
        trainer = encoder_probe._make_trainer(len(source_manifest.get("variable_names", [])))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        encoder_state = state.get("encoder", state)
        trainer.encoder.load_state_dict(encoder_state)
        trainer.encoder.eval()
        # Load and validate the normalization statistics even though the source
        # normalized cache is used directly; this records the exact protocol.
        input_stats = source / "input_statistics.npz"
        if source_manifest.get("input_statistics_sha256") != _sha256(input_stats):
            raise ValueError(f"{task}: input statistics binding is stale")
        with np.load(input_stats, allow_pickle=False) as stats:
            input_mean = np.asarray(stats["mean"], dtype=np.float32)
            input_scale = np.asarray(stats["scale"], dtype=np.float32)
        if not np.isfinite(input_mean).all() or not np.isfinite(input_scale).all() or np.any(input_scale <= 0):
            raise ValueError(f"{task}: input statistics are invalid")
        probe.trainer = trainer
        features = np.empty((len(prefix), len(source_manifest["variable_names"]),
                             int(source_manifest.get("cache_feature_dim", probe.config.layout.total_dim))), dtype=np.float32)
        batch_size = max(1, min(int(probe.config.batch_size), 256))
        with torch.no_grad():
            for begin in range(0, len(positions), batch_size):
                batch_endpoints = positions[begin:begin + batch_size]
                windows = np.stack([normalized[e - seq_len + 1:e + 1] for e in batch_endpoints])
                values = trainer.encoder.encode_views(torch.as_tensor(windows, dtype=torch.float32))[
                    "features"].detach().cpu().numpy().astype(np.float32, copy=False)
                features[begin:begin + len(values)] = values
        # GMM scores need the frozen feature statistics, not the trainer's
        # freshly constructed (unfitted) scaler.
        probe.trainer = None
        nll = np.asarray(probe._gmm_scores(features, split_name="cold_start"))
        center = np.asarray(probe._probe_stats["center"])
        scale = np.asarray(probe._probe_stats["scale"])
        threshold = float(probe._probe_stats["threshold"])
        prefix_scores = np.max((nll - center) / scale, axis=1)
        prefix_predictions = (prefix_scores > threshold).astype(np.int64)
    with np.load(source / "results/aligned_scores.npz", allow_pickle=False) as old:
        combined = {name: np.asarray(old[name]) for name in old.files}
    combined_scores = np.asarray(combined["scores"]).copy()
    combined_predictions = np.asarray(combined["predictions"]).copy()
    combined_mask = np.asarray(combined["valid_mask"], dtype=bool).copy()
    combined_scores[prefix] = prefix_scores
    combined_predictions[prefix] = prefix_predictions
    combined_mask[prefix] = True
    combined["scores"] = combined_scores
    combined["predictions"] = combined_predictions
    combined["valid_mask"] = combined_mask
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{task}-cold-start-", dir=str(output.parent)))
    try:
        (temporary / "results").mkdir()
        np.savez_compressed(temporary / "results" / "aligned_scores.npz", **combined)
        record = {"schema": "mnrf-comparison-adapter-cold-start-v1", "task": task,
                  "source_path": str(source), "source_run_id": source.name,
                  "mode": "frozen-encoder-prefix-forward-gmm-no-fit-no-calibration",
                  "prefix_local_indices": prefix.tolist(), "prefix_global_indices": [int(x) for x in positions],
                  "prefix_count": int(len(prefix)), "threshold_reused": threshold,
                  "input_statistics_sha256": _sha256(source / "input_statistics.npz"),
                  "checkpoint_sha256": _sha256(checkpoint),
                  "source_aligned_scores_sha256": _sha256(source / "results/aligned_scores.npz"),
                  "source_valid_endpoint_reproduction": True,
                  "history_policy": "within-frozen-segment-only",
                  "future_rows": 0, "noise_seed": int(probe.config.noise_seed),
                  "gmm_compute_dtype": probe.config.gmm_compute_dtype,
                  "no_training_fit_or_recalibration": True}
        (temporary / "adapter_manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, output)
        record["output_dir"] = str(output.resolve())
        return record
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def synthetic_check():
    all_global = np.arange(12, dtype=np.int64)
    all_segments = np.array([0] * 6 + [1] * 6, dtype=np.int32)
    test_global = all_global[6:]
    test_segments = all_segments[6:]
    valid = np.array([False, False, True, True, True, True], dtype=bool)
    assert np.array_equal(test_global, all_global[6:])
    assert np.array_equal(test_segments, all_segments[6:])
    assert np.flatnonzero(valid).tolist() == [2, 3, 4, 5]
    # A prefix endpoint would cross the train/test segment boundary and is not
    # silently admitted as a score.
    assert not np.all(all_segments[5:7] == all_segments[6])
    with tempfile.TemporaryDirectory(prefix="mnrf-adapter-synthetic-") as directory:
        root = Path(directory)
        source = root / "source"
        bundle = root / "bundle"
        source.mkdir(parents=True)
        (bundle / "results").mkdir(parents=True)
        labels = np.array([0, 0, 0, 1, 1, 0], dtype=np.int8)
        time_values = np.arange(6, dtype=np.int64) * 60_000_000_000
        global_indices = np.arange(6, dtype=np.int64)
        segments = np.zeros(6, dtype=np.int32)
        np.savez(source / "analysis_inputs.npz", test_labels=labels,
                 test_global_indices=global_indices,
                 test_time_values_ns=time_values,
                 test_segment_ids=segments)
        scores = np.array([np.nan, np.nan, 0.1, 0.9, 0.8, 0.2], dtype=np.float32)
        mask = np.array([False, False, True, True, True, True], dtype=bool)
        predictions = np.array([-1, -1, 0, 1, 1, 0], dtype=np.int64)
        source_npz = source / "results" / "aligned_scores.npz"
        (source / "results").mkdir()
        np.savez(source_npz, scores=scores, predictions=predictions,
                 valid_mask=mask, threshold=np.asarray(0.5),
                 time_indices=global_indices, time_values_ns=time_values,
                 actual=labels)
        shutil.copy2(source_npz, bundle / "results" / "aligned_scores.npz")
        source_manifest = {"schema": "mnrf-test", "artifact_type": "probe_head",
                           "config": {"task_name": "synthetic.csv"}}
        (source / "probe_manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
        (source / "run_status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        (source / "heads").mkdir()
        np.savez(source / "heads" / "calibration.npz", threshold=np.asarray(0.5))
        mask_hash = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
        adapter_manifest = {"schema": "mnrf-comparison-adapter-score-v1",
                            "task": "synthetic", "source_path": str(source.resolve()),
                            "source_run_id": source.name,
                            "source_aligned_scores_sha256": _sha256(source_npz),
                            "output_aligned_scores_sha256": _sha256(bundle / "results" / "aligned_scores.npz"),
                            "source_valid_mask_sha256": mask_hash}
        (bundle / "adapter_manifest.json").write_text(json.dumps(adapter_manifest), encoding="utf-8")
        report = root / "evaluation.json"
        result = evaluate(bundle, source, report)
        assert result["valid_points"] == 4 and report.is_file()
        bad = root / "bad"
        shutil.copytree(bundle, bad)
        with np.load(bad / "results/aligned_scores.npz", allow_pickle=False) as values:
            bad_payload = {name: np.asarray(values[name]) for name in values.files}
        bad_payload["time_indices"] = bad_payload["time_indices"].copy()
        bad_payload["time_indices"][2] += 1
        np.savez(bad / "results/aligned_scores.npz", **bad_payload)
        bad_manifest = _json(bad / "adapter_manifest.json")
        bad_manifest["output_aligned_scores_sha256"] = _sha256(bad / "results/aligned_scores.npz")
        (bad / "adapter_manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
        try:
            evaluate(bad, source, root / "bad.json")
        except ValueError:
            pass
        else:
            raise AssertionError("coordinate-shifted same-length bundle was accepted")
    return {"status": "passed", "valid_local": [2, 3, 4, 5], "cold_start": [0, 1],
            "boundary_crossing_rejected": True, "evaluation_bundle_written": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inventory", "plan", "common-compare", "score",
                                            "score-cold-start", "evaluate", "synthetic"), required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "inventory":
        print(json.dumps(inventory(args.output_dir), indent=2, sort_keys=True))
    elif args.mode == "plan":
        print(json.dumps(plan(args.output_dir, ROOT / "result/score/MNRF-ViewProbe"), indent=2, sort_keys=True))
    elif args.mode == "common-compare":
        print(json.dumps(common_compare(args.output_dir), indent=2, sort_keys=True))
    elif args.mode == "synthetic":
        print(json.dumps(synthetic_check(), indent=2, sort_keys=True))
    elif args.mode == "score-cold-start":
        if not args.task or not args.source_dir or not args.output_dir:
            parser.error("score-cold-start requires --task, --source-dir and --output-dir")
        print(json.dumps(score_cold_start(args.task, args.source_dir.resolve(), args.output_dir.resolve()),
                         indent=2, sort_keys=True))
    elif args.mode == "score":
        if not args.task or not args.source_dir or not args.output_dir:
            parser.error("score requires --task, --source-dir and --output-dir")
        print(json.dumps(score_cached(args.task, args.source_dir.resolve(), args.output_dir.resolve()), indent=2, sort_keys=True))
    else:
        if not args.bundle or not args.source_dir or not args.output:
            parser.error("evaluate requires --bundle, --source-dir and --output")
        print(json.dumps(evaluate(args.bundle.resolve(), args.source_dir.resolve(), args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
