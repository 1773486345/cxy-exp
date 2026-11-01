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
    # A frozen segment id is not, by itself, proof that the row immediately
    # before a test sequence belongs to the same physical stream.  Keep that
    # distinction explicit in the plan so a later cold-start command cannot
    # silently borrow history across known/unknown channel boundaries.
    continuity_evidence = {
        "MSL": ("blocked-known-cross-channel",
                "T-9 training tail and C-1 test head are different telemetry channels; "
                "the shared concatenated segment id is not permission to cross the boundary"),
        "Creditcard": ("blocked-entity-session-unproven",
                       "no authoritative entity/session boundary is recorded for the row adjacency"),
        "CICIDS": ("blocked-entity-session-unproven",
                   "no authoritative entity/session boundary is recorded for the row adjacency"),
    }
    continuity_status, continuity_reason = continuity_evidence.get(
        task, ("within-segment-monotonic-only",
               "global rows, timestamps and frozen segment ids are continuous; no additional entity column is recorded"))
    cold_status = continuity_status if continuity_status.startswith("blocked-") else ("ready-frozen-forward" if prefix_safe else "blocked-boundary")
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
        "continuity_evidence": {"status": continuity_status, "reason": continuity_reason,
                                 "prefix_window_check": {"passed": prefix_safe, "reason": prefix_reason}},
        "information_policy": {"window": f"[{seq_len}-1 past rows, current endpoint]",
                                "future_rows": 0,
                                "missing_history_mask_supported": False,
                                "missing_history_mask_reason": "frozen encoder receives normalized window values only; no mask channel is recorded",
                                "source_temporal_rule": identity.get("temporal_rule", {}),
                                "permission": "only rows within the frozen source segment and saved endpoint cache"},
        "cold_start_extension": {"status": cold_status,
                                  "coordinates": prefix_coordinates.tolist(),
                                  "reason": (continuity_reason if cold_status.startswith("blocked-") else
                                              ("source has no saved prefix features; explicit frozen-encoder forward is required"
                                               if prefix_safe else prefix_reason))},
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
    schema = adapter_manifest.get("schema")
    if schema not in {"mnrf-comparison-adapter-score-v1", "mnrf-comparison-adapter-cold-start-v1"}:
        raise ValueError("unsupported adapter manifest schema")
    cold_start = schema == "mnrf-comparison-adapter-cold-start-v1"
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
    source_mask = np.asarray(source_payload["valid_mask"], dtype=bool)
    bundle_mask = np.asarray(bundle_payload["valid_mask"], dtype=bool)
    if source_mask.ndim != 1 or bundle_mask.ndim != 1 or len(source_mask) != len(bundle_mask):
        raise ValueError("source and adapter valid masks have incompatible full-axis lengths")
    if not cold_start and not np.array_equal(source_mask, bundle_mask):
        raise ValueError("adapter valid mask differs from source")
    if not cold_start and not np.isclose(float(source_payload["threshold"]), float(bundle_payload["threshold"]), rtol=0, atol=1e-12):
        raise ValueError("adapter threshold differs from source")
    if adapter_manifest.get("output_aligned_scores_sha256") != _sha256(bundle / "results" / "aligned_scores.npz"):
        raise ValueError("adapter output score hash is stale")
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
    if labels.ndim != 1 or len(labels) != len(test_global) or len(test_times) != len(test_global):
        raise ValueError("source test axis fields have incompatible lengths")
    def _compact(payload, name):
        mask = np.asarray(payload["valid_mask"], dtype=bool)
        aligned_scores = np.asarray(payload.get("aligned_scores", payload["scores"]))
        aligned_predictions = np.asarray(payload.get("aligned_predictions", payload["predictions"]))
        if len(aligned_scores) != len(mask) or len(aligned_predictions) != len(mask):
            raise ValueError(f"{name}: aligned arrays and valid mask have incompatible lengths")
        if np.any(~mask & np.isfinite(aligned_scores)) or np.any(mask & ~np.isfinite(aligned_scores)):
            raise ValueError(f"{name}: invalid finite aligned score state")
        scores = np.asarray(payload["scores"])
        predictions = np.asarray(payload["predictions"])
        if len(scores) == len(mask):
            if not np.array_equal(scores[mask], aligned_scores[mask], equal_nan=True):
                raise ValueError(f"{name}: full scores do not map to aligned valid scores")
            scores = scores[mask]
        elif len(scores) != int(mask.sum()):
            raise ValueError(f"{name}: scores are neither full-axis nor compact")
        if len(predictions) == len(mask):
            if not np.array_equal(predictions[mask], aligned_predictions[mask]):
                raise ValueError(f"{name}: full predictions do not map to aligned valid predictions")
            predictions = predictions[mask]
        elif len(predictions) != int(mask.sum()):
            raise ValueError(f"{name}: predictions are neither full-axis nor compact")
        threshold = float(np.asarray(payload["threshold"]))
        if not np.isfinite(threshold) or not np.array_equal(predictions.astype(np.int8), (scores > threshold).astype(np.int8)):
            raise ValueError(f"{name}: predictions do not implement strict score > threshold")
        local = np.asarray(payload.get("local_indices", np.flatnonzero(mask)))
        local = _strict_indices(local, f"{name} local_indices")
        if len(local) != len(scores) or not np.array_equal(local, np.flatnonzero(mask)):
            raise ValueError(f"{name}: compact local indices do not match valid mask")
        global_values = _strict_indices(payload.get("time_indices", test_global[local]), f"{name} time_indices")
        if not np.array_equal(global_values, test_global[local]):
            raise ValueError(f"{name}: global coordinates do not map to test axis")
        times = np.asarray(payload.get("time_values_ns", test_times[local]))
        actual = np.asarray(payload.get("actual", labels[local]))
        if not np.array_equal(times, test_times[local]) or not np.array_equal(actual, labels[local]):
            raise ValueError(f"{name}: compact time/label fields do not map to test axis")
        return {"mask": mask, "scores": scores, "predictions": predictions, "local": local,
                "global": global_values, "times": times, "actual": actual,
                "aligned_scores": aligned_scores, "aligned_predictions": aligned_predictions,
                "threshold": threshold}

    source_compact = _compact(source_payload, "source")
    bundle_compact = _compact(bundle_payload, "adapter")
    if not np.isclose(source_compact["threshold"], bundle_compact["threshold"], rtol=0, atol=1e-12):
        raise ValueError("adapter threshold differs from source")
    if cold_start:
        prefix = _strict_indices(adapter_manifest.get("prefix_local_indices", []), "cold-start prefix_local_indices")
        if len(prefix) == 0 or np.any(prefix >= len(labels)):
            raise ValueError("cold-start prefix is empty or outside the test axis")
        expected_mask = source_compact["mask"].copy()
        if np.any(expected_mask[prefix]) or not np.array_equal(prefix, np.flatnonzero(~expected_mask)[:len(prefix)]):
            raise ValueError("cold-start prefix is not exactly the declared missing prefix")
        expected_mask[prefix] = True
        if not np.array_equal(bundle_compact["mask"], expected_mask):
            raise ValueError("cold-start added points are not exactly the declared prefix")
        old_positions = np.searchsorted(bundle_compact["local"], source_compact["local"])
        if (len(old_positions) != len(source_compact["local"]) or
                not np.array_equal(bundle_compact["local"][old_positions], source_compact["local"])):
            raise ValueError("cold-start output dropped or reordered an old valid point")
        for field in ("scores", "predictions", "global", "times", "actual"):
            if not np.array_equal(bundle_compact[field][old_positions], source_compact[field], equal_nan=True):
                raise ValueError(f"cold-start changed an old valid {field} value")
        if not np.array_equal(bundle_compact["global"], test_global[bundle_compact["local"]]):
            raise ValueError("cold-start global coordinate mapping is inconsistent")
    else:
        for field in ("scores", "predictions", "local", "global", "times", "actual"):
            if not np.array_equal(source_compact[field], bundle_compact[field], equal_nan=True):
                raise ValueError(f"adapter {field} differs from source")
    mask = bundle_compact["mask"]
    scores = bundle_compact["scores"]
    predictions = bundle_compact["predictions"]
    local_indices = bundle_compact["local"]
    threshold = bundle_compact["threshold"]
    with np.load(calibration_path, allow_pickle=False) as calibration:
        calibration_threshold = float(np.asarray(calibration["threshold"]))
    if not np.isclose(threshold, calibration_threshold, rtol=0, atol=1e-12):
        raise ValueError("adapter threshold differs from frozen calibration threshold")
    y = labels[local_indices]
    s = scores
    result = {"valid_points": int(len(scores)), "total_test_points": int(len(labels)),
              "coverage": float(mask.sum() / max(len(labels), 1)), "threshold": threshold,
              "roc_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, s)) if np.unique(y).size == 2 else None,
              "average_precision": float(average_precision_score(y, s)) if np.unique(y).size else None,
              "precision": float(precision_score(y, predictions, zero_division=0)) if len(y) else None,
              "recall": float(recall_score(y, predictions, zero_division=0)) if len(y) else None,
              "f1": float(f1_score(y, predictions, zero_division=0)) if len(y) else None,
              "alarm_points": int(predictions.sum()),
              "invalid_points": int((~mask).sum()), "new_valid_points": int(mask.sum() - source_mask.sum()) if cold_start else 0,
              "old_valid_points": int(source_mask.sum()),
              "protocol": "MNRF-cold-start-q995" if cold_start else "MNRF-valid-endpoint-q995",
              "source_run_id": source.name, "source_manifest_sha256": _sha256(source_manifest_path),
              "source_time_binding_checked": True, "source_label_binding_checked": True,
              "source_segment_binding_checked": test_segment_ids is not None}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _common_metrics(method: str, score_path: Path, contract_dir: Path, labels: np.ndarray,
                    expected_indices: np.ndarray, official_test_points: int):
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
            "official_test_points": int(official_test_points),
            "full_input_points": int(len(contract_labels)),
            "total_points": int(official_test_points),
            "coverage": float(len(indices) / max(official_test_points, 1)),
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
        source_inputs = source / "analysis_inputs.npz"
        with np.load(source_inputs, allow_pickle=False) as source_axis:
            official_test_points = int(len(source_axis["test_labels"]))
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
                         "official_test_points": official_test_points,
                         "full_input_points": len(labels), "total_points": official_test_points,
                         "coverage": float(strict["common_count"] / max(official_test_points, 1)),
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
                                     contract_dir, labels, expected, official_test_points)
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
    fields = ["dataset", "method", "protocol", "source_run_id", "valid_points",
              "official_test_points", "full_input_points", "total_points", "coverage",
              "threshold", "auc_roc", "auc_pr", "precision", "recall", "f1", "fpr",
              "affiliation_f", "affiliation_defined_segments", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return {"rows": len(rows), "datasets": len(TASKS), "path": str(csv_path), "json": str(json_path)}


def refresh_common_coverage(output_dir: Path):
    """Refresh denominator fields from existing rows without rescoring anything."""
    json_path = output_dir / "common_comparison.json"
    csv_path = output_dir / "common_comparison.csv"
    if not json_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError("existing common comparison JSON/CSV is required")
    payload = _json(json_path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("existing common comparison has no rows")
    official = {}
    full_input = {}
    for task in TASKS:
        task_status = _json(COMMON_BATCH / task / "task_status.json")
        source = Path(task_status["source_path"]).resolve()
        with np.load(source / "analysis_inputs.npz", allow_pickle=False) as values:
            official[task] = int(len(values["test_labels"]))
        contract = COMMON_BATCH / task / "common_contract" / "labels.npy"
        full_input[task] = int(len(np.load(contract, mmap_mode="r", allow_pickle=False)))
    for row in rows:
        task = row.get("dataset")
        if task not in official:
            raise ValueError(f"unknown dataset in existing common comparison: {task}")
        row["official_test_points"] = official[task]
        row["full_input_points"] = full_input[task]
        row["total_points"] = official[task]
        row["coverage"] = float(row["valid_points"] / max(official[task], 1))
    payload["coverage_definition"] = "valid_points / official_test_points"
    payload["full_input_points_definition"] = "length of the complete input/label axis; not a coverage denominator"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    fields = ["dataset", "method", "protocol", "source_run_id", "valid_points",
              "official_test_points", "full_input_points", "total_points", "coverage",
              "threshold", "auc_roc", "auc_pr", "precision", "recall", "f1", "fpr",
              "affiliation_f", "affiliation_defined_segments", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return {"rows": len(rows), "path": str(csv_path), "read_only_refresh": True}


def _load_npy_memmap(path: Path):
    """Load a read-only .npy array without treating the memmap as a context manager."""
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(values, np.ndarray) or values.ndim == 0:
        raise ValueError(f"expected a readable ndarray at {path}")
    return values


def _assemble_cold_start_payload(old, prefix, prefix_scores, prefix_predictions,
                                 test_global, test_times, test_labels, threshold):
    """Fill the full test axis, then derive the compact native representation."""
    old_mask = np.asarray(old["valid_mask"], dtype=bool)
    old_aligned_scores = np.asarray(old.get("aligned_scores", old["scores"])).copy()
    old_aligned_predictions = np.asarray(old.get("aligned_predictions", old["predictions"])).copy()
    if len(old_mask) != len(test_global) or len(old_aligned_scores) != len(old_mask):
        raise ValueError("cold-start source full-axis fields have incompatible lengths")
    prefix = _strict_indices(prefix, "cold-start prefix_local_indices")
    if len(prefix) == 0 or np.any(prefix >= len(old_mask)):
        raise ValueError("cold-start prefix is empty or outside the test axis")
    if not np.array_equal(prefix, np.flatnonzero(~old_mask)[:len(prefix)]):
        raise ValueError("cold-start prefix is not exactly the initial invalid range")
    if len(prefix_scores) != len(prefix) or len(prefix_predictions) != len(prefix):
        raise ValueError("cold-start prefix score lengths differ")
    if not np.isfinite(prefix_scores).all():
        raise ValueError("cold-start prefix scores are non-finite")
    new_mask = old_mask.copy()
    new_aligned_scores = old_aligned_scores.copy()
    new_aligned_predictions = old_aligned_predictions.copy()
    new_mask[prefix] = True
    new_aligned_scores[prefix] = np.asarray(prefix_scores)
    new_aligned_predictions[prefix] = np.asarray(prefix_predictions)
    if not np.array_equal(new_aligned_predictions[new_mask],
                          (new_aligned_scores[new_mask] > threshold).astype(new_aligned_predictions.dtype)):
        raise ValueError("cold-start predictions do not implement strict score > threshold")
    new_aligned_actual = np.asarray(old.get("aligned_actual", np.full(len(old_mask), -1, dtype=np.int8))).copy()
    if len(new_aligned_actual) != len(old_mask):
        raise ValueError("cold-start aligned_actual has wrong length")
    new_aligned_actual[prefix] = np.asarray(test_labels)[prefix]
    local = np.flatnonzero(new_mask).astype(np.int64)
    old_local = np.flatnonzero(old_mask).astype(np.int64)
    old_scores = np.asarray(old["scores"])
    old_predictions = np.asarray(old["predictions"])
    if len(old_scores) == len(old_mask):
        old_scores = old_scores[old_mask]
    if len(old_predictions) == len(old_mask):
        old_predictions = old_predictions[old_mask]
    if len(old_scores) != len(old_local) or not np.array_equal(old_scores, old_aligned_scores[old_mask], equal_nan=True):
        raise ValueError("cold-start source compact scores do not map to its aligned scores")
    if len(old_predictions) != len(old_local) or not np.array_equal(old_predictions, old_aligned_predictions[old_mask]):
        raise ValueError("cold-start source compact predictions do not map to its aligned predictions")
    compact_scores = new_aligned_scores[new_mask].copy()
    compact_predictions = new_aligned_predictions[new_mask].copy()
    compact_actual = new_aligned_actual[new_mask].copy()
    payload = {name: np.asarray(value).copy() for name, value in old.items()}
    payload.update({"scores": compact_scores, "predictions": compact_predictions,
                    "time_indices": np.asarray(test_global)[local],
                    "time_values_ns": np.asarray(test_times)[local],
                    "local_indices": local, "actual": compact_actual,
                    "valid_mask": new_mask, "aligned_scores": new_aligned_scores,
                    "aligned_predictions": new_aligned_predictions,
                    "aligned_actual": new_aligned_actual,
                    "threshold": np.asarray(threshold)})
    return payload, old_local


def _validate_prefix_windows(prefix, test_global, all_global, all_segments, all_times, seq_len):
    """Return global endpoints after proving every prefix window is legal."""
    positions = []
    for local in _strict_indices(prefix, "cold-start prefix_local_indices"):
        if local >= len(test_global):
            raise ValueError(f"prefix endpoint {int(local)} is outside the test axis")
        endpoint = int(test_global[local])
        start = endpoint - int(seq_len) + 1
        if start < 0 or endpoint >= len(all_global):
            raise ValueError(f"prefix endpoint {int(local)} has no complete history window")
        if not np.array_equal(all_global[start:endpoint + 1], np.arange(all_global[start], endpoint + 1, dtype=np.int64)):
            raise ValueError(f"prefix endpoint {int(local)} crosses a global coordinate gap")
        if np.any(all_segments[start:endpoint + 1] != all_segments[endpoint]):
            raise ValueError(f"prefix endpoint {int(local)} crosses a known cross-channel/segment boundary")
        differences = np.diff(all_times[start:endpoint + 1])
        if len(differences) and (not np.isfinite(differences).all() or np.any(differences <= 0)):
            raise ValueError(f"prefix endpoint {int(local)} has invalid time continuity")
        positions.append(endpoint)
    return positions


def _write_cold_start_artifact(output: Path, source: Path, payload: dict, record: dict):
    """Write and self-validate a cold-start artifact before publishing it."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cold-start output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{record['task']}-cold-start-", dir=str(output.parent)))
    try:
        (temporary / "results").mkdir()
        score_path = temporary / "results" / "aligned_scores.npz"
        np.savez_compressed(score_path, **payload)
        published_record = dict(record)
        published_record["output_aligned_scores_sha256"] = _sha256(score_path)
        (temporary / "adapter_manifest.json").write_text(
            json.dumps(published_record, indent=2, sort_keys=True), encoding="utf-8")
        # Validate the exact bytes and mappings that will be published.  The
        # temporary report is removed with the staging directory.
        validation_report = temporary / "_cold_start_validation.json"
        evaluate(temporary, source, validation_report)
        validation_report.unlink()
        os.replace(temporary, output)
        published_record["output_dir"] = str(output.resolve())
        return published_record
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


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
        test_times = np.asarray(inputs["test_time_values_ns"])
        test_labels = np.asarray(inputs["test_labels"])
        all_global = _strict_indices(inputs["all_global_indices"], f"{task} all_global_indices")
        all_segments = np.asarray(inputs["all_segment_ids"])
        all_times = np.asarray(inputs["all_time_values_ns"])
        if len(test_segments) != len(test_global) or not np.array_equal(test_segments, all_segments[int(task_record["official_train_length"]):]):
            raise ValueError(f"{task}: test segment IDs are not the frozen full-axis slice")
    normalized = _load_npy_memmap(source / "cache/normalized_values.npy")
    try:
        if normalized.ndim != 2 or normalized.shape[0] != len(all_global) or normalized.dtype != np.float32:
            raise ValueError(f"{task}: normalized source cache is not a complete float32 row axis")
        prefix = np.asarray(task_record["cold_start_extension"]["coordinates"], dtype=np.int64)
        if len(prefix) == 0:
            raise ValueError(f"{task}: no prefix coordinates require cold-start generation")
        seq_len = int(task_record["seq_len"])
        positions = _validate_prefix_windows(prefix, test_global, all_global,
                                              all_segments, all_times, seq_len)

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
        # Re-encode a bounded set of already-saved endpoints with the newly
        # loaded frozen encoder.  Cached-score agreement alone cannot prove
        # that this forward path is reproducible.
        test_cache = _load_npy_memmap(source / "cache/test_features.npy")
        saved_endpoints = _load_npy_memmap(source / "cache/test_window_end_indices.npy")
        if len(test_cache) != len(saved_endpoints):
            raise ValueError(f"{task}: test cache and endpoint arrays differ")
        check_rows = np.unique(np.linspace(0, len(saved_endpoints) - 1,
                                           num=min(8, len(saved_endpoints)), dtype=np.int64))
        with torch.no_grad():
            check_windows = np.stack([normalized[int(saved_endpoints[i]) - seq_len + 1:int(saved_endpoints[i]) + 1]
                                      for i in check_rows])
            check_features = trainer.encoder.encode_views(torch.as_tensor(check_windows, dtype=torch.float32))["features"].detach().cpu().numpy()
        max_abs = float(np.max(np.abs(check_features - np.asarray(test_cache[check_rows], dtype=np.float32)))) if len(check_rows) else 0.0
        if not np.allclose(check_features, np.asarray(test_cache[check_rows]), rtol=1e-5, atol=1e-6):
            raise ValueError(f"{task}: frozen encoder endpoint reproduction failed (max_abs={max_abs})")

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
            old_payload = {name: np.asarray(old[name]) for name in old.files}
        combined, old_local = _assemble_cold_start_payload(
            old_payload, prefix, prefix_scores, prefix_predictions,
            test_global, test_times, test_labels, threshold)
    finally:
        del normalized
    record = {"schema": "mnrf-comparison-adapter-cold-start-v1", "task": task,
              "source_path": str(source), "source_run_id": source.name,
              "mode": "frozen-encoder-prefix-forward-gmm-no-fit-no-calibration",
              "prefix_local_indices": prefix.tolist(), "prefix_global_indices": [int(x) for x in positions],
              "prefix_count": int(len(prefix)), "threshold_reused": threshold,
              "input_statistics_sha256": _sha256(source / "input_statistics.npz"),
              "checkpoint_sha256": _sha256(checkpoint),
              "source_aligned_scores_sha256": _sha256(source / "results/aligned_scores.npz"),
              "source_valid_endpoint_reproduction": True,
              "encoder_endpoint_reproduction": {"checked": int(len(check_rows)),
                                                 "max_abs": max_abs, "rtol": 1e-5, "atol": 1e-6},
              "old_valid_local_indices": old_local.tolist(),
              "source_valid_mask_sha256": hashlib.sha256(
                  np.ascontiguousarray(old_payload["valid_mask"], dtype=bool).tobytes()).hexdigest(),
              "history_policy": "within-frozen-segment-only",
              "future_rows": 0, "noise_seed": int(probe.config.noise_seed),
              "gmm_compute_dtype": probe.config.gmm_compute_dtype,
              "no_training_fit_or_recalibration": True}
    return _write_cold_start_artifact(output, source, combined, record)


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
        scores = np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float32)
        mask = np.array([False, False, True, True, True, True], dtype=bool)
        predictions = np.array([0, 1, 1, 0], dtype=np.int64)
        aligned_scores = np.array([np.nan, np.nan, 0.1, 0.9, 0.8, 0.2], dtype=np.float32)
        aligned_predictions = np.array([-1, -1, 0, 1, 1, 0], dtype=np.int64)
        source_npz = source / "results" / "aligned_scores.npz"
        (source / "results").mkdir()
        np.savez(source_npz, scores=scores, predictions=predictions,
                 valid_mask=mask, threshold=np.asarray(0.5),
                 time_indices=global_indices[2:], time_values_ns=time_values[2:],
                 local_indices=global_indices[2:], aligned_scores=aligned_scores,
                 aligned_predictions=aligned_predictions, aligned_actual=np.array([-1, -1, 0, 1, 1, 0], dtype=np.int8),
                 actual=labels[2:])
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
    cold = synthetic_cold_start_check()
    return {"status": "passed", "valid_local": [2, 3, 4, 5], "cold_start": [0, 1],
            "boundary_crossing_rejected": True, "evaluation_bundle_written": True,
            "cold_start_e2e": cold}


def synthetic_cold_start_check():
    """Exercise the native compact/full-axis cold-start contract without a model."""
    with tempfile.TemporaryDirectory(prefix="mnrf-cold-start-synthetic-") as directory:
        root = Path(directory)
        source = root / "source"
        output = root / "cold-output"
        (source / "results").mkdir(parents=True)
        (source / "heads").mkdir()
        (source / "cache").mkdir()
        labels = np.array([0, 0, 0, 1, 1, 0], dtype=np.int8)
        times = np.arange(6, dtype=np.int64) * 60_000_000_000
        global_indices = np.arange(6, dtype=np.int64)
        segments = np.zeros(6, dtype=np.int32)
        np.savez(source / "analysis_inputs.npz", test_labels=labels,
                 test_global_indices=global_indices, test_time_values_ns=times,
                 test_segment_ids=segments, all_global_indices=global_indices,
                 all_time_values_ns=times, all_segment_ids=segments)
        np.save(source / "cache" / "normalized_values.npy", np.arange(12, dtype=np.float32).reshape(6, 2))
        assert isinstance(_load_npy_memmap(source / "cache" / "normalized_values.npy"), np.ndarray)
        old_mask = np.array([False, False, True, True, True, True], dtype=bool)
        old_scores = np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float32)
        old_predictions = (old_scores > 0.5).astype(np.int64)
        aligned_scores = np.array([np.nan, np.nan, 0.1, 0.9, 0.8, 0.2], dtype=np.float32)
        aligned_predictions = np.array([-1, -1, 0, 1, 1, 0], dtype=np.int64)
        aligned_actual = np.array([-1, -1, 0, 1, 1, 0], dtype=np.int8)
        source_npz = source / "results" / "aligned_scores.npz"
        np.savez(source_npz, scores=old_scores, predictions=old_predictions,
                 time_indices=np.array([2, 3, 4, 5], dtype=np.int64),
                 time_values_ns=times[2:], local_indices=np.array([2, 3, 4, 5], dtype=np.int64),
                 valid_mask=old_mask, aligned_scores=aligned_scores,
                 aligned_predictions=aligned_predictions, aligned_actual=aligned_actual,
                 actual=labels[2:], threshold=np.asarray(0.5))
        (source / "probe_manifest.json").write_text(json.dumps({"artifact_type": "probe_head",
            "config": {"task_name": "synthetic.csv"}}), encoding="utf-8")
        (source / "run_status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        np.savez(source / "heads" / "calibration.npz", threshold=np.asarray(0.5))
        with np.load(source_npz, allow_pickle=False) as old:
            old_payload = {name: np.asarray(old[name]) for name in old.files}
        payload, old_local = _assemble_cold_start_payload(
            old_payload, np.array([0, 1], dtype=np.int64), np.array([0.7, 0.6], dtype=np.float32),
            np.array([1, 1], dtype=np.int64), global_indices, times, labels, 0.5)
        manifest = {"schema": "mnrf-comparison-adapter-cold-start-v1", "task": "synthetic",
                    "source_path": str(source.resolve()), "source_run_id": source.name,
                    "source_aligned_scores_sha256": _sha256(source_npz),
                    "source_valid_mask_sha256": hashlib.sha256(np.ascontiguousarray(old_mask).tobytes()).hexdigest(),
                    "prefix_local_indices": [0, 1]}
        published = _write_cold_start_artifact(output, source, payload, manifest)
        output_npz = output / "results" / "aligned_scores.npz"
        assert published["output_aligned_scores_sha256"] == _sha256(output_npz)
        report = root / "cold-evaluation.json"
        result = evaluate(output, source, report)
        assert result["new_valid_points"] == 2 and result["old_valid_points"] == 4
        with np.load(output_npz, allow_pickle=False) as saved:
            assert saved["scores"].shape == (6,) and saved["aligned_scores"].shape == (6,)
            assert np.array_equal(saved["scores"][2:], old_scores)
        try:
            _validate_prefix_windows(np.array([0]), np.array([4]), np.arange(8),
                                     np.array([0, 0, 0, 1, 1, 1, 1, 1]),
                                     np.arange(8, dtype=np.int64), 4)
        except ValueError as error:
            assert "boundary" in str(error)
        else:
            raise AssertionError("cross-channel prefix history was accepted")
        return {"status": "passed", "old_valid_unchanged": True,
                "new_prefix_points": 2, "evaluation": str(report),
                "written_output": str(output), "cross_channel_rejected": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inventory", "plan", "common-compare", "refresh-coverage", "score",
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
    elif args.mode == "refresh-coverage":
        print(json.dumps(refresh_common_coverage(args.output_dir), indent=2, sort_keys=True))
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
