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
import hashlib
import json
import os
from pathlib import Path
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
    if len(test_labels) != len(test_global) or len(test_segments) != len(test_global) or len(valid_mask) != len(test_global):
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
    boundary_safe = True
    boundary_reason = "within-segment history only"
    for local in valid_local:
        global_position = train_length + int(local)
        start = global_position - seq_len + 1
        if start < 0 or global_position >= len(all_global):
            boundary_safe = False
            boundary_reason = "endpoint window falls outside complete axis"
            break
        if all_global[global_position] - all_global[start] != seq_len - 1:
            boundary_safe = False
            boundary_reason = "global row coordinates contain a gap in an endpoint window"
            break
        if np.any(all_segments[start:global_position + 1] != all_segments[global_position]):
            boundary_safe = False
            boundary_reason = "endpoint window crosses a frozen segment boundary"
            break
        diffs = np.diff(all_times[start:global_position + 1])
        if len(diffs) and (not np.isfinite(diffs).all() or np.any(diffs <= 0)):
            boundary_safe = False
            boundary_reason = "endpoint window has invalid or non-increasing time values"
            break
    first_valid = int(valid_local[0]) if len(valid_local) else None
    last_valid = int(valid_local[-1]) if len(valid_local) else None
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
                                "source_temporal_rule": identity.get("temporal_rule", {}),
                                "permission": "only rows within the frozen source segment and saved endpoint cache"},
        "cold_start_extension": {"status": "blocked-no-saved-features",
                                  "coordinates": list(range(int(first_valid or 0))),
                                  "reason": "source has no encoded features for the invalid prefix; no score is invented"},
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
    from sklearn.metrics import average_precision_score, roc_auc_score
    with np.load(bundle / "results/aligned_scores.npz", allow_pickle=False) as scored:
        scores = np.asarray(scored["aligned_scores"])
        predictions = np.asarray(scored["aligned_predictions"])
        mask = np.asarray(scored["valid_mask"], dtype=bool)
        threshold = float(np.asarray(scored["threshold"]))
    with np.load(source / "analysis_inputs.npz", allow_pickle=False) as inputs:
        labels = np.asarray(inputs["test_labels"])
    if labels.shape != mask.shape or not np.array_equal(predictions[mask], (scores[mask] > threshold).astype(predictions.dtype)):
        raise ValueError("adapter score bundle does not bind to labels or strict threshold rule")
    y, s = labels[mask], scores[mask]
    result = {"valid_points": int(mask.sum()), "threshold": threshold,
              "roc_auc": float(roc_auc_score(y, s)) if np.unique(y).size == 2 else None,
              "average_precision": float(average_precision_score(y, s)),
              "alarm_points": int(predictions[mask].sum()),
              "invalid_points": int((~mask).sum()), "protocol": "MNRF-valid-endpoint-q995"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


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
        np.savez(source / "analysis_inputs.npz", test_labels=labels)
        scores = np.array([np.nan, np.nan, 0.1, 0.9, 0.8, 0.2], dtype=np.float32)
        mask = np.array([False, False, True, True, True, True], dtype=bool)
        predictions = np.array([-1, -1, 0, 1, 1, 0], dtype=np.int64)
        np.savez(bundle / "results" / "aligned_scores.npz",
                 aligned_scores=scores, aligned_predictions=predictions,
                 valid_mask=mask, threshold=np.asarray(0.5))
        report = root / "evaluation.json"
        result = evaluate(bundle, source, report)
        assert result["valid_points"] == 4 and report.is_file()
    return {"status": "passed", "valid_local": [2, 3, 4, 5], "cold_start": [0, 1],
            "boundary_crossing_rejected": True, "evaluation_bundle_written": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inventory", "plan", "score", "evaluate", "synthetic"), required=True)
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
    elif args.mode == "synthetic":
        print(json.dumps(synthetic_check(), indent=2, sort_keys=True))
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
