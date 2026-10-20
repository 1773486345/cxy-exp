#!/usr/bin/env python
"""Restore bound analysis-axis evidence for a completed full ViewProbe run.

This command never fits, trains, scores, or recalibrates. It requires the
training-time provenance receipt written by the original CLI run and verifies
the parsed source against the frozen run identity before writing the small
analysis input files.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_mnrf_view_probe_analysis import _validate_output_target, export
from scripts.run_mnrf_view_probe import (
    _read_task,
    _sha256_file,
    _write_run_supporting_evidence,
)
from scripts.validate_mnrf_view_probe import validate
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
from ts_benchmark.baselines.mnrf_view_probe.probe import _chunked_array_hash, _hash_strings
from ts_benchmark.baselines.mnrf_view_probe.integrity import canonical_segment_ids, normalize_binary_labels


def _source_facts(task, source, frame, train_length, data_path, metadata_path, metadata_row):
    values, variables = MNRFLiteV2._features_from_frame(frame)
    time_index = MNRFLiteV2._time_index(frame)
    boundary_columns = list(map(str, MNRFLiteV2._boundary_columns(frame)))
    boundary_ids = MNRFLiteV2._boundary_ids(frame)
    return {
        "task": task, "data_source": source, "rows": int(len(frame)),
        "train_length": int(train_length), "variable_names": list(map(str, variables)),
        "time_kind": time_index.kind, "time_source_timezone": time_index.source_timezone,
        "time_values_sha256": _chunked_array_hash(time_index.values_ns),
        "feature_data_sha256": _chunked_array_hash(values),
        "boundary_columns": boundary_columns,
        "boundary_values_sha256": None if boundary_ids is None else _hash_strings(boundary_ids),
        "selected_metadata_row": {str(k): str(v) for k, v in (metadata_row or {}).items()},
        "data_file": str(data_path), "data_file_sha256": _sha256_file(data_path),
        "metadata_file": str(metadata_path), "metadata_file_sha256": _sha256_file(metadata_path),
    }


def _known_old_segment_numbering(run_dir, manifest):
    """Recognize only the released v1 local-test-ID defect, not arbitrary damage."""
    npz_path, metadata_path = run_dir / "analysis_inputs.npz", run_dir / "analysis_inputs.json"
    if not npz_path.is_file() or not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    bound = manifest.get("analysis_artifacts") or {}
    if (metadata.get("schema") != "mnrf-view-probe-analysis-inputs-v1"
            or metadata.get("source_manifest_sha256") != _sha256_file(run_dir / "probe_manifest.json")
            or metadata.get("source_identity_sha256") != manifest.get("source_identity_sha256")
            or metadata.get("labels_sha256") != _sha256_file(npz_path)
            or bound.get("analysis_inputs_sha256") != _sha256_file(npz_path)):
        return False
    provenance_path = run_dir / "training_provenance.json"
    if (not provenance_path.is_file()
            or bound.get("training_provenance_sha256") != _sha256_file(provenance_path)
            or metadata.get("provenance_sha256") != _sha256_file(provenance_path)):
        return False
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_data = provenance.get("source_data") or {}
    if (metadata.get("data_file") != source_data.get("data_file")
            or metadata.get("data_file_sha256") != source_data.get("data_file_sha256")
            or metadata.get("metadata_file") != source_data.get("metadata_file")
            or metadata.get("metadata_file_sha256") != source_data.get("metadata_file_sha256")):
        return False
    identity = manifest.get("source_identity") or {}
    train, total = int(identity.get("train_length", -1)), int(identity.get("input_length", -1))
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            full_ids = np.asarray(payload["all_segment_ids"])
            test_ids = np.asarray(payload["test_segment_ids"])
            full_labels = normalize_binary_labels(payload["all_labels"], total, "legacy labels")
            test_labels = normalize_binary_labels(payload["test_labels"], total - train, "legacy test labels")
            all_times = np.asarray(payload["all_time_values_ns"])
            test_times = np.asarray(payload["test_time_values_ns"])
            all_indices = np.asarray(payload["all_global_indices"])
            test_indices = np.asarray(payload["test_global_indices"])
    except Exception:
        return False
    if (full_ids.shape != (total,) or test_ids.shape != (total - train,)
            or not np.array_equal(full_labels[train:], test_labels)
            or all_times.shape != (total,) or test_times.shape != (total - train,)
            or not np.array_equal(all_times[train:], test_times)
            or not np.array_equal(all_indices, np.arange(total))
            or not np.array_equal(test_indices, np.arange(train, total))):
        return False
    hashes = bound.get("full_axis_sha256") or {}
    from scripts.export_mnrf_view_probe_analysis import _array_sha256
    if any(hashes.get(key) != _array_sha256(value) for key, value in (
            ("labels", full_labels), ("time_values_ns", all_times),
            ("global_indices", all_indices), ("segment_ids", full_ids))):
        return False
    segments = identity.get("test_segments", [])
    old_test_ids = np.full(total - train, -1, dtype=np.int32)
    for local_id, (left, right) in enumerate(segments):
        a, b = max(train, int(left)) - train, min(total, int(right)) - train
        if b > a:
            old_test_ids[a:b] = local_id
    old_full_ids = np.full(total, -1, dtype=np.int32)
    for segment_id, (left, right) in enumerate(identity.get("segments", [])):
        old_full_ids[int(left):int(right)] = segment_id
    # This is the exact historical signature: complete IDs are canonical, but
    # test IDs were re-enumerated from zero independently.
    return (np.array_equal(full_ids, old_full_ids) and np.array_equal(test_ids, old_test_ids)
            and not np.array_equal(test_ids, full_ids[train:]))


def supplement(run_dir, output_dir, overwrite=False, recover_legacy=False,
               migrate_segment_ids=False, data_root=None):
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "probe_manifest.json"
    provenance_path = run_dir / "training_provenance.json"
    if not manifest_path.is_file():
        raise ValueError("run is missing its original manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_output_target(run_dir, Path(output_dir), overwrite)
    if manifest.get("artifact_policy") != "full" or manifest.get("stage_status") != "complete":
        raise ValueError("only a completed full-artifact run can receive supplemental analysis inputs")
    checked = validate(run_dir)
    if checked.get("status") != "complete":
        raise ValueError("completed run validation failed: {}".format(checked))

    config = manifest.get("config") or {}
    task = str(manifest.get("task_name", config.get("task_name", "")))
    if task.endswith(".csv"):
        task = task[:-4]
    source = str(config.get("data_source", ""))
    frame, train_length, test_labels, data_path, metadata_path, metadata_row = _read_task(task, source, data_root)
    test_labels = normalize_binary_labels(test_labels, len(frame) - train_length, "test labels")
    full_labels = normalize_binary_labels(frame[next(column for column in frame.columns if str(column).lower() == "label")].to_numpy(), len(frame), "full source labels")
    current = _source_facts(task, source, frame, train_length, data_path, metadata_path, metadata_row)
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    else:
        provenance = None
    partial_history = provenance is None
    if partial_history and not recover_legacy:
        raise ValueError("training receipt is missing; use explicit --recover-legacy after verifying the original data")
    if partial_history:
        provenance = {
            "receipt_schema": "mnrf-view-probe-historical-recovery-v1",
            "provenance_type": "recovered_partial", "captured_before_training": False,
            "training_facts": "unknown: original CLI failed before its post-training support write",
            "training_environment": None,
            "startup_script": {"path": None, "sha256": None,
                               "unknown_reason": "historical run did not persist a training-time script snapshot"},
            "source_data": current,
            "source_data_scope": "recovery-time parsed loader identity; not evidence of training-time raw-file bytes",
            "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
            "recovery_context": {"current_git_head": __import__("subprocess").run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip() or None,
                "label_axis_sha256": _chunked_array_hash(full_labels)},
        }
    else:
        expected = provenance.get("source_data") or {}
        if provenance.get("captured_before_training") is not True or not expected:
            raise ValueError("original training-time source receipt is incomplete; use explicit historical recovery")
        for key, value in current.items():
            if expected.get(key) != value:
                raise ValueError("current parsed source differs from immutable training provenance field {}".format(key))
    identity = manifest.get("source_identity") or {}
    startup = provenance.get("startup_script") or {}
    startup_snapshot = run_dir / str(startup.get("snapshot", ""))
    if (not partial_history and (not startup.get("sha256") or not startup_snapshot.is_file()
            or _sha256_file(startup_snapshot) != startup.get("snapshot_sha256")
            or startup.get("snapshot_sha256") != startup.get("sha256"))):
        raise ValueError("original startup script snapshot is unavailable or failed its training-time hash")
    values, variables = MNRFLiteV2._features_from_frame(frame)
    times = MNRFLiteV2._time_index(frame).values_ns
    boundary_columns = list(map(str, MNRFLiteV2._boundary_columns(frame)))
    boundary_ids = MNRFLiteV2._boundary_ids(frame)
    for key in ("file_name", "train_lens"):
        if str(metadata_row.get(key)) != str(current.get("selected_metadata_row", {}).get(key)):
            raise ValueError("selected metadata row differs from recovered source identity")
    if current["rows"] != identity.get("input_length") or current["train_length"] != identity.get("train_length"):
        raise ValueError("parsed source lengths differ from the frozen manifest")
    if current["variable_names"] != identity.get("variable_names"):
        raise ValueError("parsed variable order differs from the frozen manifest")
    if current["time_values_sha256"] != (identity.get("time_index") or {}).get("values_sha256"):
        raise ValueError("parsed time axis differs from the frozen manifest")
    if current["feature_data_sha256"] != identity.get("feature_data_sha256"):
        raise ValueError("parsed feature values differ from the frozen manifest")
    if current["boundary_columns"] != identity.get("boundary_columns", []) or current["boundary_values_sha256"] != identity.get("boundary_values_sha256"):
        raise ValueError("parsed entity/session boundaries differ from the frozen manifest")

    # Existing scoring labels are immutable evidence. Historical full-axis
    # labels beyond these endpoints are sourced from the now-verified loader,
    # not retroactively claimed as training-time evidence.
    aligned_path = run_dir / "results" / "aligned_scores.npz"
    with np.load(aligned_path, allow_pickle=False) as aligned:
        endpoints = np.asarray(aligned["time_indices"])
        actual = np.asarray(aligned["actual"])
    if (endpoints.ndim != 1 or endpoints.dtype.kind not in "iu" or actual.shape != endpoints.shape
            or not np.array_equal(actual, test_labels[endpoints.astype(np.int64) - train_length])):
        raise ValueError("recovery source labels disagree with immutable scored endpoint labels")

    existing_inputs = run_dir / "analysis_inputs.npz"
    existing_metadata = run_dir / "analysis_inputs.json"
    migration_backup_tmp = None
    if existing_inputs.exists() or existing_metadata.exists():
        if not (existing_inputs.is_file() and existing_metadata.is_file() and _known_old_segment_numbering(run_dir, manifest)
                and migrate_segment_ids):
            raise ValueError("existing analysis inputs are not the recognized legacy segment-ID defect; use export or provide evidence")
        expected_full_ids = canonical_segment_ids(identity)
        with np.load(existing_inputs, allow_pickle=False) as old_inputs:
            old_full_labels = normalize_binary_labels(old_inputs["all_labels"], len(frame), "legacy full labels")
            old_test_labels = normalize_binary_labels(old_inputs["test_labels"], len(test_labels), "legacy test labels")
            old_full_times = np.asarray(old_inputs["all_time_values_ns"])
            old_full_indices = np.asarray(old_inputs["all_global_indices"])
            old_full_ids = np.asarray(old_inputs["all_segment_ids"])
        if (not np.array_equal(old_full_labels, full_labels)
                or not np.array_equal(old_test_labels, test_labels)
                or not np.array_equal(old_full_times, times)
                or not np.array_equal(old_full_indices, np.arange(len(frame), dtype=np.int64))
                or not np.array_equal(old_full_ids, expected_full_ids)):
            raise ValueError("legacy analysis axis differs from the revalidated original data; refusing migration")
        backup = run_dir / "provenance" / ("legacy-analysis-inputs-" + _sha256_file(existing_inputs)[:16])
        if backup.exists():
            raise ValueError("legacy analysis backup already exists; inspect it before retrying migration")
        migration_backup_tmp = run_dir / "provenance" / (".legacy-migration-tmp-" + uuid.uuid4().hex)
        migration_backup_tmp.mkdir(parents=True, exist_ok=False)
        shutil.copy2(existing_inputs, migration_backup_tmp / "analysis_inputs.npz")
        shutil.copy2(existing_metadata, migration_backup_tmp / "analysis_inputs.json")
        migration = {"operation": "canonicalize_full_axis_segment_ids", "old_analysis_inputs_sha256": _sha256_file(existing_inputs),
                     "old_analysis_metadata_sha256": _sha256_file(existing_metadata),
                     "backup_path": str(backup.relative_to(run_dir)),
                     "at_utc": datetime.now(timezone.utc).isoformat(), "schema": "legacy-segment-id-migration-v1"}
        provenance.setdefault("migrations", []).append(migration)

    try:
        _write_run_supporting_evidence(run_dir, frame, train_length, test_labels,
                                       data_path, metadata_path, provenance)
    except BaseException:
        if migration_backup_tmp is not None:
            shutil.rmtree(migration_backup_tmp, ignore_errors=True)
        raise
    if migration_backup_tmp is not None:
        backup = run_dir / migration["backup_path"]
        os.replace(migration_backup_tmp, backup)
    result = export(run_dir, output_dir, overwrite=overwrite)
    return {"status": "complete", "supplemented_run": str(run_dir), **result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--recover-legacy", action="store_true",
                        help="recover a completed historical source without a training receipt; missing training facts remain unknown")
    parser.add_argument("--migrate-segment-ids", action="store_true",
                        help="migrate only the recognized released test-axis re-numbering defect")
    parser.add_argument("--data-root", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        result = supplement(args.run_dir, args.output_dir, args.overwrite,
                            args.recover_legacy, args.migrate_segment_ids,
                            args.data_root or None)
    except Exception as error:
        print(json.dumps({"status": "invalid", "reason": str(error)}, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
