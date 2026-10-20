#!/usr/bin/env python
"""Restore bound analysis-axis evidence for a completed full ViewProbe run.

This command never fits, trains, scores, or recalibrates. It requires the
training-time provenance receipt written by the original CLI run and verifies
the parsed source against the frozen run identity before writing the small
analysis input files.
"""

import argparse
import json
from pathlib import Path
import sys

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


def supplement(run_dir, output_dir, overwrite=False):
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "probe_manifest.json"
    provenance_path = run_dir / "training_provenance.json"
    if not manifest_path.is_file() or not provenance_path.is_file():
        raise ValueError("run is missing its original manifest or training-time provenance; refusing to infer training facts")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_output_target(run_dir, Path(output_dir), overwrite)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (run_dir / "analysis_inputs.npz").is_file() and (run_dir / "analysis_inputs.json").is_file():
        raise ValueError("bound analysis inputs already exist; use the read-only exporter directly")
    if provenance.get("captured_before_training") is not True or not provenance.get("source_data"):
        raise ValueError("original training-time source identity is unavailable; do not backfill from current environment")
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
    frame, train_length, test_labels, data_path, metadata_path, metadata_row = _read_task(task, source)
    expected = provenance["source_data"]
    identity = manifest.get("source_identity") or {}
    startup = provenance.get("startup_script") or {}
    startup_snapshot = run_dir / str(startup.get("snapshot", ""))
    if (not startup.get("sha256") or not startup_snapshot.is_file()
            or _sha256_file(startup_snapshot) != startup.get("snapshot_sha256")
            or startup.get("snapshot_sha256") != startup.get("sha256")):
        raise ValueError("original startup script snapshot is unavailable or failed its training-time hash")
    values, variables = MNRFLiteV2._features_from_frame(frame)
    times = MNRFLiteV2._time_index(frame).values_ns
    boundary_columns = list(map(str, MNRFLiteV2._boundary_columns(frame)))
    boundary_ids = MNRFLiteV2._boundary_ids(frame)
    current = {
        "task": task,
        "data_source": source,
        "rows": int(len(frame)),
        "train_length": int(train_length),
        "variable_names": list(map(str, variables)),
        "time_values_sha256": _chunked_array_hash(times),
        "feature_data_sha256": _chunked_array_hash(values),
        "boundary_columns": boundary_columns,
        "boundary_values_sha256": None if boundary_ids is None else _hash_strings(boundary_ids),
        "data_file": str(data_path),
        "data_file_sha256": _sha256_file(data_path),
        "metadata_file": str(metadata_path),
        "metadata_file_sha256": _sha256_file(metadata_path),
    }
    for key, value in current.items():
        if expected.get(key) != value:
            raise ValueError("current parsed source differs from the immutable training provenance field {}".format(key))
    for key in ("file_name", "train_lens"):
        if str(metadata_row.get(key)) != str(expected.get("selected_metadata_row", {}).get(key)):
            raise ValueError("selected metadata row differs from training provenance")
    if expected.get("time_kind") != MNRFLiteV2._time_index(frame).kind:
        raise ValueError("parsed source time kind differs from training provenance")
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

    _write_run_supporting_evidence(run_dir, frame, train_length, test_labels,
                                   data_path, metadata_path, provenance)
    result = export(run_dir, output_dir, overwrite=overwrite)
    return {"status": "complete", "supplemented_run": str(run_dir), **result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = supplement(args.run_dir, args.output_dir, args.overwrite)
    except Exception as error:
        print(json.dumps({"status": "invalid", "reason": str(error)}, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
