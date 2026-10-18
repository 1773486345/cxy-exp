#!/usr/bin/env python
"""Export frozen Probe split coordinates for the independent CATCH adapter."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
from ts_benchmark.baselines.mnrf_view_probe.probe import (
    _all_boundary_ids, _chunked_array_hash, _hash_strings, load_probe,
)
from ts_benchmark.baselines.mnrf_view_probe.common_contract import endpoints_for_ranges


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _endpoints_for_ranges(segment_ranges, lower, upper, seq_len):
    return endpoints_for_ranges(segment_ranges, lower, upper, seq_len)


def export(probe_run, data_csv, output_dir, catch_seq_len=None):
    run = Path(probe_run).resolve()
    probe = load_probe(run, device="cpu")
    manifest = json.loads((run / "probe_manifest.json").read_text(encoding="utf-8"))
    source = manifest.get("source_identity") or {}
    frame = pd.read_csv(data_csv)
    label_column = next((column for column in frame.columns if str(column).lower() == "label"), None)
    if label_column is None:
        raise ValueError("source CSV must contain a label column for full-coordinate evaluation")
    values, columns = MNRFLiteV2._features_from_frame(frame)
    if list(map(str, columns)) != source.get("variable_names"):
        raise ValueError("source CSV variable order differs from the frozen Probe run")
    if _chunked_array_hash(values) != source.get("feature_data_sha256"):
        raise ValueError("source CSV feature values differ from the frozen Probe run")
    time_index = MNRFLiteV2._time_index(frame)
    if _chunked_array_hash(time_index.values_ns) != source.get("time_index", {}).get("values_sha256"):
        raise ValueError("source CSV canonical time values differ from the frozen Probe run")
    boundary_ids, boundary_columns = _all_boundary_ids(frame)
    if boundary_columns != source.get("boundary_columns", []):
        raise ValueError("source CSV entity/session boundary columns differ from the frozen Probe run")
    boundary_hash = None if boundary_ids is None else _hash_strings(boundary_ids)
    if boundary_hash != source.get("boundary_values_sha256"):
        raise ValueError("source CSV entity/session values differ from the frozen Probe run")
    if len(frame) != int(source.get("input_length", -1)):
        raise ValueError("source CSV row count differs from the frozen Probe run")
    labels = frame[label_column].to_numpy()
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("source CSV labels must be one-dimensional integers")
    root = Path(output_dir).resolve()
    if root == run or root in run.parents or run in root.parents:
        raise ValueError("common export output must be outside the read-only Probe source tree")
    if root.exists():
        raise FileExistsError(f"common export directory already exists: {root}")
    # Derive every CATCH split before creating any output files.  A too-short
    # frozen range therefore fails without leaving a partial contract.
    cache_root = Path(manifest.get("source_dir") or run).resolve()
    catch_seq_len = int(source.get("seq_len") if catch_seq_len is None else catch_seq_len)
    if catch_seq_len <= 0:
        raise ValueError("catch_seq_len must be positive")
    segment_ranges = source.get("segment_ranges", {})
    derived_catch = {}
    for name in ("train", "validation", "calibration"):
        raw_range = source.get("ranges", {}).get(name)
        segments = segment_ranges.get(name, [])
        if not raw_range or not segments:
            raise ValueError(f"Probe contract lacks frozen segment ranges for {name}")
        endpoints = _endpoints_for_ranges(segments, raw_range[0], raw_range[1], catch_seq_len)
        if len(endpoints) == 0:
            raise ValueError(f"catch_seq_len={catch_seq_len} leaves no complete {name} windows in frozen range {raw_range}")
        derived_catch[name] = endpoints
    test_start = int(source.get("train_length", -1))
    catch_test = _endpoints_for_ranges(source.get("test_segments", []), test_start, len(frame), catch_seq_len)
    if len(catch_test) == 0:
        raise ValueError(f"catch_seq_len={catch_seq_len} leaves no complete test windows")
    root.mkdir(parents=True)
    records = {}
    for name in ("train", "validation", "calibration"):
        endpoints = np.load(cache_root / "cache" / f"{name}_window_end_indices.npy", allow_pickle=False)
        target = root / f"{name}_indices.npy"
        np.save(target, endpoints)
        records[name] = {"path": str(target), "sha256": _sha256(target), "count": int(len(endpoints))}
    catch_records = {}
    for name in ("train", "validation", "calibration"):
        endpoints = derived_catch[name]
        target = root / f"catch_{name}_indices.npy"
        np.save(target, endpoints)
        catch_records[name] = {"path": str(target), "sha256": _sha256(target), "count": int(len(endpoints))}
    catch_test_path = root / "catch_test_indices.npy"
    np.save(catch_test_path, catch_test)
    start, stop = source["ranges"]["train"]
    raw_rows = np.arange(int(start), int(stop), dtype=np.int64)
    rows_path = root / "train_rows.npy"
    np.save(rows_path, raw_rows)
    labels_path = root / "labels.npy"
    np.save(labels_path, labels)
    contract = {
        "schema": "mnrf-view-probe-common-export-v2", "status": "complete",
        "task_name": source.get("task_name"), "probe_run": str(run),
        "probe_manifest_sha256": _sha256(run / "probe_manifest.json"),
        "probe_source_identity": source,
        "data_identity": source.get("feature_data_sha256"),
        "time_identity": source.get("time_index", {}).get("values_sha256"),
        "time_kind": source.get("time_index", {}).get("kind"),
        "source_timezone": source.get("time_index", {}).get("source_timezone"),
        "boundary_columns": list(source.get("boundary_columns", [])),
        "boundary_identity": source.get("boundary_values_sha256"),
        "variable_names": source.get("variable_names"), "seq_len": int(source["seq_len"]),
        "catch_seq_len": catch_seq_len,
        "train_length": int(source["train_length"]),
        "input_length": int(source.get("input_length", len(frame))),
        "raw_training_range": [int(start), int(stop)],
        "ranges": source.get("ranges"),
        "segment_ranges": source.get("segment_ranges"),
        "test_segments": source.get("test_segments"),
        "temporal_rule": source.get("temporal_rule"),
        "splits": records, "catch_splits": catch_records,
        "catch_test": {"path": str(catch_test_path), "sha256": _sha256(catch_test_path), "count": int(len(catch_test))},
        "train_rows": {"path": str(rows_path), "sha256": _sha256(rows_path), "count": int(len(raw_rows))},
        "labels": {"path": str(labels_path), "sha256": _sha256(labels_path), "count": int(len(labels))},
    }
    (root / "common_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-run", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--catch-seq-len", type=int, default=192)
    args = parser.parse_args()
    contract = export(args.probe_run, args.data_csv, args.output_dir, args.catch_seq_len)
    print(json.dumps({"status": "complete", "output_dir": str(Path(args.output_dir).resolve()),
                      "train_rows": contract["train_rows"]["count"],
                      "train_windows": contract["splits"]["train"]["count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
