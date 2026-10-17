#!/usr/bin/env python
"""Register, but do not fabricate, a CATCH-common execution manifest.

This helper only accepts artifacts produced by an explicit common-split
training/score adapter.  It never trains CATCH and it refuses score files
without immutable training and calibration evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.mnrf_view_probe.evaluation import validate_common_splits


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_coordinates(path):
    with np.load(path, allow_pickle=False) as payload:
        if not {"scores", "time_indices"}.issubset(payload.files):
            raise ValueError(f"{path} lacks scores/time_indices")
        indices = np.asarray(payload["time_indices"])
        scores = np.asarray(payload["scores"])
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(np.unique(indices)) != len(indices):
        raise ValueError(f"{path} has invalid endpoint coordinates")
    if scores.ndim != 1 or len(scores) != len(indices) or not np.isfinite(scores).all():
        raise ValueError(f"{path} has invalid scores")
    return indices


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
    manifest = {
        "schema": "mnrf-view-probe-catch-common-v1",
        "protocol": "CATCH-common",
        "status": "complete",
        "task_name": args.task_name,
        "probe_score_sha256": _sha256(args.probe),
        "catch_score_sha256": _sha256(args.catch),
        "labels_sha256": _sha256(args.labels),
        "train_score_identity": {"path": str(Path(args.training_evidence).resolve()),
                                  "sha256": _sha256(args.training_evidence)},
        "calibration_identity": {"path": str(Path(args.calibration_evidence).resolve()),
                                  "sha256": _sha256(args.calibration_evidence)},
        "splits": {
            name: {"path": str(Path(path).resolve()), "sha256": _sha256(path), **split_record[name]}
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

