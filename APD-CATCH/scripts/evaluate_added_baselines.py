#!/usr/bin/env python
"""Read-only same-point evaluation for archived baseline score artifacts.

This command never fits or scores a model. Each input is an NPZ emitted by
run_archived_baseline.py (or the equivalent MNRF/common artifact schema).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REQUIRED = {"global_indices", "labels"}


def _load(path, task):
    values = np.load(path, allow_pickle=False)
    missing = REQUIRED.difference(values.files)
    if "global_indices" not in values.files and "time_indices" in values.files:
        missing.discard("global_indices")
    if "labels" not in values.files and "actual" in values.files:
        missing.discard("labels")
    if missing:
        raise ValueError(f"{path}: missing artifact fields {sorted(missing)}")
    indices = values["global_indices"] if "global_indices" in values.files else values["time_indices"]
    labels = values["labels"] if "labels" in values.files else values["actual"]
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError(f"{path}: global_indices must be a one-dimensional integer array")
    if labels.shape != indices.shape:
        raise ValueError(f"{path}: label/coordinate shape mismatch")
    if "valid_mask" in values.files:
        valid_mask = values["valid_mask"]
        if valid_mask.shape != indices.shape or valid_mask.dtype.kind != "b":
            raise ValueError(f"{path}: valid_mask/coordinate shape or dtype mismatch")
    for field in ("scores", "predictions"):
        if field in values.files and values[field].reshape(-1).size != indices.size:
            raise ValueError(f"{path}: {field}/coordinate shape mismatch")
    if indices.size == 0 or np.any(np.diff(indices) <= 0):
        raise ValueError(f"{path}: coordinates must be sorted and unique")
    if "task" in values.files and str(values["task"].reshape(-1)[0]) != task:
        raise ValueError(f"{path}: task binding mismatch")
    return values


def evaluate(task, entries, output):
    loaded = {name: _load(Path(path), task) for name, path in entries}
    common = None
    for name, values in loaded.items():
        coordinates = (values["global_indices"] if "global_indices" in values.files else values["time_indices"]).astype(np.int64, copy=False)
        valid = values["valid_mask"] if "valid_mask" in values.files else np.ones(coordinates.size, dtype=bool)
        coordinates = coordinates[valid]
        common = set(coordinates.tolist()) if common is None else common.intersection(coordinates.tolist())
    common = np.asarray(sorted(common or []), dtype=np.int64)
    if common.size < 2:
        raise ValueError("no usable common coordinate intersection")
    reference_labels = None
    reference_times = None
    rows = []
    for name, values in loaded.items():
        full_coords = (values["global_indices"] if "global_indices" in values.files else values["time_indices"]).astype(np.int64, copy=False)
        valid = values["valid_mask"] if "valid_mask" in values.files else np.ones(full_coords.size, dtype=bool)
        valid_positions = np.flatnonzero(valid)
        coords = full_coords[valid]
        positions = np.searchsorted(coords, common)
        if np.any(positions >= coords.size) or not np.array_equal(coords[positions], common):
            raise ValueError(f"{name}: coordinate intersection mapping failed")
        absolute_positions = valid_positions[positions]
        labels = (values["labels"] if "labels" in values.files else values["actual"])[absolute_positions].astype(np.int8, copy=False)
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{name}: labels are not binary")
        if reference_labels is None:
            reference_labels = labels.copy()
        elif not np.array_equal(labels, reference_labels):
            raise ValueError(f"{name}: labels disagree on common coordinates")
        if "time_values_ns" in values.files:
            times = values["time_values_ns"][absolute_positions]
            if reference_times is None:
                reference_times = times.copy()
            elif not np.array_equal(times, reference_times):
                raise ValueError(f"{name}: time coordinates disagree on common coordinates")
        elif "time_values" in values.files:
            times = values["time_values"][absolute_positions].astype(str)
            if reference_times is None:
                reference_times = times.copy()
            elif not np.array_equal(times, reference_times):
                raise ValueError(f"{name}: time coordinates disagree on common coordinates")
        row = {"task": task, "method": name, "common_points": int(common.size),
               "positive_points": int(labels.sum()), "roc_auc": "", "average_precision": "",
               "threshold": "", "precision": "", "recall": "", "fpr": "", "status": "ranking_only"}
        if "scores" in values.files:
            scores = values["scores"].reshape(-1)[absolute_positions].astype(np.float64, copy=False)
            if not np.isfinite(scores).all():
                raise ValueError(f"{name}: scores contain non-finite values")
            if np.unique(labels).size < 2:
                row["status"] = "undefined_single_class"
            else:
                row["roc_auc"] = float(roc_auc_score(labels, scores))
                row["average_precision"] = float(average_precision_score(labels, scores))
            if "threshold" in values.files and np.isfinite(values["threshold"]).all():
                threshold = float(values["threshold"].reshape(-1)[0])
                predictions = scores > threshold
                positives = labels == 1
                negatives = ~positives
                row.update({"threshold": threshold, "precision": float((predictions & positives).sum() / max(predictions.sum(), 1)),
                            "recall": float((predictions & positives).sum() / max(positives.sum(), 1)),
                            "fpr": float((predictions & negatives).sum() / max(negatives.sum(), 1)), "status": "ranking_and_alert"})
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("PSM", "MSL"))
    parser.add_argument("--artifact", action="append", required=True, metavar="NAME=NPZ")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    entries = []
    for value in args.artifact:
        if "=" not in value:
            parser.error("--artifact must be NAME=NPZ")
        entries.append(tuple(value.split("=", 1)))
    evaluate(args.task, entries, Path(args.output))


if __name__ == "__main__":
    main()
