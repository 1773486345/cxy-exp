#!/usr/bin/env python
"""Read-only coordinate and identity-bound evaluation of saved artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _text(values, key):
    if key not in values.files:
        return None
    data = values[key].reshape(-1)
    return str(data[0]) if data.size else None


def _manifest(path, task):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    found_task = data.get("task") or data.get("task_name")
    if found_task and str(found_task).replace(".csv", "") != task:
        raise ValueError(f"{path}: manifest task binding mismatch")
    status = data.get("status") or data.get("stage_status") or data.get("head_status")
    allowed = {"complete", "complete_native_score_artifact", "complete_native_label_no_threshold_export"}
    if status and status not in allowed:
        raise ValueError(f"{path}: manifest is not complete ({status})")
    return True


def _load(path, task):
    values = np.load(path, allow_pickle=False)
    index_key = "global_indices" if "global_indices" in values.files else "time_indices"
    label_key = "labels" if "labels" in values.files else "actual"
    if index_key not in values.files or label_key not in values.files:
        raise ValueError(f"{path}: coordinate and labels are required")
    indices, labels = values[index_key], values[label_key]
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or labels.shape != indices.shape
            or not indices.size or np.any(np.diff(indices) <= 0)):
        raise ValueError(f"{path}: invalid sorted coordinate/label arrays")
    if "valid_mask" in values.files:
        mask = values["valid_mask"]
        compact = mask.shape == indices.shape
        aligned = "aligned_scores" in values.files and mask.shape == values["aligned_scores"].shape
        if mask.dtype.kind != "b" or not (compact or aligned):
            raise ValueError(f"{path}: valid_mask shape or dtype mismatch")
    for field in ("scores", "predictions"):
        if field in values.files and values[field].reshape(-1).size != indices.size:
            raise ValueError(f"{path}: {field}/coordinate mismatch")
    if "task" in values.files and _text(values, "task") != task:
        raise ValueError(f"{path}: task binding mismatch")
    if ("scores" in values.files and "score_direction" in values.files
            and _text(values, "score_direction") != "higher_is_anomalous"):
        raise ValueError(f"{path}: score direction is not higher_is_anomalous")
    return values


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(task, entries, output, manifests=None, data_file=None, directions=None, require_identity=False):
    manifests, directions = manifests or {}, directions or {}
    loaded = {name: _load(Path(path), task) for name, path in entries}
    data_digest = _sha256(data_file) if data_file else None
    identities = {}
    for name, values in loaded.items():
        bound = _text(values, "source_data_sha256") is not None
        if name in manifests:
            _manifest(manifests[name], task)
            bound = True
        artifact_digest = _text(values, "source_data_sha256")
        if data_digest and artifact_digest and artifact_digest != data_digest:
            raise ValueError(f"{name}: source_data_sha256 does not match data file")
        direction = _text(values, "score_direction") or directions.get(name)
        if require_identity and not bound:
            raise ValueError(f"{name}: source identity manifest or digest is required")
        if require_identity and "scores" in values.files and direction != "higher_is_anomalous":
            raise ValueError(f"{name}: score direction is unbound")
        identities[name] = ("bound" if bound else "unbound", direction)

    common = None
    for values in loaded.values():
        indices = (values["global_indices"] if "global_indices" in values.files
                   else values["time_indices"]).astype(np.int64, copy=False)
        mask = (values["valid_mask"] if "valid_mask" in values.files
                and values["valid_mask"].shape == indices.shape
                else np.ones(indices.size, dtype=bool))
        coords = indices[mask]
        common = set(coords.tolist()) if common is None else common.intersection(coords.tolist())
    common = np.asarray(sorted(common or []), dtype=np.int64)
    if common.size < 2:
        raise ValueError("no common coordinates")

    reference_labels = reference_times = None
    rows = []
    for name, values in loaded.items():
        indices = (values["global_indices"] if "global_indices" in values.files
                   else values["time_indices"]).astype(np.int64, copy=False)
        mask = (values["valid_mask"] if "valid_mask" in values.files
                and values["valid_mask"].shape == indices.shape
                else np.ones(indices.size, dtype=bool))
        valid_positions, coords = np.flatnonzero(mask), indices[mask]
        positions = np.searchsorted(coords, common)
        if np.any(positions >= coords.size) or not np.array_equal(coords[positions], common):
            raise ValueError(f"{name}: coordinate intersection mapping failed")
        absolute = valid_positions[positions]
        labels = (values["labels"] if "labels" in values.files else values["actual"])[absolute].astype(np.int8, copy=False)
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{name}: labels are not binary")
        if reference_labels is None:
            reference_labels = labels.copy()
        elif not np.array_equal(labels, reference_labels):
            raise ValueError(f"{name}: labels disagree on common coordinates")
        if "time_values_ns" in values.files:
            times = values["time_values_ns"][absolute].astype(str)
        elif "time_values" in values.files:
            times = values["time_values"][absolute].astype(str)
        else:
            times = None
        if times is not None:
            if reference_times is None:
                reference_times = times.copy()
            elif not np.array_equal(times, reference_times):
                raise ValueError(f"{name}: time coordinates disagree on common coordinates")
        direction = identities[name][1]
        row = {"task": task, "method": name, "common_points": int(common.size),
               "positive_points": int(labels.sum()), "roc_auc": "", "average_precision": "",
               "threshold": "", "precision": "", "recall": "", "fpr": "",
               "identity_status": identities[name][0], "score_direction": direction or "",
               "status": "ranking_only"}
        if "scores" in values.files:
            scores = values["scores"].reshape(-1)[absolute].astype(np.float64, copy=False)
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
                positives, negatives = labels == 1, labels == 0
                row.update({"threshold": threshold,
                            "precision": float((predictions & positives).sum() / max(predictions.sum(), 1)),
                            "recall": float((predictions & positives).sum() / max(positives.sum(), 1)),
                            "fpr": float((predictions & negatives).sum() / max(negatives.sum(), 1)),
                            "status": "ranking_and_alert"})
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("PSM", "MSL"))
    parser.add_argument("--artifact", action="append", required=True, metavar="NAME=NPZ")
    parser.add_argument("--source-manifest", action="append", default=[], metavar="NAME=JSON")
    parser.add_argument("--score-direction", action="append", default=[], metavar="NAME=DIRECTION")
    parser.add_argument("--data-file")
    parser.add_argument("--require-identity", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    entries = [tuple(value.split("=", 1)) for value in args.artifact]
    manifests = dict(tuple(value.split("=", 1)) for value in args.source_manifest)
    directions = dict(tuple(value.split("=", 1)) for value in args.score_direction)
    evaluate(args.task, entries, Path(args.output), manifests=manifests,
             data_file=args.data_file, directions=directions, require_identity=args.require_identity)


if __name__ == "__main__":
    main()
