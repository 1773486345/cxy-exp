#!/usr/bin/env python
"""Read-only metric recheck and paired summary for frozen MNRF ablations."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.run_mnrf_branch_ablation import COMBINATIONS, _json, selected_indices


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp-{}".format(__import__("os").getpid()))
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_csv(path, rows):
    path = Path(path)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(path.name + ".tmp-{}".format(__import__("os").getpid()))
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _npz(path):
    return np.load(path, allow_pickle=False)


def _evaluation(run_dir, pinned_source, dataset, combination, seed):
    run_dir = Path(run_dir)
    pinned_source = Path(pinned_source)
    manifest = _json(run_dir / "probe_manifest.json")
    aligned = _npz(run_dir / "results/aligned_scores.npz")
    source_manifest = _json(pinned_source / "probe_manifest.json")
    if run_dir.resolve() != pinned_source.resolve():
        if manifest.get("source_identity") != source_manifest.get("source_identity"):
            raise ValueError("run source_identity differs from pinned full source")
        if manifest.get("source_feature_fingerprint", manifest.get("feature_fingerprint")) != source_manifest.get("feature_fingerprint"):
            raise ValueError("run feature fingerprint differs from pinned full source")
        if manifest.get("source_dir") and Path(manifest["source_dir"]).resolve() != pinned_source.resolve():
            raise ValueError("run source_dir differs from pinned full source")
    views = tuple(manifest.get("selected_views") or manifest.get("config", {}).get("selected_views") or ())
    expected_views = dict(COMBINATIONS)[combination]
    indices = manifest.get("selected_indices") or manifest.get("config", {}).get("selected_indices")
    if indices is None:
        indices = selected_indices(views)
    if views != expected_views or [int(i) for i in indices] != selected_indices(expected_views):
        raise ValueError("run combination or selected feature coordinates differ")
    config = manifest.get("config", {})
    if int(config.get("seed", -1)) != int(seed):
        raise ValueError("run head seed differs from summary cell")
    scores = np.asarray(aligned["scores"])
    predictions = np.asarray(aligned["predictions"])
    actual = np.asarray(aligned["actual"])
    global_indices = np.asarray(aligned["time_indices"])
    local_indices = np.asarray(aligned["local_indices"])
    valid_mask = np.asarray(aligned["valid_mask"], dtype=bool)
    threshold = float(aligned["threshold"])
    if scores.ndim != 1 or actual.shape != scores.shape or predictions.shape != scores.shape:
        raise ValueError("saved endpoint score arrays do not align")
    if global_indices.shape != scores.shape or local_indices.shape != scores.shape:
        raise ValueError("saved endpoint coordinate arrays do not align")
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("saved scores or threshold are non-finite")
    if actual.dtype.kind not in "iu" or not np.isin(actual, (0, 1)).all():
        raise ValueError("saved endpoint labels are not binary integers")
    if not np.array_equal(predictions, scores > threshold):
        raise ValueError("saved predictions disagree with calibrated threshold")

    with _npz(pinned_source / "analysis_inputs.npz") as inputs:
        all_labels = np.asarray(inputs["all_labels"])
        all_times = np.asarray(inputs["all_time_values_ns"])
        all_global = np.asarray(inputs["all_global_indices"])
        all_segments = np.asarray(inputs["all_segment_ids"])
        test_labels = np.asarray(inputs["test_labels"])
        test_global = np.asarray(inputs["test_global_indices"])
        test_segments = np.asarray(inputs["test_segment_ids"])
    if (all_labels.ndim != 1 or all_times.shape != all_labels.shape
            or all_global.shape != all_labels.shape or all_segments.shape != all_labels.shape
            or test_labels.ndim != 1 or test_global.shape != test_labels.shape or test_segments.shape != test_labels.shape):
        raise ValueError("pinned source full-axis label/coordinate inputs are invalid")
    if not np.array_equal(test_global, all_global[len(all_labels) - len(test_labels):]):
        raise ValueError("source test coordinate axis is not the suffix of its full axis")
    local = local_indices.astype(np.int64, copy=False)
    if np.any(local < 0) or np.any(local >= len(test_labels)):
        raise ValueError("scored local coordinate is outside pinned source test axis")
    if not np.array_equal(global_indices.astype(np.int64), test_global[local]):
        raise ValueError("scored global coordinates differ from pinned source")
    if not np.array_equal(actual.astype(np.int8), test_labels[local].astype(np.int8)):
        raise ValueError("saved scored labels differ from pinned full label axis")
    expected_mask = np.zeros(len(test_labels), dtype=bool)
    expected_mask[local] = True
    if not np.array_equal(expected_mask, valid_mask):
        raise ValueError("saved valid mask differs from scored local coordinates")
    if len(test_labels) != len(valid_mask):
        raise ValueError("candidate test axis length differs from pinned full source")

    tp = int(np.sum((actual == 1) & (predictions == 1)))
    fp = int(np.sum((actual == 0) & (predictions == 1)))
    fn = int(np.sum((actual == 1) & (predictions == 0)))
    tn = int(np.sum((actual == 0) & (predictions == 0)))
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = float(roc_auc_score(actual, scores)) if np.unique(actual).size == 2 else None
    ap = float(average_precision_score(actual, scores)) if len(actual) else None
    input_metadata = _json(pinned_source / "analysis_inputs.json")
    segment_identity = {int(item["segment_id"]): item.get("boundary_value")
                        for item in input_metadata.get("test_segment_identity", [])}
    segment = segment_counts_from_axis(all_labels, all_global, all_segments, test_global,
                                       global_indices, predictions, all_times,
                                       segment_identity)
    result = {
        "dataset": dataset, "combination": combination, "head_seed": int(seed),
        "candidate_or_output_path": str(run_dir.resolve()), "status": "complete",
        "selected_views": ",".join(expected_views), "selected_indices": json.dumps(selected_indices(expected_views)),
        "selected_dim": len(selected_indices(expected_views)), "threshold": threshold,
        "auc_roc": auc, "average_precision": ap, "precision": precision, "recall": recall,
        "point_f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "normal_fpr": fp / float(fp + tn) if fp + tn else None,
        "coverage": float(valid_mask.mean()) if len(valid_mask) else 0.0,
        "anomaly_segments_total": segment["total"], "anomaly_segments_scored": segment["scored"],
        "anomaly_segments_hit": segment["hit"], "segment_details": segment["details"],
        "source_run_id": pinned_source.name, "source_encoder_seed": 2021,
        "independent_recheck": True,
    }
    bundle = {
        "scores": scores.astype(np.float32, copy=False), "predictions": predictions.astype(np.int8, copy=False),
        "labels": actual.astype(np.int8, copy=False), "global_indices": global_indices.astype(np.int64, copy=False),
        "local_indices": local.astype(np.int64, copy=False), "valid_mask": valid_mask,
        "threshold": np.asarray(threshold, dtype=np.float64),
    }
    return result, bundle


def segment_counts_from_axis(all_labels, all_global, all_segments, test_global, scored_global, predictions,
                             all_times=None, segment_identity=None):
    """Count complete test-axis anomaly intervals without joining gaps/entities."""
    test_start = int(test_global[0]) if len(test_global) else int(all_global[-1]) + 1
    test_end = int(test_global[-1]) + 1 if len(test_global) else test_start
    label_by_global = {int(g): int(label) for g, label in zip(all_global, all_labels)}
    segment_by_global = {int(g): int(s) for g, s in zip(all_global, all_segments)}
    prediction_by_global = {int(g): int(p) for g, p in zip(scored_global, predictions)}
    test_positions = [int(g) for g in test_global]
    time_by_global = {} if all_times is None else {int(g): int(t) for g, t in zip(all_global, all_times)}
    details = []
    i = 0
    while i < len(test_positions):
        g = test_positions[i]
        if label_by_global.get(g) != 1:
            i += 1
            continue
        segid = segment_by_global[g]
        start_i = i
        i += 1
        while i < len(test_positions):
            prev_g, cur_g = test_positions[i - 1], test_positions[i]
            if (cur_g != prev_g + 1 or segment_by_global.get(cur_g) != segid
                    or label_by_global.get(cur_g) != 1):
                break
            i += 1
        points = test_positions[start_i:i]
        predicted = [(point, prediction_by_global[point]) for point in points if point in prediction_by_global]
        hit_points = sum(pred for _, pred in predicted)
        details.append({"segment_id": segid, "boundary_value": (segment_identity or {}).get(segid),
                        "start_global": points[0], "end_global": points[-1],
                        "start_time_ns": time_by_global.get(points[0]),
                        "end_time_ns": time_by_global.get(points[-1]),
                        "points": len(points), "valid_points": len(predicted), "hit_points": int(hit_points),
                        "hit": bool(hit_points), "no_valid_scores": len(predicted) == 0})
    return {"total": len(details), "scored": sum(d["valid_points"] > 0 for d in details),
            "hit": sum(d["hit"] for d in details), "details": details}


def segment_counts(source_path, scored_global, predictions):
    """Small adapter shared by execution-time result recording and summaries."""
    source_path = Path(source_path)
    with _npz(source_path / "analysis_inputs.npz") as inputs:
        all_labels = np.asarray(inputs["all_labels"])
        all_times = np.asarray(inputs["all_time_values_ns"])
        all_global = np.asarray(inputs["all_global_indices"])
        all_segments = np.asarray(inputs["all_segment_ids"])
        test_global = np.asarray(inputs["test_global_indices"])
    result = segment_counts_from_axis(all_labels, all_global, all_segments, test_global,
                                      np.asarray(scored_global), np.asarray(predictions), all_times)
    return result


def contribution_deltas(metrics_by_combo):
    """Return prescribed added/deleted AP and ROC deltas; positive deletion means removal hurts."""
    result = {}
    full = metrics_by_combo.get("FULL")
    s = metrics_by_combo.get("S")
    epr = metrics_by_combo.get("WITHOUT_S")
    for view in ("S", "E", "P", "R"):
        if view == "S":
            added = {"average_precision": None, "auc_roc": None}
            removed_combo = "WITHOUT_S"
        else:
            added_combo = "S" + view
            added = {}
            for metric in ("average_precision", "auc_roc"):
                left, right = metrics_by_combo.get(added_combo), s
                added[metric] = (left.get(metric) - right.get(metric)
                                if left and right and left.get(metric) is not None and right.get(metric) is not None else None)
            removed_combo = "WITHOUT_" + view
        removed = {}
        for metric in ("average_precision", "auc_roc"):
            left, right = full, metrics_by_combo.get(removed_combo)
            removed[metric] = (left.get(metric) - right.get(metric)
                               if left and right and left.get(metric) is not None and right.get(metric) is not None else None)
        result[view] = {"added": added, "deleted": removed, "removal_combination": removed_combo}
    return result


def summarize(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = _json(plan_path)
    source_records = plan["source_records"]
    rows = []
    bundles = {}
    evals = {}
    for planned in plan["rows"]:
        row = dict(planned)
        if row.get("status") not in {"blocked", "conflict", "failed", "planned/pending", "running"}:
            task, combo, seed = row["dataset"], row["combination"], int(row["head_seed"])
            source = source_records[task]
            path = row.get("candidate_or_output_path")
            try:
                result, bundle = _evaluation(path, source["path"], task, combo, seed)
                for key, value in result.items():
                    if key != "segment_details":
                        row[key] = value
                row["segment_details"] = json.dumps(result["segment_details"], separators=(",", ":"))
                row["status"] = "reused" if planned.get("status") == "reused" else "complete"
                row["elapsed_seconds_kind"] = "historical_or_reused"
                key = "{}_{}_{}".format(task, combo, seed)
                bundles[key] = bundle
                evals[(task, combo, seed)] = result
            except Exception as error:
                row["status"] = "blocked"
                row["reason"] = "read-only summary verification failed: {}: {}".format(type(error).__name__, error)
                for key in ("auc_roc", "average_precision", "precision", "recall", "point_f1",
                            "fp", "fn", "tp", "tn", "normal_fpr", "coverage"):
                    row[key] = None
        rows.append(row)

    contributions = []
    paired = []
    for task in plan["tasks"]:
        for seed in plan["seeds"]:
            table = {combo: evals.get((task, combo, int(seed))) for combo, _ in COMBINATIONS}
            delta = contribution_deltas(table)
            for view, values in delta.items():
                row = {"dataset": task, "head_seed": int(seed), "branch": view,
                       "added_ap": values["added"]["average_precision"],
                       "deleted_ap": values["deleted"]["average_precision"],
                       "added_auc_roc": values["added"]["auc_roc"],
                       "deleted_auc_roc": values["deleted"]["auc_roc"],
                       "removal_combination": values["removal_combination"]}
                contributions.append(row)
            comparisons = [("S", "SE"), ("S", "SP"), ("S", "SR"), ("FULL", "WITHOUT_S"),
                           ("FULL", "WITHOUT_E"), ("FULL", "WITHOUT_P"), ("FULL", "WITHOUT_R")]
            for base, changed in comparisons:
                left, right = table.get(base), table.get(changed)
                if not left or not right:
                    continue
                paired.append({"dataset": task, "head_seed": int(seed), "base": base, "changed": changed,
                               "delta_ap_changed_minus_base": (right["average_precision"] - left["average_precision"]
                                   if right["average_precision"] is not None and left["average_precision"] is not None else None),
                               "delta_auc_changed_minus_base": (right["auc_roc"] - left["auc_roc"]
                                   if right["auc_roc"] is not None and left["auc_roc"] is not None else None),
                               "delta_tp": right["tp"] - left["tp"], "delta_fp": right["fp"] - left["fp"],
                               "delta_fn": right["fn"] - left["fn"],
                               "delta_segments_hit": right["anomaly_segments_hit"] - left["anomaly_segments_hit"]})

    repeat_rows = []
    if len(plan["seeds"]) > 1:
        for task in plan["tasks"]:
            for combo, _ in COMBINATIONS:
                available = [evals[(task, combo, seed)] for seed in plan["seeds"] if (task, combo, seed) in evals]
                for metric in ("average_precision", "auc_roc", "point_f1", "normal_fpr", "recall"):
                    values = [item[metric] for item in available if item[metric] is not None]
                    repeat_rows.append({"dataset": task, "combination": combo, "metric": metric,
                                        "n_complete": len(values), "mean": float(np.mean(values)) if values else None,
                                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                        "per_seed": json.dumps({str(seed): evals[(task, combo, seed)].get(metric)
                                            for seed in plan["seeds"] if (task, combo, seed) in evals})})

    return rows, contributions, paired, repeat_rows, bundles


def _write_bundle(path, bundles):
    temp = Path(path).with_name(Path(path).name + ".tmp")
    np.savez_compressed(temp, **{key + "_" + name: value for key, bundle in bundles.items()
                                 for name, value in bundle.items()})
    # numpy appends .npz when the path lacks that suffix.
    generated = Path(str(temp) + ".npz")
    generated.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="plan.json from run_mnrf_branch_ablation.py")
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    rows, contributions, paired, repeats, bundles = summarize(plan_path)
    batch_dir = plan_path.parent
    _atomic_csv(batch_dir / "long_table.csv", rows)
    _atomic_csv(batch_dir / "branch_contributions.csv", contributions)
    _atomic_csv(batch_dir / "paired_changes.csv", paired)
    if repeats:
        _atomic_csv(batch_dir / "repeat_summary.csv", repeats)
    point_path = batch_dir / "paired_point_scores.npz"
    if bundles:
        _write_bundle(point_path, bundles)
    complete = sum(row["status"] in {"complete", "reused"} for row in rows)
    report = {"batch_id": _json(plan_path).get("batch_id"), "positions": len(rows),
              "complete_or_reused": complete, "blocked": sum(row["status"] == "blocked" for row in rows),
              "pending": sum(row["status"] == "planned/pending" for row in rows),
              "conflict": sum(row["status"] == "conflict" for row in rows),
              "datasets_complete": len({row["dataset"] for row in rows if row["status"] in {"complete", "reused"}}),
              "interpretation": "Frozen encoder representations with independently refit GMM heads; not encoder-seed variance or causal proof.",
              "contribution_sign": "added = AP(S+branch)-AP(S); deleted = AP(FULL)-AP(FULL-minus-branch); positive deletion means removal hurts."}
    _atomic_json(batch_dir / "summary.json", report)
    _atomic_json(batch_dir / "status.json", report)
    review_root = ROOT / "result/review/MNRF-ViewProbe"
    review_root.mkdir(parents=True, exist_ok=True)
    archive = review_root / ("MNRF_BRANCH_ABLATION_{}.zip".format(report["batch_id"]))
    temp_archive = archive.with_name(archive.name + ".tmp-{}".format(os.getpid()))
    with zipfile.ZipFile(str(temp_archive), "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name in ("plan.json", "matrix.csv", "long_table.csv", "branch_contributions.csv",
                     "paired_changes.csv", "repeat_summary.csv", "summary.json"):
            candidate = batch_dir / name
            if candidate.is_file():
                output.write(str(candidate), arcname=name)
        if point_path.is_file():
            output.write(str(point_path), arcname=point_path.name)
    if archive.exists():
        try:
            with zipfile.ZipFile(str(archive), "r") as existing:
                prior = json.loads(existing.read("plan.json").decode("utf-8"))
            if prior.get("batch_id") != report["batch_id"]:
                raise ValueError("refusing to replace a review archive from another batch")
        except Exception as error:
            temp_archive.unlink(missing_ok=True)
            raise ValueError("existing review archive is not this batch's generated package: {}".format(error))
    os.replace(str(temp_archive), str(archive))
    print("summary_dir={}".format(batch_dir))
    print("review_package={}".format(archive))
    print("positions={} complete_or_reused={} blocked={} pending={} conflict={}".format(
        report["positions"], report["complete_or_reused"], report["blocked"], report["pending"], report["conflict"]))
    # A plan with pending cells is an intentional, usable partial inventory;
    # blocked/conflicting cells remain non-success so callers cannot mistake
    # an invalid source for a merely unrun combination.
    return 1 if report["blocked"] or report["conflict"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
