#!/usr/bin/env python
"""Read-only four-cell comparison for concatenated vs sequence-bounded MSL."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mnrf_msl_sequence_boundaries import DEFAULT_MAPPING, load_mapping
from scripts.validate_mnrf_view_probe import validate
from ts_benchmark.baselines.mnrf_view_probe.evaluation import affiliation_f_segment_macro

OLD_MSL_RUN_ID = "full_gmm-full-20261021T112147Z-2470335"
FIXED_CONFIG = {
    "seed": 2021, "seq_len": 192, "pattern_len": 32, "hidden_dim": 32,
    "view_dim": 8, "evolution_dim": 8, "pattern_dim": 8, "relation_dim": 8,
    "batch_size": 128, "encoding_batch_size": 128, "cache_chunk_size": 256,
    "stage1_epochs": 50, "stage2_epochs": 30, "patience": 5,
    "learning_rate": 0.001, "weight_decay": 0.0001,
    "gmm_components": 4, "gmm_covariance_type": "full", "gmm_reg_covar": 0.001,
    "gmm_n_init": 1, "gmm_max_iter": 100, "gmm_fit_max_windows": 20000,
    "feature_std_floor": 0.05, "score_mad_floor": 0.001,
    "calibration_quantile": 0.995, "train_noise_std": 0.01,
    "noise_seed": 20261013,
    "artifact_policy": "full", "data_source": "large_detect", "device": "cuda:0",
}

def derive_score_views(validation_nll, calibration_nll, test_nll, calibration, quantile=0.995):
    """Reproduce frozen median/MAD calibration and derive full/target scores."""
    validation = np.asarray(validation_nll)
    calibration_values = np.asarray(calibration_nll)
    test = np.asarray(test_nll)
    if validation.ndim != 2 or calibration_values.ndim != 2 or test.ndim != 2:
        raise ValueError("per-variable NLL inputs must be rank two")
    if validation.shape[1] != 55 or calibration_values.shape[1] != 55 or test.shape[1] != 55:
        raise ValueError("MSL scoring comparison requires all 55 per-variable NLL columns")
    if len(validation) == 0 or len(calibration_values) == 0 or not np.isfinite(validation).all() \
            or not np.isfinite(calibration_values).all() or not np.isfinite(test).all():
        raise ValueError("NLL inputs must be non-empty where required and finite")
    center = np.median(validation, axis=0).astype(np.float32)
    scale = np.maximum(1.4826 * np.median(np.abs(validation - center), axis=0),
                       float(calibration.get("score_mad_floor", 0.001))).astype(np.float32)
    stored_center = np.asarray(calibration["center"])
    stored_scale = np.asarray(calibration["scale"])
    if stored_center.shape != (55,) or stored_scale.shape != (55,) or np.any(stored_scale <= 0):
        raise ValueError("saved calibration center/scale are invalid")
    if not np.allclose(center, stored_center, rtol=0, atol=1e-7) \
            or not np.allclose(scale, stored_scale, rtol=0, atol=1e-7):
        raise ValueError("saved calibration statistics do not reproduce from validation NLL")
    q = float(calibration["quantile"])
    if not np.isclose(q, float(quantile), rtol=0, atol=1e-12):
        raise ValueError("saved calibration quantile differs from this comparison protocol")
    cal_z = (calibration_values - center) / scale
    test_z = (test - center) / scale
    specs = {
        "all_55_max": (np.max(cal_z, axis=1), np.max(test_z, axis=1)),
        "col_0_only": (cal_z[:, 0], test_z[:, 0]),
    }
    result = {}
    for name, (cal_scores, test_scores) in specs.items():
        threshold = float(np.quantile(cal_scores, q, method="linear"))
        result[name] = {"scores": np.asarray(test_scores, dtype=np.float32),
                        "threshold": threshold,
                        "calibration_score_count": int(len(cal_scores))}
    saved_threshold = float(calibration["threshold"])
    if not np.isclose(result["all_55_max"]["threshold"], saved_threshold,
                      rtol=1e-6, atol=1e-7):
        raise ValueError("replayed 55-variable max threshold differs from saved threshold")
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_run(run_dir, label):
    run = Path(run_dir).resolve()
    checked = validate(run)
    if checked.get("status") != "complete" or checked.get("head_type") != "gmm":
        raise ValueError(f"{label} is not a natively validated complete GMM source")
    manifest = json.loads((run / "probe_manifest.json").read_text())
    if manifest.get("task_name") != "MSL.csv" and manifest.get("config", {}).get("task_name") != "MSL.csv":
        raise ValueError(f"{label} task identity is not MSL")
    if manifest.get("config", {}).get("mode") != "full_gmm" or manifest.get("config", {}).get("selected_views") != ["S", "E", "P", "R"]:
        raise ValueError(f"{label} is not the expected four-view Full GMM")
    if int(manifest.get("config", {}).get("seed", -1)) != 2021:
        raise ValueError(f"{label} seed differs from fixed MSL source seed")
    nll = np.load(run / "results/test_nll.npy", mmap_mode="r", allow_pickle=False)
    validation = np.load(run / "heads/validation_nll.npy", mmap_mode="r", allow_pickle=False)
    calibration_nll = np.load(run / "heads/calibration_nll.npy", mmap_mode="r", allow_pickle=False)
    for split, scores in (("validation", validation), ("calibration", calibration_nll)):
        endpoints_path = run / f"cache/{split}_window_end_indices.npy"
        endpoints_for_split = np.load(endpoints_path, mmap_mode="r", allow_pickle=False)
        if scores.shape != (len(endpoints_for_split), 55):
            raise ValueError(f"{label} {split} NLL rows do not match bound endpoints")
    with np.load(run / "heads/calibration.npz", allow_pickle=False) as cal_file:
        calibration = {key: np.asarray(cal_file[key]) for key in cal_file.files}
    calibration["score_mad_floor"] = float(manifest["config"]["score_mad_floor"])
    calibration["threshold"] = float(calibration["threshold"])
    calibration["quantile"] = float(calibration["quantile"])
    scores_file = np.load(run / "results/aligned_scores.npz", allow_pickle=False)
    endpoints = np.asarray(scores_file["time_indices"])
    if endpoints.ndim != 1 or endpoints.dtype.kind not in "iu" or len(np.unique(endpoints)) != len(endpoints) \
            or (len(endpoints) > 1 and np.any(np.diff(endpoints) <= 0)):
        raise ValueError(f"{label} endpoint coordinates are not unique sorted integers")
    if nll.shape != (len(endpoints), 55):
        raise ValueError(f"{label} test_nll shape does not match saved endpoints")
    identity = manifest["source_identity"]
    if identity.get("variable_names") != [f"col_{i}" for i in range(55)]:
        raise ValueError(f"{label} variable order differs from verified MSL order")
    with np.load(run / "analysis_inputs.npz", allow_pickle=False) as axes:
        full_indices = np.asarray(axes["all_global_indices"])
        full_labels = np.asarray(axes["all_labels"])
        full_times = np.asarray(axes["all_time_values_ns"])
    if full_indices.dtype.kind not in "iu" or not np.array_equal(full_indices, np.arange(len(full_indices))):
        raise ValueError(f"{label} full-axis global coordinate binding is invalid")
    if len(full_labels) != len(full_indices) or len(full_times) != len(full_indices):
        raise ValueError(f"{label} full-axis labels/times are misaligned")
    if not np.array_equal(np.asarray(scores_file["actual"]), full_labels[endpoints]):
        raise ValueError(f"{label} endpoint labels disagree with its complete axis")
    if not np.array_equal(np.asarray(scores_file["time_values_ns"]), full_times[endpoints]):
        raise ValueError(f"{label} endpoint times disagree with its complete axis")
    derived = derive_score_views(validation, calibration_nll, nll, calibration,
                                 quantile=float(manifest["config"]["calibration_quantile"]))
    saved_scores = np.asarray(scores_file["scores"])
    if not np.allclose(derived["all_55_max"]["scores"], saved_scores, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{label} 55-variable max does not reproduce its saved score")
    saved_predictions = np.asarray(scores_file["predictions"])
    expected_predictions = (saved_scores > calibration["threshold"]).astype(saved_predictions.dtype)
    if not np.array_equal(saved_predictions, expected_predictions):
        raise ValueError(f"{label} saved predictions do not match strict score > threshold")
    replayed_max_predictions = (derived["all_55_max"]["scores"] >
                               derived["all_55_max"]["threshold"]).astype(saved_predictions.dtype)
    if not np.array_equal(saved_predictions, replayed_max_predictions):
        raise ValueError(f"{label} replayed max predictions differ from the saved binary result")
    return {"run": run, "manifest": manifest, "identity": identity,
            "nll": nll, "endpoints": endpoints.astype(np.int64), "labels": full_labels,
            "times": full_times, "scores_file": scores_file, "derived": derived,
            "validation_status": checked}


def _test_channel_ids(mapping, train_length, total_length):
    channels, counts = mapping["channels"], mapping["channel_row_counts"]
    expected_test = sum(int(counts[channel]["test"]) for channel in channels)
    if total_length - int(train_length) != expected_test:
        raise ValueError("MSL full axis is inconsistent with verified test sequence lengths")
    boundaries = np.empty(expected_test, dtype=object)
    cursor = 0
    for channel in channels:
        stop = cursor + int(counts[channel]["test"])
        boundaries[cursor:stop] = channel
        cursor = stop
    return boundaries


def common_endpoint_coordinates(old_endpoints, new_endpoints):
    for label, endpoints in (("old", old_endpoints), ("new", new_endpoints)):
        array = np.asarray(endpoints)
        if array.ndim != 1 or array.dtype.kind not in "iu" or (len(array) > 1 and np.any(np.diff(array) <= 0)):
            raise ValueError(f"{label} endpoint coordinates must be sorted unique integers")
    return np.intersect1d(old_endpoints, new_endpoints, assume_unique=True)


def _metric_row(dataset, scoring_object, run, coords, labels, score, threshold,
                local_coords, full_test_labels, full_boundary_ids):
    prediction = (score > threshold).astype(np.int64)
    if len(np.unique(labels)) == 2:
        auc = float(roc_auc_score(labels, score))
        ap = float(average_precision_score(labels, score))
    else:
        auc, ap = None, None
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    valid = np.zeros(len(full_test_labels), dtype=bool)
    aligned_prediction = np.zeros(len(full_test_labels), dtype=np.int64)
    valid[local_coords] = True
    aligned_prediction[local_coords] = prediction
    affiliation = affiliation_f_segment_macro(
        full_test_labels, aligned_prediction, valid_mask=valid,
        boundary_ids=full_boundary_ids)
    defined_segments = [item for item in affiliation["segments"] if item["defined"]]
    precision = None if not defined_segments else float(np.mean([item["precision"] for item in defined_segments]))
    recall = None if not defined_segments else float(np.mean([item["recall"] for item in defined_segments]))
    return {
        "dataset": dataset, "scoring_object": scoring_object, "run_id": run.name,
        "common_valid_points": int(len(coords)), "threshold": float(threshold),
        "official_test_points": int(len(full_test_labels)),
        "official_test_coverage": float(len(coords) / max(len(full_test_labels), 1)),
        "roc_auc": auc, "average_precision": ap,
        "point_precision": float(precision_score(labels, prediction, zero_division=0)),
        "point_recall": float(recall_score(labels, prediction, zero_division=0)),
        "point_f1": float(f1_score(labels, prediction, zero_division=0)),
        "normal_fpr": None if int(tn + fp) == 0 else float(fp / (tn + fp)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "affiliation_precision_segment_mean": precision,
        "affiliation_recall_segment_mean": recall,
        "affiliation_f_segment_macro": affiliation["affiliation_f_segment_macro"],
        "affiliation_defined_segments": affiliation["defined_segment_count"],
        "affiliation_total_segments": affiliation["total_segment_count"],
    }


def compare(old_run, new_run, output_dir, mapping_path=DEFAULT_MAPPING):
    old = _load_run(old_run, "old concatenated MSL run")
    new = _load_run(new_run, "new segmented MSL run")
    old_identity, new_identity = old["identity"], new["identity"]
    for key in ("feature_data_sha256", "variable_names", "train_length", "input_length"):
        if old_identity.get(key) != new_identity.get(key):
            raise ValueError(f"old/new source identity mismatch for {key}")
    if old_identity.get("time_index") != new_identity.get("time_index"):
        raise ValueError("old/new time axis differs")
    if not np.array_equal(old["labels"], new["labels"]):
        raise ValueError("old/new complete labels differ")
    if not np.array_equal(old["times"], new["times"]):
        raise ValueError("old/new complete time axes differ")
    mapping, mapping_hash, mapping_file = load_mapping(mapping_path)
    if old["run"].name != OLD_MSL_RUN_ID:
        raise ValueError(f"old comparison source must be the registered MSL Full run {OLD_MSL_RUN_ID}")
    if old_identity.get("feature_data_sha256") != mapping.get("expected_feature_sha256"):
        raise ValueError("registered old Full source does not match the verified raw MSL feature identity")
    for label, run in (("old", old), ("new", new)):
        config = run["manifest"].get("config", {})
        for key, expected in FIXED_CONFIG.items():
            actual = config.get(key)
            if isinstance(expected, float):
                same = actual is not None and np.isclose(float(actual), expected, rtol=0, atol=1e-12)
            else:
                same = actual == expected
            if not same:
                raise ValueError(f"{label} MSL Full configuration differs for {key}: {actual!r} != {expected!r}")
    if new_identity.get("feature_data_sha256") != mapping.get("expected_feature_sha256"):
        raise ValueError("new segmented source does not match the verified raw MSL feature identity")
    if new_identity.get("boundary_columns") != ["entity_id"]:
        raise ValueError("new MSL source is not bound to explicit entity_id sequence boundaries")
    train_length = int(new_identity["train_length"])
    full_length = int(new_identity["input_length"])
    channel_ids = _test_channel_ids(mapping, train_length, full_length)
    old_endpoints, new_endpoints = old["endpoints"], new["endpoints"]
    common = common_endpoint_coordinates(old_endpoints, new_endpoints)
    if not len(common):
        raise ValueError("old and new runs have no common valid test endpoints")
    old_rows = np.searchsorted(old_endpoints, common)
    new_rows = np.searchsorted(new_endpoints, common)
    old_labels = old["labels"][common]
    new_labels = new["labels"][common]
    old_times = old["times"][common]
    new_times = new["times"][common]
    if not np.array_equal(old_labels, new_labels) or not np.array_equal(old_times, new_times):
        raise ValueError("old/new common test coordinates do not have identical labels and times")
    local = common - train_length
    if np.any(local < 0) or np.any(local >= len(channel_ids)):
        raise ValueError("common test endpoint falls outside the verified 27-sequence test axis")
    boundary_ids = channel_ids[local]
    full_test_labels = old["labels"][train_length:]
    cells = []
    for run in (old, new):
        rows = old_rows if run is old else new_rows
        labels = old_labels if run is old else new_labels
        for scoring_object in ("all_55_max", "col_0_only"):
            spec = run["derived"][scoring_object]
            cells.append(_metric_row("MSL", scoring_object, run["run"], common,
                                     labels, spec["scores"][rows], spec["threshold"],
                                     local, full_test_labels, channel_ids))
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to replace MSL boundary comparison output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    fields = list(cells[0])
    with (target / "four_cell_comparison.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader(); writer.writerows(cells)
    contract = {
        "schema": "msl-boundary-score-object-comparison-v1",
        "status": "complete", "comparison_scope": "common valid test endpoint intersection",
        "window_boundary_change": "new run only; sequence ids from fixed 27-channel train/test mapping",
        "scoring_change": "same fitted Full GMM and all 55 input fields; compare max over 55 calibrated variables vs col_0 calibrated score",
        "threshold_rule": "per-run/per-score-object validation median/MAD; calibration linear quantile q=0.995; predict strictly score > threshold",
        "old_run": str(old["run"]), "new_run": str(new["run"]),
        "old_source_identity_sha256": old["manifest"].get("source_identity_sha256"),
        "new_source_identity_sha256": new["manifest"].get("source_identity_sha256"),
        "mapping_path": str(mapping_file), "mapping_sha256": mapping_hash,
        "old_valid_test_points": int(len(old_endpoints)), "new_valid_test_points": int(len(new_endpoints)),
        "common_valid_test_points": int(len(common)), "official_test_points": int(len(old["labels"])),
        "common_endpoint_global_sha256": hashlib.sha256(np.ascontiguousarray(common).tobytes()).hexdigest(),
        "common_endpoint_global_indices": common.tolist(),
        "common_time_values_ns": old_times.tolist(),
        "channel_ids": boundary_ids.tolist(), "cells": cells,
        "native_validation": {"old": old["validation_status"], "new": new["validation_status"]},
        "source_files_sha256": {"old_manifest": _sha256(old["run"] / "probe_manifest.json"),
                                "new_manifest": _sha256(new["run"] / "probe_manifest.json")},
    }
    (target / "comparison.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    readme = [
        "# MSL Boundary × Scoring-Object Comparison", "",
        "This comparison changes window boundary treatment only in the new Full run and compares two score aggregations from the same 55-field fitted GMM. All 55 fields remain model inputs; `col_0`-only changes only final scoring.", "",
        f"- Old run: `{old['run']}`", f"- New run: `{new['run']}`",
        f"- Common valid test endpoints: {len(common)} / official test points {len(old['labels'])}.",
        "- Four cells: old concatenated × 55-variable max; old concatenated × col_0; new sequence-bounded × 55-variable max; new sequence-bounded × col_0.",
        "- Every cell derives its own validation median/MAD and calibration q=0.995 threshold; prediction is strictly score > threshold.",
        "- The common endpoint intersection is used without filling missing scores. Event metrics respect the 27 mapped test-channel boundaries.",
        "- Old 0.159/0.279 figures are not inserted; all rows are recomputed from the actual saved per-variable NLL on the common endpoints.",
        "- The new result does not exist until the manual launcher completes; this tool itself only reads and validates saved artifacts.",
    ]
    (target / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", required=True)
    parser.add_argument("--new-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    args = parser.parse_args()
    result = compare(args.old_run, args.new_run, args.output_dir, args.mapping)
    print(json.dumps({"status": result["status"], "output_dir": str(Path(args.output_dir).resolve()),
                      "common_valid_test_points": result["common_valid_test_points"],
                      "cells": len(result["cells"])}, sort_keys=True))


if __name__ == "__main__":
    main()
