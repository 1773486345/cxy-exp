#!/usr/bin/env python
"""Run CATCH, Isolation Forest and PCA on the frozen eight-task Probe protocol."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tarfile
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts.mnrf_common_data import load_task_frame
from scripts.export_mnrf_view_probe_common import export as export_common_contract
from scripts.run_mnrf_catch_common import run as run_catch_common
from scripts.register_mnrf_catch_common import register as register_catch_common
from scripts.evaluate_mnrf_view_probe_common import evaluate as evaluate_common
from scripts.validate_mnrf_view_probe import validate as validate_probe
from ts_benchmark.baselines.catch.CATCH import DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
from ts_benchmark.baselines.mnrf_view_probe.evaluation import (
    affiliation_f_segment_macro, calibrated_quantile, strict_integer_indices,
)
from ts_benchmark.baselines.mnrf_view_probe.probe import _all_boundary_ids

TASKS = ("PSM", "MSL", "GECCO", "Genesis", "NYC", "CalIt2", "Creditcard", "CICIDS")
SOURCE_RUNS = {
    "PSM": "full_gmm-full-20261024T115119Z-3976714",
    "MSL": "full_gmm-full-20261021T112147Z-2470335",
    "GECCO": "full_gmm-full-20261020T204951Z-1642355",
    "Genesis": "full_gmm-full-20261020T205005Z-1645522",
    "NYC": "full_gmm-full-20261020T211341Z-1931786",
    "CalIt2": "full_gmm-full-20261021T112212Z-2474004",
    "Creditcard": "full_gmm-full-20261024T115131Z-3977864",
    "CICIDS": "full_gmm-full-20261024T115157Z-3980319",
}
EXPECTED_SEQ_LEN = {"GECCO": 96, **{task: 192 for task in TASKS if task != "GECCO"}}
ARCHIVE_SCORE_RUNS = {
    "PSM": "run-20260716T222700Z-660928-22488",
    "MSL": "run-20260716T213635Z-58595-28964",
    "GECCO": "run-20260716T213302Z-16971-16047",
    "Genesis": "run-20260716T213208Z-6088-16524",
    "NYC": "run-20260716T222643Z-657156-32728",
    "CalIt2": "run-20260715T183704Z-2374669-5874",
    "Creditcard": "run-20260716T202925Z-3426561-275",
    "CICIDS": "run-20260715T120722Z-1345491-4045",
}
ARCHIVE_LABEL_RUNS = {
    "PSM": "run-20260716T223734Z-776688-6344",
    "MSL": "run-20260716T232646Z-1365820-1457",
    "GECCO": "run-20260716T213559Z-51278-24592",
    "Genesis": "run-20260716T221132Z-484571-23537",
    "NYC": "run-20260716T223043Z-701607-26288",
    "CalIt2": "run-20260715T184135Z-2437260-7989",
    "Creditcard": "run-20260717T085033Z-4154569-18693",
    "CICIDS": "run-20260715T133707Z-2321802-10349",
}
DATASET_FILES = {task: ROOT / "dataset" / "anomaly_detect" / "data" / (task + ".csv") for task in TASKS}
CALIBRATION_QUANTILE = 0.995
SEED = 2021


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(path):
    array = np.asarray(np.load(path, allow_pickle=False))
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _existing_catch_run(output, task):
    """Return recorded runtime for a complete run, rejecting partial output."""
    output = Path(output)
    required = (
        output / "checkpoints" / "catch_common.pt",
        output / "normalization.npz",
        output / "evidence" / "training.json",
        output / "evidence" / "calibration.json",
        output / "results" / "aligned_scores.npz",
        output / "results" / "calibration_scores.npz",
        output / "results" / "metrics.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileExistsError("refusing to overwrite incomplete CATCH-common output; missing: "
                              + ", ".join(missing))
    metrics = _load_json(output / "results" / "metrics.json")
    if metrics.get("status") != "complete" or metrics.get("task_name") != task:
        raise ValueError("existing CATCH-common output is not a completed run for this task")
    return float(metrics.get("training_time_seconds", 0.0)) + float(
        metrics.get("calibration_and_inference_time_seconds", 0.0))


def _existing_light_run(output, method, contract):
    """Validate and reuse one completed IF/PCA result without touching it."""
    output = Path(output)
    manifest_path = output / "evidence" / "baseline.json"
    if not manifest_path.is_file():
        raise FileExistsError("refusing to overwrite incomplete {} output".format(method))
    manifest = _load_json(manifest_path)
    expected = {
        "method": method,
        "seed": SEED,
        "common_contract_path": contract["common_contract_path"],
        "common_contract_sha256": _sha256(contract["common_contract_path"]),
        "probe_run": contract["probe_run"],
        "probe_manifest_sha256": contract["probe_manifest_sha256"],
        "data_csv_sha256": contract["data_csv_sha256"],
        "data_identity": contract["data_identity"],
        "time_identity": contract["time_identity"],
        "boundary_identity": contract["boundary_identity"],
        "calibration_quantile": CALIBRATION_QUANTILE,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError("existing {} run {} binding mismatch: {}".format(method, output, key))
    artifacts = {
        "normalization_sha256": output / "normalization.npz",
        "model_sha256": output / "model.joblib",
        "calibration_scores_sha256": output / "results" / "calibration_scores.npz",
        "score_sha256": output / "results" / "aligned_scores.npz",
        "train_rows_sha256": output / "evidence" / "train_fit_rows.npy",
    }
    for field, path in artifacts.items():
        if not path.is_file() or _sha256(path) != manifest.get(field):
            raise ValueError("existing {} artifact is missing or changed: {}".format(method, path))
    if manifest.get("status") != "complete":
        raise ValueError("existing {} run is not complete".format(method))
    with np.load(output / "results" / "calibration_scores.npz", allow_pickle=False) as calibration:
        cal_scores = np.asarray(calibration["scores"])
        cal_indices = strict_integer_indices(calibration["time_indices"], "calibration coordinates",
                                             upper_bound=int(contract["input_length"]), require_sorted=True)
        threshold = float(np.asarray(calibration["threshold"]))
    with np.load(output / "results" / "aligned_scores.npz", allow_pickle=False) as scored:
        scores = np.asarray(scored["scores"])
        pred = np.asarray(scored["predictions"])
        indices = strict_integer_indices(scored["time_indices"], "test coordinates",
                                         upper_bound=int(contract["input_length"]), require_sorted=True)
        score_threshold = float(np.asarray(scored["threshold"]))
    expected_cal = np.load(contract["catch_splits"]["calibration"]["path"], allow_pickle=False)
    expected_test = np.load(contract["catch_test"]["path"], allow_pickle=False)
    if (not np.array_equal(cal_indices, expected_cal) or not np.array_equal(indices, expected_test)
            or not np.isfinite(cal_scores).all() or not np.isfinite(scores).all()
            or not math.isclose(threshold, score_threshold, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(threshold, calibrated_quantile(cal_scores, CALIBRATION_QUANTILE),
                                rel_tol=0, abs_tol=1e-12)
            or not np.array_equal(pred.astype(np.int8), (scores > threshold).astype(np.int8))):
        raise ValueError("existing {} scores do not match the bound axes or calibration".format(method))
    return output / "results" / "aligned_scores.npz", manifest


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _archive_metadata(run_dir):
    result = {}
    for line in (Path(run_dir) / "metadata.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _archive_result(run_dir):
    run_dir = Path(run_dir)
    archives = sorted(run_dir.glob("*.csv.tar.gz"))
    if len(archives) != 1:
        raise ValueError("expected one archived result CSV in {}".format(run_dir))
    with tarfile.open(archives[0], "r:gz") as archive:
        members = [item for item in archive.getmembers() if item.isfile() and item.name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError("expected one CSV member in {}".format(archives[0]))
        frame = pd.read_csv(archive.extractfile(members[0]))
    if len(frame) != 1:
        raise ValueError("archived physical-task result must have one row")
    return archives[0], members[0].name, frame.iloc[0].to_dict()


def _archive_references():
    catch_root = ROOT.parent / "CATCH-master"
    summary_path = catch_root / "result" / "CATCH_reproduction_summary" / "physical_task_metrics.csv"
    summary = pd.read_csv(summary_path)
    summary_by_task = {str(row.physical_task): row for row in summary.itertuples(index=False)}
    rows = []
    configs = {}
    for task in TASKS:
        data_path = DATASET_FILES[task]
        score_dir = catch_root / "result" / "score" / "CATCH" / task / ARCHIVE_SCORE_RUNS[task]
        label_dir = catch_root / "result" / "label" / "CATCH" / task / ARCHIVE_LABEL_RUNS[task]
        score_meta = _archive_metadata(score_dir)
        label_meta = _archive_metadata(label_dir)
        for kind, metadata in (("score", score_meta), ("label", label_meta)):
            if metadata.get("dataset_file") != data_path.name:
                raise ValueError("{} archive task file mismatch for {}".format(kind, task))
            if metadata.get("dataset_sha256") != _sha256(data_path):
                raise ValueError("{} archive raw dataset hash differs for {}".format(kind, task))
            script = Path(metadata["official_script"])
            if not script.is_file() or _sha256(script) != metadata.get("official_script_sha256"):
                raise ValueError("{} archive official script binding differs for {}".format(kind, task))
            config = Path(metadata["official_config"])
            if not config.is_file() or _sha256(config) != metadata.get("official_config_sha256"):
                raise ValueError("{} archive global config binding differs for {}".format(kind, task))
        score_archive, score_member, score = _archive_result(score_dir)
        label_archive, label_member, label = _archive_result(label_dir)
        score_params = json.loads(score["model_params"])
        strategy_args = json.loads(score["strategy_args"])
        if int(strategy_args.get("seed", -1)) != SEED:
            raise ValueError("historical score archive seed differs for {}".format(task))
        shell_text = Path(score_meta["official_script"]).read_text(encoding="utf-8")
        match = re.search(r"--model-hyper-params\s+'([^']+)'", shell_text)
        if not match or json.loads(match.group(1)) != score_params:
            raise ValueError("score archive model_params differ from its official task command for {}".format(task))
        resolved_params = dict(DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS)
        resolved_params.update(score_params)
        resolved_params["seed"] = SEED
        checkpoint_paths = [str(path) for root in (score_dir, label_dir)
                            for pattern in ("*.pt", "*.pth", "*.ckpt") for path in root.rglob(pattern)]
        summary_row = summary_by_task[task]
        if (ARCHIVE_SCORE_RUNS[task] not in str(summary_row.score_report)
                or ARCHIVE_LABEL_RUNS[task] not in str(summary_row.label_report)):
            raise ValueError("physical summary selected different score/label runs for {}".format(task))
        if not math.isclose(float(summary_row.reproduced_score_auc_roc), float(score["auc_roc"]), rel_tol=0, abs_tol=1e-12):
            raise ValueError("physical summary score AUC differs from archived score run for {}".format(task))
        if not math.isclose(float(summary_row.reproduced_label_affiliation_f), float(label["affiliation_f"]), rel_tol=0, abs_tol=1e-12):
            raise ValueError("physical summary Aff-F differs from archived label run for {}".format(task))
        configs[task] = {"archive_score_model_params": score_params,
                         "archive_strategy_args": strategy_args,
                         "resolved_common_run_model_config": resolved_params,
                         "score_run_id": ARCHIVE_SCORE_RUNS[task],
                         "label_run_id": ARCHIVE_LABEL_RUNS[task]}
        rows.extend([
            {"dataset": task, "reference": "CATCH archive score", "protocol": "unfixed_detect_score archive",
             "run_id": ARCHIVE_SCORE_RUNS[task], "roc_auc": float(score["auc_roc"]),
             "average_precision": float(score["auc_pr"]), "affiliation_precision": None,
             "affiliation_recall": None, "affiliation_f": None,
             "fit_seconds": float(score.get("fit_time", np.nan)),
             "inference_seconds": float(score.get("inference_time", np.nan)),
             "model_params_json": json.dumps(score_params, sort_keys=True),
             "result_archive": str(score_archive.relative_to(catch_root)), "archive_member": score_member,
             "archive_sha256": _sha256(score_archive), "metadata_path": str(score_dir / "metadata.txt"),
             "metadata_sha256": _sha256(score_dir / "metadata.txt"),
             "official_script": score_meta["official_script"], "official_script_sha256": score_meta["official_script_sha256"],
             "checkpoint_count": len(checkpoint_paths)} ,
            {"dataset": task, "reference": "CATCH archive label", "protocol": "unfixed_detect_label archive",
             "run_id": ARCHIVE_LABEL_RUNS[task], "roc_auc": None, "average_precision": None,
             "affiliation_precision": float(label["affiliation_precision"]),
             "affiliation_recall": float(label["affiliation_recall"]),
             "affiliation_f": float(label["affiliation_f"]),
             "fit_seconds": float(label.get("fit_time", np.nan)),
             "inference_seconds": float(label.get("inference_time", np.nan)),
             "model_params_json": json.dumps(json.loads(label["model_params"]), sort_keys=True),
             "result_archive": str(label_archive.relative_to(catch_root)), "archive_member": label_member,
             "archive_sha256": _sha256(label_archive), "metadata_path": str(label_dir / "metadata.txt"),
             "metadata_sha256": _sha256(label_dir / "metadata.txt"),
             "official_script": label_meta["official_script"], "official_script_sha256": label_meta["official_script_sha256"],
             "checkpoint_count": len(checkpoint_paths)},
        ])
    return configs, rows, summary_path


def _read_contract(contract_path):
    contract_path = Path(contract_path)
    contract = _load_json(contract_path)
    for name in ("labels", "train_rows", "time_values_ns"):
        record = contract[name]
        if _sha256(record["path"]) != record["sha256"]:
            raise ValueError("common contract {} hash mismatch".format(name))
    for name in ("train", "validation", "calibration"):
        record = contract["catch_splits"][name]
        if _sha256(record["path"]) != record["sha256"]:
            raise ValueError("common contract {} split hash mismatch".format(name))
    if _sha256(contract["catch_test"]["path"]) != contract["catch_test"]["sha256"]:
        raise ValueError("common contract test endpoint hash mismatch")
    boundary = contract.get("boundary_ids")
    if boundary and _sha256(boundary["path"]) != boundary["sha256"]:
        raise ValueError("common contract boundary identity mismatch")
    return contract


def _verify_frame_for_contract(task, data_path, contract):
    frame, label_column = load_task_frame(data_path)
    values, columns = MNRFLiteV2._features_from_frame(frame)
    if _sha256(data_path) != contract.get("data_csv_sha256"):
        raise ValueError("raw CSV bytes differ from frozen contract")
    if list(map(str, columns)) != contract["variable_names"]:
        raise ValueError("variable order differs from frozen contract")
    source = contract["probe_source_identity"]
    from ts_benchmark.baselines.mnrf_view_probe.probe import _chunked_array_hash, _hash_strings
    if _chunked_array_hash(values) != source.get("feature_data_sha256"):
        raise ValueError("feature values differ from frozen Full source")
    times = np.asarray(MNRFLiteV2._time_index(frame).values_ns, dtype=np.int64)
    if _chunked_array_hash(times) != contract.get("time_identity"):
        raise ValueError("canonical time axis differs from frozen contract")
    boundaries, boundary_columns = _all_boundary_ids(frame)
    if boundary_columns != contract.get("boundary_columns"):
        raise ValueError("entity/session columns differ from frozen contract")
    actual_boundary_hash = None if boundaries is None else _hash_strings(boundaries)
    if actual_boundary_hash != contract.get("boundary_identity"):
        raise ValueError("entity/session identity differs from frozen contract")
    labels = frame[label_column].to_numpy(dtype=np.int64)
    frozen = np.load(contract["labels"]["path"], allow_pickle=False)
    if frozen.ndim != 1 or frozen.dtype.kind not in "iu" or not np.array_equal(frozen.astype(np.int64), labels):
        raise ValueError("full label axis differs from frozen contract")
    return frame, values, labels, times


def _full_axis_metrics(labels, endpoints, scores, threshold, test_segments, boundary_ids):
    endpoints = strict_integer_indices(endpoints, "baseline test endpoints", upper_bound=len(labels), require_sorted=True)
    scores = np.asarray(scores)
    if scores.ndim != 1 or len(scores) != len(endpoints) or not np.isfinite(scores).all():
        raise ValueError("baseline test score shape or values are invalid")
    prediction = (scores > float(threshold)).astype(np.int64)
    actual = labels[endpoints].astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(actual, prediction, labels=[0, 1]).ravel()
    normal = actual == 0
    anomalous = actual == 1
    full_prediction = np.zeros(len(labels), dtype=np.int64)
    valid = np.zeros(len(labels), dtype=bool)
    full_prediction[endpoints] = prediction
    valid[endpoints] = True
    aff = affiliation_f_segment_macro(labels, full_prediction, valid, boundary_ids)
    defined_affiliation = [segment for segment in aff["segments"] if segment.get("defined")]
    aff["affiliation_precision_segment_macro"] = (
        None if not defined_affiliation else float(np.mean([segment["precision"] for segment in defined_affiliation])))
    aff["affiliation_recall_segment_macro"] = (
        None if not defined_affiliation else float(np.mean([segment["recall"] for segment in defined_affiliation])))
    total_segments = covered_segments = hit_segments = 0
    for left, right in test_segments:
        left, right = int(left), int(right)
        row = left
        while row < right:
            if labels[row] == 0:
                row += 1
                continue
            event_start = row
            while row + 1 < right and labels[row + 1] == 1:
                row += 1
            event_stop = row + 1
            total_segments += 1
            first = int(np.searchsorted(endpoints, event_start, side="left"))
            last = int(np.searchsorted(endpoints, event_stop, side="left"))
            if last > first:
                covered_segments += 1
                if np.any(prediction[first:last]):
                    hit_segments += 1
            row += 1
    return {
        "roc_auc": None if np.unique(actual).size < 2 else float(roc_auc_score(actual, scores)),
        "average_precision": float(average_precision_score(actual, scores)),
        "precision": float(precision_score(actual, prediction, zero_division=0)),
        "recall": float(recall_score(actual, prediction, zero_division=0)),
        "point_f1": float(f1_score(actual, prediction, zero_division=0)),
        "fpr": None if not np.any(normal) else float(np.sum((prediction == 1) & normal) / np.sum(normal)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "valid_points": int(len(endpoints)),
        "threshold": float(threshold), "affiliation": aff,
        "total_anomaly_segments": int(total_segments),
        "scored_anomaly_segments": int(covered_segments), "hit_anomaly_segments": int(hit_segments),
    }


def _fit_light_method(method, output, values, labels, times, contract):
    output = Path(output)
    if output.exists():
        raise FileExistsError("baseline output already exists: {}".format(output))
    train_rows = strict_integer_indices(np.load(contract["train_rows"]["path"], allow_pickle=False),
                                       "raw train rows", upper_bound=len(values), require_sorted=True)
    test_indices = strict_integer_indices(np.load(contract["catch_test"]["path"], allow_pickle=False),
                                          "test endpoints", upper_bound=len(values), require_sorted=True)
    calibration_indices = strict_integer_indices(np.load(contract["catch_splits"]["calibration"]["path"], allow_pickle=False),
                                                 "calibration endpoints", upper_bound=len(values), require_sorted=True)
    if output.exists():
        return _existing_light_run(output, method, contract)
    scaler = StandardScaler().fit(values[train_rows])
    normalized = scaler.transform(values).astype(np.float32, copy=False)
    fit_start = time.perf_counter()
    if method == "IsolationForest":
        estimator = IsolationForest(n_estimators=100, max_samples=min(256, len(train_rows)),
                                    contamination="auto", random_state=SEED, n_jobs=1)
        estimator.fit(normalized[train_rows])
        fit_seconds = time.perf_counter() - fit_start

        def score_rows(indices):
            return -estimator.score_samples(normalized[indices])

        params = {"n_estimators": 100, "max_samples": min(256, len(train_rows)),
                  "contamination": "auto", "random_state": SEED, "n_jobs": 1,
                  "score": "negative score_samples"}
    else:
        estimator, params = _fit_residual_pca(normalized[train_rows], normalized.shape[1])
        fit_seconds = time.perf_counter() - fit_start

        def score_rows(indices):
            output_scores = np.empty(len(indices), dtype=np.float64)
            for start in range(0, len(indices), 8192):
                block_indices = indices[start:start + 8192]
                block = normalized[block_indices]
                reconstruction = estimator.inverse_transform(estimator.transform(block))
                output_scores[start:start + len(block)] = np.mean((block - reconstruction) ** 2, axis=1)
            return output_scores

    inference_start = time.perf_counter()
    calibration_scores = np.asarray(score_rows(calibration_indices), dtype=np.float64)
    threshold = calibrated_quantile(calibration_scores, CALIBRATION_QUANTILE)
    test_scores = np.asarray(score_rows(test_indices), dtype=np.float64)
    inference_seconds = time.perf_counter() - inference_start
    predictions = (test_scores > threshold).astype(np.int64)
    output.mkdir(parents=True)
    (output / "results").mkdir()
    (output / "evidence").mkdir()
    normalization_path = output / "normalization.npz"
    np.savez(normalization_path, mean=scaler.mean_, scale=scaler.scale_,
             train_rows_sha256=_sha256(contract["train_rows"]["path"]))
    model_path = output / "model.joblib"
    joblib.dump({"scaler": scaler, "estimator": estimator}, model_path)
    calibration_path = output / "results" / "calibration_scores.npz"
    np.savez_compressed(calibration_path, scores=calibration_scores, time_indices=calibration_indices,
                        time_values_ns=times[calibration_indices], threshold=np.asarray(threshold))
    scores_path = output / "results" / "aligned_scores.npz"
    np.savez_compressed(scores_path, scores=test_scores, predictions=predictions,
                        time_indices=test_indices, time_values_ns=times[test_indices],
                        threshold=np.asarray(threshold), actual=labels[test_indices])
    np.save(output / "evidence" / "train_fit_rows.npy", train_rows)
    manifest = {
        "protocol": "MNRF-common-light-baseline-v1", "status": "complete", "task_name": contract["task_name"],
        "method": method, "seed": SEED, "run_identity": hashlib.sha256(
            json.dumps({"task": contract["task_name"], "method": method,
                        "contract": _sha256(contract["common_contract_path"]), "seed": SEED}, sort_keys=True).encode()
        ).hexdigest(),
        "probe_run": contract["probe_run"], "probe_manifest_sha256": contract["probe_manifest_sha256"],
        "common_contract_path": contract["common_contract_path"],
        "common_contract_sha256": _sha256(contract["common_contract_path"]),
        "data_csv_path": contract["data_csv_path"], "data_csv_sha256": contract["data_csv_sha256"],
        "data_identity": contract["data_identity"], "time_identity": contract["time_identity"],
        "variable_names": contract["variable_names"], "boundary_identity": contract["boundary_identity"],
        "train_rows_path": str((output / "evidence" / "train_fit_rows.npy").resolve()),
        "train_rows_sha256": _sha256(output / "evidence" / "train_fit_rows.npy"),
        "train_rows_count": int(len(train_rows)), "normalization_path": str(normalization_path.resolve()),
        "normalization_sha256": _sha256(normalization_path), "model_path": str(model_path.resolve()),
        "model_sha256": _sha256(model_path), "parameters": params,
        "training_protocol": "fit only on the exact frozen raw training rows; no internal split",
        "context_protocol": "row-wise statistical score at the frozen endpoint; no window/future context",
        "calibration_quantile": CALIBRATION_QUANTILE,
        "calibration_scores_path": str(calibration_path.resolve()), "calibration_scores_sha256": _sha256(calibration_path),
        "calibration_indices_sha256": _sha256(contract["catch_splits"]["calibration"]["path"]),
        "score_path": str(scores_path.resolve()), "score_sha256": _sha256(scores_path),
        "test_indices_sha256": _sha256(contract["catch_test"]["path"]),
        "threshold": float(threshold), "fit_seconds": float(fit_seconds),
        "inference_seconds": float(inference_seconds), "training_environment": {
            "python": os.sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "torch": torch.__version__,
        },
    }
    _atomic_json(output / "evidence" / "baseline.json", manifest)
    return scores_path, manifest


def _fit_residual_pca(train_values, feature_count):
    """Fit PCA on training rows, retaining a residual dimension on every task."""
    feature_count = int(feature_count)
    if feature_count < 2:
        raise ValueError("PCA residual scoring requires at least two input variables")
    preliminary = PCA(n_components=0.95, svd_solver="full")
    preliminary.fit(train_values)
    component_count = min(int(preliminary.n_components_), feature_count - 1)
    if component_count < 1:
        raise ValueError("PCA residual scoring selected no retained dimensions")
    estimator = PCA(n_components=component_count, svd_solver="full")
    estimator.fit(train_values)
    params = {"variance_target": 0.95, "maximum_components": feature_count - 1,
              "selection_rule": "min(components_for_95_percent_variance, D-1)",
              "svd_solver": "full", "selected_components": int(estimator.n_components_),
              "residual_dimensions": feature_count - int(estimator.n_components_),
              "score": "per-row mean squared reconstruction error"}
    return estimator, params


def _metrics_from_evaluation(method, evaluation):
    key = "probe" if method == "MNRF Full" else "catch"
    payload = evaluation[key]
    aff = payload["affiliation"]
    return {"roc_auc": payload.get("auc_roc"), "average_precision": payload.get("auc_pr"),
            "threshold": payload["threshold"], "affiliation_precision": aff.get("affiliation_precision"),
            "affiliation_recall": aff.get("affiliation_recall"),
            "affiliation_f_segment_macro": aff.get("affiliation_f_segment_macro"),
            "affiliation_total_segments": aff.get("total_segment_count"),
            "affiliation_defined_segments": aff.get("defined_segment_count")}


def _method_row(task, method, source_run, metrics, fit_seconds=None, inference_seconds=None,
                parameter_summary=None, protocol="common calibrated protocol"):
    affiliation = metrics.get("affiliation") or {}
    return {"dataset": task, "method": method, "protocol": protocol,
            "source_full_run_id": source_run, "seed": SEED,
            "roc_auc": metrics.get("roc_auc"), "average_precision": metrics.get("average_precision"),
            "threshold": metrics.get("threshold"), "precision": metrics.get("precision"),
            "recall": metrics.get("recall"), "point_f1": metrics.get("point_f1"),
            "normal_fpr": metrics.get("fpr"), "tp": metrics.get("tp"), "fp": metrics.get("fp"),
            "fn": metrics.get("fn"), "tn": metrics.get("tn"),
            "affiliation_precision": affiliation.get("affiliation_precision_segment_macro",
                                                       metrics.get("affiliation_precision")),
            "affiliation_recall": affiliation.get("affiliation_recall_segment_macro",
                                                    metrics.get("affiliation_recall")),
            "affiliation_f_segment_macro": affiliation.get("affiliation_f_segment_macro",
                                                           metrics.get("affiliation_f_segment_macro")),
            "affiliation_total_segments": affiliation.get("total_segment_count",
                                                           metrics.get("affiliation_total_segments")),
            "affiliation_defined_segments": affiliation.get("defined_segment_count",
                                                              metrics.get("affiliation_defined_segments")),
            "total_anomaly_segments": metrics.get("total_anomaly_segments"),
            "scored_anomaly_segments": metrics.get("scored_anomaly_segments"),
            "hit_anomaly_segments": metrics.get("hit_anomaly_segments"),
            "fit_seconds": fit_seconds, "inference_seconds": inference_seconds,
            "parameter_summary": parameter_summary}


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _task_pipeline(task, batch_name, analysis_root, score_root, catch_configs, method_rows, task_records):
    source_dir = ROOT / "result" / "score" / "MNRF-ViewProbe" / task / SOURCE_RUNS[task]
    data_path = DATASET_FILES[task]
    task_analysis = analysis_root / task
    task_score = score_root / task
    task_analysis.mkdir(parents=True, exist_ok=True)
    task_score.mkdir(parents=True, exist_ok=True)
    task_record = {"dataset": task, "source_run_id": SOURCE_RUNS[task], "source_path": str(source_dir),
                   "status": "running", "methods": {}}
    task_records.append(task_record)
    source_validation = validate_probe(source_dir)
    if source_validation.get("status") != "complete":
        raise ValueError("native Full source validation did not return complete")
    source_manifest = _load_json(source_dir / "probe_manifest.json")
    source_identity = source_manifest.get("source_identity") or {}
    if (source_dir.name != SOURCE_RUNS[task]
            or source_identity.get("task_name") != data_path.name
            or source_manifest.get("task_name") != data_path.name):
        raise ValueError("fixed source path/task identity mismatch")
    if int(source_identity.get("seq_len", -1)) != EXPECTED_SEQ_LEN[task]:
        raise ValueError("frozen Full window length differs from the task protocol")
    contract_dir = task_analysis / "common_contract"
    contract_path = contract_dir / "common_contract.json"
    if contract_path.is_file():
        contract = _read_contract(contract_path)
        if (Path(contract.get("probe_run", "")).resolve() != source_dir.resolve()
                or contract.get("probe_manifest_sha256") != _sha256(source_dir / "probe_manifest.json")
                or int(contract.get("catch_seq_len", -1)) != EXPECTED_SEQ_LEN[task]):
            raise ValueError("existing common contract is not bound to the fixed Full source")
    else:
        contract = export_common_contract(source_dir, data_path, contract_dir,
                                          catch_seq_len=EXPECTED_SEQ_LEN[task])
    _verify_frame_for_contract(task, data_path, contract)
    contract = _read_contract(contract_path)
    contract["common_contract_path"] = str(contract_path.resolve())
    # The full-axis MNRF scores must cover precisely the common test endpoint set.
    probe_scores_path = source_dir / "results" / "aligned_scores.npz"
    with np.load(probe_scores_path, allow_pickle=False) as probe_payload:
        probe_indices = strict_integer_indices(probe_payload["time_indices"], "MNRF test endpoints",
                                               upper_bound=int(contract["input_length"]), require_sorted=True)
        probe_values = np.asarray(probe_payload["scores"])
        probe_threshold = float(np.asarray(probe_payload["threshold"]))
        probe_actual = np.asarray(probe_payload["actual"], dtype=np.int64)
        probe_times = np.asarray(probe_payload["time_values_ns"], dtype=np.int64)
        probe_predictions = (np.asarray(probe_payload["predictions"], dtype=np.int64)
                             if "predictions" in probe_payload.files else None)
    test_indices = np.load(contract["catch_test"]["path"], allow_pickle=False)
    time_axis = np.load(contract["time_values_ns"]["path"], allow_pickle=False)
    labels = np.load(contract["labels"]["path"], allow_pickle=False).astype(np.int64)
    if not np.array_equal(probe_indices, test_indices):
        raise ValueError("MNRF and common test endpoints are not identical")
    if not np.array_equal(probe_times, time_axis[test_indices]):
        raise ValueError("MNRF score timestamps differ from common test timestamps")
    if not np.array_equal(probe_actual, labels[test_indices]):
        raise ValueError("MNRF score labels differ from the frozen full label axis")
    full_predictions = (probe_values.astype(np.float64) > probe_threshold).astype(np.int64)
    if probe_predictions is not None and not np.array_equal(probe_predictions, full_predictions):
        raise ValueError("MNRF predictions do not match its frozen threshold")
    boundary_record = contract.get("boundary_ids")
    boundary_ids = (None if not boundary_record else
                    np.load(boundary_record["path"], allow_pickle=False))
    test_segments = source_identity.get("test_segments", [])
    source_run = source_dir.name

    catch_config = catch_configs[task]["resolved_common_run_model_config"]
    catch_output = task_score / "CATCH-common"
    model_config = dict(catch_config)
    catch_args = argparse.Namespace(
        data_csv=str(data_path), train_indices=contract["catch_splits"]["train"]["path"],
        validation_indices=contract["catch_splits"]["validation"]["path"],
        calibration_indices=contract["catch_splits"]["calibration"]["path"],
        train_rows=contract["train_rows"]["path"], output=str(catch_output),
        common_contract=str(contract_path), task_name=task, test_start=None,
        device="cuda:0", seed=SEED, seq_len=int(catch_config["seq_len"]),
        patch_size=int(catch_config["patch_size"]), patch_stride=int(catch_config["patch_stride"]),
        d_model=int(catch_config["d_model"]), cf_dim=int(catch_config["cf_dim"]),
        d_ff=int(catch_config["d_ff"]), e_layers=int(catch_config["e_layers"]),
        n_heads=int(catch_config["n_heads"]), head_dim=int(catch_config["head_dim"]),
        dropout=float(catch_config["dropout"]), learning_rate=float(catch_config["lr"]),
        epochs=int(catch_config["num_epochs"]), batch_size=int(catch_config["batch_size"]),
        patience=int(catch_config["patience"]), calibration_quantile=CALIBRATION_QUANTILE,
        score_lambda=float(catch_config["score_lambda"]), mask_learning_rate=float(catch_config["Mlr"]),
        auxi_lambda=float(catch_config["auxi_lambda"]), dc_lambda=float(catch_config["dc_lambda"]),
        temperature=float(catch_config["temperature"]), head_dropout=float(catch_config["head_dropout"]),
        anomaly_ratio=catch_config["anomaly_ratio"], use_multi_scale=catch_config.get("use_multi_scale", "False"),
        small_kernel_merged=catch_config.get("small_kernel_merged", "False"), itr=int(catch_config.get("itr", 1)),
        inference_patch_size=int(catch_config["inference_patch_size"]),
        inference_patch_stride=int(catch_config["inference_patch_stride"]),
        pct_start=float(catch_config["pct_start"]), model_config=model_config,
    )
    started = time.perf_counter()
    catch_score_path = catch_output / "results" / "aligned_scores.npz"
    common_manifest_path = catch_output / "results" / "common_manifest.json"
    if catch_output.exists():
        catch_elapsed = _existing_catch_run(catch_output, task)
        if not common_manifest_path.is_file():
            register_catch_common(argparse.Namespace(
                probe=str(probe_scores_path), catch=str(catch_score_path), labels=contract["labels"]["path"],
                train_indices=contract["catch_splits"]["train"]["path"],
                validation_indices=contract["catch_splits"]["validation"]["path"],
                calibration_indices=contract["catch_splits"]["calibration"]["path"],
                training_evidence=str(catch_output / "evidence" / "training.json"),
                calibration_evidence=str(catch_output / "evidence" / "calibration.json"),
                task_name=task, output=str(common_manifest_path),
            ))
    else:
        catch_output = run_catch_common(catch_args)
        catch_elapsed = time.perf_counter() - started
        register_catch_common(argparse.Namespace(
            probe=str(probe_scores_path), catch=str(catch_score_path), labels=contract["labels"]["path"],
            train_indices=contract["catch_splits"]["train"]["path"],
            validation_indices=contract["catch_splits"]["validation"]["path"],
            calibration_indices=contract["catch_splits"]["calibration"]["path"],
            training_evidence=str(catch_output / "evidence" / "training.json"),
            calibration_evidence=str(catch_output / "evidence" / "calibration.json"),
            task_name=task, output=str(common_manifest_path),
        ))
    strict_result = evaluate_common(
        str(probe_scores_path), str(catch_score_path), contract["labels"]["path"],
        boundary_path=None if not boundary_record else boundary_record["path"],
        split_paths=tuple(contract["catch_splits"][name]["path"] for name in ("train", "validation", "calibration")),
        common_manifest=str(common_manifest_path), strict_common=True,
    )
    if strict_result["protocol"] != "CATCH-common" or strict_result["common_count"] != len(test_indices):
        raise ValueError("native strict CATCH-common evaluator did not accept the full test axis")
    catch_metrics = _full_axis_metrics(labels, test_indices,
                                      np.load(catch_score_path, allow_pickle=False)["scores"],
                                      strict_result["catch"]["threshold"], test_segments, boundary_ids)
    method_rows.append(_method_row(task, "MNRF Full", source_run,
                                   _full_axis_metrics(labels, test_indices, probe_values,
                                                      probe_threshold, test_segments, boundary_ids),
                                   parameter_summary="saved Full; independently validated"))
    method_rows.append(_method_row(task, "CATCH-common", source_run, catch_metrics,
                                   fit_seconds=catch_elapsed,
                                   inference_seconds=json.loads((catch_output / "results" / "metrics.json").read_text()).get("calibration_and_inference_time_seconds"),
                                   parameter_summary=json.dumps(catch_config, sort_keys=True)))
    task_record["methods"]["CATCH-common"] = {"status": "complete", "output": str(catch_output),
                                               "strict_evaluation": str(catch_output / "results" / "common_evaluation.json"),
                                               "fit_and_pipeline_seconds": catch_elapsed,
                                               "archive_params": catch_configs[task]["archive_score_model_params"]}
    _atomic_json(catch_output / "results" / "common_evaluation.json", strict_result)

    # The two unsupervised statistical references use the identical raw train rows,
    # calibration endpoints and full test endpoints, but do not use window context.
    frame, values, labels, times = _verify_frame_for_contract(task, data_path, contract)
    for method in ("IsolationForest", "PCA"):
        baseline_output = task_score / method
        if method == "PCA" and baseline_output.is_dir():
            old_model_path = baseline_output / "model.joblib"
            if old_model_path.is_file():
                old_estimator = joblib.load(old_model_path).get("estimator")
                if old_estimator is not None and int(old_estimator.n_components_) >= values.shape[1]:
                    # Preserve the historical full-reconstruction result and
                    # write the residual-subspace correction beside it.
                    baseline_output = task_score / "PCA-residual"
        score_path, evidence = _fit_light_method(method, baseline_output, values, labels, times, contract)
        with np.load(score_path, allow_pickle=False) as scored:
            scored_indices = strict_integer_indices(scored["time_indices"], "{} endpoints".format(method),
                                                     upper_bound=len(labels), require_sorted=True)
            scored_times = np.asarray(scored["time_values_ns"], dtype=np.int64)
            scored_actual = np.asarray(scored["actual"], dtype=np.int64)
            scores = np.asarray(scored["scores"], dtype=np.float64)
            threshold = float(np.asarray(scored["threshold"]))
        if (not np.array_equal(scored_indices, test_indices)
                or not np.array_equal(scored_times, time_axis[test_indices])
                or not np.array_equal(scored_actual, labels[test_indices])
                or evidence["common_contract_sha256"] != _sha256(contract_path)):
            raise ValueError("{} score coordinates/identity differ from the common contract".format(method))
        metrics = _full_axis_metrics(labels, test_indices, scores, threshold, test_segments, boundary_ids)
        method_rows.append(_method_row(task, method, source_run, metrics,
                                       fit_seconds=evidence["fit_seconds"],
                                       inference_seconds=evidence["inference_seconds"],
                                       parameter_summary=json.dumps(evidence["parameters"], sort_keys=True)))
        task_record["methods"][method] = {"status": "complete", "output": str(baseline_output),
                                         "fit_seconds": evidence["fit_seconds"],
                                         "inference_seconds": evidence["inference_seconds"],
                                         "parameters": evidence["parameters"]}
    task_record["status"] = "complete"
    task_record["source_validation"] = source_validation
    task_record["coordinate_count"] = int(len(test_indices))
    task_record["calibration_count"] = int(contract["catch_splits"]["calibration"]["count"])
    _atomic_json(task_analysis / "task_status.json", task_record)
    stale_failure = analysis_root / "{}.failure.json".format(task)
    if stale_failure.is_file():
        stale_failure.unlink()


def _write_readme(path, batch_name, method_rows, archive_rows, task_records, catch_configs, archive_summary):
    by_task = {task: {row["method"]: row for row in method_rows if row["dataset"] == task} for task in TASKS}
    lines = ["# 八数据集共同协议基线对照", "",
             "批次：`{}`。共同协议使用各固定 Full 源绑定的数据文件、变量顺序、原始训练行、冻结 train/validation/calibration/test 端点；测试标签只用于指标。".format(batch_name),
             "CATCH 每组使用 score 归档中的任务参数、seed=2021，按冻结训练范围拟合、冻结 validation 早停、calibration 的 0.995 分位数阈值。原 benchmark 的 score/label 归档来自不同运行，分开列示，不拼成一行。",
             "Isolation Forest 和 PCA 使用相同原始训练行拟合 StandardScaler 与模型，并在共同 calibration/test 端点评分。它们是逐点统计基线，不使用窗口上下文；与 CATCH/MNRF 的过去窗口上下文不同。三者都使用 score > 各自阈值。Aff-F 此处为 `affiliation_f_segment_macro`，只在有效连续评分段计算，不是论文原始协议结果。", "",
             "## 共同协议结果", "",
             "| 数据集 | 方法 | ROC-AUC | AP | 点 Precision | Recall | F1 | 正常点 FPR | TP/FP/FN/TN | Aff-F（有效段宏平均） | 命中/有分数/异常段 | 阈值 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        for method in ("MNRF Full", "CATCH-common", "IsolationForest", "PCA"):
            row = by_task.get(task, {}).get(method)
            if not row:
                lines.append("| {} | {} | blocked | | | | | | | | | |".format(task, method))
                continue
            confusion = "{}/{}/{}/{}".format(row.get("tp"), row.get("fp"), row.get("fn"), row.get("tn"))
            hit = "{}/{}/{}".format(row.get("hit_anomaly_segments"), row.get("scored_anomaly_segments"),
                                    row.get("total_anomaly_segments"))
            vals = [row.get("roc_auc"), row.get("average_precision"), row.get("precision"), row.get("recall"),
                    row.get("point_f1"), row.get("normal_fpr"), row.get("affiliation_f_segment_macro"), row.get("threshold")]
            fmt = ["" if value is None else "{:.6f}".format(float(value)) for value in vals]
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                task, method, fmt[0], fmt[1], fmt[2], fmt[3], fmt[4], fmt[5], confusion, fmt[6], hit, fmt[7]))
    lines += ["", "## 固定数据轴与标注污染", "",
              "下表的标注计数按固定 Full 源的完整原始区间统计，未用于清洗或训练/阈值选择；test 原始标签数包含冷启动前缀，指标仍只在共同有效 test endpoints 上计算。`boundary_columns` 为空表示当前 CSV/冻结合同没有可用实体或 session 分段列。", "",
              "| 数据集 | 变量数 | train/validation/calibration/test 原始异常点 | 时间轴/冻结间隔 | 边界列 |", "|---|---:|---:|---|---|"]
    contract_root = ROOT / "result" / "analysis" / "MNRF-ViewProbe-Common" / batch_name
    for task in TASKS:
        contract = _load_json(contract_root / task / "common_contract" / "common_contract.json")
        source = contract["probe_source_identity"]
        labels = np.load(contract["labels"]["path"], mmap_mode="r", allow_pickle=False)
        ranges = dict(source.get("ranges", {}))
        ranges["test"] = [int(source["train_length"]), int(source["input_length"])]
        anomaly_counts = []
        for split in ("train", "validation", "calibration", "test"):
            begin, end = map(int, ranges[split])
            anomaly_counts.append(int(np.asarray(labels[begin:end]).sum()))
        cadence = source.get("temporal_rule", {}).get("expected_interval_ns")
        time_desc = "{} / {} ns".format(source.get("time_index", {}).get("kind"), cadence)
        boundary_columns = ",".join(source.get("boundary_columns", [])) or "none"
        lines.append("| {} | {} | {}/{}/{}/{} | {} | {} |".format(
            task, len(source.get("variable_names", [])), *anomaly_counts, time_desc, boundary_columns))
    lines += ["", "七组非 CalIt2 CSV 的 `date` 是重复在变量长表行上的整数位置号（每个任务 1..T），loader 将它解释成 1ns 等距索引；这只证明保留了行顺序，不能证明物理采样间隔或相邻事件属于同一过程。CalIt2 的日期为真实格式化时间戳，冻结间隔为 30 分钟。Creditcard 交易与 CICIDS 网络流在本 benchmark 文件中没有 session/entity 边界，窗口因此是固定行序列假设，不应解读为同一交易主体或网络会话的连续动态。", ""]
    lines += ["", "## 原始 CATCH 归档参照", "",
              "这些是 `CATCH-master` 原生 `unfixed_detect_score` 与 `unfixed_detect_label` 的独立归档运行。score 行只含 ROC-AUC/AP；label 行只含 Affiliation P/R/F。它们的运行 ID、脚本和压缩 CSV 哈希见 `archive_references.csv`；当前未发现归档 checkpoint，因此本轮按 score 归档参数补跑共同协议 CATCH。原归档分数使用原任务协议阈值/评估过程，不等同于下表独立 calibration 0.995。", "",
              "| 数据集 | 历史 score ROC-AUC | 历史 score AP | score run | 历史 label Aff-P | Aff-R | Aff-F | label run |", "|---|---:|---:|---|---:|---:|---:|---|"]
    archive_by = {}
    for row in archive_rows:
        archive_by.setdefault(row["dataset"], {})[row["reference"]] = row
    for task in TASKS:
        score = archive_by[task]["CATCH archive score"]
        label = archive_by[task]["CATCH archive label"]
        lines.append("| {} | {:.6f} | {:.6f} | `{}` | {:.6f} | {:.6f} | {:.6f} | `{}` |".format(
            task, score["roc_auc"], score["average_precision"], score["run_id"],
            label["affiliation_precision"], label["affiliation_recall"], label["affiliation_f"], label["run_id"]))
    lines += ["", "## 结果读取与边界", "",
              "共同协议使用相同测试端点和完整标签轴；没有按标签清洗训练数据，也没有内部再次划分训练行。Validation 仅用于 CATCH 早停，calibration 仅用于各自 0.995 阈值。CalIt2/Genesis 校准端点较少，分位阈值方差较高，仍按统一规则报告，不作特例调整。", "",
              "不能把历史 score 和 label 两个独立运行的结果合成一个历史 CATCH 运行。`archive_references.csv` 同时记录每组 score/label 的参数和压缩归档来源；当前归档中权重文件计数为零。共同协议 CATCH 是本次新训练，保存可重载 checkpoint、训练证据、归一化统计、分数和注册验证。", "",
              "Isolation Forest 参数固定为 100 trees、max_samples=min(256,n_train)、contamination=auto、seed=2021；PCA 只在原始训练行拟合，先按 95% 方差目标选维，再统一限制保留维数不超过 D−1，确保存在非空残差子空间，并按每行重建 MSE 排序。旧的全维重建 PCA 结果保留为历史退化结果；该规则不根据测试表现选维。它们均没有内部 holdout。", "",
              "逐点 AP/ROC 描述排序；点报警指标和有效段 affiliation 指标使用 calibration 阈值。原 CATCH score 归档还可能使用合并 train/test 能量及 anomaly_ratio 阈值，协议不同。数值并列/接近不能解释为论文协议复现。", "",
              "## 每组配置与失败状态", ""]
    for task in TASKS:
        record = next((item for item in task_records if item["dataset"] == task), {})
        config = catch_configs[task]["resolved_common_run_model_config"]
        lines.append("- **{}**：source `{}`；状态 `{}`；CATCH 配置 `{}`。校准端点数：{}。".format(
            task, SOURCE_RUNS[task], record.get("status", "blocked"),
            json.dumps(config, sort_keys=True, separators=(",", ":")), record.get("calibration_count", "未知")))
        if record.get("error"):
            lines.append("  失败原因：`{}`".format(record["error"].replace("`", "'")))
    lines += ["", "## 协议解释", "",
              "对比只涉及当前 Full 表示与共同切分下的 CATCH、Isolation Forest、PCA。若排名接近但报警差，首先体现独立校准尾部分位与 score 分布差异；若 IF/PCA 排名更强，提示点统计结构可能足以排序，但不等价于证明 MNRF 表示无效。该诊断不改 MNRF、不做分支消融，也不据此宣称 SOTA。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_all(args):
    batch_name = args.batch_name or "common-baseline-{}-{}".format(
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), os.getpid())
    analysis_root = ROOT / "result" / "analysis" / "MNRF-ViewProbe-Common" / batch_name
    score_root = ROOT / "result" / "score" / "MNRF-ViewProbe-Common" / batch_name
    existing = analysis_root.exists() or score_root.exists()
    if existing and not args.resume_batch:
        raise FileExistsError("batch output exists; choose a new batch name or pass --resume-batch")
    if bool(analysis_root.exists()) != bool(score_root.exists()):
        raise ValueError("only one side of this result batch exists; refusing mixed resume")
    analysis_root.mkdir(parents=True, exist_ok=args.resume_batch)
    score_root.mkdir(parents=True, exist_ok=args.resume_batch)
    prior_status_path = analysis_root / "task_status.json"
    prior_status = _load_json(prior_status_path) if args.resume_batch and prior_status_path.is_file() else None
    published_complete_batch = bool(
        prior_status and int(prior_status.get("completed_tasks", 0)) == len(TASKS)
        and (analysis_root / "common_protocol_metrics.csv").is_file()
        and (analysis_root / "README.md").is_file())
    catch_configs, archive_rows, archive_summary = _archive_references()
    archive_csv = analysis_root / "archive_references.csv"
    archive_summary_file = analysis_root / "archive_summary_source.txt"
    archive_config_file = analysis_root / "archived_score_configs.json"
    if args.resume_batch:
        if archive_csv.is_file() and archive_config_file.is_file() and archive_summary_file.is_file():
            saved_config = _load_json(archive_config_file)
            if saved_config != catch_configs:
                raise ValueError("archived configuration changed; refusing to resume this batch")
            if archive_summary_file.read_text(encoding="utf-8").strip() != "{} sha256={}".format(
                    archive_summary, _sha256(archive_summary)):
                raise ValueError("archived summary source changed; refusing to resume this batch")
        elif any(path.exists() for path in (archive_csv, archive_summary_file, archive_config_file)):
            raise ValueError("batch archive evidence is partial; refusing to overwrite it")
        else:
            _write_csv(archive_csv, archive_rows)
            archive_summary_file.write_text("{} sha256={}\n".format(archive_summary, _sha256(archive_summary)), encoding="utf-8")
            archive_config_file.write_text(json.dumps(catch_configs, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    else:
        _write_csv(archive_csv, archive_rows)
        archive_summary_file.write_text("{} sha256={}\n".format(archive_summary, _sha256(archive_summary)), encoding="utf-8")
        archive_config_file.write_text(json.dumps(catch_configs, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    method_rows = []
    task_records = []
    for task in TASKS:
        try:
            _task_pipeline(task, batch_name, analysis_root, score_root,
                           catch_configs, method_rows, task_records)
        except Exception as error:
            record = next((item for item in task_records if item["dataset"] == task), None)
            if record is None:
                record = {"dataset": task, "source_run_id": SOURCE_RUNS[task], "status": "failed", "methods": {}}
                task_records.append(record)
            record["status"] = "failed"
            record["error"] = "{}: {}".format(type(error).__name__, error)
            record["traceback"] = traceback.format_exc()
            _atomic_json(analysis_root / "{}.failure.json".format(task), record)
            print("{} failed: {}".format(task, record["error"]), flush=True)
    failed = any(item.get("status") != "complete" for item in task_records)
    if failed and published_complete_batch:
        # A failed continuation must not replace the last published complete
        # report. Keep this attempt separately inspectable and retryable.
        attempt = {"batch": batch_name, "status": "partial_resume_failed",
                   "task_records": task_records,
                   "environment": {"python": os.sys.version.split()[0], "torch": torch.__version__,
                                   "numpy": np.__version__, "pandas": pd.__version__,
                                   "scikit_learn": sklearn.__version__}}
        attempt_path = analysis_root / "resume-attempt-{}-{}.json".format(
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), os.getpid())
        _atomic_json(attempt_path, attempt)
        print("preserved_previous_complete_report=true")
        raise RuntimeError("resume attempt incomplete; previous complete batch report was preserved")
    else:
        _write_csv(analysis_root / "common_protocol_metrics.csv", method_rows)
        _atomic_json(analysis_root / "task_status.json", {"batch": batch_name,
                                                        "expected_tasks": list(TASKS),
                                                        "completed_tasks": sum(item.get("status") == "complete" for item in task_records),
                                                        "task_records": task_records,
                                                        "environment": {"python": os.sys.version.split()[0],
                                                                        "torch": torch.__version__,
                                                                        "numpy": np.__version__,
                                                                        "pandas": pd.__version__,
                                                                        "scikit_learn": sklearn.__version__},
                                                        "device": "cuda:0", "seed": SEED,
                                                        "calibration_quantile": CALIBRATION_QUANTILE})
        _write_readme(analysis_root / "README.md", batch_name, method_rows, archive_rows,
                      task_records, catch_configs, archive_summary)
    print("analysis_dir={}".format(analysis_root.resolve()))
    print("score_dir={}".format(score_root.resolve()))
    print("complete_tasks={}/{}".format(sum(item.get("status") == "complete" for item in task_records), len(TASKS)))
    return analysis_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-name", default="", help="optional unique result batch identifier")
    parser.add_argument("--resume-batch", action="store_true",
                        help="continue the explicitly named batch created by this adapter")
    args = parser.parse_args()
    try:
        output = run_all(args)
    except Exception as error:
        raise SystemExit("common baseline run failed: {}: {}".format(type(error).__name__, error))
    status = _load_json(output / "task_status.json")
    if int(status["completed_tasks"]) != len(TASKS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
