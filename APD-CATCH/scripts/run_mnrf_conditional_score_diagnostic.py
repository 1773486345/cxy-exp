#!/usr/bin/env python
"""Read-only, fixed conditional-density contrast over the eight Full sources."""

import csv
import ast
import hashlib
import io
import json
import math
import pickle
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular, cho_solve
from scipy.special import logsumexp
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mnrf_common_data import load_task_frame
from scripts.run_mnrf_common_baselines import SOURCE_RUNS, TASKS
from scripts.validate_mnrf_view_probe import validate as validate_source
from ts_benchmark.baselines.mnrf_view_probe.evaluation import affiliation_f_segment_macro
from ts_benchmark.baselines.mnrf_view_probe.probe import _gmm_score_values

SOURCE_ROOT = ROOT / "result" / "score" / "MNRF-ViewProbe"
Q = 0.995
CHUNK = 512
BRANCH_TARGET = np.asarray([0] + list(range(9, 17)), dtype=np.int64)  # S,P
BRANCH_CONTEXT = np.asarray(list(range(1, 9)) + list(range(17, 25)), dtype=np.int64)  # E,R
NYC_SOURCE_URL = (
    "https://raw.githubusercontent.com/chickenbestlover/"
    "Online-Recurrent-Extreme-Learning-Machine/master/data/nyc_taxi.csv"
)
CALIT2_UCI_URL = "https://archive.ics.uci.edu/static/public/156/calit2+building+people+counts.zip"
MSL_LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _logpdf_cholesky(values, means, chol):
    centered = values - means
    whitened = solve_triangular(chol, centered.T, lower=True, check_finite=False).T
    logdet = 2.0 * np.log(np.diag(chol)).sum()
    return -0.5 * (values.shape[1] * math.log(2.0 * math.pi) + logdet
                   + np.einsum("ij,ij->i", whitened, whitened))


def conditional_mixture_log_density(model, values, target_indices, condition_indices):
    """Evaluate log p(target | condition) as a posterior-weighted mixture.

    This directly forms each conditional Gaussian and posterior component
    weights. It never subtracts two large joint/marginal NLL values.
    """
    if model.covariance_type != "full":
        raise ValueError("conditional diagnostic requires the saved full-covariance GMM")
    x = np.asarray(values, dtype=np.float64)
    target = np.asarray(target_indices, dtype=np.int64)
    condition = np.asarray(condition_indices, dtype=np.int64)
    if (x.ndim != 2 or not len(target) or not len(condition)
            or len(np.unique(np.r_[target, condition])) != len(target) + len(condition)):
        raise ValueError("conditional density dimensions must be non-empty and disjoint")
    if x.shape[1] != model.means_.shape[1] or not np.isfinite(x).all():
        raise ValueError("conditional density input has wrong width or non-finite values")

    component_log_posterior = []
    component_conditional_logpdf = []
    for weight, mean, covariance in zip(model.weights_, model.means_, model.covariances_):
        mu_c = mean[condition]
        mu_d = mean[target]
        sigma_cc = covariance[np.ix_(condition, condition)]
        sigma_dc = covariance[np.ix_(target, condition)]
        sigma_cd = covariance[np.ix_(condition, target)]
        sigma_dd = covariance[np.ix_(target, target)]
        chol_c = cholesky(sigma_cc, lower=True, check_finite=False)
        beta = cho_solve((chol_c, True), sigma_cd, check_finite=False)
        conditional_cov = sigma_dd - sigma_dc.dot(beta)
        conditional_cov = (conditional_cov + conditional_cov.T) * 0.5
        chol_d_given_c = cholesky(conditional_cov, lower=True, check_finite=False)
        log_p_c = _logpdf_cholesky(x[:, condition], mu_c, chol_c)
        conditional_mean = mu_d + (x[:, condition] - mu_c).dot(beta)
        log_p_d_given_c = _logpdf_cholesky(x[:, target], conditional_mean, chol_d_given_c)
        component_log_posterior.append(math.log(float(weight)) + log_p_c)
        component_conditional_logpdf.append(log_p_d_given_c)
    log_posterior = np.stack(component_log_posterior, axis=1)
    log_posterior -= logsumexp(log_posterior, axis=1, keepdims=True)
    conditional = logsumexp(log_posterior + np.stack(component_conditional_logpdf, axis=1), axis=1)
    if not np.isfinite(conditional).all():
        raise FloatingPointError("conditional mixture produced non-finite log density")
    return conditional


def _raw_scores(run_dir, split, models, mean, scale, noise, dtype, target_dims=None):
    features = np.load(run_dir / "cache" / (split + "_features.npy"),
                       mmap_mode="r", allow_pickle=False)
    variables = features.shape[1]
    selected = np.arange(features.shape[2], dtype=np.int64)
    if target_dims is None:
        target_dims = BRANCH_TARGET
    nll = np.zeros((len(features), variables), dtype=np.float64)
    conditional = np.zeros_like(nll)
    with threadpool_limits(limits=1):
        for noise_vector in noise:
            for start in range(0, len(features), CHUNK):
                stop = min(len(features), start + CHUNK)
                for variable, model in enumerate(models):
                    values = _gmm_score_values(
                        features[start:stop, variable], mean[variable], scale[variable],
                        noise_vector, dtype,
                    )
                    # Accumulate each of four fixed full-layout noise draws in float64.
                    nll[start:stop, variable] += -model.score_samples(values) / len(noise)
                    conditional[start:stop, variable] += -conditional_mixture_log_density(
                        model, values, target_dims, BRANCH_CONTEXT
                    ) / len(noise)
    if not np.isfinite(nll).all() or not np.isfinite(conditional).all():
        raise FloatingPointError(split + " scoring produced non-finite NLL")
    return nll, conditional


def _score_metrics(scores, threshold, labels, point_indices, full_labels,
                   segment_ids, valid_full):
    prediction = (scores > float(threshold)).astype(np.int8)
    actual = labels.astype(np.int8, copy=False)
    auc = float(roc_auc_score(actual, scores)) if len(np.unique(actual)) == 2 else None
    ap = float(average_precision_score(actual, scores)) if len(np.unique(actual)) == 2 else None
    normal = actual == 0
    fp = int(np.sum(prediction[normal] == 1))
    tn = int(np.sum(prediction[normal] == 0))
    tp = int(np.sum((actual == 1) & (prediction == 1)))
    fn = int(np.sum((actual == 1) & (prediction == 0)))
    # Rebuild on the complete test axis so cold-start and segment boundaries remain explicit.
    full_score = np.full(len(full_labels), np.nan, dtype=np.float64)
    full_prediction = np.full(len(full_labels), -1, dtype=np.int8)
    local_positions = point_indices
    full_score[local_positions] = scores
    full_prediction[local_positions] = prediction
    affiliation = affiliation_f_segment_macro(
        full_labels, full_prediction, valid_full, segment_ids
    )
    fpr_at = {}
    if len(np.unique(actual)) == 2:
        fpr, tpr, _ = roc_curve(actual, scores, drop_intermediate=False)
        for budget in (0.001, 0.005, 0.01, 0.05):
            eligible = tpr[fpr <= budget]
            fpr_at[str(budget)] = None if not len(eligible) else float(np.max(eligible))
    else:
        fpr_at = {str(b): None for b in (0.001, 0.005, 0.01, 0.05)}
    return {
        "roc_auc": auc, "average_precision": ap,
        "precision": float(precision_score(actual, prediction, zero_division=0)),
        "recall": float(recall_score(actual, prediction, zero_division=0)),
        "f1": float(f1_score(actual, prediction, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall_at_fpr": fpr_at,
        "affiliation_f_valid_segment_macro": affiliation["affiliation_f_segment_macro"],
        "affiliation_defined_segments": affiliation["defined_segment_count"],
        "affiliation_total_segments": affiliation["total_segment_count"],
    }


def _field_inventory(task, run_dir, manifest):
    frame, label_column = load_task_frame(ROOT / "dataset" / "anomaly_detect" / "data" / (task + ".csv"))
    names = list(manifest["variable_names"])
    features = frame.drop(columns=[label_column])
    if list(features.columns) != names:
        raise ValueError(task + " loaded variables differ from frozen source order")
    arrays = [np.asarray(features[name]) for name in names]
    duplicate_of = {}
    for i, values in enumerate(arrays):
        for j in range(i):
            if np.array_equal(values, arrays[j]):
                duplicate_of[names[i]] = names[j]
                break
    fields = []
    for i, (name, values) in enumerate(zip(names, arrays)):
        finite = np.isfinite(values).all() if values.dtype.kind in "fiu" else False
        unique = int(pd.Series(values).nunique(dropna=False))
        constant = bool(unique == 1)
        record = {
            "name": str(name), "index": i, "dtype": str(values.dtype),
            "min": float(np.min(values)) if finite else None,
            "max": float(np.max(values)) if finite else None,
            "unique_count_full_axis": unique,
            "constant_full_axis": constant,
            "duplicate_of": duplicate_of.get(name),
            "responsibility": "observation_preserved_unknown",
            "evidence": "benchmark input field; no field-level exogenous-role evidence found",
        }
        if task == "NYC":
            if name == "channnel1":
                record.update(responsibility="target_observation", evidence="exact match to upstream passenger_count")
            elif name == "channnel2":
                record.update(responsibility="exogenous_condition", evidence="exact match to upstream timeofday")
            elif name == "channnel3":
                record.update(responsibility="exogenous_condition", evidence="exact match to upstream dayofweek")
        elif task == "MSL":
            record.update(
                responsibility="observation_preserved_mapping_unknown",
                evidence="local conversion renamed variables col_N; upstream defines channel value plus command context, but per-column mapping/raw arrays are unavailable locally",
            )
        elif constant:
            record.update(responsibility="constant_observed_field", evidence="full loaded axis exact cardinality=1; retained in scoring")
        elif name in duplicate_of:
            record.update(responsibility="duplicate_observed_field", evidence="exact duplicate over full loaded axis; retained in scoring")
        fields.append(record)
    return fields, features


def _nyc_row_time_evidence(frame, out_dir):
    raw = urllib.request.urlopen(NYC_SOURCE_URL, timeout=30).read()
    source = pd.read_csv(io.BytesIO(raw), skiprows=[1, 2])
    names = {"channnel1": "passenger_count", "channnel2": "timeofday", "channnel3": "dayofweek"}
    if len(source) != len(frame) or any(
        not np.array_equal(pd.to_numeric(frame[local]).to_numpy(),
                           pd.to_numeric(source[remote]).to_numpy())
        for local, remote in names.items()
    ):
        raise ValueError("NYC rows no longer match the specified upstream source")
    mapping = pd.DataFrame({
        "original_row": np.arange(len(source), dtype=np.int64),
        "timestamp": source["timestamp"].astype(str),
        "passenger_count": source["passenger_count"].to_numpy(),
        "timeofday": source["timeofday"].to_numpy(),
        "dayofweek": source["dayofweek"].to_numpy(),
    })
    mapping.to_csv(out_dir / "nyc_original_row_time.csv", index=False)
    return {"url": NYC_SOURCE_URL, "sha256": _sha256_bytes(raw), "rows": len(source),
            "local_column_matches": names, "match": "exact all rows"}


def _calit2_conversion_evidence(frame):
    raw_zip = urllib.request.urlopen(CALIT2_UCI_URL, timeout=30).read()
    archive = zipfile.ZipFile(io.BytesIO(raw_zip))
    raw = archive.read("CalIt2.data")
    original = pd.read_csv(io.BytesIO(raw), header=None, names=["flow", "date", "time", "count"])
    details = {}
    for flow in ("7", "9"):
        source_counts = original.loc[original["flow"] == int(flow), "count"].to_numpy()
        local_counts = pd.to_numeric(frame[flow]).to_numpy()
        details[flow] = {
            "count": len(source_counts), "raw_min": int(source_counts.min()),
            "raw_max": int(source_counts.max()), "raw_unique_count": int(np.unique(source_counts).size),
            "local_min": int(local_counts.min()), "local_max": int(local_counts.max()),
            "local_unique_count": int(np.unique(local_counts).size),
            "exact_mod10_match": bool(np.array_equal(source_counts % 10, local_counts)),
            "exact_raw_match": bool(np.array_equal(source_counts, local_counts)),
        }
    return {"url": CALIT2_UCI_URL, "archive_sha256": _sha256_bytes(raw_zip),
            "data_sha256": _sha256_bytes(raw), "raw_rows": len(original),
            "conversion": "local flow counts equal original integer counts modulo 10",
            "flows": details}


def _msl_sequence_evidence(out_dir, frozen_test_labels):
    raw = urllib.request.urlopen(MSL_LABELS_URL, timeout=30).read()
    table = pd.read_csv(io.BytesIO(raw))
    table = table.loc[table["spacecraft"] == "MSL"].sort_values("chan_id", kind="stable")
    sequences = []
    rebuilt_labels = []
    cursor = 0
    for row in table.itertuples(index=False):
        length = int(row.num_values)
        sequence_labels = np.zeros(length, dtype=np.int8)
        for first, last in ast.literal_eval(row.anomaly_sequences):
            first, last = int(first), int(last)
            if first < 0 or last < first or last >= length:
                raise ValueError("MSL source anomaly interval is outside its channel test sequence")
            sequence_labels[first:last + 1] = 1
        sequences.append({"channel_id": row.chan_id, "test_length": length,
                          "test_local_start": cursor, "test_local_stop": cursor + length,
                          "test_global_start": 58317 + cursor,
                          "test_global_stop": 58317 + cursor + length,
                          "anomaly_points": int(sequence_labels.sum()),
                          "train_length": None,
                          "train_mapping_status": "not recoverable from labels file; original per-channel train arrays absent locally"})
        rebuilt_labels.append(sequence_labels)
        cursor += length
    if len(sequences) != 27 or cursor != 73729:
        raise ValueError("Telemanom MSL channel lengths do not match the frozen 73,729-point test axis")
    expected = np.concatenate(rebuilt_labels)
    if not np.array_equal(expected, np.asarray(frozen_test_labels, dtype=np.int8)):
        raise ValueError("sorted official Telemanom labels do not match the frozen MSL test label axis")
    pd.DataFrame(sequences).to_csv(out_dir / "msl_test_sequence_map.csv", index=False)
    return {"url": MSL_LABELS_URL, "sha256": _sha256_bytes(raw),
            "test_sequences": len(sequences), "test_length_sum": cursor,
            "sort": "channel id lexical", "label_axis_match": "previously established and rechecked by exact source/test label identity",
            "train_side": "per-channel train lengths unavailable; global train_length=58,317 is verified, channel-wise mapping unknown",
            "sequences": sequences}


def _msl_boundary_sensitivity(out_dir, sequence_evidence):
    task_dir = ROOT / "result" / "score" / "MNRF-ViewProbe-Common" / "common-baseline-20261029T-newrun" / "MSL"
    source_dir = SOURCE_ROOT / "MSL" / SOURCE_RUNS["MSL"]
    with np.load(source_dir / "analysis_inputs.npz", allow_pickle=False) as archive:
        full_labels = np.asarray(archive["test_labels"], dtype=np.int8)
        segment_ids = np.full(len(full_labels), -1, dtype=np.int32)
    allowed = np.zeros(len(full_labels), dtype=bool)
    for segment_id, sequence in enumerate(sequence_evidence["sequences"]):
        start, stop = int(sequence["test_local_start"]), int(sequence["test_local_stop"])
        segment_ids[start:stop] = segment_id
        # A 192-point window must lie entirely within one original test channel.
        allowed[min(stop, start + 191):stop] = True
    paths = {
        "MNRF-Full": source_dir / "results" / "aligned_scores.npz",
        "CATCH-common": task_dir / "CATCH-common" / "results" / "aligned_scores.npz",
        "IsolationForest": task_dir / "IsolationForest" / "results" / "aligned_scores.npz",
        "PCA": task_dir / "PCA" / "results" / "aligned_scores.npz",
    }
    methods = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError("MSL boundary sensitivity input missing: " + str(path))
        with np.load(path, allow_pickle=False) as archive:
            methods[name] = {key: np.asarray(archive[key]) for key in
                             ("scores", "predictions", "time_indices", "time_values_ns", "actual", "threshold")}
    coordinate_sets = [value["time_indices"] for value in methods.values()]
    coordinates = coordinate_sets[0]
    for name, value in methods.items():
        if not np.array_equal(value["time_indices"], coordinates):
            raise ValueError("MSL baseline coordinates differ before sequence-boundary filtering: " + name)
    local = coordinates - 58317
    if np.any(local < 0) or np.any(local >= len(full_labels)):
        raise ValueError("MSL baseline coordinate lies outside the frozen test axis")
    for name, value in methods.items():
        if (not np.array_equal(value["actual"].astype(np.int8), full_labels[local])
                or not np.array_equal(value["time_values_ns"], np.load(
                    source_dir / "analysis_inputs.npz", allow_pickle=False)["test_time_values_ns"][local])):
            raise ValueError("MSL baseline labels/time do not match frozen Full axis: " + name)
    keep = allowed[local]
    common_local = local[keep]
    if len(common_local) == 0:
        raise ValueError("MSL common sequence-interior score set is empty")
    valid = np.zeros(len(full_labels), dtype=bool)
    valid[common_local] = True
    rows = []
    for name, value in methods.items():
        scores = np.asarray(value["scores"], dtype=np.float64)[keep]
        threshold = float(np.asarray(value["threshold"]))
        predictions = (scores > threshold).astype(np.int8)
        labels = full_labels[common_local]
        metrics = _score_metrics(scores, threshold, labels, common_local, full_labels,
                                 segment_ids, valid)
        rows.append({"dataset": "MSL", "method": name,
                     "boundary_rule": "192-row windows confined to official per-channel test sequence",
                     "points_before": len(local), "points_after": int(keep.sum()),
                     "excluded_cross_sequence_or_cold_start": int(len(local) - keep.sum()),
                     "threshold_source": "method's saved independent calibration threshold",
                     **metrics})
    _save_csv(out_dir / "msl_boundary_sensitivity.csv", rows)
    return {"status": "complete", "common_points": int(keep.sum()),
            "excluded_points": int(len(local) - keep.sum()),
            "methods": {row["method"]: row for row in rows},
            "interpretation": "test-only sensitivity analysis; frozen encoder and head were trained on concatenated axis"}


def _save_csv(path, rows):
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_run(task, out_dir, checked_runs):
    run_dir = (SOURCE_ROOT / task / SOURCE_RUNS[task]).resolve()
    if task not in checked_runs:
        validation = validate_source(run_dir)
        if validation.get("status") != "complete":
            raise ValueError(task + " Full source native validation is not complete")
        checked_runs.add(task)
    manifest = json.loads((run_dir / "probe_manifest.json").read_text())
    if manifest.get("head_type") != "gmm" or manifest.get("config", {}).get("gmm_covariance_type") != "full":
        raise ValueError(task + " source is not the fixed full-covariance Full GMM")
    fields, frame = _field_inventory(task, run_dir, manifest)
    with open(run_dir / "heads" / "gmm_models.pkl", "rb") as handle:
        models = pickle.load(handle)
    if len(models) != len(manifest["variable_names"]):
        raise ValueError(task + " GMM variable count differs from source manifest")
    for i, model in enumerate(models):
        if model.covariance_type != "full" or model.means_.shape != (4, 25):
            raise ValueError("{} variable {} has incompatible saved GMM".format(task, i))
    with np.load(run_dir / "heads" / "feature_statistics.npz", allow_pickle=False) as stats:
        feature_mean = np.asarray(stats["mean"], dtype=np.float32)
        feature_scale = np.asarray(stats["scale"], dtype=np.float32)
    noise = np.load(run_dir / "heads" / "fixed_noise.npy", allow_pickle=False)
    if feature_mean.shape != (len(models), 25) or feature_scale.shape != feature_mean.shape:
        raise ValueError(task + " frozen feature statistic shape mismatch")
    source_config = manifest["config"]
    compute_dtype = "float64"
    score_floor = float(source_config["score_mad_floor"])
    q = float(source_config["calibration_quantile"])
    if q != Q:
        raise ValueError(task + " saved calibration q does not equal fixed 0.995")

    with np.load(run_dir / "analysis_inputs.npz", allow_pickle=False) as evidence:
        full_test_labels = np.asarray(evidence["test_labels"], dtype=np.int8)
        full_test_times = np.asarray(evidence["test_time_values_ns"], dtype=np.int64)
        full_test_global = np.asarray(evidence["test_global_indices"], dtype=np.int64)
        full_test_segments = np.asarray(evidence["test_segment_ids"], dtype=np.int32)
    test_endpoints = np.load(run_dir / "cache" / "test_window_end_indices.npy", allow_pickle=False)
    validation_endpoints = np.load(run_dir / "cache" / "validation_window_end_indices.npy", allow_pickle=False)
    calibration_endpoints = np.load(run_dir / "cache" / "calibration_window_end_indices.npy", allow_pickle=False)
    with np.load(run_dir / "results" / "aligned_scores.npz", allow_pickle=False) as aligned:
        saved_time_indices = np.asarray(aligned["time_indices"], dtype=np.int64)
        saved_time_values = np.asarray(aligned["time_values_ns"], dtype=np.int64)
        saved_valid_mask = np.asarray(aligned["valid_mask"], dtype=bool)
        saved_scores = np.asarray(aligned["scores"], dtype=np.float64)
        saved_predictions = np.asarray(aligned["predictions"], dtype=np.int64)
        saved_actual = np.asarray(aligned["actual"], dtype=np.int8)
        original_threshold = float(np.asarray(aligned["threshold"]))
    if not np.array_equal(saved_predictions, (saved_scores > original_threshold).astype(np.int64)):
        raise ValueError(task + " saved predictions do not match strict score > saved threshold")
    train_length = int(manifest.get("train_length", manifest["source_identity"]["train_length"]))
    local_test = test_endpoints - train_length
    if (not np.array_equal(test_endpoints, saved_time_indices)
            or np.any(local_test < 0) or np.any(local_test >= len(full_test_labels))
            or not np.array_equal(full_test_global[local_test], test_endpoints)
            or not np.array_equal(full_test_times[local_test], saved_time_values)
            or not np.array_equal(full_test_labels[local_test], saved_actual)
            or len(saved_valid_mask) != len(full_test_labels)
            or not np.array_equal(np.flatnonzero(saved_valid_mask), local_test)):
        raise ValueError(task + " full-axis labels/time/mask do not match saved score coordinates")

    started = time.perf_counter()
    validation_joint, validation_conditional = _raw_scores(
        run_dir, "validation", models, feature_mean, feature_scale, noise, compute_dtype
    )
    calibration_joint, calibration_conditional = _raw_scores(
        run_dir, "calibration", models, feature_mean, feature_scale, noise, compute_dtype
    )
    test_joint, test_conditional = _raw_scores(
        run_dir, "test", models, feature_mean, feature_scale, noise, compute_dtype
    )
    scoring_seconds = time.perf_counter() - started
    saved_test_nll = np.load(run_dir / "results" / "test_nll.npy", mmap_mode="r", allow_pickle=False)
    saved_validation_nll = np.load(run_dir / "heads" / "validation_nll.npy", mmap_mode="r", allow_pickle=False)
    saved_calibration_nll = np.load(run_dir / "heads" / "calibration_nll.npy", mmap_mode="r", allow_pickle=False)
    if (saved_test_nll.shape != test_joint.shape
            or saved_validation_nll.shape != validation_joint.shape
            or saved_calibration_nll.shape != calibration_joint.shape
            or len(validation_endpoints) != len(validation_joint)
            or len(calibration_endpoints) != len(calibration_joint)):
        raise ValueError(task + " NLL row counts do not match frozen validation/calibration/test endpoints")
    replay = {
        "test_max_abs": float(np.max(np.abs(test_joint - saved_test_nll))),
        "validation_max_abs": float(np.max(np.abs(validation_joint - saved_validation_nll))),
        "calibration_max_abs": float(np.max(np.abs(calibration_joint - saved_calibration_nll))),
    }
    replay["test_max_rel"] = float(np.max(np.abs(test_joint - saved_test_nll)
                                          / np.maximum(np.abs(saved_test_nll), 1.0)))
    if not (np.allclose(test_joint, saved_test_nll, rtol=2e-5, atol=2e-5)
            and np.allclose(validation_joint, saved_validation_nll, rtol=2e-5, atol=2e-5)
            and np.allclose(calibration_joint, saved_calibration_nll, rtol=2e-5, atol=2e-5)):
        raise ValueError(task + " float64 joint GMM replay differs from saved NLL beyond rtol=2e-5, atol=2e-5")
    # Use serialized role definitions only for evidenced exogenous columns.
    target_variables = np.asarray([
        item["index"] for item in fields if item["responsibility"] != "exogenous_condition"
    ], dtype=np.int64)
    if not len(target_variables):
        raise ValueError(task + " field responsibility map has no scored observation variables")
    rows = []
    original_metrics = _score_metrics(
        saved_scores, original_threshold, saved_actual, local_test,
        full_test_labels, full_test_segments, saved_valid_mask,
    )
    rows.append({"dataset": task, "method": "original_joint_saved", "status": "verified_saved",
                 "source_run_id": SOURCE_RUNS[task], "target_variable_count": len(models),
                 "threshold": original_threshold, "scoring_seconds": 0.0,
                 "score_compute_dtype": compute_dtype,
                 **replay, **original_metrics})
    methods = (
        ("responsibility_joint", validation_joint, calibration_joint, test_joint),
        ("responsibility_conditional_S_P_given_E_R", validation_conditional,
         calibration_conditional, test_conditional),
    )
    for method, val_nll, cal_nll, test_nll in methods:
        val = val_nll[:, target_variables]
        cal = cal_nll[:, target_variables]
        tst = test_nll[:, target_variables]
        center = np.median(val, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(val - center), axis=0), score_floor)
        calibration_score = np.max((cal - center) / scale, axis=1)
        threshold = float(np.quantile(calibration_score, q, method="linear"))
        test_score = np.max((tst - center) / scale, axis=1)
        metrics = _score_metrics(test_score, threshold, saved_actual, local_test,
                                 full_test_labels, full_test_segments, saved_valid_mask)
        rows.append({"dataset": task, "method": method, "status": "complete",
                     "source_run_id": SOURCE_RUNS[task],
                     "target_variable_count": len(target_variables),
                     "target_variables": json.dumps([manifest["variable_names"][i]
                                                      for i in target_variables]),
                     "threshold": threshold, "scoring_seconds": scoring_seconds,
                     "score_compute_dtype": compute_dtype,
                     "calibration_quantile": q, "center_rule": "validation median",
                     "scale_rule": "max(1.4826*MAD, score_mad_floor)", **replay, **metrics})
    if task == "NYC":
        if len(target_variables) != 1 or len(manifest["variable_names"]) != 3:
            raise ValueError("NYC role mapping must score one observed field and retain two exogenous fields as context")
    return rows, fields, replay, {"target_variables": [manifest["variable_names"][i]
                                                         for i in target_variables],
                                 "source_manifest_sha256": _sha256_bytes((run_dir / "probe_manifest.json").read_bytes()),
                                 "feature_fingerprint": manifest.get("feature_fingerprint"),
                                 "source_identity": manifest.get("source_identity"),
                                 "scoring_seconds": scoring_seconds}


def run(output_dir=None):
    batch_id = datetime.now(timezone.utc).strftime("conditional-score-%Y%m%dT%H%M%SZ")
    output_dir = Path(output_dir) if output_dir else ROOT / "result" / "analysis" / "MNRF-ViewProbe-ConditionalScore" / batch_id
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    checked = set()
    all_rows, field_rows, source_records = [], [], {}
    for task in TASKS:
        print("[{}] validating source and replaying frozen GMM scores".format(task), flush=True)
        try:
            run_dir = SOURCE_ROOT / task / SOURCE_RUNS[task]
            pre_evidence = {}
            if task == "NYC":
                frame, _ = load_task_frame(ROOT / "dataset" / "anomaly_detect" / "data" / "NYC.csv")
                pre_evidence["upstream_row_time"] = _nyc_row_time_evidence(frame, output_dir)
            elif task == "CalIt2":
                frame, _ = load_task_frame(ROOT / "dataset" / "anomaly_detect" / "data" / "CalIt2.csv")
                pre_evidence["raw_count_conversion"] = _calit2_conversion_evidence(frame)
            elif task == "MSL":
                with np.load(run_dir / "analysis_inputs.npz", allow_pickle=False) as archive:
                    labels = np.asarray(archive["test_labels"], dtype=np.int8)
                pre_evidence["test_sequence_evidence"] = _msl_sequence_evidence(output_dir, labels)
            rows, fields, replay, record = _dataset_run(task, output_dir, checked)
            all_rows.extend(rows)
            field_rows.extend({"dataset": task, **field} for field in fields)
            source_records[task] = {"status": "complete", "score_replay": replay,
                                    **record, **pre_evidence}
            if task == "MSL":
                source_records[task]["boundary_sensitivity"] = _msl_boundary_sensitivity(
                    output_dir, pre_evidence["test_sequence_evidence"]
                )
        except Exception as error:
            source_records[task] = {"status": "failed", "reason": "{}: {}".format(type(error).__name__, error)}
            print("[{}] FAILED: {}".format(task, error), flush=True)
    _save_csv(output_dir / "conditional_scores.csv", all_rows)
    _save_csv(output_dir / "field_responsibilities.csv", field_rows)
    (output_dir / "source_evidence.json").write_text(
        json.dumps(source_records, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    complete = sum(1 for task in TASKS if source_records.get(task, {}).get("status") == "complete")
    readme = [
        "# Frozen Full conditional-score diagnostic", "",
        "This is a fixed post-hoc CPU diagnostic using the eight registered Full sources. It does not fit a GMM, encode, retrain, change feature/cache identities, or tune on test labels.", "",
        "## Scoring", "",
        "The original Full rows reuse saved scores and thresholds after native source validation. The two diagnostic rows replay saved GMMs against float32-normalized features plus the original four fixed noise vectors, widening to float64 only at likelihood evaluation. The replay is checked against saved validation/calibration/test per-variable NLL (`rtol=2e-5`, `atol=2e-5`).", "",
        "For the responsibility-aware rows, raw variables are scored only when role is evidenced as an observation; NYC's time-of-day and day-of-week columns are explicitly exogenous. Unknown MSL column mappings and other unknown responsibilities remain scored. Conditional density is computed directly as a posterior-weighted conditional Gaussian mixture: `p(S,P | E,R)`, with S/P dimensions `[0,1) U [9,17)` and E/R context dimensions `[1,9) U [17,25)`. No subtraction of two large NLLs is used.", "",
        "Each diagnostic method gets its own validation median/MAD and calibration 0.995 linear quantile; variable aggregation remains max and prediction uses strict `score > threshold`. Metrics include AUC/AP, point metrics, recall at fixed FPR budgets, and `affiliation_f_valid_segment_macro`; this segment macro is not the paper's original Affiliation-F.", "",
        "## Field responsibilities", "",
        "| Dataset | Field-role evidence | Scoring responsibility in this diagnostic | Sequence/entity evidence |",
        "|---|---|---|---|",
        "| PSM | The source is the pooled server-metrics benchmark (25 channels), but this frozen conversion names channels `feature_N`; a channel dictionary is not present. | All 25 retained as observations; individual system metric meanings unknown. | Timestamped aggregate series; the benchmark does not identify host/application entities in this frozen source. |",
        "| MSL | Telemanom defines 27 telemetry channels with command context, but this run has 55 `col_N` inputs and no local column-to-source map or raw channel arrays. | All 55 retained as observations; mapping and exogenous/telemetry split unknown. | Official 27 test sequence ranges reproduced in sorted channel order; corresponding train per-channel lengths unavailable, so encoder training-boundary sensitivity remains unresolved. |",
        "| GECCO | Named inputs include water-quality measurements (for example pH, Redox, conductivity/turbidity-related fields); the local source has no independent exogenous covariate mapping. | All 9 retained as observations; physical channels are not discarded based on names. | Environmental sensor stream; frozen source has no separate entity/session key. |",
        "| Genesis | Local names identify `MotorData.*` process measurements and discrete status-like channels; no source dictionary proves any status field is an external condition. | All 18 retained as observations, including low-cardinality status channels. | Process sequence identity beyond the frozen continuous axis is unavailable. |",
        "| NYC | Every row matches the upstream three-column file: passenger_count is passenger flow; timeofday and dayofweek encode time context. | passenger_count is the target observation; the two evidenced time fields are conditions, retained as context but excluded from role-aware target aggregation. | Half-hour timestamps map row-for-row; no entity ID. |",
        "| CalIt2 | UCI documents flow ID 7 as outflow and 9 as inflow, with half-hour count aggregates. Original raw counts range 0–62; local values equal `count % 10`. | Both lossy-converted streams remain target observations; no exogenous channel is inferred from low cardinality. | Real half-hour time index; two flow directions are separate variables, not entity IDs. |",
        "| Creditcard | ULB schema is transactions with PCA-anonymized V1–V28 and Amount; the original Time column is absent from these 29 model inputs. | All 29 retained as transaction observations; PCA components are not interpretable field-by-field. | Transaction row order is not a per-card sequence: card/customer IDs are absent. |",
        "| CICIDS | Inputs are flow-statistic fields; the frozen conversion drops identifiers/time/session keys. Full-axis inventory finds 10 constant and 4 exact-duplicate channels. | All 72 retained, including constants and duplicates; they are integrity/representation findings, not grounds for silent channel removal. | Flow records do not establish a continuous physical time series or entity boundaries. |", "",
        "The field-level `field_responsibilities.csv` records full-axis range, cardinality, constant status, exact duplicate identity, and whether a role is evidenced. Sources: [PSM/Time-Series-Library](https://github.com/thuml/Time-Series-Library), [Telemanom](https://github.com/khundman/telemanom), [GECCO water-quality channel descriptions](https://www.mdpi.com/2073-4441/18/6/726), [UCI CalIt2](https://archive.ics.uci.edu/dataset/156/calit2), [ULB credit-card schema](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), and [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html). NYC is directly matched to the upstream file supplied in the task. For PSM, Genesis, Creditcard and CICIDS, exact frozen CSV provenance/transform lineage beyond the local loader is incomplete; semantic scope is not claimed as a per-column ground truth. Low cardinality alone never makes a field exogenous. No constant or duplicate field was removed.", "",
        "The current CalIt2 representation is lossy modulo-10 bucketing; no score-only change can restore count magnitude. The conditional comparison diagnoses the frozen representation, not a corrected raw-count model.", "",
        "## Sequence limits", "",
        "The MSL report gives 27 test sequence boundaries and common-coordinate rescoring after excluding windows whose 192-row receptive field crosses a test sequence boundary. This is only a test boundary sensitivity check: the frozen encoder/heads were trained under the original concatenated axis, so it is not an independently retrained segmented experiment. Per-channel training sequence lengths are not present in the local evidence, so no training-side boundary correction is claimed. NYC's `nyc_original_row_time.csv` preserves row-to-source-time mapping. Creditcard and CICIDS lack entity IDs in the frozen source, so within-entity continuity cannot be reconstructed; especially CICIDS consists of flow records, for which row adjacency is not established as a physical time series.", "",
        "## Source evidence", "",
        "See `source_evidence.json`, `field_responsibilities.csv`, and the MSL/NYC evidence CSVs. A failed dataset remains explicitly failed and is not silently omitted.", "",
        "Completed datasets: {}/8. No model weights, feature caches, or per-window NLL matrices are copied here.".format(complete),
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (output_dir / "status.json").write_text(json.dumps({"status": "complete" if complete == 8 else "partial",
                                                         "datasets_complete": complete,
                                                         "datasets_expected": 8,
                                                         "rows_written": len(all_rows)},
                                                        indent=2) + "\n")
    print("output_dir={}".format(output_dir))
    print("datasets_complete={}/8".format(complete))
    return 0 if complete == 8 else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.output_dir))
