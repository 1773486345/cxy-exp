#!/usr/bin/env python
"""Fail the one-task shell command when benchmark artifacts are incomplete."""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import (
    ARCHITECTURE_VERSION, TIME_PROTOCOL_VERSION, _hash_array,
)
from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
from ts_benchmark.baselines.mnrf_lite_v2.layout import FeatureLayout, require_layout


def validate_artifact_shapes(layout, variable_count, cache_shapes, index_shapes,
                             expected_counts, input_mean_shape, input_scale_shape,
                             feature_mean_shape, feature_scale_shape,
                             calibration_center_shape, calibration_scale_shape,
                             noise_shape, context):
    """Validate shape-only contracts, also used directly by focused tests."""
    d, f = variable_count, layout.total_dim
    for name, shape in cache_shapes.items():
        expected = (expected_counts[name], d, f)
        if tuple(shape) != expected:
            raise ValueError(f"{context} cache {name} shape mismatch: expected {expected}, got {tuple(shape)}")
        if tuple(index_shapes[name]) != (expected_counts[name],):
            raise ValueError(
                f"{context} window indices {name} shape mismatch: expected {(expected_counts[name],)}, got {tuple(index_shapes[name])}"
            )
    expected_d = (d,)
    for name, actual in (("input mean", input_mean_shape), ("input scale", input_scale_shape),
                         ("calibration_center", calibration_center_shape),
                         ("calibration_scale", calibration_scale_shape)):
        if tuple(actual) != expected_d:
            raise ValueError(f"{context} {name} shape mismatch: expected {expected_d}, got {tuple(actual)}")
    expected_feature = (d, f)
    for name, actual in (("feature_mean", feature_mean_shape), ("feature_scale", feature_scale_shape)):
        if tuple(actual) != expected_feature:
            raise ValueError(f"{context} {name} shape mismatch: expected {expected_feature}, got {tuple(actual)}")
    expected_noise = (4, f)
    if tuple(noise_shape) != expected_noise:
        raise ValueError(f"{context} fixed scoring noise shape mismatch: expected {expected_noise}, got {tuple(noise_shape)}")


def _config_layout(config, context):
    if not isinstance(config, dict):
        raise ValueError(f"{context}: saved config is missing or not an object")
    try:
        return MNRFConfig.from_kwargs(**config).layout
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: invalid saved MNRF configuration: {error}") from error


def _array_equal_at_path(left, right, left_name, right_name):
    if not np.array_equal(np.asarray(left), np.asarray(right)):
        raise ValueError(f"{left_name} differs from {right_name}")


def main(run_path: str, task: str) -> None:
    root = Path(run_path)
    required = [
        root / "run_identity.json", root / "benchmark_context.json", root / "run_metadata.json",
        root / "cache_manifest.json", root / "input_statistics.npz", root / "deployment.pt",
        root / "normalization_state.npz", root / "feature_statistics.npz", root / "stage2_binding.json",
        root / "checkpoints" / "best_encoder.pt", root / "checkpoints" / "best_fusion.pt",
        root / "checkpoints" / "best_stage1.pt",
        root / "checkpoints" / "stage1_latest.pt", root / "checkpoints" / "stage2_latest.pt",
        root / "logs" / "training.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit(f"Missing required run artifacts: {missing}")
    identity = json.loads((root / "run_identity.json").read_text(encoding="utf-8"))
    context = json.loads((root / "benchmark_context.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "cache_manifest.json").read_text(encoding="utf-8"))
    identity_config_layout = _config_layout(identity.get("config"), str(root / "run_identity.json"))
    metadata_config_layout = _config_layout(metadata.get("config"), str(root / "run_metadata.json"))
    try:
        layout = require_layout(identity.get("feature_layout"), identity_config_layout,
                               f"{root / 'run_identity.json'} feature_layout")
        require_layout(metadata.get("feature_layout"), layout,
                       f"{root / 'run_metadata.json'} feature_layout")
        require_layout(manifest.get("feature_layout"), layout,
                       f"{root / 'cache_manifest.json'} feature_layout")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if metadata_config_layout.to_dict() != layout.to_dict():
        raise SystemExit(f"{root / 'run_metadata.json'} config layout differs from run identity")
    architecture_config = manifest.get("architecture_config")
    if not isinstance(architecture_config, dict):
        raise SystemExit(f"{root / 'cache_manifest.json'} architecture_config is missing")
    for key, expected in (("hidden_dim", 32), ("evolution_dim", layout.evolution_dim),
                          ("pattern_dim", layout.pattern_dim), ("relation_dim", layout.relation_dim)):
        actual = architecture_config.get(key)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise SystemExit(
                f"{root / 'cache_manifest.json'} architecture_config.{key} mismatch: expected {expected}, got {actual}"
            )
    binding_path = root / "stage2_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    try:
        require_layout(binding.get("feature_layout"), layout, f"{binding_path} feature_layout")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if context.get("task") != f"{task}.csv" or context.get("run_id") != root.name:
        raise SystemExit(f"Task/run identity mismatch in {root / 'benchmark_context.json'}")
    if context.get("strategy_name") != "mnrf_valid_detect" or metadata.get("config", {}).get("task_name") != f"{task}.csv":
        raise SystemExit("Strategy or model-config task does not match this run")
    if identity.get("architecture_version") != ARCHITECTURE_VERSION:
        raise SystemExit("run architecture version is missing or incompatible")
    if identity.get("temporal_rule", {}).get("version") != TIME_PROTOCOL_VERSION:
        raise SystemExit("run frozen time protocol is missing or incompatible")
    if (manifest.get("status") != "complete"
            or manifest.get("architecture_version") != ARCHITECTURE_VERSION
            or manifest.get("temporal_rule") != identity.get("temporal_rule")
            or manifest.get("encoder_fingerprint") != metadata.get("feature_fingerprint")):
        raise SystemExit("feature cache manifest is incomplete or fingerprint-mismatched")
    for phase in ("stage1", "stage2"):
        if metadata.get("phase_states", {}).get(phase, {}).get("status") != "completed":
            raise SystemExit(f"training phase {phase} is not marked complete")
    cache_shapes = {}
    index_shapes = {}
    actual_cache_hashes = {}
    actual_index_hashes = {}
    cache_paths = metadata.get("feature_cache_paths", {})
    if set(cache_paths) != {"train", "validation", "calibration"}:
        raise SystemExit(f"Unexpected feature cache set in {root / 'run_metadata.json'}: {sorted(cache_paths)}")
    expected_counts = metadata.get("feature_cache_window_counts", {})
    variable_count = len(metadata.get("variable_names", []))
    if variable_count < 2 or metadata.get("variable_names") != manifest.get("variable_names"):
        raise SystemExit("Run metadata and cache manifest variable lists are missing or inconsistent")
    for name, path in cache_paths.items():
        index_path = Path(metadata.get("feature_cache_window_end_indices", {}).get(name, ""))
        if not Path(path).is_file() or not index_path.is_file():
            raise SystemExit(f"Missing cache or original-index file for {name}")
        expected_cache_sha = manifest.get("feature_cache_sha256", {}).get(name)
        actual_cache_hashes[name] = _sha256_file(path)
        if expected_cache_sha is None or actual_cache_hashes[name] != expected_cache_sha:
            raise SystemExit(f"Feature cache missing or fingerprint mismatch for {name}")
        cache = np.load(path, mmap_mode="r")
        indices = np.load(index_path, mmap_mode="r")
        if len(cache) != expected_counts.get(name) or len(indices) != len(cache):
            raise SystemExit(f"Cache/index length mismatch for {name}")
        expected_cache_shape = (expected_counts[name], variable_count, layout.total_dim)
        if cache.shape != expected_cache_shape:
            raise SystemExit(f"{path} shape mismatch: expected {expected_cache_shape}, got {cache.shape}")
        if indices.shape != (expected_counts[name],):
            raise SystemExit(f"{index_path} shape mismatch: expected {(expected_counts[name],)}, got {indices.shape}")
        cache_shapes[name] = cache.shape
        index_shapes[name] = indices.shape
        if not _finite_memmap(cache):
            raise SystemExit(f"Non-finite cached feature values in {path}")
        previous = None
        for start in range(0, len(indices), 65536):
            chunk = np.asarray(indices[start:start + 65536])
            if (previous is not None and len(chunk) and chunk[0] <= previous) or np.any(np.diff(chunk) <= 0):
                raise SystemExit(f"Invalid or non-increasing window indices in {index_path}")
            if len(chunk):
                previous = chunk[-1]
        expected_sha = manifest.get("window_end_index_sha256", {}).get(name)
        if expected_sha is None:
            raise SystemExit(f"Missing cache index fingerprint for {name}")
        digest = __import__("hashlib").sha256(Path(index_path).read_bytes()).hexdigest()
        actual_index_hashes[name] = digest
        if digest != expected_sha:
            raise SystemExit(f"Cache index fingerprint mismatch for {name}")
        split = metadata.get("split_ranges", {}).get(name)
        if split is None:
            raise SystemExit(f"Missing split bounds for cache {name}")
        from ts_benchmark.baselines.mnrf_lite_v2.windows import window_endpoints, intersect_segments
        expected_indices = window_endpoints(
            intersect_segments(manifest["continuous_segments"], split["start"], split["end_exclusive"]),
            int(metadata["config"]["seq_len"]),
        )
        if not np.array_equal(indices, expected_indices):
            raise SystemExit(f"Cache positions do not match continuous split windows for {name}")
    try:
        import torch
        deployment = torch.load(root / "deployment.pt", map_location="cpu", weights_only=False)
    except Exception as error:
        raise SystemExit(f"Deployment package cannot be loaded: {error}")
    with np.load(root / "input_statistics.npz", allow_pickle=False) as stats:
        for key in ("mean", "scale", "feature_mean", "feature_scale", "calibration_center", "calibration_scale", "threshold", "variable_names"):
            if key not in stats.files:
                raise SystemExit(f"Missing {key} in input_statistics.npz")
        if "noise_vectors" not in stats.files:
            raise SystemExit("Missing noise_vectors in input_statistics.npz")
        if not np.isfinite(stats["threshold"]).all() or not np.isfinite(stats["calibration_center"]).all():
            raise SystemExit("Non-finite calibration artifact")
        if (not np.isfinite(stats["mean"]).all() or not np.isfinite(stats["scale"]).all()
                or not np.isfinite(stats["feature_mean"]).all() or not np.isfinite(stats["feature_scale"]).all()
                or not np.isfinite(stats["calibration_scale"]).all() or not np.isfinite(stats["noise_vectors"]).all()):
            raise SystemExit("Non-finite normalization, feature, calibration, or noise artifact")
        if np.any(stats["scale"] <= 0) or np.any(stats["feature_scale"] <= 0) or np.any(stats["calibration_scale"] <= 0):
            raise SystemExit("Normalization, feature, and calibration scales must be positive")
        if not np.array_equal(stats["variable_names"].astype(str), np.asarray(metadata["variable_names"], dtype=str)):
            raise SystemExit("input_statistics.npz variable_names differ from run metadata")
        expected_feature_shape = (variable_count, layout.total_dim)
        for name, values in (("input_statistics.npz feature_mean", stats["feature_mean"]),
                             ("input_statistics.npz feature_scale", stats["feature_scale"]),
                             ("cache_manifest.json feature_mean", np.asarray(manifest.get("feature_mean"))),
                             ("cache_manifest.json feature_scale", np.asarray(manifest.get("feature_scale")))):
            if values.shape != expected_feature_shape:
                raise SystemExit(f"{root / name.split()[0]} {name.split()[-1]} shape mismatch: expected {expected_feature_shape}, got {values.shape}")
        if not np.allclose(stats["feature_mean"], manifest["feature_mean"], rtol=0, atol=0):
            raise SystemExit("Feature mean differs between statistics and cache manifest")
        if not np.allclose(stats["feature_scale"], manifest["feature_scale"], rtol=0, atol=0):
            raise SystemExit("Feature scale differs between statistics and cache manifest")
        with np.load(root / "feature_statistics.npz", allow_pickle=False) as feature_stats:
            for key in ("feature_mean", "feature_scale"):
                if key not in feature_stats.files:
                    raise SystemExit(f"Missing {key} in feature_statistics.npz")
                if not np.array_equal(feature_stats[key], stats[key]):
                    raise SystemExit(f"feature_statistics.npz {key} differs from input_statistics.npz")
        with np.load(root / "normalization_state.npz", allow_pickle=False) as input_state:
            input_mean = input_state["input_mean"]
            input_scale = input_state["input_scale"]
            if (not np.array_equal(input_mean, stats["mean"])
                    or not np.array_equal(input_scale, stats["scale"])
                    or str(input_state["resume_id"].item()) != identity.get("resume_id")):
                raise SystemExit("Frozen input statistics differ between run artifacts")
            fingerprint = _hash_array(np.concatenate((input_mean, input_scale)))
            if identity.get("input_statistics_fingerprint") != fingerprint:
                raise SystemExit("Frozen input statistics fingerprint differs from run identity")
            if input_mean.shape != (variable_count,) or input_scale.shape != (variable_count,):
                raise SystemExit(
                    f"{root / 'normalization_state.npz'} input statistics shape mismatch: expected {(variable_count,)}, "
                    f"got mean={input_mean.shape}, scale={input_scale.shape}"
                )
        binding_checks = {
            "resume_id": identity.get("resume_id"),
            "encoder_fingerprint": manifest.get("encoder_fingerprint"),
            "input_mean_sha256": _hash_array(stats["mean"]),
            "input_scale_sha256": _hash_array(stats["scale"]),
            "feature_mean_sha256": _hash_array(stats["feature_mean"]),
            "feature_scale_sha256": _hash_array(stats["feature_scale"]),
            "variable_names": metadata["variable_names"],
            "temporal_rule": identity.get("temporal_rule"),
            "feature_cache_sha256": actual_cache_hashes,
            "window_index_sha256": actual_index_hashes,
        }
        for key, expected_value in binding_checks.items():
            if binding.get(key) != expected_value:
                raise SystemExit(f"{binding_path} {key} differs from run/cache/statistics identity")
        try:
            validate_artifact_shapes(
                layout, variable_count, cache_shapes, index_shapes, expected_counts,
                stats["mean"].shape, stats["scale"].shape,
                stats["feature_mean"].shape, stats["feature_scale"].shape,
                stats["calibration_center"].shape, stats["calibration_scale"].shape,
                stats["noise_vectors"].shape, "input_statistics.npz",
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if deployment["noise_vectors"].shape != (4, layout.total_dim):
            raise SystemExit(
                f"{root / 'deployment.pt'} noise_vectors shape mismatch: expected {(4, layout.total_dim)}, "
                f"got {tuple(deployment['noise_vectors'].shape)}"
            )
        if not torch.isfinite(deployment["noise_vectors"]).all():
            raise SystemExit(f"{root / 'deployment.pt'} contains non-finite noise_vectors")
        for key, stats_key in (("input_mean", "mean"), ("input_scale", "scale"),
                               ("feature_mean", "feature_mean"), ("feature_scale", "feature_scale"),
                               ("calibration_center", "calibration_center"),
                               ("calibration_scale", "calibration_scale")):
            value = deployment.get(key)
            if value is None or tuple(value.shape) != tuple(stats[stats_key].shape):
                raise SystemExit(
                    f"{root / 'deployment.pt'} {key} shape mismatch: expected {stats[stats_key].shape}, "
                    f"got {None if value is None else tuple(value.shape)}"
                )
            if not np.array_equal(value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value), stats[stats_key]):
                raise SystemExit(f"{root / 'deployment.pt'} {key} differs from input_statistics.npz {stats_key}")
        deployment_noise = deployment["noise_vectors"].detach().cpu().numpy()
        if not np.array_equal(deployment_noise, stats["noise_vectors"]):
            raise SystemExit("Deployment fixed scoring noise differs from input_statistics.npz")
        manifest_feature_mean = np.asarray(manifest.get("feature_mean"))
        manifest_feature_scale = np.asarray(manifest.get("feature_scale"))
        manifest_input_mean = np.asarray(manifest.get("input_mean"))
        manifest_input_scale = np.asarray(manifest.get("input_scale"))
        if manifest_feature_mean.shape != (variable_count, layout.total_dim):
            raise SystemExit(f"{root / 'cache_manifest.json'} feature_mean shape mismatch: expected {(variable_count, layout.total_dim)}, got {manifest_feature_mean.shape}")
        if manifest_feature_scale.shape != (variable_count, layout.total_dim):
            raise SystemExit(f"{root / 'cache_manifest.json'} feature_scale shape mismatch: expected {(variable_count, layout.total_dim)}, got {manifest_feature_scale.shape}")
        if manifest_input_mean.shape != (variable_count,) or manifest_input_scale.shape != (variable_count,):
            raise SystemExit(f"{root / 'cache_manifest.json'} input mean/scale shape mismatch: expected {(variable_count,)}")
        _array_equal_at_path(manifest_input_mean, stats["mean"],
                             f"{root / 'cache_manifest.json'} input_mean", "input_statistics.npz mean")
        _array_equal_at_path(manifest_input_scale, stats["scale"],
                             f"{root / 'cache_manifest.json'} input_scale", "input_statistics.npz scale")
        if not np.array_equal(np.asarray(deployment["input_mean"]), stats["mean"]):
            raise SystemExit("Deployment input_mean differs from input_statistics.npz")
        if not np.array_equal(np.asarray(deployment["input_scale"]), stats["scale"]):
            raise SystemExit("Deployment input_scale differs from input_statistics.npz")
    if deployment.get("architecture_version") != identity["architecture_version"]:
        raise SystemExit("Deployment/run architecture mismatch")
    try:
        deployment_layout = require_layout(deployment.get("feature_layout"), layout,
                                           f"{root / 'deployment.pt'} feature_layout")
        deployment_config_layout = _config_layout(
            deployment.get("config"), f"{root / 'deployment.pt'} config"
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if deployment_config_layout.to_dict() != layout.to_dict():
        raise SystemExit(f"{root / 'deployment.pt'} config layout differs from run layout")
    if deployment_layout.to_dict() != layout.to_dict():
        raise SystemExit(f"{root / 'deployment.pt'} feature layout differs from run layout")
    if deployment.get("time_protocol_version") != TIME_PROTOCOL_VERSION or deployment.get("temporal_rule") != identity.get("temporal_rule"):
        raise SystemExit("Deployment frozen time protocol differs from the run identity")
    if deployment.get("training_metadata", {}).get("feature_fingerprint") != manifest.get("encoder_fingerprint"):
        raise SystemExit("Deployment encoder fingerprint differs from cache manifest")
    if tuple(deployment.get("variable_names", ())) != tuple(manifest.get("variable_names", ())):
        raise SystemExit("Deployment and cache variable order differ")
    if "auxiliary_state" in deployment or "auxiliary" in deployment:
        raise SystemExit("Auxiliary pretraining head must not be part of deployment")
    for checkpoint in (root / "checkpoints" / "best_encoder.pt", root / "checkpoints" / "best_fusion.pt",
                       root / "checkpoints" / "stage1_latest.pt", root / "checkpoints" / "stage2_latest.pt"):
        try:
            torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception as error:
            raise SystemExit(f"Checkpoint cannot be loaded ({checkpoint.name}): {error}")
    stage2_checkpoint = torch.load(root / "checkpoints" / "stage2_latest.pt",
                                  map_location="cpu", weights_only=False)
    checkpoint_binding = stage2_checkpoint.get("stage2_identity")
    if not isinstance(checkpoint_binding, dict):
        raise SystemExit("stage2_latest.pt is missing stage2_identity")
    try:
        require_layout(checkpoint_binding.get("feature_layout"), layout,
                       f"{root / 'checkpoints' / 'stage2_latest.pt'} stage2_identity")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if checkpoint_binding != binding:
        raise SystemExit("stage2_latest.pt identity differs from stage2_binding.json")
    result_files = sorted((root / "results").glob("*_metrics.json"))
    if len(result_files) != 1:
        raise SystemExit(f"Expected one per-task metrics JSON, found {len(result_files)} in {root}")
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("task") != f"{task}.csv" or payload.get("run_id") != root.name:
        raise SystemExit(f"Incomplete or mismatched task metrics: {payload}")
    for key in ("precision_at_calibrated_threshold", "recall_at_calibrated_threshold",
                "f1_at_calibrated_threshold", "normal_false_alarm_rate", "threshold"):
        if payload.get(key) is None or not math.isfinite(float(payload[key])):
            raise SystemExit(f"Non-finite {key} in {result_files[0]}")
    reports = sorted(root.glob("test_report*.csv"))
    reference_path = root / "report_reference.json"
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        report_path = Path(reference.get("report_path", ""))
        if (reference.get("task") != f"{task}.csv" or reference.get("model_name") != "MNRFLiteV2"
                or reference.get("run_id") != root.name
                or reference.get("benchmark_invocation_id") != context.get("benchmark_invocation_id")):
            raise SystemExit("Current report reference belongs to another task, model, run, or benchmark invocation")
        if not report_path.is_file():
            raise SystemExit(f"Referenced leaderboard is missing: {report_path}")
        if (report_path.resolve().parent != root.resolve()
                or not report_path.name.startswith("test_report")
                or report_path.suffix != ".csv"):
            raise SystemExit("Referenced leaderboard is not a report artifact inside this run directory")
        if _sha256_file(report_path) != reference.get("report_sha256"):
            raise SystemExit(f"Referenced leaderboard is damaged or changed: {report_path}")
        reports_for_run = [report_path]
        model_columns = [reference.get("model_column")]
    else:
        # Compatibility for legacy runs that predate explicit report references.
        # A unique report is the only unambiguous legacy case; never pick by mtime.
        if context.get("benchmark_invocation_id"):
            raise SystemExit("Current report reference is missing for this benchmark invocation")
        if len(reports) != 1:
            raise SystemExit(f"Legacy run has no report reference and {len(reports)} reports; current report is ambiguous")
        reports_for_run = reports
        model_columns = [column for column in pd.read_csv(reports[0], nrows=0).columns
                         if "MNRFLiteV2;" in str(column)]
    report_path = reports_for_run[0]
    report = pd.read_csv(report_path)
    if len(model_columns) != 1 or model_columns[0] not in report.columns or "metric_name" not in report.columns:
        raise SystemExit("Leaderboard does not contain the referenced MNRFLiteV2 result column")
    for metric in ("auc_roc", "auc_pr"):
        rows = report.loc[report.metric_name == metric]
        if len(rows) != 1 or not math.isfinite(float(rows.iloc[0][model_columns[0]])):
            raise SystemExit(f"Missing or non-finite {metric} in {report_path}")
        if payload[metric] is None or not math.isclose(float(rows.iloc[0][model_columns[0]]), float(payload[metric]), rel_tol=1e-9, abs_tol=1e-9):
            raise SystemExit(f"Leaderboard {metric} differs from aligned task metrics")
        if reference_path.is_file() and not math.isclose(
                float(reference.get("metrics", {}).get(metric, float("nan"))),
                float(rows.iloc[0][model_columns[0]]), rel_tol=1e-9, abs_tol=1e-9):
            raise SystemExit(f"Referenced report metric {metric} differs from its registered value")
    aligned_files = sorted((root / "results").glob("*_aligned_scores.npz"))
    if len(aligned_files) != 1:
        raise SystemExit(f"Expected one aligned score artifact, found {len(aligned_files)}")
    with np.load(aligned_files[0], allow_pickle=False) as aligned:
        expected = {"scores", "predictions", "time_indices", "local_time_indices", "time_index_values", "valid_mask", "local_valid_mask", "actual", "threshold"}
        if not expected.issubset(aligned.files):
            raise SystemExit(f"Aligned artifact missing fields: {expected - set(aligned.files)}")
        scores = aligned["scores"]
        predictions = aligned["predictions"]
        local = aligned["local_time_indices"]
        indices = aligned["time_indices"]
        actual = aligned["actual"]
        if not (len(scores) == len(predictions) == len(local) == len(indices) == payload["valid_points"]):
            raise SystemExit("Aligned score/index/prediction lengths do not match metrics")
        if not np.isfinite(scores).all() or not np.all(np.diff(indices) > 0) or not np.all(np.diff(local) > 0):
            raise SystemExit("Aligned scores or indices are invalid")
        if np.any(local < 0) or np.any(local >= len(actual)) or not np.array_equal(np.flatnonzero(aligned["local_valid_mask"]), local):
            raise SystemExit("Valid mask does not match scored local indices")
        if not np.array_equal(np.flatnonzero(aligned["valid_mask"]), indices):
            raise SystemExit("Full-series valid mask does not match global scored indices")
        if not np.array_equal(predictions, (scores > float(payload["threshold"])).astype(np.int64)):
            raise SystemExit("Saved predictions do not match the calibrated threshold")
        selected = actual[local]
        if np.unique(selected).size != 2:
            raise SystemExit("Metrics are not complete: aligned labels contain only one class")
        if not math.isclose(roc_auc_score(selected, scores), float(payload["auc_roc"]), rel_tol=1e-9, abs_tol=1e-9):
            raise SystemExit("Saved aligned scores do not reproduce AUC-ROC")
        if not math.isclose(average_precision_score(selected, scores), float(payload["auc_pr"]), rel_tol=1e-9, abs_tol=1e-9):
            raise SystemExit("Saved aligned scores do not reproduce AUC-PR")
        if len(aligned["time_index_values"]) != len(local):
            raise SystemExit("Saved timestamp coordinates do not match scored indices")
    aligned_task = f"{task}.csv_aligned_scores.npz"
    if aligned_files[0].name != aligned_task:
        raise SystemExit(f"Aligned artifact belongs to another task: {aligned_files[0].name}")
    if metadata.get("config", {}).get("output_dir") != str(root.resolve()):
        raise SystemExit("Run config output_dir does not match validated run path")
    print(f"MNRF-Lite V2 run complete: {root}")


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = __import__("hashlib").sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _finite_memmap(array, chunk_rows=4096):
    for start in range(0, len(array), chunk_rows):
        if not np.isfinite(array[start:start + chunk_rows]).all():
            return False
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: validate_mnrf_lite_v2_run.py RUN_DIR TASK")
    main(sys.argv[1], sys.argv[2])
