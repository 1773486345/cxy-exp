#!/usr/bin/env python
"""Evaluate the frozen Stage-1 E/R auxiliary decoder without retraining.

The command is deliberately one-task-at-a-time.  It reads the registered Full
run's cached raw E/R latents and normalized targets, restores the serialized
encoder/auxiliary state for identity checking, and evaluates the decoder in
bounded chunks.  It never calls the encoder, fits a GMM, changes a source run,
or uses test labels for fitting or calibration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mnrf_branch_source_registry import selected_source
from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
from ts_benchmark.baselines.mnrf_lite_v2.layout import FeatureLayout, require_layout
from ts_benchmark.baselines.mnrf_lite_v2.training import MNRFTrainer


TASKS = {
    "PSM": (None, 192),
    "MSL": ("full_gmm-full-20261021T112147Z-2470335", 192),
    "GECCO": ("full_gmm-full-20261020T204951Z-1642355", 96),
    "Genesis": ("full_gmm-full-20261020T205005Z-1645522", 192),
    "NYC": ("full_gmm-full-20261020T211341Z-1931786", 192),
    "CalIt2": ("full_gmm-full-20261021T112212Z-2474004", 192),
    "Creditcard": (None, 192),
    "CICIDS": (None, 192),
}
TASK_INDEX = {"E": 0, "P": 1, "R": 2}
DEFAULT_CHUNK = 256
Q = 0.995


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite(name, values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def _finite_in_chunks(name, values, chunk_rows=256):
    if values.ndim == 0:
        _finite(name, values)
        return
    for start in range(0, values.shape[0], chunk_rows):
        _finite(name, values[start:start + chunk_rows])


def source_path(task: str) -> tuple[Path, dict | None]:
    if task not in TASKS:
        raise ValueError(f"unsupported task {task!r}; choose one of {sorted(TASKS)}")
    legacy, _ = TASKS[task]
    path, registration = selected_source(task, legacy)
    if path is None:
        raise FileNotFoundError(f"{task} has no registered Full source")
    return Path(path).resolve(), registration


def _config_from_manifest(manifest: dict) -> MNRFConfig:
    saved = manifest.get("config")
    if not isinstance(saved, dict):
        raise ValueError("manifest has no persisted MNRF config")
    allowed = {
        "seed", "seq_len", "pattern_len", "hidden_dim", "view_dim",
        "evolution_dim", "pattern_dim", "relation_dim", "batch_size",
        "stage1_epochs", "stage2_epochs", "patience", "learning_rate",
        "weight_decay", "feature_std_floor", "train_noise_std", "noise_seed",
        "score_mad_floor", "calibration_quantile", "cache_chunk_size",
        "encoding_batch_size", "device", "output_dir", "resume_from", "task_name",
    }
    kwargs = {key: saved[key] for key in allowed if key in saved}
    layout = saved.get("resolved_layout") or manifest.get("layout")
    if isinstance(layout, dict):
        kwargs.update(layout)
    kwargs["device"] = "cpu"
    return MNRFConfig.from_kwargs(**kwargs)


def _check_declared_file(run: Path, rel: str, declared: str | None):
    path = run / rel
    if not path.is_file():
        raise FileNotFoundError(f"missing required source file: {rel}")
    if declared and sha256_file(path) != declared:
        raise ValueError(f"{rel} SHA256 differs from manifest")
    return path


def validate_source_contract(run: Path) -> dict:
    """Validate source identity and decoder/cache bindings without scoring."""
    manifest = _json(run / "probe_manifest.json")
    if manifest.get("stage_status") != "complete" or manifest.get("head_status") != "complete":
        raise ValueError("source is not a complete Stage-1/full-GMM run")
    if manifest.get("artifact_policy") != "full" or manifest.get("mode") not in {"full_gmm", "pretrained"}:
        raise ValueError("source must be artifact_policy=full with a full/pretrained head")
    if manifest.get("head_status") != "complete" or str(manifest.get("head", "")).lower() != "gmm":
        raise ValueError("source must contain a completed GMM head")
    identity = manifest.get("source_identity")
    if (not isinstance(identity, dict)
            or _json_hash(identity) != manifest.get("source_identity_sha256")):
        raise ValueError("source_identity is missing or its declared digest is invalid")
    config_values = manifest.get("config", {})
    selected_views = manifest.get("selected_views") or config_values.get("selected_views")
    if selected_views is not None and list(selected_views) != ["S", "E", "P", "R"]:
        raise ValueError("source is not the registered four-view Full representation")
    try:
        layout = FeatureLayout.from_dict(manifest.get("layout"))
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(f"source: invalid feature layout: {error}") from error
    config = _config_from_manifest(manifest)
    if config.layout.to_dict() != layout.to_dict():
        raise ValueError("manifest config and source layout differ")
    if float(config.calibration_quantile) != Q:
        raise ValueError(f"registered Full source calibration quantile must be {Q}")
    variables = len(manifest.get("variable_names", []))
    if variables <= 0:
        raise ValueError("source has no variable_names")
    required = {
        "checkpoints/best_stage1.pt": None,
        "checkpoints/best_encoder.pt": manifest.get("checkpoint_sha256", {}).get("best_encoder.pt"),
        "cache/normalized_values.npy": manifest.get("normalized_cache_sha256"),
        "analysis_inputs.npz": None,
        "input_statistics.npz": manifest.get("input_statistics_sha256"),
    }
    for rel, digest in required.items():
        _check_declared_file(run, rel, digest)
    normalized = np.load(run / "cache/normalized_values.npy", mmap_mode="r", allow_pickle=False)
    if normalized.ndim != 2 or normalized.shape[1] != variables or normalized.dtype != np.dtype("float32"):
        raise ValueError("normalized_values.npy shape/dtype does not match source variables")
    _finite_in_chunks("normalized_values.npy", normalized)
    cache_shapes = manifest.get("cache_shapes", {})
    endpoints = {}
    for split in ("train", "validation", "calibration", "test"):
        feature_path = _check_declared_file(run, f"cache/{split}_features.npy",
                                            manifest.get("cache_sha256", {}).get(split))
        endpoint_path = _check_declared_file(run, f"cache/{split}_window_end_indices.npy",
                                             manifest.get("endpoint_sha256", {}).get(split))
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        endpoint_array = np.load(endpoint_path, mmap_mode="r", allow_pickle=False)
        endpoints[split] = endpoint_array
        if features.ndim != 3 or features.shape[1:] != (variables, layout.total_dim):
            raise ValueError(f"{split}_features.npy shape does not match source layout")
        if features.dtype != np.dtype("float32"):
            raise ValueError(f"{split}_features.npy must remain float32")
        if cache_shapes and list(features.shape) != list(cache_shapes.get(split, [])):
            raise ValueError(f"{split} cache shape differs from manifest")
        if endpoint_array.ndim != 1 or len(endpoint_array) != features.shape[0] or endpoint_array.dtype.kind not in "iu":
            raise ValueError(f"{split} endpoints are malformed")
        if np.any(endpoint_array < 0) or np.any(endpoint_array >= normalized.shape[0]):
            raise ValueError(f"{split} endpoints are outside normalized input")
        _finite_in_chunks(f"{split}_features.npy", features)
    with np.load(run / "analysis_inputs.npz", allow_pickle=False) as axes:
        all_global = np.asarray(axes["all_global_indices"])
        all_labels = np.asarray(axes["all_labels"])
        all_times = np.asarray(axes["all_time_values_ns"])
        all_segments = np.asarray(axes["all_segment_ids"])
        test_global = np.asarray(axes["test_global_indices"])
        test_labels = np.asarray(axes["test_labels"])
        test_times = np.asarray(axes["test_time_values_ns"])
        test_segments = np.asarray(axes["test_segment_ids"])
    if not (all_global.ndim == all_labels.ndim == all_times.ndim == all_segments.ndim == 1):
        raise ValueError("analysis axis arrays must be one-dimensional")
    if not (len(all_global) == len(all_labels) == len(all_times) == len(all_segments)):
        raise ValueError("analysis axis arrays have inconsistent lengths")
    if test_global.ndim != 1 or any(
            values.ndim != 1 or values.shape != test_global.shape
            for values in (test_labels, test_times, test_segments)):
        raise ValueError("test analysis axis is malformed")
    if (all_global.dtype.kind not in "iu" or test_global.dtype.kind not in "iu"
            or (len(all_global) > 1 and np.any(np.diff(all_global.astype(np.int64)) <= 0))
            or (len(test_global) > 1 and np.any(np.diff(test_global.astype(np.int64)) <= 0))):
        raise ValueError("analysis global coordinates must be sorted unique integer axes")
    if (all_segments.dtype.kind not in "iu" or test_segments.dtype.kind not in "iu"
            or all_times.dtype.kind not in "iu" or test_times.dtype.kind not in "iu"):
        raise ValueError("analysis times and segment IDs must have integer dtypes")
    if not np.isin(all_labels, (0, 1)).all() or not np.isin(test_labels, (0, 1)).all():
        raise ValueError("analysis labels are not binary")
    full_positions = np.searchsorted(all_global, test_global)
    if (np.any(full_positions >= len(all_global))
            or not np.array_equal(all_global[full_positions], test_global)):
        raise ValueError("test coordinates are not a subset of the complete source axis")
    if not np.array_equal(test_labels, all_labels[full_positions]):
        raise ValueError("test labels differ from the complete source axis mapping")
    if not np.array_equal(test_times, all_times[full_positions]):
        raise ValueError("test times differ from the complete source axis mapping")
    if not np.array_equal(test_segments, all_segments[full_positions]):
        raise ValueError("test_segment_ids differ from the complete source axis mapping")
    test_endpoints = np.asarray(endpoints["test"], dtype=np.int64)
    if not np.isin(test_endpoints, test_global).all():
        raise ValueError("test feature endpoints are outside the frozen test axis")
    global_label_map = {int(index): int(label) for index, label in zip(all_global, all_labels)}
    endpoint_labels = np.asarray([global_label_map[int(index)] for index in test_endpoints], dtype=np.int8)
    if not np.array_equal(endpoint_labels, test_labels[np.searchsorted(test_global, test_endpoints)]):
        raise ValueError("test endpoint labels differ between full and test analysis axes")
    aligned_path = run / "results" / "aligned_scores.npz"
    declared_aligned = manifest.get("aligned_scores_sha256")
    if declared_aligned and sha256_file(aligned_path) != declared_aligned:
        raise ValueError("results/aligned_scores.npz SHA256 differs from manifest")
    with np.load(aligned_path, allow_pickle=False) as aligned:
        if not np.array_equal(np.asarray(aligned["time_indices"], dtype=np.int64), test_endpoints):
            raise ValueError("test cache endpoints differ from saved aligned score coordinates")
        if not np.array_equal(np.asarray(aligned["actual"], dtype=np.int8), endpoint_labels):
            raise ValueError("test endpoint labels differ from saved aligned scores")
        if "valid_mask" in aligned and "local_indices" in aligned:
            valid = np.asarray(aligned["valid_mask"])
            local = np.asarray(aligned["local_indices"])
            if valid.dtype.kind != "b" or not np.array_equal(np.flatnonzero(valid), local):
                raise ValueError("saved Full valid mask/local coordinates are inconsistent")
    return {"run": run, "manifest": manifest, "config": config, "layout": layout,
            "variables": variables, "normalized": normalized, "endpoints": endpoints,
            "axes": {"all_global": all_global, "all_labels": all_labels, "all_times": all_times,
                     "all_segments": all_segments, "test_global": test_global, "test_labels": test_labels,
                     "test_times": test_times, "test_segments": test_segments}}


def load_decoder(source: dict):
    """Restore state for identity checking; no encoder forward is performed."""
    import torch
    run = source["run"]
    if "layout" in source and source["config"].layout.to_dict() != source["layout"].to_dict():
        raise ValueError("checkpoint config/layout mismatch")
    checkpoint_path = run / "checkpoints" / "best_stage1.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not {"encoder", "auxiliary"}.issubset(checkpoint):
        raise ValueError("best_stage1.pt lacks encoder/auxiliary state")
    declared_encoder = torch.load(run / "checkpoints" / "best_encoder.pt",
                                  map_location="cpu", weights_only=False)
    for component in ("encoder", "auxiliary"):
        left, right = checkpoint[component], declared_encoder.get(component)
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError(f"best_stage1.pt and best_encoder.pt {component} identities differ")
        if any(not torch.equal(left[key], right[key]) for key in left):
            raise ValueError(f"best_stage1.pt and best_encoder.pt {component} tensors differ")
    trainer = MNRFTrainer(source["config"], source["variables"], torch.device("cpu"))
    try:
        trainer.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        trainer.auxiliary.load_state_dict(checkpoint["auxiliary"], strict=True)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"checkpoint does not match source layout/variables: {error}") from error
    statistics_path = run / "input_statistics.npz"
    with np.load(statistics_path, allow_pickle=False) as statistics:
        mean = np.asarray(statistics["mean"], dtype=np.float32)
        scale = np.asarray(statistics["scale"], dtype=np.float32)
    if mean.shape != (source["variables"],) or scale.shape != mean.shape:
        raise ValueError("input statistics do not match source variable order/count")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("input statistics must be finite with positive scales")
    trainer.fixed_input.mean_ = mean
    trainer.fixed_input.scale_ = scale
    if trainer.fingerprint() != source["manifest"].get("feature_fingerprint"):
        raise ValueError("checkpoint, input statistics and feature fingerprint do not bind")
    trainer.encoder.eval(); trainer.auxiliary.eval()
    return trainer, hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()


def decoder_nll(trainer, latent, targets, task: str):
    """Apply exactly GaussianAuxiliaryDecoder's mean/sigma/NLL formula."""
    import torch
    if task not in TASK_INDEX:
        raise ValueError("task must be E or R")
    latent = torch.as_tensor(np.array(latent, dtype=np.float32, copy=True))
    targets = torch.as_tensor(np.array(targets, dtype=np.float32, copy=True))
    if latent.ndim != 3 or targets.shape != latent.shape[:2]:
        raise ValueError("latent must be [N,C,D] and targets [N,C]")
    task_index = TASK_INDEX[task]
    with torch.no_grad():
        z = trainer.auxiliary.input_adapters[task_index](latent)
        n, c, _ = z.shape
        identity = trainer.encoder.point.variable_identity.view(1, c, -1).expand(n, -1, -1)
        task_identity = trainer.auxiliary.task_identity[task_index].view(1, 1, -1).expand(n, c, -1)
        position = torch.zeros((n, c, 1), dtype=z.dtype)
        output = trainer.auxiliary.decoder(torch.cat((z, task_identity, identity, position), dim=-1))
        mean = output[..., 0]
        sigma = 0.05 + torch.nn.functional.softplus(output[..., 1])
        nll = (targets - mean).square() / (2.0 * sigma.square()) + torch.log(sigma) + 0.5 * math.log(2.0 * math.pi)
    return mean.numpy(), sigma.numpy(), nll.numpy()


def _split_predictions(source, trainer, split, task, chunk_size, latent_override=None):
    features = np.load(source["run"] / "cache" / f"{split}_features.npy", mmap_mode="r", allow_pickle=False)
    endpoints = source["endpoints"][split]
    normalized = source["normalized"]
    start, end = source["layout"].slices[task]
    latent_means = latent_override
    means = np.empty((len(features), source["variables"]), dtype=np.float32)
    sigmas = np.empty_like(means)
    nll = np.empty_like(means)
    for left in range(0, len(features), chunk_size):
        right = min(left + chunk_size, len(features))
        latent = np.asarray(features[left:right, :, start:end])
        if latent_means is not None:
            latent = np.array(latent, copy=True)
            latent[:] = latent_means[None, :, :]
        target = np.asarray(normalized[np.asarray(endpoints[left:right], dtype=np.int64)])
        means[left:right], sigmas[left:right], nll[left:right] = decoder_nll(trainer, latent, target, task)
    return {"mean": means, "sigma": sigmas, "nll": nll, "target": np.asarray(normalized[np.asarray(endpoints, dtype=np.int64)])}


def _latent_mean(source, task, chunk_size):
    features = np.load(source["run"] / "cache/train_features.npy", mmap_mode="r", allow_pickle=False)
    start, end = source["layout"].slices[task]
    total = np.zeros((source["variables"], end - start), dtype=np.float64)
    count = 0
    for left in range(0, len(features), chunk_size):
        block = np.asarray(features[left:left + chunk_size, :, start:end], dtype=np.float64)
        _finite("training latent", block)
        total += block.sum(axis=0)
        count += len(block)
    if count == 0:
        raise ValueError("training cache has no rows")
    return (total / count).astype(np.float32)


def _score_from_matrix(matrix, validation, calibration, q=Q, floor=0.001):
    center = np.median(validation, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(validation - center), axis=0), float(floor))
    val_score = np.max((validation - center) / scale, axis=1)
    cal_score = np.max((calibration - center) / scale, axis=1)
    threshold = float(np.quantile(cal_score, q, method="linear"))
    test_score = np.max((matrix - center) / scale, axis=1)
    return test_score, threshold, center, scale


def _endpoint_labels(source):
    """Map legal cached test endpoints onto the complete, label-bearing axis."""
    axes = source["axes"]
    lookup = {int(index): int(label) for index, label in zip(axes["all_global"], axes["all_labels"])}
    endpoints = np.asarray(source["endpoints"]["test"], dtype=np.int64)
    try:
        labels = np.asarray([lookup[int(index)] for index in endpoints], dtype=np.int8)
    except KeyError as error:
        raise ValueError(f"test endpoint is absent from the complete label axis: {error}") from error
    return endpoints, labels


def _persistence_prediction(source, endpoints):
    """Previous-row baseline, undefined across any frozen segment boundary."""
    normalized = source["normalized"]
    axes = source["axes"]
    segments = {int(index): int(segment) for index, segment in zip(axes["all_global"], axes["all_segments"])}
    result = np.full((len(endpoints), source["variables"]), np.nan, dtype=np.float32)
    for row, endpoint in enumerate(np.asarray(endpoints, dtype=np.int64)):
        previous = int(endpoint) - 1
        if previous in segments and segments.get(int(endpoint)) == segments[previous]:
            result[row] = normalized[previous]
    return result


def _error_summary(target, mean, sigma, nll, variable_names=None):
    residual = target - mean
    names = list(variable_names or [str(index) for index in range(target.shape[1])])
    return {
        "mae": float(np.mean(np.abs(residual))),
        "mse": float(np.mean(residual ** 2)),
        "sigma_quantiles": [float(value) for value in np.quantile(sigma, [0.01, 0.5, 0.99])],
        "nll_quantiles": [float(value) for value in np.quantile(nll, [0.01, 0.5, 0.99])],
        "per_variable": [
            {"index": index, "variable": names[index],
             "mae": float(np.mean(np.abs(residual[:, index]))),
             "mse": float(np.mean(residual[:, index] ** 2)),
             "sigma_median": float(np.median(sigma[:, index])),
             "nll_median": float(np.median(nll[:, index]))}
            for index in range(target.shape[1])
        ],
    }


def _metrics(scores, labels, threshold):
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
    labels = np.asarray(labels, dtype=np.int8)
    pred = np.asarray(scores > threshold, dtype=np.int8)
    positives = labels == 1
    negatives = ~positives
    tp = int(np.sum(positives & (pred == 1))); fp = int(np.sum(negatives & (pred == 1)))
    fn = int(np.sum(positives & (pred == 0))); tn = int(np.sum(negatives & (pred == 0)))
    result = {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
              "precision": float(tp / (tp + fp)) if tp + fp else None,
              "recall": float(tp / (tp + fn)) if tp + fn else None,
              "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
              "fpr": float(fp / (fp + tn)) if fp + tn else None,
              "auc_roc": None, "average_precision": None}
    if len(np.unique(labels)) == 2:
        result["auc_roc"] = float(roc_auc_score(labels, scores))
        result["average_precision"] = float(average_precision_score(labels, scores))
    else:
        result["undefined_reason"] = "test labels contain one class"
    return result


def _boundary_metrics(source, scores, threshold):
    from scripts.summarize_mnrf_branch_ablation import segment_counts_from_axis
    from ts_benchmark.baselines.mnrf_view_probe.evaluation import affiliation_f_segment_macro

    axes = source["axes"]
    endpoints = np.asarray(source["endpoints"]["test"], dtype=np.int64)
    predictions = np.asarray(scores > threshold, dtype=np.int8)
    test_axis = np.asarray(axes["test_global"], dtype=np.int64)
    local = np.searchsorted(test_axis, endpoints)
    if np.any(local >= len(test_axis)) or not np.array_equal(test_axis[local], endpoints):
        raise ValueError("scored endpoint is not on the frozen Full test axis")
    valid = np.zeros(len(test_axis), dtype=bool)
    full_prediction = np.zeros(len(test_axis), dtype=np.int8)
    valid[local] = True
    full_prediction[local] = predictions
    labels = np.asarray(axes["test_labels"], dtype=np.int8)
    boundaries = np.asarray(axes["test_segments"], dtype=np.int64)
    # A source segment is authoritative; a jump in original row coordinates is
    # also a hard boundary even if adjacent entries carry the same segment ID.
    boundary_ids = np.empty(len(test_axis), dtype=np.int64)
    boundary_id = -1
    for index in range(len(test_axis)):
        if (index == 0 or boundaries[index] != boundaries[index - 1]
                or test_axis[index] != test_axis[index - 1] + 1):
            boundary_id += 1
        boundary_ids[index] = boundary_id
    aff = affiliation_f_segment_macro(labels, full_prediction, valid, boundary_ids)
    defined = [segment for segment in aff["segments"] if segment["defined"]]
    precision = (float(np.mean([segment["precision"] for segment in defined]))
                 if defined else None)
    recall = (float(np.mean([segment["recall"] for segment in defined]))
              if defined else None)
    undefined = [
        {"start": int(segment["start"]), "end": int(segment["end"]),
         "reason": segment["undefined_reason"]}
        for segment in aff["segments"] if not segment["defined"]
    ]
    segments = segment_counts_from_axis(
        axes["all_labels"], axes["all_global"], axes["all_segments"], test_axis,
        endpoints, predictions, axes["all_times"])
    return {
        "affiliation_precision_valid_segment_macro": precision,
        "affiliation_recall_valid_segment_macro": recall,
        "affiliation_f_valid_segment_macro": aff["affiliation_f_segment_macro"],
        "affiliation_defined_segments": aff["defined_segment_count"],
        "affiliation_total_segments": aff["total_segment_count"],
        "affiliation_undefined_segments": undefined,
        "affiliation_aggregation": "macro mean of per-segment precision, recall, and F over defined segments",
        "anomaly_segments_total": segments["total"],
        "anomaly_segments_scored": segments["scored"],
        "anomaly_segments_hit": segments["hit"],
    }


def _write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _run_task_impl(task: str, output_dir: Path | None = None, chunk_size: int = DEFAULT_CHUNK,
                   thread_record: dict | None = None):
    source_run, registration = source_path(task)
    source = validate_source_contract(source_run)
    trainer, checkpoint_sha256 = load_decoder(source)
    output = Path(output_dir) if output_dir else ROOT / "result/analysis/MNRF-FrozenDecoder" / task / source_run.name
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"diagnostic output exists; choose a new path: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    rows = []
    evidence = {"task": task, "source_run": str(source_run), "source_registration": registration,
                "checkpoint_sha256": checkpoint_sha256, "chunk_size": chunk_size,
                "cpu_threads": thread_record,
                "calibration_quantile": Q, "test_labels_used_only_for_metrics": True,
                "encoder_forward_called": False, "gmm_fit_called": False}
    evidence["variable_names"] = source["manifest"]["variable_names"]
    evidence["task_role_note"] = ({"MSL": "col_0 is a telemetry-target diagnostic only; other columns remain context/unknown.",
                                   "NYC": "passenger-count role may be interpreted separately; calendar covariates remain in the frozen encoder context."}
                                  .get(task, "All variables retain their frozen source roles."))
    test_axis, test_labels = _endpoint_labels(source)
    with np.load(source_run / "results" / "aligned_scores.npz", allow_pickle=False) as saved_full:
        full_scores = np.asarray(saved_full["scores"])
        full_predictions = np.asarray(saved_full["predictions"], dtype=np.int8)
        full_threshold = float(np.asarray(saved_full["threshold"]).reshape(()))
    rows.append({"task": task, "view": "FULL", "variant": "saved_full_gmm_reference",
                 "score": "saved_original_aggregate", "source_run_id": source_run.name,
                 "variables": source["variables"], "valid_test_points": len(full_scores),
                 "prediction_rule": "score > threshold",
                 **_metrics(full_scores, test_labels, full_threshold),
                 **_boundary_metrics(source, full_scores, full_threshold)})
    variable_summaries = []
    for view in ("E", "R"):
        train_latent_mean = _latent_mean(source, view, chunk_size)
        for variant, override in (("decoder", None), ("decoder_latent_train_mean", train_latent_mean)):
            split_data = {split: _split_predictions(source, trainer, split, view, chunk_size, override)
                          for split in ("validation", "calibration", "test")}
            for score_name in ("squared_residual", "nll"):
                matrices = {split: split_data[split]["mean"] for split in split_data} if score_name == "squared_residual" else {split: split_data[split]["nll"] for split in split_data}
                if score_name == "squared_residual":
                    matrices = {split: (split_data[split]["target"] - split_data[split]["mean"]) ** 2 for split in split_data}
                score, threshold, center, scale = _score_from_matrix(
                    matrices["test"], matrices["validation"], matrices["calibration"],
                    floor=source["config"].score_mad_floor)
                metric = _metrics(score, test_labels, threshold)
                residual = split_data["test"]["target"] - split_data["test"]["mean"]
                rows.append({"task": task, "view": view, "variant": variant, "score": score_name,
                             "source_run_id": source_run.name, "variables": source["variables"],
                             "valid_test_points": len(test_axis), "threshold": threshold,
                             "center_rule": "validation median", "scale_rule": "max(1.4826*MAD,0.001)",
                             "prediction_rule": "score > threshold",
                             "prediction_mae": float(np.mean(np.abs(residual))),
                             "prediction_mse": float(np.mean(residual ** 2)), **metric,
                             **_boundary_metrics(source, score, threshold)})
                variable_summaries.append({"view": view, "variant": variant, "score": score_name,
                                           "split": "test", **_error_summary(
                                               split_data["test"]["target"], split_data["test"]["mean"],
                                               split_data["test"]["sigma"], split_data["test"]["nll"],
                                               source["manifest"]["variable_names"])})
                if variant == "decoder":
                    np.savez_compressed(output / f"{view}_{score_name}_test_scores.npz",
                                        scores=score.astype(np.float32), labels=test_labels.astype(np.int8),
                                        global_indices=test_axis.astype(np.int64), threshold=np.asarray(threshold),
                                        center=center.astype(np.float32), scale=scale.astype(np.float32))
                    if score_name == "nll":
                        residual = split_data["test"]["target"] - split_data["test"]["mean"]
                        np.savez_compressed(output / f"{view}_test_prediction_evidence.npz",
                                            residual=residual.astype(np.float32),
                                            per_variable_nll=split_data["test"]["nll"].astype(np.float32),
                                            endpoints=test_axis.astype(np.int64),
                                            variable_names=np.asarray(source["manifest"]["variable_names"], dtype=str))
            # Save bounded distribution summaries, not raw per-variable matrices.
            evidence.setdefault("prediction_error", {}).setdefault(view, {})[variant] = {
                **_error_summary(split_data["test"]["target"], split_data["test"]["mean"],
                                 split_data["test"]["sigma"], split_data["test"]["nll"],
                                 source["manifest"]["variable_names"]),
                "latent_train_mean_shape": list(train_latent_mean.shape),
            }
        # Persistence/mean references use exactly the same target axes.
        for ref_name in ("persistence", "train_target_mean"):
            split_data = {}
            for split in ("validation", "calibration", "test"):
                endpoints = source["endpoints"][split]
                target = np.asarray(source["normalized"][np.asarray(endpoints, dtype=np.int64)])
                if ref_name == "persistence":
                    pred = _persistence_prediction(source, endpoints)
                else:
                    train_target = np.asarray(source["normalized"][np.asarray(source["endpoints"]["train"], dtype=np.int64)])
                    pred = np.broadcast_to(np.mean(train_target, axis=0), target.shape)
                split_data[split] = (target - pred) ** 2
                if not np.isfinite(split_data[split]).all():
                    raise ValueError(f"{view} {ref_name} crosses a source segment boundary")
            score, threshold, _, _ = _score_from_matrix(
                split_data["test"], split_data["validation"], split_data["calibration"],
                floor=source["config"].score_mad_floor)
            test_endpoints = np.asarray(source["endpoints"]["test"], dtype=np.int64)
            test_target = np.asarray(source["normalized"][test_endpoints])
            if ref_name == "persistence":
                reference_prediction = _persistence_prediction(source, test_endpoints)
            else:
                train_target = np.asarray(source["normalized"][np.asarray(source["endpoints"]["train"], dtype=np.int64)])
                reference_prediction = np.broadcast_to(np.mean(train_target, axis=0), test_target.shape)
            reference_residual = test_target - reference_prediction
            rows.append({"task": task, "view": view, "variant": ref_name, "score": "squared_residual",
                         "source_run_id": source_run.name, "variables": source["variables"],
                         "valid_test_points": len(test_axis), "threshold": threshold,
                         "center_rule": "validation median", "scale_rule": "max(1.4826*MAD,0.001)",
                         "prediction_rule": "score > threshold",
                         "prediction_mae": float(np.mean(np.abs(reference_residual))),
                         "prediction_mse": float(np.mean(reference_residual ** 2)),
                         **_metrics(score, test_labels, threshold),
                         **_boundary_metrics(source, score, threshold)})
    evidence["elapsed_seconds"] = time.perf_counter() - started
    evidence["notes"] = [
        "E/R decoder consumes raw cached latent slices before GMM feature standardization/noise.",
        "Decoder latent-train-mean is a diagnostic intervention, not a new model.",
        "Test labels are used only after calibration for reported metrics.",
        "P is intentionally not evaluated in this round; future masking/recovery diagnostics require re-encoding.",
    ]
    _write_csv(output / "metrics.csv", rows)
    (output / "per_variable_summary.json").write_text(
        json.dumps(variable_summaries, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "diagnostic_manifest.json").write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        f"# Frozen E/R decoder diagnostic: {task}\n\n"
        "This CPU diagnostic reads a registered Full source and does not train, encode, fit a GMM, or alter the source. "
        "The decoder uses raw E/R latent coordinates and the Stage-1 Gaussian formula. `decoder_latent_train_mean` is an intervention to test whether the decoder uses conditional representation. "
        "Squared-residual and NLL rankings are reported separately; each has validation median/MAD and calibration q=0.995. "
        "P is not evaluated because the existing cache cannot be treated as an occluded-input representation.\n\n"
        f"Source: `{source_run}`\nOutput: `{output}`\n",
        encoding="utf-8")
    print(json.dumps({"task": task, "source_dir": str(source_run), "output_dir": str(output), "rows": len(rows), "elapsed_seconds": evidence["elapsed_seconds"]}, indent=2))
    return output


def run_task(task: str, output_dir: Path | None = None, chunk_size: int = DEFAULT_CHUNK,
             cpu_threads: int = 1):
    """Run one diagnostic with process-local CPU library limits."""
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(cpu_threads)
    except RuntimeError:
        # PyTorch only permits setting inter-op threads before its first work;
        # record the effective value instead of claiming the requested value.
        pass
    limiter = threadpool_limits(limits=cpu_threads)
    try:
        thread_record = {
            "requested": cpu_threads,
            "torch_intraop": int(torch.get_num_threads()),
            "torch_interop": int(torch.get_num_interop_threads()),
            "native_threadpools": [
                {key: pool.get(key) for key in ("user_api", "internal_api", "num_threads", "prefix")}
                for pool in threadpool_info()
            ],
        }
        if thread_record["torch_intraop"] != cpu_threads or any(
                pool["num_threads"] != cpu_threads for pool in thread_record["native_threadpools"]):
            raise RuntimeError("requested CPU thread limit did not take effect")
        return _run_task_impl(task, output_dir, chunk_size, thread_record)
    finally:
        limiter.restore_original_limits()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    run_task(args.task, args.output_dir, args.chunk_size, args.cpu_threads)


if __name__ == "__main__":
    main()
