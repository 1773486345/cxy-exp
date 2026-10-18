"""Frozen four-view representation probes.

This module deliberately owns a separate artifact format. It imports the
existing MNRF-Lite V2 encoder/trainer primitives but never writes their run
directories or changes their model/scoring code.
"""

import hashlib
import json
import math
import numbers
import os
import pickle
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_info, threadpool_limits

from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
from ts_benchmark.baselines.mnrf_lite_v2.layout import FeatureLayout, require_layout
from ts_benchmark.baselines.mnrf_lite_v2.training import (
    EndWindowDataset, MNRFTrainer, _capture_rng, _restore_rng,
    atomic_torch_save, purged_ranges, seed_all,
)
from ts_benchmark.baselines.mnrf_lite_v2.windows import (
    TimeIndex, continuous_segments, infer_expected_interval, intersect_segments, window_endpoints,
)
from .config import VALID_VIEWS, ViewProbeConfig


PROBE_SCHEMA = "mnrf-view-probe-v2"
ARTIFACT_SOURCE = "source_encoder"
ARTIFACT_HEAD = "probe_head"
ARTIFACT_RESULTS = "result_bundle"
RESULT_SCHEMA = "mnrf-view-probe-results-v1"


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_pickle(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(temp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _chunked_array_hash(value, chunk_size=65536):
    """Hash array metadata and bounded chunks without a second full copy."""
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True).encode())
    flat = array.reshape(-1)
    for start in range(0, flat.size, chunk_size):
        digest.update(np.ascontiguousarray(flat[start:start + chunk_size]).tobytes())
    return digest.hexdigest()


def _hash_strings(values, chunk_size=65536):
    digest = hashlib.sha256()
    values = np.asarray(values, dtype=object).reshape(-1)
    for start in range(0, len(values), chunk_size):
        for value in values[start:start + chunk_size]:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def _json_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _runtime_free_config(config):
    """Return the persisted configuration fields that define an artifact."""
    value = dict(config.to_dict() if hasattr(config, "to_dict") else config)
    # Logging cadence and retention are execution/storage metadata. They must
    # be persisted, but changing them must not invalidate numerical recovery.
    for key in ("output_dir", "resume_from", "resolved_layout", "selected_indices", "selected_dim",
                "artifact_policy", "log_every_steps", "log_every_seconds", "gmm_log_every_iters"):
        value.pop(key, None)
    return value


def _strict_integer_indices(value, name):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains coordinates that overflow int64")
    if np.any(array < 0):
        raise ValueError(f"{name} contains negative coordinates")
    return np.asarray(array, dtype=np.int64)


def _safe_name(value):
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))


def _validate_checkpoint_state_dict(payload, module, name):
    """Strictly validate a serialized module state without mutating a model."""
    if not isinstance(payload, dict):
        raise ValueError(f"resume rejected: Flow {name} state payload is not a mapping")
    expected = module.state_dict()
    actual_keys = set(payload)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"resume rejected: Flow {name} state keys differ; missing={missing}, unexpected={unexpected}"
        )
    for key, reference in expected.items():
        value = payload[key]
        if not torch.is_tensor(value):
            raise ValueError(f"resume rejected: Flow {name} state {key} is not a tensor")
        if tuple(value.shape) != tuple(reference.shape) or value.dtype != reference.dtype:
            raise ValueError(
                f"resume rejected: Flow {name} state {key} shape/dtype differs; "
                f"expected={tuple(reference.shape)}/{reference.dtype}, "
                f"actual={tuple(value.shape)}/{value.dtype}"
            )
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise ValueError(f"resume rejected: non-finite Flow {name} state {key}")


def _validate_optimizer_step(value, parameter_id):
    """Validate AdamW's scalar step before PyTorch can coerce its type."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"resume rejected: Flow optimizer step for parameter {parameter_id} is bool")
    if torch.is_tensor(value):
        if value.numel() != 1 or torch.is_complex(value) or value.dtype == torch.bool:
            raise ValueError(f"resume rejected: Flow optimizer step for parameter {parameter_id} is not a scalar")
        scalar = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, numbers.Real):
        scalar = value
    else:
        raise ValueError(f"resume rejected: Flow optimizer step for parameter {parameter_id} is not numeric")
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"resume rejected: Flow optimizer step for parameter {parameter_id} is not numeric") from error
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"resume rejected: Flow optimizer step for parameter {parameter_id} must be a non-negative integer")


def _validate_flow_optimizer_state(serialized, trainer, optimizer):
    """Validate AdamW groups and lazy state tensors before a resume write."""
    if not isinstance(serialized, dict):
        raise ValueError("resume rejected: Flow optimizer state is not a mapping")
    groups = serialized.get("param_groups")
    states = serialized.get("state")
    if not isinstance(groups, list) or not isinstance(states, dict):
        raise ValueError("resume rejected: Flow optimizer state has invalid groups/state")
    expected_groups = optimizer.param_groups
    if len(groups) != len(expected_groups):
        raise ValueError("resume rejected: Flow optimizer parameter-group count differs")
    referenced = set()
    for index, (saved, expected) in enumerate(zip(groups, expected_groups)):
        if not isinstance(saved, dict) or not isinstance(saved.get("params"), list):
            raise ValueError(f"resume rejected: Flow optimizer group {index} is invalid")
        try:
            referenced.update(saved["params"])
        except TypeError as error:
            raise ValueError(f"resume rejected: Flow optimizer group {index} parameter binding is invalid") from error
        if len(saved["params"]) != len(expected["params"]):
            raise ValueError(f"resume rejected: Flow optimizer group {index} parameter count differs")
        if set(saved).difference(expected) - {"params"}:
            raise ValueError(f"resume rejected: Flow optimizer group {index} has unknown options")
        for field, expected_value in expected.items():
            if field == "params":
                continue
            if field not in saved:
                raise ValueError(f"resume rejected: Flow optimizer {field} is missing")
            if field == "lr":
                if not math.isclose(float(saved[field]), float(expected_value), rel_tol=0.0, abs_tol=1e-15):
                    raise ValueError("resume rejected: Flow optimizer learning rate differs")
            elif saved[field] != expected_value:
                raise ValueError(f"resume rejected: Flow optimizer {field} differs")
    if set(states) - referenced:
        raise ValueError("resume rejected: Flow optimizer contains unbound parameter state")
    parameter_amsgrad = {}
    for group in groups:
        for parameter_id in group["params"]:
            parameter_amsgrad[parameter_id] = bool(group.get("amsgrad", False))
    allowed_state_fields = {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
    for parameter_id, state in states.items():
        if not isinstance(state, dict):
            raise ValueError(f"resume rejected: Flow optimizer state for parameter {parameter_id} is invalid")
        # No state entry or an empty entry is the legitimate lazy-optimizer
        # case. Once any field exists, AdamW must have a complete state.
        if not state:
            continue
        missing = {"step", "exp_avg", "exp_avg_sq"}.difference(state)
        if parameter_amsgrad.get(parameter_id, False):
            missing.add("max_exp_avg_sq")
        if missing:
            raise ValueError(
                f"resume rejected: Flow optimizer state for parameter {parameter_id} "
                f"is missing required field(s) {sorted(missing)}"
            )
        unknown = set(state).difference(allowed_state_fields)
        if unknown:
            raise ValueError(
                f"resume rejected: Flow optimizer state for parameter {parameter_id} "
                f"has unknown field(s) {sorted(unknown)}"
            )
        _validate_optimizer_step(state["step"], parameter_id)
    try:
        optimizer.load_state_dict(serialized)
    except Exception as error:
        raise ValueError("resume rejected: Flow optimizer state is incompatible") from error
    for parameter, state in optimizer.state.items():
        if not isinstance(state, dict):
            raise ValueError("resume rejected: Flow optimizer parameter state is invalid")
        for key, value in state.items():
            if key == "step":
                if torch.is_tensor(value):
                    if value.numel() != 1 or not torch.isfinite(value).all():
                        raise ValueError("resume rejected: Flow optimizer step is invalid")
                elif not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError("resume rejected: Flow optimizer step is invalid")
            elif key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
                if not torch.is_tensor(value) or tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"resume rejected: Flow optimizer {key} shape differs")
                if torch.is_floating_point(value) and not torch.isfinite(value).all():
                    raise ValueError(f"resume rejected: Flow optimizer {key} is non-finite")
            else:
                raise ValueError(f"resume rejected: unknown Flow optimizer state field {key}")


def _validate_global_rng_state(rng_state):
    """Load all serialized RNG states into private objects only."""
    if not isinstance(rng_state, dict):
        raise ValueError("resume rejected: Flow global RNG state is invalid")
    try:
        if not isinstance(rng_state.get("torch"), torch.Tensor) or rng_state["torch"].dtype != torch.uint8:
            raise ValueError("torch RNG state is invalid")
        torch.Generator(device="cpu").set_state(rng_state["torch"])
        private_python = random.Random()
        private_python.setstate(rng_state["python"])
        private_numpy = np.random.RandomState()
        private_numpy.set_state(rng_state["numpy"])
        if "cuda" in rng_state:
            if not torch.cuda.is_available() or not isinstance(rng_state["cuda"], (list, tuple)):
                raise ValueError("CUDA RNG state cannot be validated on this runtime")
            for index, state in enumerate(rng_state["cuda"]):
                if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8:
                    raise ValueError(f"CUDA RNG state {index} is invalid")
                torch.Generator(device=f"cuda:{index}").set_state(state)
    except Exception as error:
        raise ValueError("resume rejected: Flow global RNG state is not loadable") from error


def _all_boundary_ids(frame):
    """Build one stable boundary identity from every declared boundary column."""
    columns = [column for column in frame.columns
               if str(column).lower() in {"entity_id", "session_id", "record_id", "scenario_id"}]
    if not columns:
        return None, []
    values = frame[columns].astype(object).where(frame[columns].notna(), "<NA>")
    combined = values.astype(str).agg("\x1f".join, axis=1).to_numpy(dtype=object)
    return combined, [str(column) for column in columns]


def _select_stratified(endpoints, segments, limit, seed):
    endpoints = np.asarray(endpoints, dtype=np.int64)
    limit = min(int(limit), len(endpoints))
    if limit <= 0:
        return np.empty(0, dtype=np.int64)
    if len(endpoints) <= limit:
        return endpoints.copy()
    rng = np.random.default_rng(seed)
    groups = []
    for left, right in segments:
        group = endpoints[(endpoints >= left) & (endpoints < right)]
        if len(group):
            groups.append(group)
    if not groups:
        return np.sort(rng.choice(endpoints, size=limit, replace=False))
    sizes = np.asarray([len(group) for group in groups], dtype=np.int64)
    raw = sizes.astype(np.float64) * float(limit) / float(sizes.sum())
    quotas = np.floor(raw).astype(np.int64)
    quotas = np.minimum(quotas, sizes)
    remaining = int(limit - quotas.sum())
    # Largest-remainder allocation with capacity checks handles more segments
    # than requested samples without ever producing negative quotas.
    order = np.argsort(-(raw - np.floor(raw)))
    while remaining:
        progressed = False
        for index in order:
            if quotas[index] < sizes[index]:
                quotas[index] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break
    chosen = [np.sort(rng.choice(group, size=int(quota), replace=False))
              for group, quota in zip(groups, quotas) if quota]
    result = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    if len(result) < limit:
        available = np.setdiff1d(endpoints, result, assume_unique=False)
        result = np.concatenate((result, np.sort(rng.choice(available, size=limit - len(result), replace=False))))
    return np.sort(result)


class MNRFViewProbe:
    """One independent probe run over frozen MNRF-Lite V2 views."""

    model_name = "MNRFViewProbe"

    def __init__(self, **kwargs):
        self.config = ViewProbeConfig.from_kwargs(**kwargs)
        self.device = self._select_device(self.config.device)
        configured_output = self.config.output_dir or self.config.resume_from
        self.output_dir = Path(configured_output) if configured_output else None
        self.trainer = None
        self.variable_names = None
        self.temporal_rule = None
        self.ranges = None
        self.segments = None
        self.source_identity = None
        self._current_test_endpoints = None
        self._test_length = None
        self._fitted = False
        self._gmm = None
        self._probe_stats = None
        self._feature_mean = None
        self._feature_scale = None
        self._noise_vectors = None
        self._run_identity = None
        self._writes_started = False

    def _log_event(self, event, **fields):
        """Emit a flushed, human-readable event without an extra model pass."""
        values = [f"event={event}"]
        for key, value in fields.items():
            if isinstance(value, float):
                value = f"{value:.6g}"
            values.append(f"{key}={value}")
        print("MNRF " + " ".join(values), flush=True)

    @staticmethod
    def _tree_size(path):
        total = 0
        count = 0
        root = Path(path)
        if not root.exists():
            return count, total
        for item in root.rglob("*"):
            if item.is_file() and not item.is_symlink():
                count += 1
                total += item.stat().st_size
        return count, total

    def _finalize_results_artifact(self):
        """Commit a validated result-only bundle after all scoring is done."""
        manifest_path = self.output_dir / "probe_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        metrics_path = self.output_dir / "results" / "metrics.json"
        aligned_path = self.output_dir / "results" / "aligned_scores.npz"
        if not metrics_path.is_file() or not aligned_path.is_file():
            raise ValueError("results artifact cannot be organized before metrics and scores exist")
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("status") != "complete":
            raise ValueError("results artifact cleanup requires complete metrics; recovery files are retained")
        # Validate the saved arrays before deleting any recovery material.
        self._log_event("result_validation_start")
        _validate_result_bundle(self.output_dir, allow_in_progress=True)
        self._log_event("result_validation_complete")
        known_root = {"probe_manifest.json", "run_status.json", "run.log", "training.jsonl",
                      "results", "checkpoints", "cache", "heads", "run_request.json",
                      "input_statistics.npz"}
        unknown_root = [child.name for child in self.output_dir.iterdir() if child.name not in known_root]
        allowed_results = {"metrics.json", "aligned_scores.npz", "test_nll.npy"}
        unknown_results = [child.name for child in (self.output_dir / "results").iterdir()
                           if child.name not in allowed_results]
        if unknown_root or unknown_results:
            raise ValueError("refusing results cleanup with unknown paths: "
                             + ", ".join(unknown_root + [f"results/{name}" for name in unknown_results]))
        # The CLI appends its temporary Tee log after fit() returns; reserve
        # the final path now so retention counts include the required log.
        (self.output_dir / "run.log").touch(exist_ok=True)
        before_count, before_bytes = self._tree_size(self.output_dir)
        manifest.update({
            "schema": PROBE_SCHEMA,
            "artifact_type": ARTIFACT_RESULTS,
            "artifact_policy": "results",
            "result_bundle_schema": RESULT_SCHEMA,
            "result_cleanup_status": "in_progress",
            "capabilities": {
                "validate_results": True, "recompute_metrics": True,
                "load_model": False, "resume_training": False,
                "reuse_source": False, "strict_common_recompute": False,
            },
            "retention_before": {"file_count": before_count, "bytes": before_bytes},
        })
        _atomic_json(manifest_path, manifest)
        self._log_event("artifact_cleanup_start", files_before=before_count, bytes_before=before_bytes)
        self._set_run_status("organizing_results", artifact_type=ARTIFACT_RESULTS,
                             result_cleanup_status="in_progress")
        keep = {
            self.output_dir / "probe_manifest.json",
            self.output_dir / "run_status.json",
            self.output_dir / "run.log",
            self.output_dir / "training.jsonl",
            self.output_dir / "results" / "metrics.json",
            self.output_dir / "results" / "aligned_scores.npz",
        }
        for name in ("checkpoints", "cache", "heads"):
            candidate = self.output_dir / name
            if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
                raise ValueError(f"refusing to organize unknown artifact path: {candidate}")
            if candidate.is_dir():
                shutil.rmtree(candidate)
        request_path = self.output_dir / "run_request.json"
        if request_path.exists():
            request_path.unlink()
        results_dir = self.output_dir / "results"
        for child in results_dir.iterdir():
            if child not in {metrics_path, aligned_path}:
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
        for child in self.output_dir.iterdir():
            if child in keep or child == self.output_dir / "results":
                continue
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        after_count, after_bytes = self._tree_size(self.output_dir)
        manifest = json.loads(manifest_path.read_text())
        manifest.update({
            "result_cleanup_status": "complete",
            "retention_after": {"file_count": after_count, "bytes": after_bytes},
            "retention_released_bytes": max(0, before_bytes - after_bytes),
            "retention_released_files": max(0, before_count - after_count),
        })
        _atomic_json(manifest_path, manifest)
        self._set_run_status("complete", artifact_type=ARTIFACT_RESULTS,
                             result_cleanup_status="complete",
                             artifact_policy="results",
                             retention_released_bytes=max(0, before_bytes - after_bytes))
        self._log_event("artifact_cleanup_complete", files_before=before_count,
                        files_after=after_count, bytes_before=before_bytes,
                        bytes_after=after_bytes,
                        released_bytes=max(0, before_bytes - after_bytes))

    @staticmethod
    def required_hyper_params():
        return {}

    @classmethod
    def from_kwargs(cls, **kwargs):
        return cls(**kwargs)

    @staticmethod
    def _select_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        selected = torch.device(device)
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested device {device}, but CUDA is unavailable")
        return selected

    def _prepare_dirs(self):
        if self.output_dir is None:
            self.output_dir = Path(
                "result/score/MNRF-ViewProbe/run-"
                + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                + f"-{os.getpid()}"
            )
        output_resolved = self.output_dir.resolve()
        if self.config.source_dir:
            source_resolved = Path(self.config.source_dir).resolve()
            if output_resolved == source_resolved or source_resolved in output_resolved.parents:
                raise ValueError("probe output must not equal or be inside its read-only source directory")
        resume = bool(self.config.resume_from)
        if resume and output_resolved != Path(self.config.resume_from).resolve():
            raise ValueError("resume_from must identify the configured output directory")
        if resume:
            if not self.output_dir.is_dir():
                raise FileNotFoundError(f"resume run directory does not exist: {self.output_dir}")
        else:
            self.output_dir.mkdir(parents=True, exist_ok=False)
        for name in ("checkpoints", "cache", "heads", "results"):
            directory = self.output_dir / name
            if resume:
                if not directory.is_dir():
                    raise ValueError(f"resume run is missing required directory: {directory}")
            else:
                directory.mkdir()

    def _set_run_status(self, status, reason=None, **extra):
        if self.output_dir is None:
            return
        payload = {
            "schema": PROBE_SCHEMA, "status": status,
            "task_name": self.config.task_name, "mode": self.config.mode,
            "run_identity": self._run_identity,
        }
        if reason is not None:
            payload["reason"] = str(reason)
        payload.update(extra)
        _atomic_json(self.output_dir / "run_status.json", payload)

    def _save_run_request(self, **extra):
        """Persist recovery identity before any stage can mutate artifacts."""
        if self.output_dir is None:
            return
        requested_stage = extra.pop("stage", "created")
        request_path = self.output_dir / "run_request.json"
        previous = {}
        if request_path.is_file():
            previous = json.loads(request_path.read_text(encoding="utf-8"))
            if previous.get("run_identity") not in {None, self._run_identity}:
                raise ValueError("run request identity cannot change during a stage update")
        payload = {
            "schema": PROBE_SCHEMA,
            "run_identity": self._run_identity,
            "task_name": self.config.task_name,
            "mode": self.config.mode,
            "config": self.config.to_dict(),
            "source_identity": self.source_identity,
            "stage": requested_stage,
        }
        # Stage transitions are mutable metadata; immutable initialization
        # identities (notably initial_encoder_sha256) must survive them.
        payload.update(previous)
        payload.update({"schema": PROBE_SCHEMA, "run_identity": self._run_identity,
                        "task_name": self.config.task_name, "mode": self.config.mode,
                        "config": self.config.to_dict(), "source_identity": self.source_identity})
        payload["stage"] = requested_stage
        payload.update(extra)
        _atomic_json(request_path, payload)

    @staticmethod
    def _features(frame):
        values, columns = MNRFLiteV2._features_from_frame(frame)
        return values, tuple(map(str, columns))

    def _protocol(self, frame, train_length):
        values, columns = self._features(frame)
        train_length = int(train_length)
        if train_length <= self.config.seq_len or train_length >= len(values):
            raise ValueError("train_length must leave complete purged train/validation/calibration and test windows")
        ranges = purged_ranges(train_length, self.config.seq_len)
        time_index = MNRFLiteV2._time_index(frame)
        boundary_ids, boundary_columns = _all_boundary_ids(frame)
        a, b = ranges["train"]
        train_time = TimeIndex(time_index.values_ns[a:b], time_index.kind, time_index.source_timezone)
        expected = infer_expected_interval(train_time, b - a, None if boundary_ids is None else boundary_ids[a:b])
        segments, continuity = continuous_segments(time_index, len(values), expected_interval=expected,
                                                   tolerance=0.05, boundary_ids=boundary_ids)
        ranges["segment_ranges"] = {name: intersect_segments(segments, *ranges[name]) for name in ("train", "validation", "calibration")}
        ranges["window_counts"] = {
            name: len(EndWindowDataset(values, *ranges[name], self.config.seq_len, ranges["segment_ranges"][name]))
            for name in ("train", "validation", "calibration")
        }
        if min(ranges["window_counts"].values()) < 1:
            raise ValueError(f"no complete internal windows: {ranges['window_counts']}")
        test_start = train_length
        test_segments, _ = continuous_segments(
            TimeIndex(time_index.values_ns[test_start:], time_index.kind, time_index.source_timezone),
            len(values) - test_start, expected_interval=expected, tolerance=0.05,
            boundary_ids=None if boundary_ids is None else boundary_ids[test_start:],
        )
        test_segments = [(left + test_start, right + test_start) for left, right in test_segments]
        test_endpoints = EndWindowDataset(values, test_start, len(values), self.config.seq_len, test_segments).endpoints
        if not len(test_endpoints):
            raise ValueError("test interval has no complete windows")
        self.variable_names = columns
        self.ranges = ranges
        self.segments = segments
        self.temporal_rule = {
            "version": "train-frozen-cadence-v1", "time_kind": time_index.kind,
            "expected_interval_ns": expected, "tolerance": 0.05,
            "entity_boundary_columns": boundary_columns,
            "fit_range": [int(a), int(b)],
        }
        self._current_test_endpoints = np.asarray(test_endpoints, dtype=np.int64)
        self._test_length = int(len(values) - train_length)
        self._time_values_ns = np.asarray(time_index.values_ns, dtype=np.int64)
        self._time_index_kind = time_index.kind
        self._source_timezone = time_index.source_timezone
        self.source_identity = {
            "schema": "mnrf-view-probe-source-identity-v2",
            "task_name": self.config.task_name,
            "data_source": self.config.data_source,
            "train_length": train_length,
            "input_length": int(len(values)),
            "variable_names": list(columns),
            "feature_data_sha256": _chunked_array_hash(values),
            "time_index": {
                "kind": time_index.kind, "source_timezone": time_index.source_timezone,
                "values_sha256": _chunked_array_hash(time_index.values_ns),
            },
            "boundary_columns": boundary_columns,
            "boundary_values_sha256": None if boundary_ids is None else _hash_strings(boundary_ids),
            "ranges": {name: list(ranges[name]) for name in ("train", "validation", "calibration")},
            "segment_ranges": {name: [list(item) for item in ranges["segment_ranges"][name]]
                               for name in ("train", "validation", "calibration")},
            "segments": [list(item) for item in segments],
            "test_segments": [list(item) for item in test_segments],
            "test_endpoints_sha256": _chunked_array_hash(test_endpoints),
            "test_endpoints_count": int(len(test_endpoints)),
            "temporal_rule": self.temporal_rule,
            "layout": self.config.layout.to_dict(),
            "seq_len": int(self.config.seq_len), "pattern_len": int(self.config.pattern_len),
        }
        return values, test_endpoints, continuity

    def _make_trainer(self, variables):
        config = self.config.mnrf
        config.task_name = self.config.task_name or "view_probe"
        config.output_dir = str(self.output_dir)
        config.resume_from = str(self.output_dir) if self.config.resume_from else ""
        trainer = MNRFTrainer(config, variables, self.device)
        return trainer

    def _stage1_with_progress(self, normalized):
        """Observe the existing stage-one calls without changing its schedule."""
        original = self.trainer._pretrain_batch
        state = {"steps": 0, "last_log": time.perf_counter(), "started": time.perf_counter()}

        def observed(windows, task, *args, **kwargs):
            loss = original(windows, task, *args, **kwargs)
            # Validation uses eval mode; only training calls count here.
            if self.trainer.encoder.training and self.trainer.auxiliary.training:
                state["steps"] += 1
                elapsed = time.perf_counter() - state["started"]
                due = (state["steps"] == 1 or state["steps"] % self.config.log_every_steps == 0
                       or time.perf_counter() - state["last_log"] >= self.config.log_every_seconds)
                if due:
                    self._log_event("stage1_batch", completed_steps=state["steps"], task=task,
                                    train_nll=float(loss.detach().cpu()),
                                    seconds_per_step=elapsed / state["steps"])
                    state["last_log"] = time.perf_counter()
            return loss

        self.trainer._pretrain_batch = observed
        try:
            return self.trainer.stage1(
                normalized, self.ranges["train"], self.ranges["validation"],
                self.output_dir / "checkpoints", self.segments,
                self.output_dir / "training.jsonl", self._run_identity,
            )
        finally:
            self.trainer._pretrain_batch = original

    def _normalized_memmap(self, values):
        started = time.perf_counter()
        self._log_event("feature_normalization_start", rows=len(values), train_rows=self.ranges["train"][1] - self.ranges["train"][0])
        path = self.output_dir / "cache" / "normalized_values.npy"
        mmap = np.lib.format.open_memmap(path.with_suffix(".partial.npy"), mode="w+", dtype=np.float32, shape=values.shape)
        self.trainer.fixed_input.fit(values[self.ranges["train"][0]:self.ranges["train"][1]])
        for offset in range(0, len(values), 65536):
            mmap[offset:offset + min(65536, len(values) - offset)] = self.trainer.fixed_input.transform_chunk(values[offset:offset + 65536])
        mmap.flush(); del mmap
        os.replace(path.with_suffix(".partial.npy"), path)
        self._log_event("feature_normalization_complete", rows=len(values), path=path,
                        elapsed_seconds=time.perf_counter() - started)
        return np.load(path, mmap_mode="r")

    def _cache_all(self, normalized, values, test_endpoints):
        cache = {}
        for name in ("train", "validation", "calibration"):
            started = time.perf_counter()
            self._log_event("cache_start", split=name)
            path, count, index_path = self.trainer.extract_cache(
                normalized, self.ranges[name], name, self.output_dir / "cache", self.segments
            )
            cache[name] = np.load(path, mmap_mode="r")
            self._log_event("cache_complete", split=name, windows=len(cache[name]), path=path,
                            elapsed_seconds=time.perf_counter() - started)
        test_path = self.output_dir / "cache" / "test_features.npy"
        test_started = time.perf_counter()
        test_indices = self.output_dir / "cache" / "test_window_end_indices.npy"
        # Test windows use endpoints from the post-boundary sequence and retain
        # full-series coordinates; build the cache with the same encoder path.
        selected = np.asarray(test_endpoints, dtype=np.int64)
        np.save(test_indices, selected)
        mmap = np.lib.format.open_memmap(test_path.with_suffix(".partial.npy"), mode="w+", dtype=np.float32,
                                         shape=(len(selected), self.trainer.variables, self.trainer.feature_dim))
        batch_size = min(self.config.batch_size, self.config.encoding_batch_size or self.config.batch_size)
        for start in range(0, len(selected), batch_size):
            endpoints = selected[start:start + batch_size]
            windows = np.stack([normalized[e - self.config.seq_len + 1:e + 1] for e in endpoints])
            feature = self.trainer.encoder.encode_views(torch.as_tensor(windows, device=self.device))["features"]
            mmap[start:start + len(endpoints)] = feature.detach().cpu().numpy()
        mmap.flush(); del mmap
        os.replace(test_path.with_suffix(".partial.npy"), test_path)
        cache["test"] = np.load(test_path, mmap_mode="r")
        self._log_event("cache_complete", split="test", windows=len(cache["test"]), path=test_path,
                        elapsed_seconds=time.perf_counter() - test_started)
        return cache, {name: self.output_dir / "cache" / f"{name}_window_end_indices.npy" for name in ("train", "validation", "calibration", "test")}

    def _cache_s_only(self, normalized, test_endpoints):
        """Export the S view directly, without training an encoder."""
        cache = {}
        endpoint_paths = {}
        for name, (start, end) in [(name, self.ranges[name]) for name in ("train", "validation", "calibration")]:
            started = time.perf_counter()
            self._log_event("cache_start", split=name)
            endpoints = window_endpoints(
                intersect_segments(self.ranges["segment_ranges"][name], start, end),
                self.config.seq_len,
            )
            path = self.output_dir / "cache" / f"{name}_features.npy"
            mmap = np.lib.format.open_memmap(path.with_suffix(".partial.npy"), mode="w+", dtype=np.float32,
                                             shape=(len(endpoints), len(self.variable_names), 1))
            for offset in range(0, len(endpoints), self.config.encoding_batch_size or self.config.batch_size):
                batch = endpoints[offset:offset + (self.config.encoding_batch_size or self.config.batch_size)]
                mmap[offset:offset + len(batch)] = np.asarray(normalized[batch], dtype=np.float32)[:, :, None]
            mmap.flush(); del mmap
            os.replace(path.with_suffix(".partial.npy"), path)
            index_path = self.output_dir / "cache" / f"{name}_window_end_indices.npy"
            np.save(index_path, endpoints)
            cache[name] = np.load(path, mmap_mode="r")
            endpoint_paths[name] = index_path
            self._log_event("cache_complete", split=name, windows=len(cache[name]), path=path,
                            elapsed_seconds=time.perf_counter() - started)
        endpoints = np.asarray(test_endpoints, dtype=np.int64)
        started = time.perf_counter()
        path = self.output_dir / "cache" / "test_features.npy"
        mmap = np.lib.format.open_memmap(path.with_suffix(".partial.npy"), mode="w+", dtype=np.float32,
                                         shape=(len(endpoints), len(self.variable_names), 1))
        batch_size = self.config.encoding_batch_size or self.config.batch_size
        for offset in range(0, len(endpoints), batch_size):
            batch = endpoints[offset:offset + batch_size]
            mmap[offset:offset + len(batch)] = np.asarray(normalized[batch], dtype=np.float32)[:, :, None]
        mmap.flush(); del mmap
        os.replace(path.with_suffix(".partial.npy"), path)
        index_path = self.output_dir / "cache" / "test_window_end_indices.npy"
        np.save(index_path, endpoints)
        cache["test"] = np.load(path, mmap_mode="r")
        endpoint_paths["test"] = index_path
        self._log_event("cache_complete", split="test", windows=len(cache["test"]), path=path,
                        elapsed_seconds=time.perf_counter() - started)
        return cache, endpoint_paths

    def _expected_endpoints(self, name):
        if name == "test":
            return np.asarray(self._current_test_endpoints, dtype=np.int64)
        segment_ranges = self.ranges["segment_ranges"][name]
        start, end = self.ranges[name]
        return window_endpoints(intersect_segments(segment_ranges, start, end), self.config.seq_len)

    def _validate_cache_entry(self, path, index_path, name, expected_sha=None, expected_index_sha=None):
        path, index_path = Path(path), Path(index_path)
        if not path.is_file() or not index_path.is_file():
            raise ValueError(f"view probe cache is missing: {path} or {index_path}")
        if expected_sha is not None and _sha256_file(path) != expected_sha:
            raise ValueError(f"view probe cache hash mismatch: {name}")
        if expected_index_sha is not None and _sha256_file(index_path) != expected_index_sha:
            raise ValueError(f"view probe endpoint hash mismatch: {name}")
        values = np.load(path, mmap_mode="r")
        endpoints = np.load(index_path, mmap_mode="r")
        expected_endpoints = np.asarray(self._expected_endpoints(name), dtype=np.int64)
        expected_shape = (len(expected_endpoints), len(self.variable_names), self.config.layout.total_dim)
        if values.dtype != np.dtype("float32") or values.shape != expected_shape:
            raise ValueError(f"view probe cache {name} shape/dtype mismatch: expected {expected_shape}/float32, got {values.shape}/{values.dtype}")
        try:
            endpoints = _strict_integer_indices(endpoints, f"view probe endpoint index {name}")
        except ValueError as error:
            raise ValueError(f"view probe endpoint index {name} has invalid shape or dtype: {error}") from error
        if len(endpoints) != len(expected_endpoints):
            raise ValueError(f"view probe endpoint index {name} has invalid length")
        if np.any(np.diff(endpoints) <= 0) or not np.array_equal(endpoints, expected_endpoints):
            raise ValueError(f"view probe endpoint index {name} is not the exact protocol window sequence")
        for start in range(0, len(values), 65536):
            if not np.isfinite(values[start:start + 65536]).all():
                raise ValueError(f"view probe cache {name} contains non-finite values")
        return values

    def _save_source_metadata(self, mode, cache, endpoint_paths, initial_state=None):
        payload = {
            "schema": PROBE_SCHEMA, "mode": mode, "task_name": self.config.task_name,
            "artifact_type": ARTIFACT_SOURCE, "head_type": None,
            "artifact_policy": "full",
            "config": self.config.to_dict(), "layout": self.config.layout.to_dict(),
            "variable_names": list(self.variable_names), "temporal_rule": self.temporal_rule,
            "ranges": {name: list(self.ranges[name]) for name in ("train", "validation", "calibration")},
            "segments": self.segments, "cache_paths": {name: str(path) for name, path in endpoint_paths.items()},
            "cache_sha256": {name: _sha256_file(self.output_dir / "cache" / f"{name}_features.npy") for name in cache},
            "endpoint_sha256": {name: _sha256_file(path) for name, path in endpoint_paths.items()},
            "feature_fingerprint": self.trainer.fingerprint(),
            "source_identity": self.source_identity,
            "cache_shapes": {name: list(cache[name].shape) for name in cache},
            "cache_dtypes": {name: str(cache[name].dtype) for name in cache},
            "cache_feature_dim": int(next(iter(cache.values())).shape[-1]),
            "input_statistics_sha256": _sha256_file(self.output_dir / "input_statistics.npz"),
            "raw_input_statistics_sha256": _sha256_file(self.output_dir / "input_statistics.npz"),
            "normalized_cache_sha256": (
                _sha256_file(self.output_dir / "cache" / "normalized_values.npy")
                if (self.output_dir / "cache" / "normalized_values.npy").is_file() else None
            ),
            "checkpoint_sha256": {
                name: _sha256_file(self.output_dir / "checkpoints" / name)
                for name in ("initial_encoder.pt", "best_encoder.pt")
                if (self.output_dir / "checkpoints" / name).is_file()
            },
            "initial_encoder_saved": initial_state is not None,
            "stage_status": "stage1_complete",
            "created_with": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        }
        _atomic_json(self.output_dir / "probe_manifest.json", payload)
        return payload

    def _save_probe_metadata(self, mode, source_manifest=None):
        payload = {
            "schema": PROBE_SCHEMA, "mode": mode, "artifact_type": ARTIFACT_HEAD,
            "artifact_policy": self.config.artifact_policy,
            "head_type": "flow" if mode == "flow" else None,
            "head_status": "in_progress" if mode == "flow" else None,
            "stage_status": "flow_training" if mode == "flow" else ("source_reused" if source_manifest else "cache_ready"),
            "config": self.config.to_dict(),
            "layout": self.config.layout.to_dict(), "variable_names": list(self.variable_names),
            "temporal_rule": self.temporal_rule,
            "source_dir": str(Path(self.config.source_dir).resolve()) if self.config.source_dir else "",
            "source_identity": self.source_identity,
            "source_feature_fingerprint": None if source_manifest is None else source_manifest.get("feature_fingerprint"),
            "selected_views": list(self.config.selected_views),
            "selected_indices": self.config.selected_indices, "selected_dim": self.config.selected_dim,
            "cache_feature_dim": self.config.selected_dim if mode == "s_only" and source_manifest is None else None,
            "raw_input_statistics_sha256": _sha256_file(self.output_dir / "input_statistics.npz"),
        }
        if source_manifest is not None:
            payload["source_cache_sha256"] = source_manifest.get("cache_sha256", {})
            payload["source_endpoint_sha256"] = source_manifest.get("endpoint_sha256", {})
            payload["source_gmm_fit_endpoints_sha256"] = source_manifest.get("gmm_fit_endpoints_sha256")
        manifest_path = self.output_dir / "probe_manifest.json"
        # A resumed head already owns immutable bindings (feature statistics,
        # stage identity, cache hashes, and completion facts).  Merge the
        # current view instead of recreating a weaker initial manifest.
        if self.config.resume_from and manifest_path.is_file():
            existing = json.loads(manifest_path.read_text())
            for key, value in payload.items():
                if value is not None or key not in existing:
                    existing[key] = value
            payload = existing
        _atomic_json(manifest_path, payload)
        return payload

    def _bind_local_cache_metadata(self, cache, endpoint_paths):
        manifest_path = self.output_dir / "probe_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update({
            "cache_sha256": {name: _sha256_file(self.output_dir / "cache" / f"{name}_features.npy") for name in cache},
            "endpoint_sha256": {name: _sha256_file(path) for name, path in endpoint_paths.items()},
            "cache_shapes": {name: list(cache[name].shape) for name in cache},
            "cache_dtypes": {name: str(cache[name].dtype) for name in cache},
            "cache_feature_dim": int(next(iter(cache.values())).shape[-1]),
        })
        _atomic_json(manifest_path, manifest)

    def _load_source(self, source_dir):
        source = Path(source_dir).resolve()
        manifest_path = source / "probe_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("view probe source has no manifest; re-export stage one")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != PROBE_SCHEMA:
            raise ValueError("view probe source schema is unsupported; re-export stage one")
        current_identity = self.source_identity
        source_identity = manifest.get("source_identity")
        if source_identity is None:
            raise ValueError("view probe source has no complete source identity; re-export stage one")
        if source_identity != current_identity:
            raise ValueError("view probe source data/time/boundary/train split identity differs")
        require_layout(manifest.get("layout"), self.config.layout, "view probe source")
        source_config = dict(manifest.get("config", {}))
        for key in ("resolved_layout", "selected_indices", "selected_dim"):
            source_config.pop(key, None)
        try:
            source_cfg = ViewProbeConfig.from_kwargs(**source_config)
        except (TypeError, ValueError) as error:
            raise ValueError(f"view probe source configuration is invalid: {error}") from error
        require_layout(source_cfg.layout, self.config.layout, "view probe source config")
        if source_cfg.task_name != self.config.task_name:
            raise ValueError("view probe source task identity differs")
        if source_cfg.data_source != self.config.data_source:
            raise ValueError("view probe source data source differs")
        if source_cfg.seq_len != self.config.seq_len or source_cfg.pattern_len != self.config.pattern_len:
            raise ValueError("view probe source window configuration differs")
        if manifest.get("temporal_rule") != self.temporal_rule:
            raise ValueError("view probe source temporal protocol differs")
        if tuple(manifest.get("variable_names", ())) != tuple(self.variable_names):
            raise ValueError("view probe source variable order differs")
        if (manifest.get("mode") not in {"stage1", "pretrained", "random"}
                or manifest.get("artifact_type") not in {ARTIFACT_SOURCE, ARTIFACT_HEAD}
                or (manifest.get("artifact_type") == ARTIFACT_SOURCE
                    and manifest.get("stage_status") != "stage1_complete")
                or (manifest.get("artifact_type") == ARTIFACT_HEAD
                    and manifest.get("stage_status") != "complete")):
            raise ValueError("view probe source is not a stage-1 encoder artifact")
        self.trainer = self._make_trainer(len(self.variable_names))
        checkpoint = source / "checkpoints" / ("initial_encoder.pt" if self.config.mode == "random_gmm" else "best_encoder.pt")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"source encoder checkpoint missing: {checkpoint}")
        declared_checkpoint_hash = manifest.get("checkpoint_sha256", {}).get(checkpoint.name)
        if declared_checkpoint_hash is None or _sha256_file(checkpoint) != declared_checkpoint_hash:
            raise ValueError("view probe source encoder checkpoint hash mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        encoder_state = state.get("encoder", state)
        self.trainer.encoder.load_state_dict(encoder_state)
        self.trainer.encoder.eval()
        with np.load(source / "input_statistics.npz", allow_pickle=False) as stats:
            self.trainer.fixed_input.mean_ = stats["mean"].astype(np.float32)
            self.trainer.fixed_input.scale_ = stats["scale"].astype(np.float32)
        if _sha256_file(source / "input_statistics.npz") != manifest.get("input_statistics_sha256"):
            raise ValueError("view probe source input statistics hash mismatch")
        if np.any(~np.isfinite(self.trainer.fixed_input.mean_)) or np.any(~np.isfinite(self.trainer.fixed_input.scale_)):
            raise ValueError("view probe source input statistics are non-finite")
        if np.any(self.trainer.fixed_input.scale_ <= 0):
            raise ValueError("view probe source input statistics contain non-positive scales")
        if self.trainer.fingerprint() != manifest.get("feature_fingerprint"):
            raise ValueError("view probe source encoder/input-statistics fingerprint mismatch")
        cache = {}
        for name in ("train", "validation", "calibration", "test"):
            path = source / "cache" / f"{name}_features.npy"
            endpoint_path = source / "cache" / f"{name}_window_end_indices.npy"
            declared_cache_hash = manifest.get("cache_sha256", {}).get(name)
            declared_endpoint_hash = manifest.get("endpoint_sha256", {}).get(name)
            if not declared_cache_hash or not declared_endpoint_hash:
                raise ValueError(f"view probe source manifest is missing immutable {name} hashes")
            cache[name] = self._validate_cache_entry(
                path, endpoint_path, name,
                declared_cache_hash, declared_endpoint_hash,
            )
        declared_sample_hash = manifest.get("gmm_fit_endpoints_sha256")
        if manifest.get("head_type") == "gmm" and not declared_sample_hash:
            raise ValueError("view probe source GMM manifest is missing fitting endpoint hash")
        if declared_sample_hash is not None:
            sample_path = source / "heads" / "gmm_fit_endpoints.npy"
            if not sample_path.is_file() or _sha256_file(sample_path) != declared_sample_hash:
                raise ValueError("view probe source GMM fitting endpoint sample is missing or changed")
            if len(np.load(sample_path, mmap_mode="r")) > self.config.gmm_fit_max_windows:
                raise ValueError("view probe source GMM fitting sample exceeds the requested head limit")
        self._source_config = source_cfg
        return source, manifest, cache

    def _fit_gmm(self, cache, source_manifest=None):
        train = cache["train"]
        self._log_event("gmm_start", variables=train.shape[1], feature_dim=self.config.selected_dim,
                        sample_limit=min(self.config.gmm_fit_max_windows, len(train)),
                        components=self.config.gmm_components,
                        covariance_type=self.config.gmm_covariance_type)
        statistics_path = self.output_dir / "heads" / "feature_statistics.npz"
        if self.config.resume_from and statistics_path.is_file():
            with np.load(statistics_path, allow_pickle=False) as saved_statistics:
                saved_mean = np.asarray(saved_statistics["mean"], dtype=np.float32)
                saved_scale = np.asarray(saved_statistics["scale"], dtype=np.float32)
            self.trainer.fixed_feature.mean_ = saved_mean
            self.trainer.fixed_feature.scale_ = saved_scale
            expected_statistics_shape = (train.shape[1], train.shape[2])
            if (saved_mean.shape != expected_statistics_shape or saved_scale.shape != expected_statistics_shape
                    or not np.isfinite(saved_mean).all() or not np.isfinite(saved_scale).all()
                    or np.any(saved_scale < self.config.feature_std_floor)):
                raise ValueError("GMM resume rejected: feature statistics are invalid")
            computed_mean = saved_mean
            computed_scale = saved_scale
        else:
            with threadpool_limits(limits=self.config.cpu_threads):
                self.trainer.fixed_feature.fit(train, chunk_size=self.config.cache_chunk_size)
            computed_mean = np.asarray(self.trainer.fixed_feature.mean_, dtype=np.float32)
            computed_scale = np.asarray(self.trainer.fixed_feature.scale_, dtype=np.float32)
            np.savez(statistics_path, mean=computed_mean, scale=computed_scale)
        manifest_path = self.output_dir / "probe_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            manifest["feature_statistics_sha256"] = _sha256_file(statistics_path)
            _atomic_json(manifest_path, manifest)
        selected = np.asarray(self.config.selected_indices, dtype=np.int64)
        endpoints_path = self.output_dir / "cache" / "train_window_end_indices.npy"
        if source_manifest is not None:
            endpoints_path = Path(self.config.source_dir) / "cache" / "train_window_end_indices.npy"
        endpoints = np.load(endpoints_path, mmap_mode="r")
        source_sample = None if source_manifest is None else Path(self.config.source_dir) / "heads" / "gmm_fit_endpoints.npy"
        if source_sample is not None and source_sample.is_file():
            fit_indices = _strict_integer_indices(np.load(source_sample), "GMM fitting endpoint sample")
        else:
            fit_indices = _select_stratified(endpoints, self.segments, min(self.config.gmm_fit_max_windows, len(train)), self.config.seed)
        if fit_indices.ndim != 1 or len(fit_indices) == 0:
            raise ValueError("GMM fitting endpoint sample must be a non-empty one-dimensional array")
        positions = np.searchsorted(endpoints, fit_indices)
        if (np.any(positions >= len(endpoints))
                or not np.array_equal(endpoints[positions], fit_indices)):
            raise ValueError("GMM fitting endpoint sample is not a subset of the source train endpoints")
        sample_path = self.output_dir / "heads" / "gmm_fit_endpoints.npy"
        np.save(sample_path, fit_indices)
        sample_hash = _sha256_file(sample_path)
        self._noise_vectors = self._fixed_noise_vectors()
        np.save(self.output_dir / "heads" / "fixed_noise.npy", self._noise_vectors)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            manifest["gmm_fit_endpoints_sha256"] = sample_hash
            manifest["gmm_fit_endpoint_count"] = int(len(fit_indices))
            manifest["fixed_noise_sha256"] = _sha256_file(self.output_dir / "heads" / "fixed_noise.npy")
            _atomic_json(manifest_path, manifest)
        variable_dir = self.output_dir / "heads" / "gmm_variables"
        variable_dir.mkdir(exist_ok=True)
        progress_path = self.output_dir / "heads" / "gmm_progress.json"
        fit_config = self.config.to_dict()
        fit_config.pop("output_dir", None)
        fit_config.pop("resume_from", None)
        source_cache_identity = None if source_manifest is None else {
            "source_dir": str(Path(self.config.source_dir).resolve()),
            "source_identity": self.source_identity,
            "cache_sha256": source_manifest.get("cache_sha256"),
            "endpoint_sha256": source_manifest.get("endpoint_sha256"),
        }
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest = {}
        progress_identity = _json_hash({"config": fit_config, "sample_hash": sample_hash,
                                        "statistics_hash": _sha256_file(statistics_path),
                                        "noise_hash": _sha256_file(self.output_dir / "heads" / "fixed_noise.npy"),
                                        "cache_hashes": (manifest.get("source_cache_sha256") if source_manifest is not None
                                                         else manifest.get("cache_sha256")),
                                        "endpoint_hashes": (manifest.get("source_endpoint_sha256") if source_manifest is not None
                                                            else manifest.get("endpoint_sha256")),
                                        "selected_indices": self.config.selected_indices,
                                        "source": source_cache_identity})
        progress = {"schema": PROBE_SCHEMA, "identity": progress_identity, "completed": {}}
        if self.config.resume_from:
            if not progress_path.is_file():
                raise ValueError("GMM resume rejected: gmm_progress.json is missing")
            progress = json.loads(progress_path.read_text())
            if progress.get("identity") != progress_identity:
                raise ValueError("GMM resume rejected: head configuration or fitting sample differs")
        else:
            _atomic_json(progress_path, progress)
        request_path = self.output_dir / "run_request.json"
        if request_path.is_file():
            request = json.loads(request_path.read_text())
            request["gmm_progress_identity"] = progress_identity
            request["stage"] = "gmm_fitting"
            _atomic_json(request_path, request)
        self._gmm = [None] * train.shape[1]
        statuses = [None] * train.shape[1]
        limited_info = None
        with threadpool_limits(limits=self.config.cpu_threads):
            limited_info = threadpool_info()
            for variable in range(train.shape[1]):
                entry = progress.get("completed", {}).get(str(variable))
                model_path = variable_dir / f"variable_{variable}.pkl"
                if entry and (not model_path.is_file() or _sha256_file(model_path) != entry.get("model_sha256")):
                    raise ValueError(f"GMM resume rejected: completed variable {variable} is missing or changed")
                if entry and model_path.is_file():
                    self._log_event("gmm_variable_reused", variable=variable, path=model_path)
                    with open(model_path, "rb") as handle:
                        model = pickle.load(handle)
                    statuses[variable] = entry["status"]
                    self._gmm[variable] = model
                    continue
                values = np.asarray(train[positions, variable][:, selected], dtype=np.float32)
                mean = self.trainer.fixed_feature.mean_[variable, selected]
                scale = self.trainer.fixed_feature.scale_[variable, selected]
                values = (values - mean) / scale
                model = GaussianMixture(
                    n_components=self.config.gmm_components,
                    covariance_type=self.config.gmm_covariance_type,
                    reg_covar=self.config.gmm_reg_covar,
                    n_init=self.config.gmm_n_init,
                    max_iter=self.config.gmm_max_iter,
                    random_state=self.config.seed,
                    verbose=2,
                    verbose_interval=self.config.gmm_log_every_iters,
                )
                variable_start = time.perf_counter()
                self._log_event("gmm_variable_start", variable=variable, samples=len(values), dimension=values.shape[1])
                model.fit(values)
                if not model.converged_:
                    raise RuntimeError(f"GMM variable {variable} did not converge")
                self._gmm[variable] = model
                dimension = int(model.means_.shape[1])
                covariance_type = model.covariance_type
                if covariance_type == "full":
                    covariance_count = model.weights_.size * dimension * (dimension + 1) // 2
                elif covariance_type == "tied":
                    covariance_count = dimension * (dimension + 1) // 2
                elif covariance_type == "diag":
                    covariance_count = model.weights_.size * dimension
                elif covariance_type == "spherical":
                    covariance_count = model.weights_.size
                else:  # Config validation makes this unreachable.
                    raise ValueError(f"unsupported GMM covariance type: {covariance_type}")
                free_count = int((model.weights_.size - 1) + model.means_.size + covariance_count)
                storage_count = int(model.weights_.size + model.means_.size + model.covariances_.size)
                status = {"variable": variable, "converged": bool(model.converged_),
                          "n_iter": int(model.n_iter_), "lower_bound": float(model.lower_bound_),
                          "sample_count": len(values), "n_features_in": dimension,
                          "n_components": int(model.n_components),
                          "covariance_type": model.covariance_type,
                          "parameter_count_free": free_count,
                          "parameter_count_storage": storage_count}
                statuses[variable] = status
                _atomic_pickle(model_path, model)
                progress.setdefault("completed", {})[str(variable)] = {
                    "status": status, "model_sha256": _sha256_file(model_path),
                }
                statuses[variable]["model_sha256"] = progress["completed"][str(variable)]["model_sha256"]
                _atomic_json(progress_path, progress)
                self._log_event("gmm_variable_complete", variable=variable, n_iter=model.n_iter_,
                                converged=model.converged_, elapsed_seconds=time.perf_counter() - variable_start)
        if any(model is None for model in self._gmm) or any(status is None for status in statuses):
            raise RuntimeError("GMM fitting did not produce every variable head")
        with open(self.output_dir / "heads" / "gmm_models.pkl", "wb") as handle:
            pickle.dump(self._gmm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        gmm_models_hash = _sha256_file(self.output_dir / "heads" / "gmm_models.pkl")
        _atomic_json(self.output_dir / "heads" / "gmm_status.json", {
            "config": self.config.to_dict(), "sklearn_version": __import__("sklearn").__version__,
            "fit_endpoint_count": int(len(fit_indices)), "fit_endpoint_sha256": sample_hash,
            "fixed_noise_shape": list(self._noise_vectors.shape),
            "fixed_noise_sha256": _sha256_file(self.output_dir / "heads" / "fixed_noise.npy"),
            "threadpool_info": limited_info,
            "variables": statuses,
            "head_parameter_count": sum(item["parameter_count_free"] for item in statuses),
            "head_storage_parameter_count": sum(item["parameter_count_storage"] for item in statuses),
            "gmm_models_sha256": gmm_models_hash,
        })
        manifest_path = self.output_dir / "probe_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            manifest["gmm_fit_endpoints_sha256"] = sample_hash
            manifest["gmm_fit_endpoint_count"] = int(len(fit_indices))
            _atomic_json(manifest_path, manifest)
        self._log_event("gmm_complete", variables=len(self._gmm), sample_count=len(fit_indices))

    def _fixed_noise_vectors(self):
        if self._noise_vectors is None:
            generator = torch.Generator(device="cpu").manual_seed(self.config.noise_seed)
            self._noise_vectors = (torch.randn(
                4, self.config.layout.total_dim, generator=generator, dtype=torch.float32
            ) * self.config.train_noise_std).numpy()
        return np.asarray(self._noise_vectors, dtype=np.float32)

    def _gmm_scores(self, features, output_path=None, split_name="score"):
        if self._gmm is None:
            with open(self.output_dir / "heads" / "gmm_models.pkl", "rb") as handle:
                self._gmm = pickle.load(handle)
        selected = np.asarray(self.config.selected_indices, dtype=np.int64)
        if self.trainer is not None:
            feature_mean = self.trainer.fixed_feature.mean_
            feature_scale = self.trainer.fixed_feature.scale_
        else:
            feature_mean = self._feature_mean
            feature_scale = self._feature_scale
        output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(len(features), features.shape[1])) if output_path else np.empty((len(features), features.shape[1]), dtype=np.float32)
        output[:] = 0.0
        noises = self._fixed_noise_vectors()
        with threadpool_limits(limits=self.config.cpu_threads):
            for noise_index, noise in enumerate(noises):
                for start in range(0, len(features), self.config.batch_size):
                    stop = min(len(features), start + self.config.batch_size)
                    for variable, model in enumerate(self._gmm):
                        values = np.asarray(features[start:stop, variable][:, selected], dtype=np.float32)
                        values = (values - feature_mean[variable, selected]) / feature_scale[variable, selected]
                        values = values + noise[selected]
                        output[start:stop, variable] += -model.score_samples(values) / len(noises)
                    batch_number = start // self.config.batch_size + 1
                    if noise_index == len(noises) - 1 and (batch_number == 1 or stop == len(features)
                                                           or batch_number % self.config.log_every_steps == 0):
                        self._log_event("score_progress", split=split_name, windows=stop, total=len(features))
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite GMM NLL")
        if hasattr(output, "flush"):
            output.flush()
        return output

    def _calibrate(self, cache):
        started = time.perf_counter()
        self._log_event("calibration_start", validation_windows=len(cache["validation"]), calibration_windows=len(cache["calibration"]))
        validation = self._gmm_scores(cache["validation"], self.output_dir / "heads" / "validation_nll.npy", "validation")
        if not np.isfinite(validation).all():
            raise FloatingPointError("non-finite validation NLL")
        center = np.median(validation, axis=0).astype(np.float32)
        scale = np.maximum(1.4826 * np.median(np.abs(validation - center), axis=0), self.config.score_mad_floor).astype(np.float32)
        calibration = self._gmm_scores(cache["calibration"], self.output_dir / "heads" / "calibration_nll.npy", "calibration")
        aggregate = np.max((calibration - center) / scale, axis=1)
        threshold = float(np.quantile(aggregate, self.config.calibration_quantile))
        if not np.isfinite(threshold) or np.any(scale <= 0):
            raise FloatingPointError("invalid GMM calibration scale or threshold")
        self._probe_stats = {"center": center, "scale": scale, "threshold": threshold}
        np.savez(self.output_dir / "heads" / "calibration.npz", center=center, scale=scale,
                 threshold=np.asarray(threshold), quantile=np.asarray(self.config.calibration_quantile))
        self._log_event("calibration_complete", threshold=threshold, quantile=self.config.calibration_quantile,
                        elapsed_seconds=time.perf_counter() - started)

    def _write_scores(self, cache, test_endpoints, labels):
        started = time.perf_counter()
        self._log_event("scoring_start", test_windows=len(cache["test"]))
        inference_start = time.perf_counter()
        nll = self._gmm_scores(cache["test"], self.output_dir / "results" / "test_nll.npy", "test")
        score = np.max((nll - self._probe_stats["center"]) / self._probe_stats["scale"], axis=1)
        prediction = (score > self._probe_stats["threshold"]).astype(np.int64)
        local = np.asarray(test_endpoints, dtype=np.int64) - int(self._test_offset)
        if np.any(local < 0) or np.any(local >= int(self._test_length)):
            raise ValueError("test endpoint cannot be aligned to the local test label range")
        actual = None
        if labels is not None:
            labels = np.asarray(labels).reshape(-1)
            if len(labels) != int(self._test_length):
                raise ValueError(f"test labels must be local and have length {self._test_length}, got {len(labels)}")
            actual = labels[local]
        gmm_status = json.loads((self.output_dir / "heads" / "gmm_status.json").read_text())
        metrics = {"status": "complete", "head": "gmm", "auc_roc": None, "auc_pr": None,
                   "threshold": self._probe_stats["threshold"], "valid_points": len(score),
                   "selected_views": list(self.config.selected_views),
                   "selected_indices": self.config.selected_indices, "selected_dim": self.config.selected_dim,
                   "head_parameter_count": gmm_status["head_parameter_count"],
                   "fit_time_seconds": float(getattr(self, "_fit_elapsed", 0.0)),
                   "inference_time_seconds": float(time.perf_counter() - inference_start),
                   "gmm_fit_endpoint_count": gmm_status["fit_endpoint_count"]}
        from sklearn.metrics import average_precision_score, roc_auc_score
        metrics["coverage"] = float(len(score) / max(int(self._test_length), 1))
        metrics["excluded_points"] = int(self._test_length - len(score))
        if actual is not None:
            if np.unique(actual).size == 2:
                metrics["auc_roc"] = float(roc_auc_score(actual, score))
            else:
                metrics["status"] = "incomplete"
                metrics["auc_undefined_reason"] = "valid test labels contain one class"
            if np.unique(actual).size >= 1:
                metrics["auc_pr"] = float(average_precision_score(actual, score))
        aligned_scores = np.full(int(self._test_length), np.nan, dtype=np.float32)
        aligned_predictions = np.full(int(self._test_length), -1, dtype=np.int64)
        valid_mask = np.zeros(int(self._test_length), dtype=bool)
        aligned_scores[local] = score
        aligned_predictions[local] = prediction
        valid_mask[local] = True
        payload = {"scores": score, "predictions": prediction,
                   "time_indices": np.asarray(test_endpoints),
                   "time_values_ns": np.asarray(self._time_values_ns[test_endpoints], dtype=np.int64),
                   "local_indices": local, "valid_mask": valid_mask, "aligned_scores": aligned_scores,
                   "aligned_predictions": aligned_predictions,
                   "threshold": np.asarray(self._probe_stats["threshold"])}
        if actual is not None:
            payload["actual"] = actual
            payload["aligned_actual"] = np.full(int(self._test_length), -1, dtype=np.asarray(actual).dtype)
            payload["aligned_actual"][local] = actual
        np.savez_compressed(self.output_dir / "results" / "aligned_scores.npz", **payload)
        _atomic_json(self.output_dir / "results" / "metrics.json", metrics)
        self._log_event("scoring_complete", valid_points=len(score), auc_roc=metrics.get("auc_roc"),
                        auc_pr=metrics.get("auc_pr"), threshold=metrics.get("threshold"),
                        elapsed_seconds=time.perf_counter() - started,
                        result_path=self.output_dir / "results" / "metrics.json")
        return metrics

    def _set_protocol_identity(self, frame, train_length):
        values, test_endpoints, continuity = self._protocol(frame, int(train_length))
        self._log_event("protocol_summary", train_rows=self.ranges["train"][1] - self.ranges["train"][0],
                        validation_rows=self.ranges["validation"][1] - self.ranges["validation"][0],
                        calibration_rows=self.ranges["calibration"][1] - self.ranges["calibration"][0],
                        train_windows=self.ranges["window_counts"]["train"],
                        validation_windows=self.ranges["window_counts"]["validation"],
                        calibration_windows=self.ranges["window_counts"]["calibration"],
                        test_windows=len(test_endpoints), segments=len(self.segments),
                        temporal_rule=self.temporal_rule)
        self._test_offset = int(train_length)
        self._run_identity = _json_hash({
            "source_identity": self.source_identity,
            "config": _runtime_free_config(self.config),
        })
        return values, test_endpoints, continuity

    def _validate_resume_preflight(self, frame, train_length):
        """Read-only validation performed before any resume file is written."""
        if not self.config.resume_from or self.output_dir is None:
            return
        if self.config.mode == "full_gmm" and self.config.source_dir:
            raise ValueError("full_gmm with source_dir is unsupported; use select_gmm or s_only")
        if not self.output_dir.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {self.output_dir}")
        required = ("run_status.json", "checkpoints", "cache", "heads", "results")
        missing = [name for name in required if not (self.output_dir / name).exists()]
        if missing:
            raise ValueError("resume preflight failed; missing artifacts: " + ", ".join(missing))
        manifest_path = self.output_dir / "probe_manifest.json"
        request_path = self.output_dir / "run_request.json"
        if not manifest_path.is_file() and not request_path.is_file():
            raise ValueError("resume preflight failed: run identity metadata is missing")
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        request = json.loads(request_path.read_text()) if request_path.is_file() else None
        status = json.loads((self.output_dir / "run_status.json").read_text())
        if ((manifest or {}).get("artifact_type") == ARTIFACT_RESULTS
                or (manifest or {}).get("artifact_policy") == "results"):
            raise ValueError("resume rejected: results artifact is metrics-only and cannot resume training")
        identity_record = manifest or request
        if identity_record.get("schema") != PROBE_SCHEMA or status.get("schema") != PROBE_SCHEMA:
            raise ValueError("resume preflight failed: unsupported artifact schema")
        if status.get("run_identity") != self._run_identity:
            raise ValueError("resume rejected: run configuration or source identity differs")
        manifest_config = identity_record.get("config")
        if manifest_config is None or _runtime_free_config(manifest_config) != _runtime_free_config(self.config):
            raise ValueError("resume rejected: persisted configuration differs")
        if identity_record.get("source_identity") != self.source_identity:
            raise ValueError("resume rejected: source identity differs")
        if manifest is not None and manifest.get("artifact_type") not in {ARTIFACT_SOURCE, ARTIFACT_HEAD}:
            raise ValueError("resume rejected: unknown artifact type")
        if status.get("status") == "complete":
            raise ValueError("resume rejected: run is already complete")
        # A Flow run that has entered stage 2 is never silently restarted.  A
        # stage-2 checkpoint is the durable proof of its progress and must be
        # validated before _prepare_dirs or any status/cache write occurs.
        if self.config.mode == "flow":
            stage2_path = self.output_dir / "checkpoints" / "stage2_latest.pt"
            request_stage = (request or {}).get("stage")
            flow_started = (
                status.get("status") == "flow_training"
                or (manifest or {}).get("head_type") == "flow"
                or (manifest or {}).get("stage_status") == "flow_training"
                or request_stage == "flow_training"
            )
            if flow_started and not stage2_path.is_file():
                raise ValueError("resume rejected: Flow stage has started but stage2_latest.pt is missing")
            if stage2_path.is_file():
                try:
                    stage2 = torch.load(stage2_path, map_location="cpu", weights_only=False)
                except Exception as error:
                    raise ValueError("resume rejected: stage2_latest.pt cannot be loaded") from error
                required = {"schema_version", "stage", "resume_id", "next_epoch", "stage_status",
                            "completion_reason", "best_epoch", "best", "best_state", "patience",
                            "history", "completed_steps", "flow", "optimizer", "sampler_rng",
                            "rng", "stage2_identity"}
                if not isinstance(stage2, dict) or not required.issubset(stage2):
                    missing = sorted(required.difference(stage2 if isinstance(stage2, dict) else {}))
                    raise ValueError(f"resume rejected: incomplete Flow checkpoint fields {missing}")
                if (stage2.get("schema_version") != 2 or stage2.get("stage") != "stage2"
                        or stage2.get("resume_id") != self._run_identity
                        or isinstance(stage2.get("next_epoch"), bool)
                        or not isinstance(stage2.get("next_epoch"), int) or stage2["next_epoch"] < 1
                        or stage2.get("stage_status") not in {"in_progress", "completed"}
                        or not isinstance(stage2.get("completed_steps"), int) or stage2["completed_steps"] < 0
                        or not isinstance(stage2.get("sampler_rng"), torch.Tensor)
                        or stage2["sampler_rng"].dtype != torch.uint8):
                    raise ValueError("resume rejected: invalid Flow checkpoint phase or RNG contract")
                expected_shape = (len(self.variable_names), self.config.layout.total_dim)
                if any((not isinstance(stage2[name], dict)) for name in ("flow", "best_state")):
                    raise ValueError("resume rejected: Flow checkpoint state payload is invalid")
                if stage2["sampler_rng"].numel() == 0:
                    raise ValueError("resume rejected: Flow sampler RNG state is empty")
                try:
                    torch.Generator(device="cpu").set_state(stage2["sampler_rng"])
                except Exception as error:
                    raise ValueError("resume rejected: Flow sampler RNG state is not loadable") from error
                rng_state = stage2.get("rng")
                _validate_global_rng_state(rng_state)
                if (not np.isfinite(float(stage2.get("best", np.inf)))
                        or isinstance(stage2.get("patience"), bool)
                        or not isinstance(stage2.get("patience"), int)
                        or not isinstance(stage2.get("history"), list)):
                    raise ValueError("resume rejected: Flow best/patience/history state is invalid")
                process_rng = _capture_rng()
                try:
                    trainer_probe = self._make_trainer(len(self.variable_names))
                    _validate_checkpoint_state_dict(stage2["flow"], trainer_probe.flow, "flow")
                    _validate_checkpoint_state_dict(stage2["best_state"], trainer_probe.flow, "best_state")
                    optimizer_probe = torch.optim.AdamW(
                        trainer_probe.flow.parameters(), lr=self.config.mnrf.learning_rate,
                        weight_decay=self.config.mnrf.weight_decay,
                    )
                    _validate_flow_optimizer_state(stage2["optimizer"], trainer_probe, optimizer_probe)
                finally:
                    _restore_rng(process_rng)
                declared_identity = (manifest or {}).get("stage2_identity")
                if declared_identity is not None and stage2.get("stage2_identity") != declared_identity:
                    raise ValueError("resume rejected: Flow stage2 identity differs from manifest")
                if stage2["stage_status"] == "completed" and stage2.get("completion_reason") not in {"early_stopping", "epoch_limit"}:
                    raise ValueError("resume rejected: completed Flow checkpoint has no valid completion reason")
                feature_stats_path = self.output_dir / "heads" / "feature_statistics.npz"
                declared_feature_stats = manifest.get("feature_statistics_sha256")
                if (not declared_feature_stats or not feature_stats_path.is_file()
                        or _sha256_file(feature_stats_path) != declared_feature_stats):
                    raise ValueError("resume rejected: Flow feature statistics binding is missing or changed")
                with np.load(feature_stats_path, allow_pickle=False) as stats:
                    mean = np.asarray(stats["mean"])
                    scale = np.asarray(stats["scale"])
                if (mean.shape != expected_shape or scale.shape != expected_shape
                        or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                        or np.any(scale <= 0)):
                    raise ValueError(f"resume rejected: Flow feature statistics shape/value differs; expected {expected_shape}")
        # Validate an explicitly referenced source without creating any target
        # directories or rewriting any target artifact.
        if self.config.source_dir:
            if not Path(self.config.source_dir).is_dir():
                raise ValueError("resume rejected: source_dir is not a directory")
            source, source_manifest, _ = self._load_source(self.config.source_dir)
            for field in ("source_cache_sha256", "source_endpoint_sha256", "source_feature_fingerprint"):
                expected = (manifest or {}).get(field)
                actual = (source_manifest.get("cache_sha256") if field == "source_cache_sha256"
                          else source_manifest.get("endpoint_sha256") if field == "source_endpoint_sha256"
                          else source_manifest.get("feature_fingerprint"))
                if expected != actual:
                    raise ValueError(f"resume rejected: source {field} differs")
        # Verify immutable local artifacts against hashes recorded before the
        # interruption. Shape and finite-value checks alone are insufficient.
        if manifest is not None:
            for name in ("train", "validation", "calibration", "test"):
                cache_root = (Path(self.config.source_dir).resolve()
                              if self.config.source_dir else self.output_dir)
                cache_path = cache_root / "cache" / f"{name}_features.npy"
                endpoint_path = cache_root / "cache" / f"{name}_window_end_indices.npy"
                hash_prefix = "source_" if self.config.source_dir else ""
                declared_cache = manifest.get(hash_prefix + "cache_sha256", {}).get(name)
                declared_endpoint = manifest.get(hash_prefix + "endpoint_sha256", {}).get(name)
                if (not declared_cache or not cache_path.is_file() or _sha256_file(cache_path) != declared_cache):
                    raise ValueError(f"resume rejected: cache hash differs for {name}")
                if (not declared_endpoint or not endpoint_path.is_file() or _sha256_file(endpoint_path) != declared_endpoint):
                    raise ValueError(f"resume rejected: endpoint hash differs for {name}")
            normalized_path = self.output_dir / "cache" / "normalized_values.npy"
            declared_normalized = manifest.get("normalized_cache_sha256")
            if declared_normalized is not None and (not normalized_path.is_file() or _sha256_file(normalized_path) != declared_normalized):
                raise ValueError("resume rejected: normalized cache hash differs")
            input_stats_path = self.output_dir / "input_statistics.npz"
            declared_input_stats = manifest.get("input_statistics_sha256")
            if declared_input_stats is not None and (not input_stats_path.is_file() or _sha256_file(input_stats_path) != declared_input_stats):
                raise ValueError("resume rejected: input statistics hash differs")
            feature_stats_path = self.output_dir / "heads" / "feature_statistics.npz"
            declared_feature_stats = manifest.get("feature_statistics_sha256")
            if declared_feature_stats is not None and (not feature_stats_path.is_file() or _sha256_file(feature_stats_path) != declared_feature_stats):
                raise ValueError("resume rejected: feature statistics hash differs")
        # Validate the immutable initialization contract before fit() is
        # allowed to update run_status, normalized arrays, or checkpoints.
        # Source-backed and direct S-only heads intentionally have no local
        # encoder initialization to resume.
        needs_encoder = not self.config.source_dir and self.config.mode != "s_only"
        if needs_encoder:
            initial_path = self.output_dir / "checkpoints" / "initial_encoder.pt"
            if not initial_path.is_file():
                raise ValueError("resume rejected: initial encoder checkpoint is missing")
            payload = torch.load(initial_path, map_location="cpu", weights_only=False)
            if payload.get("run_identity") != self._run_identity:
                raise ValueError("resume rejected: initial encoder identity differs")
            declared_initial = (manifest or {}).get("checkpoint_sha256", {}).get("initial_encoder.pt")
            if declared_initial is None and request is not None:
                declared_initial = request.get("initial_encoder_sha256")
            if not declared_initial or _sha256_file(initial_path) != declared_initial:
                raise ValueError("resume rejected: initial encoder checkpoint hash differs")
            trainer = self._make_trainer(len(self.variable_names))
            expected = trainer.encoder.state_dict()
            saved = payload.get("encoder", payload)
            if set(saved) != set(expected) or any(tuple(saved[name].shape) != tuple(expected[name].shape) for name in expected):
                raise ValueError("resume rejected: initial encoder state shape differs")
            if status.get("status") == "stage1_training" or (request or {}).get("stage") in {
                    "running", "initial_encoder_saved", "stage1_training"}:
                latest_path = self.output_dir / "checkpoints" / "stage1_latest.pt"
                if not latest_path.is_file():
                    raise ValueError("resume rejected: stage1 epoch checkpoint is missing")
                latest = torch.load(latest_path, map_location="cpu", weights_only=False)
                required = {"schema_version", "stage", "resume_id", "next_epoch", "stage_status",
                            "encoder", "auxiliary", "optimizer", "loader_rng", "rng", "best",
                            "best_state", "patience", "history", "completed_steps"}
                if (not required.issubset(latest) or latest.get("schema_version") != 2
                        or latest.get("stage") != "stage1" or latest.get("resume_id") != self._run_identity
                        or not isinstance(latest.get("next_epoch"), int) or latest["next_epoch"] < 1
                        or latest.get("stage_status") not in {"in_progress", "completed"}
                        or not isinstance(latest.get("optimizer"), dict)
                        or not isinstance(latest.get("best_state"), dict)
                        or not isinstance(latest.get("history"), list)
                        or not isinstance(latest.get("completed_steps"), int)):
                    raise ValueError("resume rejected: stage1 epoch checkpoint contract is invalid")
                trainer = self._make_trainer(len(self.variable_names))
                for state_name, module in (("encoder", trainer.encoder), ("auxiliary", trainer.auxiliary)):
                    state = latest.get(state_name)
                    expected_state = module.state_dict()
                    if (not isinstance(state, dict) or set(state) != set(expected_state)
                            or any(tuple(state[name].shape) != tuple(expected_state[name].shape) for name in expected_state)):
                        raise ValueError(f"resume rejected: stage1 {state_name} checkpoint shape differs")
                best_state = latest.get("best_state")
                if not {"encoder", "auxiliary"}.issubset(best_state):
                    raise ValueError("resume rejected: stage1 best state is incomplete")
        if manifest is not None and manifest.get("artifact_type") == ARTIFACT_HEAD and not self.config.source_dir and self.config.mode != "s_only":
            if not (self.output_dir / "input_statistics.npz").is_file():
                raise ValueError("resume rejected: local head is missing input statistics")
        # A completed stage-one source must have its complete source contract.
        if manifest is not None and manifest.get("artifact_type") == ARTIFACT_SOURCE:
            checkpoint_trainer = self._make_trainer(len(self.variable_names))
            for name in ("initial_encoder.pt", "best_encoder.pt"):
                checkpoint_path = self.output_dir / "checkpoints" / name
                declared = manifest.get("checkpoint_sha256", {}).get(name)
                if not checkpoint_path.is_file() or not declared or _sha256_file(checkpoint_path) != declared:
                    raise ValueError("resume rejected: completed source checkpoint is missing or changed: " + name)
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                except Exception as error:
                    raise ValueError("resume rejected: source checkpoint cannot be loaded: " + name) from error
                # initial_encoder.pt carries the run identity; best_encoder.pt
                # is the trainer's established {encoder, auxiliary} payload.
                # The immutable manifest hash binds both formats to this run.
                if not isinstance(checkpoint, dict):
                    raise ValueError("resume rejected: completed source checkpoint identity differs")
                if name == "initial_encoder.pt":
                    if checkpoint.get("run_identity") != self._run_identity or "encoder" not in checkpoint:
                        raise ValueError("resume rejected: initial source checkpoint identity differs")
                elif not {"encoder", "auxiliary"}.issubset(checkpoint):
                    raise ValueError("resume rejected: best source checkpoint payload is invalid")
                for part in (("encoder", "auxiliary") if name == "best_encoder.pt" else ("encoder",)):
                    state = checkpoint[part] if part in checkpoint else checkpoint
                    module = checkpoint_trainer.encoder if part == "encoder" else checkpoint_trainer.auxiliary
                    expected_state = module.state_dict()
                    if (not isinstance(state, dict) or set(state) != set(expected_state)
                            or any(tuple(state[key].shape) != tuple(expected_state[key].shape)
                                   for key in expected_state) or any(
                            not torch.is_tensor(value) or not torch.isfinite(value).all()
                            for value in state.values())):
                        raise ValueError("resume rejected: source checkpoint state is invalid")
            input_stats_path = self.output_dir / "input_statistics.npz"
            declared_stats = manifest.get("input_statistics_sha256")
            if not input_stats_path.is_file() or not declared_stats or _sha256_file(input_stats_path) != declared_stats:
                raise ValueError("resume rejected: completed source input statistics are missing or changed")
            normalized_path = self.output_dir / "cache" / "normalized_values.npy"
            declared_normalized = manifest.get("normalized_cache_sha256")
            if (not normalized_path.is_file() or not declared_normalized
                    or _sha256_file(normalized_path) != declared_normalized):
                raise ValueError("resume rejected: completed source normalized cache is missing or changed")
            # Stage-one exports intentionally have no score/head artifacts;
            # use the source contract above instead of the deployment loader,
            # which also validates final results.
            for name in ("train", "validation", "calibration", "test"):
                cache_path = self.output_dir / "cache" / f"{name}_features.npy"
                endpoint_path = self.output_dir / "cache" / f"{name}_window_end_indices.npy"
                if (not cache_path.is_file() or not endpoint_path.is_file()
                        or not manifest.get("cache_sha256", {}).get(name)
                        or not manifest.get("endpoint_sha256", {}).get(name)
                        or _sha256_file(cache_path) != manifest["cache_sha256"][name]
                        or _sha256_file(endpoint_path) != manifest["endpoint_sha256"][name]):
                    raise ValueError("resume rejected: completed source cache is incomplete")
            self._resume_stage1_complete = status.get("status") == "stage1_complete"
        gmm_mode = self.config.mode in {"full_gmm", "select_gmm", "random_gmm"}
        if ((manifest is not None and manifest.get("artifact_type") == ARTIFACT_HEAD
             and manifest.get("head_type") == "gmm") or gmm_mode):
            progress_path = self.output_dir / "heads" / "gmm_progress.json"
            if not progress_path.is_file() and status.get("status") == "gmm_fitting":
                raise ValueError("resume rejected: GMM progress manifest is missing")
            if progress_path.is_file():
                progress = json.loads(progress_path.read_text())
                if progress.get("schema") != PROBE_SCHEMA or not progress.get("identity"):
                    raise ValueError("resume rejected: GMM progress identity is invalid")
                completed = progress.get("completed", {})
                if not isinstance(completed, dict):
                    raise ValueError("resume rejected: GMM progress completed map is invalid")
                if completed:
                    # These artifacts already influenced persisted heads. A
                    # missing declaration is evidence loss, not permission to
                    # bless the current files with newly computed hashes.
                    for field in ("feature_statistics_sha256", "fixed_noise_sha256",
                                  "gmm_fit_endpoints_sha256"):
                        if not manifest or not manifest.get(field):
                            raise ValueError(f"resume rejected: completed GMM variables require {field}")
                    statistics_path = self.output_dir / "heads" / "feature_statistics.npz"
                    noise_path = self.output_dir / "heads" / "fixed_noise.npy"
                    sample_path = self.output_dir / "heads" / "gmm_fit_endpoints.npy"
                    for field, artifact in (("feature_statistics_sha256", statistics_path),
                                            ("fixed_noise_sha256", noise_path),
                                            ("gmm_fit_endpoints_sha256", sample_path)):
                        if not artifact.is_file() or _sha256_file(artifact) != manifest[field]:
                            raise ValueError(f"resume rejected: {field} is missing or changed")
                for variable, entry in completed.items():
                    if not str(variable).isdigit() or int(variable) < 0 or int(variable) >= len(self.variable_names):
                        raise ValueError("resume rejected: GMM progress variable order is invalid")
                    model_path = self.output_dir / "heads" / "gmm_variables" / f"variable_{variable}.pkl"
                    if (not isinstance(entry, dict) or not model_path.is_file()
                            or not entry.get("model_sha256")
                            or _sha256_file(model_path) != entry["model_sha256"]):
                        raise ValueError(f"resume rejected: completed GMM variable {variable} is missing or changed")
                    try:
                        with open(model_path, "rb") as handle:
                            model = pickle.load(handle)
                        if not getattr(model, "converged_", False):
                            raise ValueError("model is not converged")
                    except Exception as error:
                        raise ValueError(f"resume rejected: completed GMM variable {variable} is invalid: {error}") from error
                sample_path = self.output_dir / "heads" / "gmm_fit_endpoints.npy"
                cache_root = (Path(self.config.source_dir).resolve()
                              if self.config.source_dir else self.output_dir)
                train_endpoint_path = cache_root / "cache" / "train_window_end_indices.npy"
                if not sample_path.is_file() or not train_endpoint_path.is_file():
                    raise ValueError("resume rejected: GMM fitting endpoint sample is missing")
                sample = _strict_integer_indices(np.load(sample_path, mmap_mode="r"), "GMM fitting endpoint sample")
                train_endpoints = _strict_integer_indices(np.load(train_endpoint_path, mmap_mode="r"), "GMM train endpoints")
                if len(sample) == 0 or len(sample) > self.config.gmm_fit_max_windows:
                    raise ValueError("resume rejected: GMM fitting sample size is invalid")
                positions = np.searchsorted(train_endpoints, sample)
                if np.any(positions >= len(train_endpoints)) or not np.array_equal(train_endpoints[positions], sample):
                    raise ValueError("resume rejected: GMM fitting sample is not a train-endpoint subset")
                declared_sample = (manifest or {}).get("gmm_fit_endpoints_sha256")
                if not declared_sample or _sha256_file(sample_path) != declared_sample:
                    raise ValueError("resume rejected: GMM fitting sample hash differs")
                noise_path = self.output_dir / "heads" / "fixed_noise.npy"
                if not noise_path.is_file():
                    raise ValueError("resume rejected: fixed noise artifact is missing")
                noise = np.asarray(np.load(noise_path, allow_pickle=False))
                generator = torch.Generator(device="cpu").manual_seed(self.config.noise_seed)
                expected_noise = (torch.randn(4, self.config.layout.total_dim,
                                               generator=generator, dtype=torch.float32)
                                  * self.config.train_noise_std).numpy()
                if noise.dtype != np.float32 or noise.shape != expected_noise.shape or not np.array_equal(noise, expected_noise):
                    raise ValueError("resume rejected: fixed noise artifact differs")
                fit_config = self.config.to_dict()
                fit_config.pop("output_dir", None)
                fit_config.pop("resume_from", None)
                source_cache_identity = None if not self.config.source_dir else {
                    "source_dir": str(Path(self.config.source_dir).resolve()),
                    "source_identity": self.source_identity,
                    "cache_sha256": (manifest or {}).get("source_cache_sha256"),
                    "endpoint_sha256": (manifest or {}).get("source_endpoint_sha256"),
                }
                stats_path = self.output_dir / "heads" / "feature_statistics.npz"
                noise_path = self.output_dir / "heads" / "fixed_noise.npy"
                if not stats_path.is_file() or not noise_path.is_file():
                    raise ValueError("resume rejected: GMM statistics or fixed noise is missing")
                current_cache_hashes = ((manifest or {}).get("source_cache_sha256") if self.config.source_dir
                                        else (manifest or {}).get("cache_sha256"))
                current_endpoint_hashes = ((manifest or {}).get("source_endpoint_sha256") if self.config.source_dir
                                           else (manifest or {}).get("endpoint_sha256"))
                expected_progress_identity = _json_hash({
                    "config": fit_config,
                    "sample_hash": _sha256_file(sample_path),
                    "statistics_hash": _sha256_file(stats_path),
                    "noise_hash": _sha256_file(noise_path),
                    "cache_hashes": current_cache_hashes,
                    "endpoint_hashes": current_endpoint_hashes,
                    "selected_indices": self.config.selected_indices,
                    "source": source_cache_identity,
                })
                if progress.get("identity") != expected_progress_identity:
                    raise ValueError("resume rejected: GMM progress identity differs")

    def fit(self, frame, train_length, labels=None):
        self._fit_started = time.perf_counter()
        if self.config.mode == "full_gmm" and self.config.source_dir:
            raise ValueError("full_gmm with source_dir is unsupported; use select_gmm or s_only")
        resume_preflight = bool(self.config.resume_from)
        if resume_preflight:
            self._set_protocol_identity(frame, int(train_length))
            self._validate_resume_preflight(frame, int(train_length))
            if getattr(self, "_resume_stage1_complete", False) and self.config.mode == "stage1":
                # A validated stage-one export is already complete.  Returning
                # here is intentionally read-only: no second encoder fit,
                # cache rewrite, or run-status mutation is performed.
                self._fitted = True
                return self
            self._log_event("preflight_complete", mode=self.config.mode,
                            train_windows=self.ranges["window_counts"]["train"],
                            validation_windows=self.ranges["window_counts"]["validation"],
                            calibration_windows=self.ranges["window_counts"]["calibration"],
                            test_windows=len(self._current_test_endpoints))
        self._prepare_dirs()
        self._writes_started = True
        write_allowed = True
        try:
            seed_all(self.config.seed)
            # The probe's CPU budget is explicit and intentionally scoped to
            # this process; it does not change server-wide thread settings.
            torch.set_num_threads(self.config.cpu_threads)
            values, test_endpoints, continuity = self._set_protocol_identity(frame, int(train_length))
            self._log_event("preflight_complete", mode=self.config.mode,
                            train_windows=self.ranges["window_counts"]["train"],
                            validation_windows=self.ranges["window_counts"]["validation"],
                            calibration_windows=self.ranges["window_counts"]["calibration"],
                            test_windows=len(test_endpoints))
            self._save_run_request(stage="running")
            self._set_run_status("running")
            self.trainer = self._make_trainer(len(self.variable_names))
            mode = self.config.mode
            reuse_local_cache = False
            reused_cache = None
            source_cache_mode = mode in {"select_gmm", "flow"} and bool(self.config.source_dir)
            if resume_preflight and mode in {"full_gmm", "random_gmm"}:
                manifest_path = self.output_dir / "probe_manifest.json"
                normalized_path = self.output_dir / "cache" / "normalized_values.npy"
                statistics_path = self.output_dir / "input_statistics.npz"
                cache_files = [self.output_dir / "cache" / f"{name}_features.npy"
                               for name in ("train", "validation", "calibration", "test")]
                endpoint_files = [self.output_dir / "cache" / f"{name}_window_end_indices.npy"
                                  for name in ("train", "validation", "calibration", "test")]
                if (manifest_path.is_file() and normalized_path.is_file() and statistics_path.is_file()
                        and all(path.is_file() for path in cache_files + endpoint_files)):
                    with np.load(statistics_path, allow_pickle=False) as statistics:
                        self.trainer.fixed_input.mean_ = np.asarray(statistics["mean"], dtype=np.float32)
                        self.trainer.fixed_input.scale_ = np.asarray(statistics["scale"], dtype=np.float32)
                    normalized = np.load(normalized_path, mmap_mode="r")
                    if (normalized.shape != values.shape or normalized.dtype != np.float32
                            or not np.isfinite(normalized[:min(len(normalized), 65536)]).all()):
                        raise ValueError("resume rejected: normalized cache shape or values are invalid")
                    reused_cache = {
                        name: self._validate_cache_entry(
                            self.output_dir / "cache" / f"{name}_features.npy",
                            self.output_dir / "cache" / f"{name}_window_end_indices.npy", name
                        ) for name in ("train", "validation", "calibration", "test")
                    }
                    reuse_local_cache = True
            if source_cache_mode:
                normalized = None
            elif not reuse_local_cache:
                normalized = self._normalized_memmap(values)
            initial_state = None
            if self.config.mode != "s_only":
                initial_state = {k: v.detach().cpu().clone() for k, v in self.trainer.encoder.state_dict().items()}
                initial_path = self.output_dir / "checkpoints" / "initial_encoder.pt"
                if self.config.resume_from and initial_path.is_file():
                    initial_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
                    if initial_payload.get("run_identity") != self._run_identity:
                        raise ValueError("resume rejected: initial encoder identity differs")
                    saved_initial = initial_payload.get("encoder", initial_payload)
                    if set(saved_initial) != set(initial_state) or any(not torch.equal(saved_initial[key], initial_state[key]) for key in initial_state):
                        raise ValueError("resume rejected: initial encoder state differs from the run identity")
                else:
                    atomic_torch_save({"schema": PROBE_SCHEMA, "run_identity": self._run_identity, "encoder": initial_state}, initial_path)
                request = json.loads((self.output_dir / "run_request.json").read_text())
                request["initial_encoder_sha256"] = _sha256_file(initial_path)
                request["stage"] = "initial_encoder_saved"
                _atomic_json(self.output_dir / "run_request.json", request)
            source_manifest = None
            if mode in {"random_gmm", "stage1", "full_gmm"}:
                if mode != "random_gmm":
                    self._log_event("stage1_start", epochs=self.config.stage1_epochs, patience=self.config.patience)
                    self._set_run_status("stage1_training")
                    self._save_run_request(stage="stage1_training")
                    self._stage1_with_progress(normalized)
                    self._log_event("stage1_complete", checkpoint=self.output_dir / "checkpoints" / "best_encoder.pt")
            elif mode in {"select_gmm", "flow"} or (mode == "s_only" and self.config.source_dir):
                if not self.config.source_dir:
                    raise ValueError(f"mode={mode} requires source_dir pointing to a validated stage-1 probe")
                source, source_manifest, source_cache = self._load_source(self.config.source_dir)
                with np.load(source / "input_statistics.npz", allow_pickle=False) as stats:
                    self.trainer.fixed_input.mean_ = stats["mean"].astype(np.float32)
                    self.trainer.fixed_input.scale_ = stats["scale"].astype(np.float32)
                self.trainer.encoder.eval()
            if mode in {"select_gmm", "flow"} or (mode == "s_only" and self.config.source_dir):
                cache = source_cache
                test_endpoints = self._current_test_endpoints
            elif mode == "s_only":
                cache, endpoint_paths = self._cache_s_only(normalized, test_endpoints)
            elif reused_cache is not None:
                cache = reused_cache
                endpoint_paths = {name: self.output_dir / "cache" / f"{name}_window_end_indices.npy"
                                  for name in ("train", "validation", "calibration", "test")}
            else:
                cache, endpoint_paths = self._cache_all(normalized, values, test_endpoints)
            if not reuse_local_cache:
                np.savez(self.output_dir / "input_statistics.npz", mean=self.trainer.fixed_input.mean_, scale=self.trainer.fixed_input.scale_)
            if mode == "stage1":
                self._save_source_metadata("pretrained", cache, endpoint_paths, initial_state)
                self._set_run_status("stage1_complete")
                return None
            if mode in {"select_gmm", "flow"} or (mode == "s_only" and self.config.source_dir):
                self._save_probe_metadata(mode, source_manifest)
            elif mode == "s_only":
                self._save_probe_metadata(mode, None)
                self._bind_local_cache_metadata(cache, endpoint_paths)
            else:
                self._save_source_metadata("random" if mode == "random_gmm" else "pretrained", cache, endpoint_paths, initial_state)
            if mode == "flow":
                if tuple(self.config.selected_views) != VALID_VIEWS:
                    raise ValueError("flow mode requires the complete S,E,P,R layout")
                self._set_run_status("flow_training")
                self._save_run_request(stage="flow_training")
                self._log_event("flow_start", epochs=self.config.stage2_epochs, patience=self.config.patience)
                self._fit_flow(cache, labels, test_endpoints)
                self._log_event("flow_complete")
            else:
                self._set_run_status("gmm_fitting")
                manifest_path = self.output_dir / "probe_manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text())
                    manifest.update({"head_type": "gmm", "stage_status": "gmm_fitting", "head_status": "in_progress"})
                    _atomic_json(manifest_path, manifest)
                request = json.loads((self.output_dir / "run_request.json").read_text())
                request["stage"] = "gmm_fitting"
                _atomic_json(self.output_dir / "run_request.json", request)
                self._fit_gmm(cache, source_manifest=source_manifest)
                self._calibrate(cache)
                self._fit_elapsed = time.perf_counter() - self._fit_started
                self._write_scores(cache, test_endpoints, labels)
                manifest_path = self.output_dir / "probe_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest.update({
                    "artifact_type": ARTIFACT_HEAD, "head": "gmm", "head_type": "gmm",
                    "artifact_policy": self.config.artifact_policy,
                    "head_status": "complete", "stage_status": "complete",
                    "feature_statistics_sha256": _sha256_file(self.output_dir / "heads" / "feature_statistics.npz"),
                    "gmm_status_sha256": _sha256_file(self.output_dir / "heads" / "gmm_status.json"),
                    "calibration_sha256": _sha256_file(self.output_dir / "heads" / "calibration.npz"),
                    "test_nll_sha256": _sha256_file(self.output_dir / "results" / "test_nll.npy"),
                    "aligned_scores_sha256": _sha256_file(self.output_dir / "results" / "aligned_scores.npz"),
                    "metrics_sha256": _sha256_file(self.output_dir / "results" / "metrics.json"),
                })
                _atomic_json(manifest_path, manifest)
            self._fitted = True
            if self.config.artifact_policy == "results":
                # Release memmaps before the retention transaction removes the
                # cache and head trees.  This keeps the cleanup portable to
                # platforms that do not permit unlinking an open mapping.
                cache = None
                normalized = None
                self._finalize_results_artifact()
            else:
                self._set_run_status("complete", artifact_type=ARTIFACT_HEAD,
                                     artifact_policy="full")
            return self
        except Exception as error:
            if write_allowed:
                self._set_run_status("failed", repr(error))
            raise

    def _fit_flow(self, cache, labels, test_endpoints):
        feature_stats_path = self.output_dir / "heads" / "feature_statistics.npz"
        if self.config.resume_from and feature_stats_path.is_file():
            with np.load(feature_stats_path, allow_pickle=False) as stats:
                mean = np.asarray(stats["mean"], dtype=np.float32)
                scale = np.asarray(stats["scale"], dtype=np.float32)
            expected_shape = (int(cache["train"].shape[1]), int(cache["train"].shape[2]))
            if (mean.shape != expected_shape or scale.shape != expected_shape
                    or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                    or np.any(scale <= 0)):
                raise ValueError(f"resume rejected: Flow feature statistics are invalid; expected {expected_shape}")
            self.trainer.fixed_feature.mean_ = mean
            self.trainer.fixed_feature.scale_ = scale
        else:
            with threadpool_limits(limits=self.config.cpu_threads):
                self.trainer.fixed_feature.fit(cache["train"], chunk_size=self.config.cache_chunk_size)
            np.savez(feature_stats_path,
                     mean=np.asarray(self.trainer.fixed_feature.mean_, dtype=np.float32),
                     scale=np.asarray(self.trainer.fixed_feature.scale_, dtype=np.float32))
            manifest_path = self.output_dir / "probe_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                manifest["feature_statistics_sha256"] = _sha256_file(feature_stats_path)
                _atomic_json(manifest_path, manifest)
        stage2_identity = _json_hash({"source_identity": self.source_identity,
                                      "feature_fingerprint": self.trainer.fingerprint(),
                                      "layout": self.config.layout.to_dict()})
        manifest_path = self.output_dir / "probe_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            declared_identity = manifest.get("stage2_identity")
            if self.config.resume_from and declared_identity and declared_identity != stage2_identity:
                raise ValueError("resume rejected: Flow stage2 identity differs")
            manifest["stage2_identity"] = stage2_identity
            manifest["feature_statistics_sha256"] = _sha256_file(feature_stats_path)
            _atomic_json(manifest_path, manifest)
        self.trainer.stage2(
            cache["train"], cache["validation"], self.output_dir / "checkpoints",
            log_path=self.output_dir / "training.jsonl", resume_id=self._run_identity,
            stage2_identity=stage2_identity,
        )
        self.trainer.calibrate(cache["validation"], cache["calibration"], self.output_dir / "heads")
        self._fit_elapsed = time.perf_counter() - self._fit_started
        self._noise_vectors = self.trainer._fixed_noises().detach().cpu().numpy().astype(np.float32)
        np.save(self.output_dir / "heads" / "fixed_noise.npy", self._noise_vectors)
        np.savez(self.output_dir / "heads" / "calibration.npz",
                 center=self.trainer.calibration_center, scale=self.trainer.calibration_scale,
                 threshold=np.asarray(self.trainer.threshold), quantile=np.asarray(self.config.calibration_quantile))
        inference_start = time.perf_counter()
        nll = self.trainer.per_variable_scores(cache["test"], self.output_dir / "results" / "test_nll.npy")
        if not np.isfinite(nll).all():
            raise FloatingPointError("non-finite Flow NLL")
        score = np.max((nll - self.trainer.calibration_center) / self.trainer.calibration_scale, axis=1)
        prediction = (score > self.trainer.threshold).astype(np.int64)
        payload = {"status": "complete", "head": "flow", "selected_views": list(VALID_VIEWS),
                   "selected_dim": self.config.layout.total_dim, "threshold": float(self.trainer.threshold),
                   "head_parameter_count": sum(parameter.numel() for parameter in self.trainer.flow.parameters()),
                   "fit_time_seconds": float(getattr(self, "_fit_elapsed", 0.0)),
                   "inference_time_seconds": float(time.perf_counter() - inference_start),
                   "coverage": float(len(score) / max(int(self._test_length), 1)),
                   "excluded_points": int(self._test_length - len(score))}
        local = np.asarray(test_endpoints, dtype=np.int64) - int(self._test_offset)
        if np.any(local < 0) or np.any(local >= int(self._test_length)):
            raise ValueError("Flow test endpoint cannot be aligned to the local test range")
        aligned_scores = np.full(int(self._test_length), np.nan, dtype=np.float32)
        aligned_predictions = np.full(int(self._test_length), -1, dtype=np.int64)
        valid_mask = np.zeros(int(self._test_length), dtype=bool)
        aligned_scores[local] = score
        aligned_predictions[local] = prediction
        valid_mask[local] = True
        result_payload = {"scores": score, "predictions": prediction,
                          "time_indices": np.asarray(test_endpoints),
                          "time_values_ns": np.asarray(self._time_values_ns[test_endpoints], dtype=np.int64),
                          "local_indices": local,
                          "valid_mask": valid_mask, "aligned_scores": aligned_scores,
                          "aligned_predictions": aligned_predictions,
                          "threshold": np.asarray(self.trainer.threshold)}
        if labels is not None:
            labels = np.asarray(labels).reshape(-1)
            if len(labels) != int(self._test_length):
                raise ValueError(f"test labels must be local and have length {self._test_length}, got {len(labels)}")
            actual = labels[local]
            from sklearn.metrics import average_precision_score, roc_auc_score
            payload["auc_roc"] = float(roc_auc_score(actual, score)) if np.unique(actual).size == 2 else None
            payload["auc_pr"] = float(average_precision_score(actual, score))
            if payload["auc_roc"] is None:
                payload["status"] = "incomplete"
                payload["auc_undefined_reason"] = "valid test labels contain one class"
            result_payload["actual"] = actual
            result_payload["aligned_actual"] = np.full(int(self._test_length), -1, dtype=np.asarray(actual).dtype)
            result_payload["aligned_actual"][local] = actual
        np.savez_compressed(self.output_dir / "results" / "aligned_scores.npz", **result_payload)
        _atomic_json(self.output_dir / "results" / "metrics.json", payload)
        manifest_path = self.output_dir / "probe_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            manifest.update({"head": "flow", "head_status": "complete",
                             "artifact_type": ARTIFACT_HEAD, "head_type": "flow",
                             "stage_status": "complete",
                             "flow_checkpoint_sha256": _sha256_file(self.output_dir / "checkpoints" / "best_fusion.pt"),
                             "input_statistics_sha256": _sha256_file(self.output_dir / "heads" / "feature_statistics.npz"),
                             "calibration_sha256": _sha256_file(self.output_dir / "heads" / "calibration.npz"),
                             "fixed_noise_sha256": _sha256_file(self.output_dir / "heads" / "fixed_noise.npy"),
                             "test_nll_sha256": _sha256_file(self.output_dir / "results" / "test_nll.npy"),
                             "aligned_scores_sha256": _sha256_file(self.output_dir / "results" / "aligned_scores.npz"),
                             "metrics_sha256": _sha256_file(self.output_dir / "results" / "metrics.json")})
            _atomic_json(manifest_path, manifest)

    def detect_fit(self, train_data, train_label=None):
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        # The standard benchmark calls fit with the train partition only.  We
        # retain that partition and complete the protocol when detect_score
        # receives the held-out partition, so test rows never enter training.
        self._benchmark_train_frame = frame.copy(deep=True)
        self._benchmark_train_label = train_label
        self._benchmark_train_length = len(frame)
        return self

    def detect_score(self, test_data):
        if not hasattr(self, "_benchmark_train_frame"):
            raise RuntimeError("detect_fit must be called before detect_score")
        test_frame = test_data if isinstance(test_data, pd.DataFrame) else pd.DataFrame(test_data)
        combined = pd.concat([self._benchmark_train_frame, test_frame], axis=0)
        self.fit(combined, self._benchmark_train_length, labels=None)
        result = np.load(self.output_dir / "results" / "aligned_scores.npz", allow_pickle=False)
        # The public benchmark interface returns only valid endpoint scores;
        # score_with_metadata exposes the full index/mask without imputation.
        return result["scores"], result["predictions"]

    def score_with_metadata(self):
        if self.output_dir is None or not (self.output_dir / "results" / "aligned_scores.npz").is_file():
            raise RuntimeError("view probe has no completed scoring artifact")
        result = np.load(self.output_dir / "results" / "aligned_scores.npz", allow_pickle=False)
        return {name: result[name] for name in result.files}

    def score_cached_test(self, labels=None):
        """Score a restored source-backed run without rebuilding its encoder.

        This is intentionally a read-only deployment path.  It consumes the
        validated source test cache and rewrites only this probe's result
        artifact, so a loaded Flow head is never mistaken for an unfitted
        object.
        """
        if not self._fitted:
            raise RuntimeError("probe is not a loaded, fitted artifact")
        cache_path = getattr(self, "_loaded_test_cache", None)
        endpoint_path = getattr(self, "_loaded_test_endpoints", None)
        if cache_path is None or endpoint_path is None:
            raise RuntimeError("loaded probe has no validated test cache")
        features = np.load(cache_path, mmap_mode="r")
        endpoints = np.asarray(np.load(endpoint_path, mmap_mode="r"), dtype=np.int64)
        if self._loaded_head == "gmm":
            nll = self._gmm_scores(features)
            center = self._probe_stats["center"]
            scale = self._probe_stats["scale"]
            threshold = self._probe_stats["threshold"]
        else:
            nll = self.trainer.per_variable_scores(features)
            center = self._probe_stats["center"]
            scale = self._probe_stats["scale"]
            threshold = self._probe_stats["threshold"]
        score = np.max((nll - center) / scale, axis=1)
        if not np.isfinite(score).all():
            raise FloatingPointError("restored probe produced non-finite scores")
        prediction = (score > threshold).astype(np.int64)
        return {"scores": score, "predictions": prediction, "time_indices": endpoints,
                "threshold": float(threshold)}


def _validate_result_bundle(path, allow_in_progress=False):
    """Validate a results-only bundle without constructing a model or RNG."""
    path = Path(path)
    manifest_path = path / "probe_manifest.json"
    status_path = path / "run_status.json"
    metrics_path = path / "results" / "metrics.json"
    aligned_path = path / "results" / "aligned_scores.npz"
    if not manifest_path.is_file() or not status_path.is_file():
        raise ValueError("results bundle is missing manifest or run status")
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    if manifest.get("schema") != PROBE_SCHEMA:
        raise ValueError("results bundle has unsupported probe schema")
    if manifest.get("artifact_type") not in {ARTIFACT_RESULTS, ARTIFACT_HEAD}:
        raise ValueError("artifact is not a results bundle")
    if manifest.get("artifact_type") == ARTIFACT_RESULTS and manifest.get("result_bundle_schema") != RESULT_SCHEMA:
        raise ValueError("results bundle schema is missing or unsupported")
    if manifest.get("artifact_type") == ARTIFACT_HEAD and manifest.get("artifact_policy") != "results":
        raise ValueError("probe head is not declared as artifact_policy=results")
    cleanup_status = manifest.get("result_cleanup_status", "complete")
    if cleanup_status not in {"complete", "in_progress", "failed"}:
        raise ValueError("results cleanup status is invalid")
    if not allow_in_progress and cleanup_status != "complete":
        raise ValueError(f"results cleanup is not complete: {cleanup_status}")
    allowed_status = {"complete", "organizing_results", "failed"}
    if allow_in_progress:
        allowed_status.update({"gmm_fitting", "flow_training"})
    if status.get("status") not in allowed_status:
        raise ValueError(f"results run status is invalid: {status.get('status')}")
    if not metrics_path.is_file() or not aligned_path.is_file():
        raise ValueError("results bundle is missing metrics.json or aligned_scores.npz")
    if metrics_path.is_symlink() or aligned_path.is_symlink():
        raise ValueError("results bundle must not use symlinked result files")
    if not allow_in_progress and cleanup_status == "complete":
        forbidden = ("checkpoints", "cache", "heads", "run_request.json",
                     "input_statistics.npz")
        if any((path / name).exists() for name in forbidden):
            raise ValueError("results bundle retains forbidden training artifacts")
        for child in (path / "results").iterdir():
            if child not in {metrics_path, aligned_path}:
                raise ValueError("results bundle contains an unapproved result artifact")
    for field, artifact in (("metrics_sha256", metrics_path), ("aligned_scores_sha256", aligned_path)):
        declared = manifest.get(field)
        if not declared or _sha256_file(artifact) != declared:
            raise ValueError(f"results bundle binding is missing or changed: {field}")
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("status") not in {"complete", "incomplete"}:
        raise ValueError("results metrics status is invalid")
    threshold = float(metrics.get("threshold"))
    if not np.isfinite(threshold):
        raise ValueError("results threshold is not finite")
    with np.load(aligned_path, allow_pickle=False) as payload:
        required = {"scores", "predictions", "time_indices", "local_indices", "valid_mask",
                    "aligned_scores", "aligned_predictions", "threshold"}
        if not required.issubset(payload.files):
            raise ValueError("results aligned score file is incomplete")
        scores = np.asarray(payload["scores"])
        predictions = np.asarray(payload["predictions"])
        endpoints = np.asarray(payload["time_indices"])
        local = np.asarray(payload["local_indices"])
        valid = np.asarray(payload["valid_mask"])
        aligned_scores = np.asarray(payload["aligned_scores"])
        aligned_predictions = np.asarray(payload["aligned_predictions"])
        if any(item.ndim != 1 for item in (scores, predictions, endpoints, local, valid)):
            raise ValueError("results vectors must be one-dimensional")
        if endpoints.dtype.kind not in "iu" or local.dtype.kind not in "iu" or valid.dtype.kind != "b":
            raise ValueError("results coordinates or valid_mask have invalid dtype")
        if len(scores) != len(predictions) or len(scores) != len(endpoints) or len(scores) != len(local):
            raise ValueError("results vector lengths disagree")
        if len(np.unique(endpoints)) != len(endpoints) or len(np.unique(local)) != len(local):
            raise ValueError("results coordinates are not unique")
        if np.any(local < 0) or np.any(local >= len(valid)):
            raise ValueError("results local coordinates are out of range")
        if not np.isfinite(scores).all() or not np.isfinite(aligned_scores[valid]).all():
            raise ValueError("results scores contain NaN or Inf")
        if not np.isclose(float(np.asarray(payload["threshold"])), threshold, rtol=0, atol=1e-7):
            raise ValueError("results threshold copy differs from metrics")
        expected_predictions = (scores > threshold).astype(np.int64)
        if not np.array_equal(predictions.astype(np.int64), expected_predictions):
            raise ValueError("results predictions do not match scores and threshold")
        if len(aligned_scores) != len(valid) or len(aligned_predictions) != len(valid):
            raise ValueError("results aligned vectors disagree")
        if not np.array_equal(valid[local], np.ones(len(local), dtype=bool)):
            raise ValueError("results valid mask does not cover score coordinates")
        if not np.allclose(aligned_scores[local], scores, rtol=0, atol=1e-7):
            raise ValueError("results aligned scores do not match scores")
        if not np.array_equal(aligned_predictions[local].astype(np.int64), predictions.astype(np.int64)):
            raise ValueError("results aligned predictions do not match predictions")
        if np.any(aligned_predictions[~valid] != -1) or np.any(np.isfinite(aligned_scores[~valid])):
            raise ValueError("results invalid positions have incorrect sentinels")
        if "actual" in payload.files:
            actual = np.asarray(payload["actual"])
            if actual.ndim != 1 or len(actual) != len(scores):
                raise ValueError("results actual labels are invalid")
            from sklearn.metrics import average_precision_score, roc_auc_score
            expected_pr = float(average_precision_score(actual, scores))
            if metrics.get("auc_pr") is None or not np.isclose(float(metrics["auc_pr"]), expected_pr, rtol=0, atol=1e-6):
                raise ValueError("results AUC-PR does not reproduce")
            if np.unique(actual).size == 2:
                expected_roc = float(roc_auc_score(actual, scores))
                if metrics.get("auc_roc") is None or not np.isclose(float(metrics["auc_roc"]), expected_roc, rtol=0, atol=1e-6):
                    raise ValueError("results AUC-ROC does not reproduce")
    return {"status": "valid", "artifact_type": ARTIFACT_RESULTS,
            "cleanup_status": cleanup_status, "metrics": metrics}


def organize_results(path):
    """Idempotently finish a previously interrupted results cleanup."""
    path = Path(path).resolve()
    manifest = json.loads((path / "probe_manifest.json").read_text())
    if manifest.get("artifact_policy") != "results":
        raise ValueError("only artifact_policy=results runs can be organized")
    _validate_result_bundle(path, allow_in_progress=True)
    known_root = {"probe_manifest.json", "run_status.json", "run.log", "training.jsonl",
                  "results", "checkpoints", "cache", "heads", "run_request.json",
                  "input_statistics.npz"}
    unknown_root = [child.name for child in path.iterdir() if child.name not in known_root]
    allowed_results = {"metrics.json", "aligned_scores.npz", "test_nll.npy"}
    unknown_results = [child.name for child in (path / "results").iterdir()
                       if child.name not in allowed_results]
    if unknown_root or unknown_results:
        raise ValueError("refusing results cleanup with unknown paths: "
                         + ", ".join(unknown_root + [f"results/{name}" for name in unknown_results]))
    keep = {path / "probe_manifest.json", path / "run_status.json", path / "run.log",
            path / "training.jsonl", path / "results" / "metrics.json",
            path / "results" / "aligned_scores.npz"}
    for name in ("checkpoints", "cache", "heads"):
        candidate = path / name
        if candidate.is_symlink():
            raise ValueError(f"refusing to follow symlink during cleanup: {candidate}")
        if candidate.is_dir():
            shutil.rmtree(candidate)
    request = path / "run_request.json"
    if request.exists():
        request.unlink()
    results_dir = path / "results"
    keep_results = {results_dir / "metrics.json", results_dir / "aligned_scores.npz"}
    for child in results_dir.iterdir():
        if child not in keep_results:
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    for child in path.iterdir():
        if child in keep or child == path / "results":
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    manifest = json.loads((path / "probe_manifest.json").read_text())
    count, size = MNRFViewProbe._tree_size(path)
    manifest["result_cleanup_status"] = "complete"
    manifest["retention_after"] = {"file_count": count, "bytes": size}
    _atomic_json(path / "probe_manifest.json", manifest)
    _atomic_json(path / "run_status.json", {
        "schema": PROBE_SCHEMA, "status": "complete", "artifact_type": ARTIFACT_RESULTS,
        "run_identity": manifest.get("run_identity"), "result_cleanup_status": "complete",
        "artifact_policy": "results",
    })
    return _validate_result_bundle(path)


def load_probe(path, device="cpu"):
    path = Path(path)
    manifest_path = path / "probe_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"view probe manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != PROBE_SCHEMA:
        raise ValueError("view probe artifact schema is unsupported; re-export with the current probe")
    if manifest.get("artifact_type") == ARTIFACT_RESULTS or manifest.get("artifact_policy") == "results":
        raise ValueError("results artifact is metrics-only and has no model/cache; rerun with artifact_policy=full for loading or source reuse")
    run_status_path = path / "run_status.json"
    if not run_status_path.is_file():
        raise ValueError("view probe artifact has no run_status.json")
    run_status = json.loads(run_status_path.read_text())
    if run_status.get("status") not in {"complete", "stage1_complete"}:
        raise ValueError(f"view probe artifact is not complete: {run_status.get('status')}")
    artifact_type = manifest.get("artifact_type")
    if artifact_type not in {ARTIFACT_SOURCE, ARTIFACT_HEAD}:
        raise ValueError("view probe artifact type is missing or unsupported")
    if artifact_type == ARTIFACT_SOURCE and run_status.get("status") != "stage1_complete":
        raise ValueError("source encoder artifact must have stage1_complete status")
    if artifact_type == ARTIFACT_HEAD and run_status.get("status") != "complete":
        raise ValueError("probe head artifact must have complete status")
    # Keep the artifact's original training configuration intact. The caller's
    # device is a load-time execution choice, not a persisted model parameter.
    persisted_config = dict(manifest["config"])
    config = dict(persisted_config)
    config.pop("resolved_layout", None); config.pop("selected_indices", None); config.pop("selected_dim", None)
    config["output_dir"] = str(path); config["device"] = device
    probe = MNRFViewProbe.from_kwargs(**config)
    require_layout(manifest.get("layout"), probe.config.layout, "view probe artifact")
    if manifest.get("selected_views", list(probe.config.selected_views)) != list(probe.config.selected_views):
        raise ValueError("view probe artifact selected view order does not match its configuration")
    if manifest.get("selected_indices", probe.config.selected_indices) != probe.config.selected_indices:
        raise ValueError("view probe artifact selected indices do not match its layout")
    if manifest.get("source_identity") is None:
        raise ValueError("view probe artifact has no source identity")
    source_dir = manifest.get("source_dir") or ""
    if source_dir:
        source = Path(source_dir).resolve()
        source_manifest_path = source / "probe_manifest.json"
        if not source_manifest_path.is_file():
            raise ValueError("view probe source manifest is missing")
        source_manifest = json.loads(source_manifest_path.read_text())
        source_is_frozen = (
            source_manifest.get("artifact_type") == ARTIFACT_SOURCE
            and source_manifest.get("stage_status") == "stage1_complete"
        ) or (
            source_manifest.get("artifact_type") == ARTIFACT_HEAD
            and source_manifest.get("head_type") == "gmm"
            and source_manifest.get("stage_status") == "complete"
        )
        if (source_manifest.get("schema") != PROBE_SCHEMA
                or not source_is_frozen
                or source_manifest.get("source_identity") != manifest.get("source_identity")):
            raise ValueError("view probe source identity does not match the loaded artifact")
        source_cache_hashes = manifest.get("source_cache_sha256", {})
        source_endpoint_hashes = manifest.get("source_endpoint_sha256", {})
        if any(name not in source_cache_hashes for name in ("train", "validation", "calibration", "test")):
            raise ValueError("view probe source cache hashes are incomplete")
        if any(name not in source_endpoint_hashes for name in ("train", "validation", "calibration", "test")):
            raise ValueError("view probe source endpoint hashes are incomplete")
        for name, declared_hash in source_cache_hashes.items():
            cache_path = source / "cache" / f"{name}_features.npy"
            if not cache_path.is_file() or _sha256_file(cache_path) != declared_hash:
                raise ValueError(f"view probe source cache hash mismatch: {name}")
        for name, declared_hash in source_endpoint_hashes.items():
            endpoint_path = source / "cache" / f"{name}_window_end_indices.npy"
            if not endpoint_path.is_file() or _sha256_file(endpoint_path) != declared_hash:
                raise ValueError(f"view probe source endpoint hash mismatch: {name}")
        if set(manifest.get("source_cache_sha256", {})) != {"train", "validation", "calibration", "test"}:
            raise ValueError("view probe source cache hashes are incomplete")
        if set(manifest.get("source_endpoint_sha256", {})) != {"train", "validation", "calibration", "test"}:
            raise ValueError("view probe source endpoint hashes are incomplete")
    elif artifact_type == ARTIFACT_HEAD:
        if set(manifest.get("cache_sha256", {})) != {"train", "validation", "calibration", "test"}:
            raise ValueError("view probe local cache hashes are incomplete")
        if set(manifest.get("endpoint_sha256", {})) != {"train", "validation", "calibration", "test"}:
            raise ValueError("view probe local endpoint hashes are incomplete")
        for name in ("train", "validation", "calibration", "test"):
            cache_path = path / "cache" / f"{name}_features.npy"
            endpoint_path = path / "cache" / f"{name}_window_end_indices.npy"
            if (not cache_path.is_file() or _sha256_file(cache_path) != manifest["cache_sha256"][name]
                    or not endpoint_path.is_file() or _sha256_file(endpoint_path) != manifest["endpoint_sha256"][name]):
                raise ValueError(f"view probe local cache binding mismatch: {name}")
            cache = np.load(cache_path, mmap_mode="r")
            expected_shape = tuple(manifest.get("cache_shapes", {}).get(name, ()))
            expected_dtype = manifest.get("cache_dtypes", {}).get(name)
            if (expected_shape and tuple(cache.shape) != expected_shape
                    or expected_dtype and str(cache.dtype) != expected_dtype
                    or cache.dtype != np.float32 or cache.ndim != 3 or cache.shape[1] != len(manifest.get("variable_names", ()))
                    or cache.shape[-1] != int(manifest.get("cache_feature_dim") or probe.config.layout.total_dim)
                    or not all(np.isfinite(cache[start:start + 65536]).all()
                               for start in range(0, len(cache), 65536))):
                raise ValueError(f"view probe local cache metadata mismatch: {name}")
    mode = manifest.get("mode")
    stats_path = path / "heads" / "feature_statistics.npz"
    calibration_path = path / "heads" / "calibration.npz"
    model_path = path / "heads" / "gmm_models.pkl"
    status_path = path / "heads" / "gmm_status.json"
    sample_path = path / "heads" / "gmm_fit_endpoints.npy"
    gmm_mode = artifact_type == ARTIFACT_HEAD and manifest.get("head_type") == "gmm"
    if artifact_type == ARTIFACT_SOURCE:
        if any(item.exists() for item in (stats_path, calibration_path, model_path, status_path, sample_path)):
            raise ValueError("source encoder artifact contains partial head artifacts")
        required_source = (path / "input_statistics.npz", path / "checkpoints" / "initial_encoder.pt",
                           path / "checkpoints" / "best_encoder.pt")
        if any(not item.is_file() for item in required_source):
            raise ValueError("source encoder artifact is incomplete")
        for field, rel in (("input_statistics_sha256", "input_statistics.npz"),
                           ("checkpoint_sha256", "checkpoints/initial_encoder.pt")):
            declared = manifest.get(field)
            if field == "checkpoint_sha256":
                declared = declared.get("initial_encoder.pt") if isinstance(declared, dict) else None
            artifact = path / rel
            if not declared or _sha256_file(artifact) != declared:
                raise ValueError(f"source encoder artifact hash mismatch: {rel}")
        declared_best = manifest.get("checkpoint_sha256", {}).get("best_encoder.pt")
        if not declared_best or _sha256_file(path / "checkpoints" / "best_encoder.pt") != declared_best:
            raise ValueError("source encoder best checkpoint hash mismatch")
        checkpoint_trainer = probe._make_trainer(len(manifest.get("variable_names", ())))
        for name in ("initial_encoder.pt", "best_encoder.pt"):
            checkpoint_path = path / "checkpoints" / name
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict):
                raise ValueError("source encoder checkpoint payload is invalid: " + name)
            states = (("encoder", checkpoint.get("encoder", checkpoint)),) if name == "initial_encoder.pt" else (
                ("encoder", checkpoint.get("encoder")), ("auxiliary", checkpoint.get("auxiliary")))
            for state_name, state in states:
                module = checkpoint_trainer.encoder if state_name == "encoder" else checkpoint_trainer.auxiliary
                expected_state = module.state_dict()
                if (not isinstance(state, dict) or set(state) != set(expected_state)
                        or any(tuple(state[key].shape) != tuple(expected_state[key].shape)
                               or not torch.isfinite(state[key]).all() for key in expected_state)):
                    raise ValueError("source encoder checkpoint state is invalid: " + name)
            if name == "initial_encoder.pt" and checkpoint.get("run_identity") != manifest.get("run_identity"):
                # The stage-one run identity is stored in run_status, while
                # the exported source identity is data/config based.
                status_identity = run_status.get("run_identity")
                if checkpoint.get("run_identity") != status_identity:
                    raise ValueError("source initial checkpoint run identity differs")
        if set(manifest.get("cache_sha256", {})) != {"train", "validation", "calibration", "test"} or set(manifest.get("endpoint_sha256", {})) != {"train", "validation", "calibration", "test"}:
            raise ValueError("source encoder cache bindings are incomplete")
        for name in ("train", "validation", "calibration", "test"):
            cache_path = path / "cache" / f"{name}_features.npy"
            endpoint_path = path / "cache" / f"{name}_window_end_indices.npy"
            if (not cache_path.is_file() or _sha256_file(cache_path) != manifest["cache_sha256"][name]
                    or not endpoint_path.is_file() or _sha256_file(endpoint_path) != manifest["endpoint_sha256"][name]):
                raise ValueError(f"source encoder cache binding mismatch: {name}")
            cache = np.load(cache_path, mmap_mode="r")
            endpoints = _strict_integer_indices(np.load(endpoint_path, mmap_mode="r"), f"{name} endpoints")
            finite = all(np.isfinite(cache[start:start + 65536]).all() for start in range(0, len(cache), 65536))
            expected_shape = tuple(manifest.get("cache_shapes", {}).get(name, ()))
            expected_dtype = manifest.get("cache_dtypes", {}).get(name)
            identity = manifest.get("source_identity", {})
            if name == "test":
                expected_endpoints = window_endpoints(identity.get("test_segments", ()), probe.config.seq_len)
            else:
                segment_ranges = identity.get("segment_ranges", {}).get(name, ())
                data_range = identity.get("ranges", {}).get(name)
                expected_endpoints = window_endpoints(
                    intersect_segments(segment_ranges, *data_range), probe.config.seq_len
                ) if data_range is not None else np.empty(0, dtype=np.int64)
            if (cache.dtype != np.float32 or cache.ndim != 3 or cache.shape[0] != len(endpoints)
                    or cache.shape[1] != len(manifest.get("variable_names", ()))
                    or cache.shape[-1] != probe.config.layout.total_dim
                    or expected_shape and tuple(cache.shape) != expected_shape
                    or expected_dtype and str(cache.dtype) != expected_dtype
                    or not np.array_equal(endpoints, expected_endpoints)
                    or not finite or np.any(np.diff(endpoints) <= 0)):
                raise ValueError(f"source encoder cache {name} is invalid")
        return probe
    if manifest.get("head_type") not in {"gmm", "flow"} or manifest.get("head_status") != "complete":
        raise ValueError("probe head artifact has an unknown or incomplete head type")
    if manifest.get("head_type") == "flow" and mode != "flow":
        raise ValueError("Flow head artifact has an incompatible mode")
    if manifest.get("head_type") == "gmm" and mode not in {"full_gmm", "select_gmm", "s_only", "random_gmm", "pretrained", "random"}:
        raise ValueError("GMM head artifact has an incompatible mode")
    raw_input_statistics = path / "input_statistics.npz"
    declared_raw_input = manifest.get("raw_input_statistics_sha256")
    if (not declared_raw_input or not raw_input_statistics.is_file()
            or _sha256_file(raw_input_statistics) != declared_raw_input):
        raise ValueError("view probe head input statistics are missing or changed")
    if gmm_mode:
        required = (stats_path, calibration_path, model_path, status_path, sample_path)
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise ValueError("view probe GMM artifact is incomplete: " + ", ".join(missing))
        status = json.loads(status_path.read_text())
        if status.get("sklearn_version") != sklearn.__version__:
            raise ValueError(f"view probe sklearn version mismatch: artifact={status.get('sklearn_version')} current={sklearn.__version__}")
        status_config = status.get("config")
        if status_config is None or _runtime_free_config(status_config) != _runtime_free_config(persisted_config):
            raise ValueError("view probe GMM configuration does not match the manifest")
        if status_config.get("selected_indices") != probe.config.selected_indices:
            raise ValueError("view probe GMM selected indices do not match its configuration")
        with np.load(stats_path, allow_pickle=False) as stats:
            mean = np.asarray(stats["mean"], dtype=np.float32)
            scale = np.asarray(stats["scale"], dtype=np.float32)
        variables = len(manifest.get("variable_names", ()))
        expected_width = int(manifest.get("cache_feature_dim") or probe.config.layout.total_dim)
        if expected_width not in {probe.config.layout.total_dim, probe.config.selected_dim}:
            raise ValueError("view probe cache feature width is incompatible with the layout")
        expected = (variables, expected_width)
        if (mean.shape != expected or scale.shape != expected or np.any(~np.isfinite(mean))
                or np.any(~np.isfinite(scale)) or np.any(scale < probe.config.feature_std_floor)):
            raise ValueError(f"view probe feature statistics shape mismatch: expected {expected}")
        for field, artifact in (("feature_statistics_sha256", stats_path),
                                ("gmm_status_sha256", status_path),
                                ("calibration_sha256", calibration_path)):
            if not manifest.get(field):
                raise ValueError(f"view probe manifest is missing required hash: {field}")
            if _sha256_file(artifact) != manifest[field]:
                raise ValueError(f"view probe {field} mismatch")
        sample_hash = _sha256_file(sample_path)
        if sample_hash != status.get("fit_endpoint_sha256") or sample_hash != manifest.get("gmm_fit_endpoints_sha256"):
            raise ValueError("view probe GMM fitting endpoint hash mismatch")
        with np.load(calibration_path, allow_pickle=False) as calibration:
            probe._probe_stats = {key: np.asarray(calibration[key]) for key in ("center", "scale")}
            probe._probe_stats["threshold"] = float(np.asarray(calibration["threshold"]))
        if (probe._probe_stats["center"].shape != (variables,)
                or probe._probe_stats["scale"].shape != (variables,)
                or np.any(~np.isfinite(probe._probe_stats["center"]))
                or np.any(~np.isfinite(probe._probe_stats["scale"]))
                or np.any(probe._probe_stats["scale"] < probe.config.score_mad_floor)
                or not np.isfinite(probe._probe_stats["threshold"])
                or not np.isfinite(probe._probe_stats["threshold"])):
            raise ValueError("view probe calibration state is invalid")
        noise_path = path / "heads" / "fixed_noise.npy"
        if not noise_path.is_file() or status.get("fixed_noise_sha256") != _sha256_file(noise_path):
            raise ValueError("view probe fixed-noise artifact is missing or changed")
        noise = np.asarray(np.load(noise_path), dtype=np.float32)
        expected_noise = (4, probe.config.layout.total_dim)
        if noise.shape != expected_noise or not np.isfinite(noise).all():
            raise ValueError("view probe fixed-noise shape or values are invalid")
        generator = torch.Generator(device="cpu").manual_seed(probe.config.noise_seed)
        expected_noise_values = (torch.randn(4, probe.config.layout.total_dim, generator=generator, dtype=torch.float32)
                                 * probe.config.train_noise_std).numpy()
        if not np.array_equal(noise, expected_noise_values):
            raise ValueError("view probe fixed-noise seed or amplitude does not match configuration")
        probe._noise_vectors = noise
        with open(model_path, "rb") as handle:
            probe._gmm = pickle.load(handle)
        if len(probe._gmm) != variables:
            raise ValueError("view probe GMM variable count does not match its manifest")
        if status.get("gmm_models_sha256") != _sha256_file(model_path):
            raise ValueError("view probe GMM model bundle hash mismatch")
        if len(status.get("variables", [])) != variables:
            raise ValueError("view probe GMM variable status count mismatch")
        for variable, (status_item, model) in enumerate(zip(status["variables"], probe._gmm)):
            if status_item.get("variable") != variable:
                raise ValueError("view probe GMM variable order mismatch")
            if getattr(model, "n_features_in_", None) != probe.config.selected_dim:
                raise ValueError(f"view probe GMM variable {variable} feature width mismatch")
            if (model.n_components != probe.config.gmm_components
                    or model.covariance_type != probe.config.gmm_covariance_type
                    or float(getattr(model, "reg_covar", -1)) != float(probe.config.gmm_reg_covar)
                    or int(getattr(model, "n_init", -1)) != int(probe.config.gmm_n_init)
                    or int(getattr(model, "max_iter", -1)) != int(probe.config.gmm_max_iter)
                    or int(getattr(model, "random_state", -1)) != int(probe.config.seed)):
                raise ValueError(f"view probe GMM variable {variable} configuration mismatch")
            if not getattr(model, "converged_", False):
                raise ValueError(f"view probe GMM variable {variable} is not converged")
            if (status_item.get("n_features_in") != probe.config.selected_dim
                    or status_item.get("n_components") != probe.config.gmm_components
                    or status_item.get("covariance_type") != probe.config.gmm_covariance_type
                    or not status_item.get("converged")):
                raise ValueError("view probe per-variable GMM metadata mismatch")
            dimension = int(model.means_.shape[1])
            if model.covariance_type == "full":
                covariance_count = model.n_components * dimension * (dimension + 1) // 2
            elif model.covariance_type == "tied":
                covariance_count = dimension * (dimension + 1) // 2
            elif model.covariance_type == "diag":
                covariance_count = model.n_components * dimension
            elif model.covariance_type == "spherical":
                covariance_count = model.n_components
            else:
                raise ValueError("view probe GMM covariance type is unsupported")
            expected_free = (model.n_components - 1) + model.means_.size + covariance_count
            if status_item.get("parameter_count_free") != int(expected_free):
                raise ValueError("view probe GMM parameter count metadata mismatch")
            for array in (model.weights_, model.means_, model.covariances_):
                if not np.isfinite(array).all():
                    raise ValueError("view probe GMM contains non-finite parameters")
            if np.any(model.weights_ <= 0) or not np.isclose(np.sum(model.weights_), 1.0):
                raise ValueError("view probe GMM weights are invalid")
        probe._feature_mean = mean
        probe._feature_scale = scale
        probe._loaded_head = "gmm"
        cache_root = Path(source_dir).resolve() if source_dir else path
        probe._loaded_test_cache = cache_root / "cache" / "test_features.npy"
        probe._loaded_test_endpoints = cache_root / "cache" / "test_window_end_indices.npy"
        if not probe._loaded_test_cache.is_file() or not probe._loaded_test_endpoints.is_file():
            raise ValueError("view probe GMM test cache is missing")
        probe._fitted = True
    elif mode == "flow":
        if manifest.get("head_status") != "complete":
            raise ValueError("view probe Flow head is not marked complete")
        source_dir = manifest.get("source_dir")
        if not source_dir:
            raise ValueError("view probe Flow artifact has no source encoder directory")
        source = Path(source_dir).resolve()
        source_manifest_path = source / "probe_manifest.json"
        if not source_manifest_path.is_file():
            raise ValueError("view probe Flow source manifest is missing")
        source_manifest = json.loads(source_manifest_path.read_text())
        source_is_frozen = (
            source_manifest.get("artifact_type") == ARTIFACT_SOURCE
            and source_manifest.get("stage_status") == "stage1_complete"
        ) or (
            source_manifest.get("artifact_type") == ARTIFACT_HEAD
            and source_manifest.get("head_type") == "gmm"
            and source_manifest.get("stage_status") == "complete"
        )
        if source_manifest.get("schema") != PROBE_SCHEMA or not source_is_frozen:
            raise ValueError("view probe Flow source is not a complete stage-one artifact")
        variables = len(manifest.get("variable_names", ()))
        if variables < 1:
            raise ValueError("view probe Flow artifact has no variable layout")
        source_checkpoint = source / "checkpoints" / "best_encoder.pt"
        source_hash = source_manifest.get("checkpoint_sha256", {}).get("best_encoder.pt")
        if not source_checkpoint.is_file() or source_hash is None or _sha256_file(source_checkpoint) != source_hash:
            raise ValueError("view probe Flow source encoder checkpoint is missing or changed")
        for name in ("train", "validation", "calibration", "test"):
            cache_path = source / "cache" / f"{name}_features.npy"
            endpoint_path = source / "cache" / f"{name}_window_end_indices.npy"
            if (not cache_path.is_file() or not endpoint_path.is_file()
                    or _sha256_file(cache_path) != source_manifest.get("cache_sha256", {}).get(name)
                    or _sha256_file(endpoint_path) != source_manifest.get("endpoint_sha256", {}).get(name)):
                raise ValueError(f"view probe Flow source cache binding mismatch: {name}")
        flow_checkpoint = path / "checkpoints" / "best_fusion.pt"
        flow_stats = path / "heads" / "feature_statistics.npz"
        calibration_path = path / "heads" / "calibration.npz"
        noise_path = path / "heads" / "fixed_noise.npy"
        required = (flow_checkpoint, flow_stats, calibration_path, noise_path)
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise ValueError("view probe Flow artifact is incomplete: " + ", ".join(missing))
        if manifest.get("flow_checkpoint_sha256") != _sha256_file(flow_checkpoint):
            raise ValueError("view probe Flow checkpoint hash mismatch")
        if manifest.get("input_statistics_sha256") != _sha256_file(flow_stats):
            raise ValueError("view probe Flow feature statistics hash mismatch")
        if manifest.get("calibration_sha256") != _sha256_file(calibration_path):
            raise ValueError("view probe Flow calibration hash mismatch")
        if manifest.get("fixed_noise_sha256") != _sha256_file(noise_path):
            raise ValueError("view probe Flow fixed-noise hash mismatch")
        source_input = source / "input_statistics.npz"
        if not source_input.is_file() or _sha256_file(source_input) != source_manifest.get("input_statistics_sha256"):
            raise ValueError("view probe Flow source input statistics are missing or changed")
        with np.load(flow_stats, allow_pickle=False) as stats:
            mean = np.asarray(stats["mean"], dtype=np.float32)
            scale = np.asarray(stats["scale"], dtype=np.float32)
        expected = (variables, probe.config.layout.total_dim)
        if mean.shape != expected or scale.shape != expected or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale < probe.config.feature_std_floor):
            raise ValueError("view probe Flow feature statistics are invalid")
        with np.load(calibration_path, allow_pickle=False) as calibration:
            center = np.asarray(calibration["center"], dtype=np.float32)
            scale_score = np.asarray(calibration["scale"], dtype=np.float32)
            threshold = float(np.asarray(calibration["threshold"]))
        if center.shape != (variables,) or scale_score.shape != (variables,) or not np.isfinite(center).all() or not np.isfinite(scale_score).all() or np.any(scale_score < probe.config.score_mad_floor) or not np.isfinite(threshold):
            raise ValueError("view probe Flow calibration state is invalid")
        noise = np.asarray(np.load(noise_path), dtype=np.float32)
        generator = torch.Generator(device="cpu").manual_seed(probe.config.noise_seed)
        expected_noise = (torch.randn(4, probe.config.layout.total_dim, generator=generator, dtype=torch.float32) * probe.config.train_noise_std).numpy()
        if noise.shape != expected_noise.shape or not np.array_equal(noise, expected_noise):
            raise ValueError("view probe Flow fixed-noise state does not match configuration")
        trainer = MNRFTrainer(probe.config.mnrf, variables, probe.device)
        state = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        trainer.encoder.load_state_dict(state.get("encoder", state))
        with np.load(source_input, allow_pickle=False) as stats:
            trainer.fixed_input.mean_ = np.asarray(stats["mean"], dtype=np.float32)
            trainer.fixed_input.scale_ = np.asarray(stats["scale"], dtype=np.float32)
        trainer.fixed_feature.mean_ = mean
        trainer.fixed_feature.scale_ = scale
        trainer.noise_vectors = torch.as_tensor(noise, device=probe.device)
        trainer.flow.load_state_dict(torch.load(flow_checkpoint, map_location="cpu", weights_only=False))
        trainer.encoder.eval(); trainer.flow.eval()
        probe.trainer = trainer
        probe._probe_stats = {"center": center, "scale": scale_score, "threshold": threshold}
        probe._noise_vectors = noise
        probe._feature_mean = mean
        probe._feature_scale = scale
        probe._loaded_head = "flow"
        probe._loaded_test_cache = source / "cache" / "test_features.npy"
        probe._loaded_test_endpoints = source / "cache" / "test_window_end_indices.npy"
        if not probe._loaded_test_cache.is_file() or not probe._loaded_test_endpoints.is_file():
            raise ValueError("view probe Flow source test cache is missing")
        probe._fitted = True
    _validate_saved_results(probe, path, manifest)
    return probe


def _validate_saved_results(probe, path, manifest):
    """Validate result arrays and recompute scores/metrics from frozen state."""
    result_dir = Path(path) / "results"
    metrics_path = result_dir / "metrics.json"
    aligned_path = result_dir / "aligned_scores.npz"
    nll_path = result_dir / "test_nll.npy"
    required_hashes = {
        "metrics_sha256": metrics_path,
        "aligned_scores_sha256": aligned_path,
        "test_nll_sha256": nll_path,
    }
    for field, artifact in required_hashes.items():
        declared = manifest.get(field)
        if not declared or not artifact.is_file() or _sha256_file(artifact) != declared:
            raise ValueError(f"view probe result binding is missing or changed: {field}")
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("status") not in {"complete", "incomplete"}:
        raise ValueError("view probe metrics has an invalid status")
    with np.load(aligned_path, allow_pickle=False) as payload:
        required = {"scores", "predictions", "time_indices", "local_indices", "valid_mask",
                    "aligned_scores", "aligned_predictions"}
        if not required.issubset(payload.files):
            raise ValueError("view probe aligned result is incomplete")
        scores = np.asarray(payload["scores"], dtype=np.float64).reshape(-1)
        predictions = np.asarray(payload["predictions"], dtype=np.int64).reshape(-1)
        endpoints = _strict_integer_indices(payload["time_indices"], "result time_indices")
        if "time_values_ns" in payload.files:
            raw_time_values = np.asarray(payload["time_values_ns"])
            if raw_time_values.ndim != 1 or raw_time_values.dtype.kind not in "iu":
                raise ValueError("result time_values_ns must be a one-dimensional integer array")
            if raw_time_values.dtype.kind == "u" and np.any(raw_time_values > np.iinfo(np.int64).max):
                raise ValueError("result time_values_ns overflow int64")
            time_values = np.asarray(raw_time_values, dtype=np.int64)
            if len(time_values) != len(endpoints):
                raise ValueError("view probe result time values do not match endpoints")
        local = _strict_integer_indices(payload["local_indices"], "result local_indices")
        valid_mask = np.asarray(payload["valid_mask"], dtype=bool).reshape(-1)
        aligned_scores = np.asarray(payload["aligned_scores"], dtype=np.float64).reshape(-1)
        aligned_predictions = np.asarray(payload["aligned_predictions"], dtype=np.int64).reshape(-1)
        if len(scores) != len(predictions) or len(scores) != len(endpoints) or len(scores) != len(local):
            raise ValueError("view probe result vector lengths disagree")
        if len(np.unique(local)) != len(local) or np.any(local < 0) or np.any(local >= len(valid_mask)):
            raise ValueError("view probe result local indices are invalid")
        if len(aligned_scores) != len(valid_mask) or len(aligned_predictions) != len(valid_mask):
            raise ValueError("view probe aligned result lengths disagree")
        if not np.isfinite(scores).all() or not np.isfinite(aligned_scores[valid_mask]).all():
            raise ValueError("view probe effective scores contain NaN or Inf")
        if not np.array_equal(valid_mask[local], np.ones(len(local), dtype=bool)):
            raise ValueError("view probe valid mask does not cover all effective scores")
        if not np.array_equal(aligned_scores[local], scores) or not np.array_equal(aligned_predictions[local], predictions):
            raise ValueError("view probe aligned scores do not match effective scores")
        if np.any(aligned_predictions[~valid_mask] != -1) or np.any(np.isfinite(aligned_scores[~valid_mask])):
            raise ValueError("view probe invalid positions are not represented by the required sentinels")
        actual = np.asarray(payload["actual"]).reshape(-1) if "actual" in payload.files else None
    features = np.load(probe._loaded_test_cache, mmap_mode="r")
    expected_width = int(manifest.get("cache_feature_dim") or probe.config.layout.total_dim)
    variables = len(manifest.get("variable_names", ()))
    if features.dtype != np.float32 or features.ndim != 3 or features.shape != (len(scores), variables, expected_width):
        raise ValueError("view probe frozen test cache shape/dtype does not match the result")
    endpoints_file = _strict_integer_indices(np.load(probe._loaded_test_endpoints, mmap_mode="r"), "cache endpoints")
    if not np.array_equal(endpoints, endpoints_file):
        raise ValueError("view probe result endpoints do not match the frozen test cache")
    nll = np.load(nll_path, mmap_mode="r")
    if nll.ndim != 2 or nll.shape != (len(scores), variables) or not np.isfinite(nll).all():
        raise ValueError("view probe saved NLL has invalid shape or values")
    if probe._loaded_head == "gmm":
        expected_nll = probe._gmm_scores(features)
    else:
        expected_nll = probe.trainer.per_variable_scores(features)
    if expected_nll.shape != nll.shape or not np.allclose(expected_nll, nll, rtol=0, atol=1e-6):
        raise ValueError("view probe saved NLL does not reproduce from the frozen head")
    center = probe._probe_stats["center"]
    scale = probe._probe_stats["scale"]
    threshold = float(probe._probe_stats["threshold"])
    expected_score = np.max((nll - center) / scale, axis=1)
    expected_prediction = (expected_score > threshold).astype(np.int64)
    if not np.allclose(expected_score, scores, rtol=0, atol=1e-6) or not np.array_equal(expected_prediction, predictions):
        raise ValueError("view probe saved score does not reproduce from calibration")
    if actual is not None:
        if len(actual) != len(scores) or not np.isfinite(actual).all():
            raise ValueError("view probe actual labels are invalid")
        from sklearn.metrics import average_precision_score, roc_auc_score
        expected_pr = float(average_precision_score(actual, scores))
        if metrics.get("auc_pr") is None or not np.isfinite(metrics["auc_pr"]) or not np.isclose(metrics["auc_pr"], expected_pr, rtol=0, atol=1e-7):
            raise ValueError("view probe AUC-PR does not reproduce")
        if np.unique(actual).size == 2:
            expected_roc = float(roc_auc_score(actual, scores))
            if metrics.get("auc_roc") is None or not np.isfinite(metrics["auc_roc"]) or not np.isclose(metrics["auc_roc"], expected_roc, rtol=0, atol=1e-7):
                raise ValueError("view probe AUC-ROC does not reproduce")
        elif metrics.get("auc_roc") is not None:
            raise ValueError("view probe AUC-ROC must be null for a single-class label set")
    for key in ("auc_roc", "auc_pr"):
        if metrics.get(key) is not None and (not np.isfinite(metrics[key]) or not 0 <= float(metrics[key]) <= 1):
            raise ValueError(f"view probe metric {key} is invalid")
