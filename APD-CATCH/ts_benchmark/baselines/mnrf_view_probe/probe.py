"""Frozen four-view representation probes.

This module deliberately owns a separate artifact format. It imports the
existing MNRF-Lite V2 encoder/trainer primitives but never writes their run
directories or changes their model/scoring code.
"""

import hashlib
import json
import math
import os
import pickle
import platform
import random
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
    EndWindowDataset, MNRFTrainer, atomic_torch_save, purged_ranges, seed_all,
)
from ts_benchmark.baselines.mnrf_lite_v2.windows import (
    TimeIndex, continuous_segments, infer_expected_interval, intersect_segments, window_endpoints,
)
from .config import VALID_VIEWS, ViewProbeConfig


PROBE_SCHEMA = "mnrf-view-probe-v2"
ARTIFACT_SOURCE = "source_encoder"
ARTIFACT_HEAD = "probe_head"


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
    for key in ("output_dir", "resume_from", "resolved_layout", "selected_indices", "selected_dim"):
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
        payload = {
            "schema": PROBE_SCHEMA,
            "run_identity": self._run_identity,
            "task_name": self.config.task_name,
            "mode": self.config.mode,
            "config": self.config.to_dict(),
            "source_identity": self.source_identity,
            "stage": extra.pop("stage", "created"),
        }
        payload.update(extra)
        _atomic_json(self.output_dir / "run_request.json", payload)

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
        boundary_ids = MNRFLiteV2._boundary_ids(frame)
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
            "entity_boundary_columns": [str(c) for c in MNRFLiteV2._boundary_columns(frame)],
            "fit_range": [int(a), int(b)],
        }
        self._current_test_endpoints = np.asarray(test_endpoints, dtype=np.int64)
        self._test_length = int(len(values) - train_length)
        self._time_values_ns = np.asarray(time_index.values_ns, dtype=np.int64)
        self._time_index_kind = time_index.kind
        self._source_timezone = time_index.source_timezone
        boundary_columns = [str(c) for c in MNRFLiteV2._boundary_columns(frame)]
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

    def _normalized_memmap(self, values):
        path = self.output_dir / "cache" / "normalized_values.npy"
        mmap = np.lib.format.open_memmap(path.with_suffix(".partial.npy"), mode="w+", dtype=np.float32, shape=values.shape)
        self.trainer.fixed_input.fit(values[self.ranges["train"][0]:self.ranges["train"][1]])
        for offset in range(0, len(values), 65536):
            mmap[offset:offset + min(65536, len(values) - offset)] = self.trainer.fixed_input.transform_chunk(values[offset:offset + 65536])
        mmap.flush(); del mmap
        os.replace(path.with_suffix(".partial.npy"), path)
        return np.load(path, mmap_mode="r")

    def _cache_all(self, normalized, values, test_endpoints):
        cache = {}
        for name in ("train", "validation", "calibration"):
            path, count, index_path = self.trainer.extract_cache(
                normalized, self.ranges[name], name, self.output_dir / "cache", self.segments
            )
            cache[name] = np.load(path, mmap_mode="r")
        test_path = self.output_dir / "cache" / "test_features.npy"
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
        return cache, {name: self.output_dir / "cache" / f"{name}_window_end_indices.npy" for name in ("train", "validation", "calibration", "test")}

    def _cache_s_only(self, normalized, test_endpoints):
        """Export the S view directly, without training an encoder."""
        cache = {}
        endpoint_paths = {}
        for name, (start, end) in [(name, self.ranges[name]) for name in ("train", "validation", "calibration")]:
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
        endpoints = np.asarray(test_endpoints, dtype=np.int64)
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
            "head_type": None, "stage_status": "source_reused" if source_manifest else "cache_ready",
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
        _atomic_json(self.output_dir / "probe_manifest.json", payload)
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
            cache[name] = self._validate_cache_entry(
                path, endpoint_path, name,
                manifest.get("cache_sha256", {}).get(name),
                manifest.get("endpoint_sha256", {}).get(name),
            )
        declared_sample_hash = manifest.get("gmm_fit_endpoints_sha256")
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
        progress_identity = _json_hash({"config": fit_config, "sample_hash": sample_hash,
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
                )
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

    def _fixed_noise_vectors(self):
        if self._noise_vectors is None:
            generator = torch.Generator(device="cpu").manual_seed(self.config.noise_seed)
            self._noise_vectors = (torch.randn(
                4, self.config.layout.total_dim, generator=generator, dtype=torch.float32
            ) * self.config.train_noise_std).numpy()
        return np.asarray(self._noise_vectors, dtype=np.float32)

    def _gmm_scores(self, features, output_path=None):
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
            for noise in noises:
                for start in range(0, len(features), self.config.batch_size):
                    stop = min(len(features), start + self.config.batch_size)
                    for variable, model in enumerate(self._gmm):
                        values = np.asarray(features[start:stop, variable][:, selected], dtype=np.float32)
                        values = (values - feature_mean[variable, selected]) / feature_scale[variable, selected]
                        values = values + noise[selected]
                        output[start:stop, variable] += -model.score_samples(values) / len(noises)
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite GMM NLL")
        if hasattr(output, "flush"):
            output.flush()
        return output

    def _calibrate(self, cache):
        validation = self._gmm_scores(cache["validation"], self.output_dir / "heads" / "validation_nll.npy")
        if not np.isfinite(validation).all():
            raise FloatingPointError("non-finite validation NLL")
        center = np.median(validation, axis=0).astype(np.float32)
        scale = np.maximum(1.4826 * np.median(np.abs(validation - center), axis=0), self.config.score_mad_floor).astype(np.float32)
        calibration = self._gmm_scores(cache["calibration"], self.output_dir / "heads" / "calibration_nll.npy")
        aggregate = np.max((calibration - center) / scale, axis=1)
        threshold = float(np.quantile(aggregate, self.config.calibration_quantile))
        if not np.isfinite(threshold) or np.any(scale <= 0):
            raise FloatingPointError("invalid GMM calibration scale or threshold")
        self._probe_stats = {"center": center, "scale": scale, "threshold": threshold}
        np.savez(self.output_dir / "heads" / "calibration.npz", center=center, scale=scale,
                 threshold=np.asarray(threshold), quantile=np.asarray(self.config.calibration_quantile))

    def _write_scores(self, cache, test_endpoints, labels):
        inference_start = time.perf_counter()
        nll = self._gmm_scores(cache["test"], self.output_dir / "results" / "test_nll.npy")
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
        return metrics

    def _set_protocol_identity(self, frame, train_length):
        values, test_endpoints, continuity = self._protocol(frame, int(train_length))
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
        if manifest is not None and manifest.get("artifact_type") == ARTIFACT_HEAD and not self.config.source_dir and self.config.mode != "s_only":
            if not (self.output_dir / "input_statistics.npz").is_file():
                raise ValueError("resume rejected: local head is missing input statistics")
        # A completed stage-one source must have its complete source contract.
        if manifest is not None and manifest.get("artifact_type") == ARTIFACT_SOURCE:
            for name in ("initial_encoder.pt", "best_encoder.pt"):
                if not (self.output_dir / "checkpoints" / name).is_file():
                    raise ValueError("resume rejected: completed source is missing " + name)
            if not (self.output_dir / "input_statistics.npz").is_file():
                raise ValueError("resume rejected: completed source is missing input statistics")
            # Stage-one exports intentionally have no score/head artifacts;
            # use the source contract above instead of the deployment loader,
            # which also validates final results.
            for name in ("train", "validation", "calibration", "test"):
                cache_path = self.output_dir / "cache" / f"{name}_features.npy"
                endpoint_path = self.output_dir / "cache" / f"{name}_window_end_indices.npy"
                if not cache_path.is_file() or not endpoint_path.is_file():
                    raise ValueError("resume rejected: completed source cache is incomplete")
            self._resume_stage1_complete = status.get("status") == "stage1_complete"
        gmm_mode = self.config.mode in {"full_gmm", "select_gmm", "random_gmm", "s_only"}
        if ((manifest is not None and manifest.get("artifact_type") == ARTIFACT_HEAD
             and manifest.get("head_type") == "gmm") or gmm_mode):
            progress_path = self.output_dir / "heads" / "gmm_progress.json"
            if not progress_path.is_file() and status.get("status") == "gmm_fitting":
                raise ValueError("resume rejected: GMM progress manifest is missing")
            if progress_path.is_file():
                progress = json.loads(progress_path.read_text())
                completed = progress.get("completed", {})
                if not isinstance(completed, dict):
                    raise ValueError("resume rejected: GMM progress completed map is invalid")
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
        self._prepare_dirs()
        write_allowed = True
        try:
            seed_all(self.config.seed)
            # The probe's CPU budget is explicit and intentionally scoped to
            # this process; it does not change server-wide thread settings.
            torch.set_num_threads(self.config.cpu_threads)
            values, test_endpoints, continuity = self._set_protocol_identity(frame, int(train_length))
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
                    self._set_run_status("stage1_training")
                    self.trainer.stage1(
                        normalized, self.ranges["train"], self.ranges["validation"],
                        self.output_dir / "checkpoints", self.segments,
                        self.output_dir / "training.jsonl", self._run_identity,
                    )
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
                self._fit_flow(cache, labels, test_endpoints)
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
            self._set_run_status("complete")
            return self
        except Exception as error:
            if write_allowed:
                self._set_run_status("failed", repr(error))
            raise

    def _fit_flow(self, cache, labels, test_endpoints):
        with threadpool_limits(limits=self.config.cpu_threads):
            self.trainer.fixed_feature.fit(cache["train"], chunk_size=self.config.cache_chunk_size)
        np.savez(self.output_dir / "heads" / "feature_statistics.npz",
                 mean=np.asarray(self.trainer.fixed_feature.mean_, dtype=np.float32),
                 scale=np.asarray(self.trainer.fixed_feature.scale_, dtype=np.float32))
        stage2_identity = _json_hash({"source_identity": self.source_identity,
                                      "feature_fingerprint": self.trainer.fingerprint(),
                                      "layout": self.config.layout.to_dict()})
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


def load_probe(path, device="cpu"):
    path = Path(path)
    manifest_path = path / "probe_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"view probe manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != PROBE_SCHEMA:
        raise ValueError("view probe artifact schema is unsupported; re-export with the current probe")
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
    config = dict(manifest["config"])
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
        for name, declared_hash in manifest.get("source_cache_sha256", {}).items():
            cache_path = source / "cache" / f"{name}_features.npy"
            if not cache_path.is_file() or _sha256_file(cache_path) != declared_hash:
                raise ValueError(f"view probe source cache hash mismatch: {name}")
        for name, declared_hash in manifest.get("source_endpoint_sha256", {}).items():
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
        if status_config is None or _runtime_free_config(status_config) != _runtime_free_config(probe.config):
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
