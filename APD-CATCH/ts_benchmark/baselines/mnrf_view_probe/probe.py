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
from sklearn.mixture import GaussianMixture

from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
from ts_benchmark.baselines.mnrf_lite_v2.layout import FeatureLayout, require_layout
from ts_benchmark.baselines.mnrf_lite_v2.training import (
    EndWindowDataset, MNRFTrainer, purged_ranges, seed_all,
)
from ts_benchmark.baselines.mnrf_lite_v2.windows import (
    TimeIndex, continuous_segments, infer_expected_interval, intersect_segments,
)
from .config import VALID_VIEWS, ViewProbeConfig


PROBE_SCHEMA = "mnrf-view-probe-v1"


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


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _safe_name(value):
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))


def _select_stratified(endpoints, segments, limit, seed):
    endpoints = np.asarray(endpoints, dtype=np.int64)
    if len(endpoints) <= limit:
        return endpoints.copy()
    rng = np.random.default_rng(seed)
    groups = []
    for left, right in segments:
        group = endpoints[(endpoints >= left) & (endpoints < right)]
        if len(group):
            groups.append(group)
    chosen = []
    remaining = limit
    for index, group in enumerate(groups):
        take = min(len(group), max(1, int(round(limit * len(group) / len(endpoints)))))
        if index == len(groups) - 1:
            take = min(len(group), remaining)
        selected = np.sort(rng.choice(group, size=take, replace=False))
        chosen.append(selected)
        remaining -= take
    if remaining > 0:
        available = np.setdiff1d(endpoints, np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64))
        chosen.append(np.sort(rng.choice(available, size=min(remaining, len(available)), replace=False)))
    return np.sort(np.concatenate(chosen))[:limit]


class MNRFViewProbe:
    """One independent probe run over frozen MNRF-Lite V2 views."""

    model_name = "MNRFViewProbe"

    def __init__(self, **kwargs):
        self.config = ViewProbeConfig.from_kwargs(**kwargs)
        self.device = self._select_device(self.config.device)
        self.output_dir = Path(self.config.output_dir) if self.config.output_dir else None
        self.trainer = None
        self.variable_names = None
        self.temporal_rule = None
        self.ranges = None
        self.segments = None
        self._fitted = False
        self._gmm = None
        self._probe_stats = None

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
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for name in ("checkpoints", "cache", "heads", "results"):
            (self.output_dir / name).mkdir()

    @staticmethod
    def _features(frame):
        values, columns = MNRFLiteV2._features_from_frame(frame)
        return values, tuple(map(str, columns))

    def _protocol(self, frame, train_length):
        values, columns = self._features(frame)
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
        }
        return values, test_endpoints, continuity

    def _make_trainer(self, variables):
        config = self.config.mnrf
        config.task_name = self.config.task_name or "view_probe"
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

    def _save_source_metadata(self, mode, cache, endpoint_paths, initial_state=None):
        payload = {
            "schema": PROBE_SCHEMA, "mode": mode, "task_name": self.config.task_name,
            "config": self.config.to_dict(), "layout": self.config.layout.to_dict(),
            "variable_names": list(self.variable_names), "temporal_rule": self.temporal_rule,
            "ranges": {name: list(self.ranges[name]) for name in ("train", "validation", "calibration")},
            "segments": self.segments, "cache_paths": {name: str(path) for name, path in endpoint_paths.items()},
            "cache_sha256": {name: _sha256_file(self.output_dir / "cache" / f"{name}_features.npy") for name in cache},
            "endpoint_sha256": {name: _sha256_file(path) for name, path in endpoint_paths.items()},
            "feature_fingerprint": self.trainer.fingerprint(),
            "initial_encoder_saved": initial_state is not None,
            "created_with": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        }
        _atomic_json(self.output_dir / "probe_manifest.json", payload)
        return payload

    def _save_probe_metadata(self, mode, source_manifest=None):
        payload = {
            "schema": PROBE_SCHEMA, "mode": mode, "config": self.config.to_dict(),
            "layout": self.config.layout.to_dict(), "variable_names": list(self.variable_names),
            "temporal_rule": self.temporal_rule, "source_dir": self.config.source_dir,
            "source_feature_fingerprint": None if source_manifest is None else source_manifest.get("feature_fingerprint"),
            "selected_views": list(self.config.selected_views),
            "selected_indices": self.config.selected_indices, "selected_dim": self.config.selected_dim,
        }
        if source_manifest is not None:
            payload["source_cache_sha256"] = source_manifest.get("cache_sha256", {})
            payload["source_endpoint_sha256"] = source_manifest.get("endpoint_sha256", {})
            payload["source_gmm_fit_endpoints_sha256"] = source_manifest.get("gmm_fit_endpoints_sha256")
        _atomic_json(self.output_dir / "probe_manifest.json", payload)
        return payload

    def _load_source(self, source_dir):
        source = Path(source_dir)
        manifest = json.loads((source / "probe_manifest.json").read_text())
        current_temporal_rule = self.temporal_rule
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
        if current_temporal_rule != manifest.get("temporal_rule"):
            raise ValueError("view probe source temporal protocol differs")
        if tuple(manifest.get("variable_names", ())) != tuple(self.variable_names):
            raise ValueError("view probe source variable order differs")
        if manifest.get("mode") not in {"stage1", "pretrained", "random"}:
            raise ValueError("view probe source is not a stage-1 encoder artifact")
        self.trainer = self._make_trainer(len(self.variable_names))
        checkpoint = source / "checkpoints" / ("initial_encoder.pt" if self.config.mode == "random_gmm" else "best_encoder.pt")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"source encoder checkpoint missing: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        encoder_state = state.get("encoder", state)
        self.trainer.encoder.load_state_dict(encoder_state)
        self.trainer.encoder.eval()
        with np.load(source / "input_statistics.npz", allow_pickle=False) as stats:
            self.trainer.fixed_input.mean_ = stats["mean"].astype(np.float32)
            self.trainer.fixed_input.scale_ = stats["scale"].astype(np.float32)
        if self.trainer.fingerprint() != manifest.get("feature_fingerprint"):
            raise ValueError("view probe source encoder/input-statistics fingerprint mismatch")
        self.variable_names = tuple(manifest["variable_names"])
        self.temporal_rule = manifest["temporal_rule"]
        self.ranges = {name: tuple(value) for name, value in manifest["ranges"].items()}
        self.segments = [tuple(item) for item in manifest["segments"]]
        cache = {}
        for name in ("train", "validation", "calibration", "test"):
            path = source / "cache" / f"{name}_features.npy"
            if not path.is_file() or _sha256_file(path) != manifest.get("cache_sha256", {}).get(name):
                raise ValueError(f"view probe source cache is missing or changed: {path}")
            endpoint_path = source / "cache" / f"{name}_window_end_indices.npy"
            if not endpoint_path.is_file() or _sha256_file(endpoint_path) != manifest.get("endpoint_sha256", {}).get(name):
                raise ValueError(f"view probe source endpoint index is missing or changed: {endpoint_path}")
            values = np.load(path, mmap_mode="r")
            expected = (len(values), len(self.variable_names), self.config.layout.total_dim)
            if values.shape != expected:
                raise ValueError(f"view probe source cache {name} shape mismatch: expected {expected}, got {values.shape}")
            cache[name] = values
        declared_sample_hash = manifest.get("gmm_fit_endpoints_sha256")
        if declared_sample_hash is not None:
            sample_path = source / "heads" / "gmm_fit_endpoints.npy"
            if not sample_path.is_file() or _sha256_file(sample_path) != declared_sample_hash:
                raise ValueError("view probe source GMM fitting endpoint sample is missing or changed")
        return source, manifest, cache

    def _fit_gmm(self, cache, source_manifest=None):
        train = cache["train"]
        self.trainer.fixed_feature.fit(train, chunk_size=self.config.cache_chunk_size)
        np.savez(
            self.output_dir / "heads" / "feature_statistics.npz",
            mean=np.asarray(self.trainer.fixed_feature.mean_, dtype=np.float32),
            scale=np.asarray(self.trainer.fixed_feature.scale_, dtype=np.float32),
        )
        selected = np.asarray(self.config.selected_indices, dtype=np.int64)
        endpoints_path = self.output_dir / "cache" / "train_window_end_indices.npy"
        if source_manifest is not None:
            endpoints_path = Path(self.config.source_dir) / "cache" / "train_window_end_indices.npy"
        endpoints = np.load(endpoints_path, mmap_mode="r")
        source_sample = None if source_manifest is None else Path(self.config.source_dir) / "heads" / "gmm_fit_endpoints.npy"
        if source_sample is not None and source_sample.is_file():
            fit_indices = np.asarray(np.load(source_sample), dtype=np.int64)
        else:
            fit_indices = _select_stratified(endpoints, self.segments, min(self.config.gmm_fit_max_windows, len(train)), self.config.seed)
        if fit_indices.ndim != 1 or len(fit_indices) == 0:
            raise ValueError("GMM fitting endpoint sample must be a non-empty one-dimensional array")
        positions = np.searchsorted(endpoints, fit_indices)
        if (np.any(positions >= len(endpoints))
                or not np.array_equal(np.asarray(endpoints)[positions], fit_indices)):
            raise ValueError("GMM fitting endpoint sample is not a subset of the source train endpoints")
        sample_path = self.output_dir / "heads" / "gmm_fit_endpoints.npy"
        np.save(sample_path, fit_indices)
        sample_hash = _sha256_file(sample_path)
        self._gmm = []
        statuses = []
        old_threads = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = str(self.config.cpu_threads)
        try:
            for variable in range(train.shape[1]):
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
                self._gmm.append(model)
                parameter_count = int(model.weights_.size + model.means_.size + model.covariances_.size)
                statuses.append({"variable": variable, "converged": bool(model.converged_), "n_iter": int(model.n_iter_), "lower_bound": float(model.lower_bound_), "sample_count": len(values), "parameter_count": parameter_count})
        finally:
            if old_threads is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = old_threads
        with open(self.output_dir / "heads" / "gmm_models.pkl", "wb") as handle:
            pickle.dump(self._gmm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _atomic_json(self.output_dir / "heads" / "gmm_status.json", {
            "config": self.config.to_dict(), "sklearn_version": __import__("sklearn").__version__,
            "fit_endpoint_count": int(len(fit_indices)), "fit_endpoint_sha256": sample_hash,
            "variables": statuses,
            "head_parameter_count": sum(item["parameter_count"] for item in statuses),
        })
        manifest_path = self.output_dir / "probe_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            manifest["gmm_fit_endpoints_sha256"] = sample_hash
            manifest["gmm_fit_endpoint_count"] = int(len(fit_indices))
            _atomic_json(manifest_path, manifest)

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
        for start in range(0, len(features), self.config.batch_size):
            stop = min(len(features), start + self.config.batch_size)
            for variable, model in enumerate(self._gmm):
                values = np.asarray(features[start:stop, variable][:, selected], dtype=np.float32)
                values = (values - feature_mean[variable, selected]) / feature_scale[variable, selected]
                output[start:stop, variable] = -model.score_samples(values)
        if hasattr(output, "flush"):
            output.flush()
        return output

    def _calibrate(self, cache):
        validation = self._gmm_scores(cache["validation"], self.output_dir / "heads" / "validation_nll.npy")
        center = np.median(validation, axis=0).astype(np.float32)
        scale = np.maximum(1.4826 * np.median(np.abs(validation - center), axis=0), self.config.score_mad_floor).astype(np.float32)
        calibration = self._gmm_scores(cache["calibration"], self.output_dir / "heads" / "calibration_nll.npy")
        aggregate = np.max((calibration - center) / scale, axis=1)
        threshold = float(np.quantile(aggregate, self.config.calibration_quantile))
        self._probe_stats = {"center": center, "scale": scale, "threshold": threshold}
        np.savez(self.output_dir / "heads" / "calibration.npz", center=center, scale=scale, threshold=np.asarray(threshold))

    def _write_scores(self, cache, test_endpoints, labels):
        inference_start = time.perf_counter()
        nll = self._gmm_scores(cache["test"], self.output_dir / "results" / "test_nll.npy")
        score = np.max((nll - self._probe_stats["center"]) / self._probe_stats["scale"], axis=1)
        prediction = (score > self._probe_stats["threshold"]).astype(np.int64)
        labels = np.asarray(labels).reshape(-1)
        # Endpoints remain full-series coordinates; labels are test-local.
        local = np.asarray(test_endpoints) - int(self._test_offset)
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
        if np.unique(actual).size == 2:
            metrics["auc_roc"] = float(roc_auc_score(actual, score))
        else:
            metrics["status"] = "incomplete"
            metrics["auc_undefined_reason"] = "valid test labels contain one class"
        if np.unique(actual).size >= 1:
            metrics["auc_pr"] = float(average_precision_score(actual, score))
        np.savez_compressed(self.output_dir / "results" / "aligned_scores.npz", scores=score, predictions=prediction,
                            time_indices=np.asarray(test_endpoints), actual=actual, threshold=np.asarray(self._probe_stats["threshold"]))
        _atomic_json(self.output_dir / "results" / "metrics.json", metrics)
        return metrics

    def fit(self, frame, train_length, labels=None):
        self._fit_started = time.perf_counter()
        self._prepare_dirs()
        seed_all(self.config.seed)
        # The probe's CPU budget is explicit and intentionally single-threaded
        # by default; this also prevents a short validation from occupying all
        # shared server cores through PyTorch's intra-op pool.
        torch.set_num_threads(self.config.cpu_threads)
        values, test_endpoints, continuity = self._protocol(frame, int(train_length))
        self._test_offset = int(train_length)
        self.trainer = self._make_trainer(len(self.variable_names))
        normalized = self._normalized_memmap(values)
        initial_state = {k: v.detach().cpu().clone() for k, v in self.trainer.encoder.state_dict().items()}
        mode = self.config.mode
        source_manifest = None
        if mode in {"random_gmm", "stage1", "full_gmm"}:
            if mode != "random_gmm":
                self.trainer.stage1(normalized, self.ranges["train"], self.ranges["validation"], self.output_dir / "checkpoints", self.segments, self.output_dir / "training.jsonl", "probe")
                torch.save({"encoder": initial_state}, self.output_dir / "checkpoints" / "initial_encoder.pt")
            else:
                torch.save({"encoder": initial_state}, self.output_dir / "checkpoints" / "initial_encoder.pt")
        elif mode in {"select_gmm", "s_only", "flow"}:
            if not self.config.source_dir:
                raise ValueError(f"mode={mode} requires source_dir pointing to a validated stage-1 probe")
            source, source_manifest, source_cache = self._load_source(self.config.source_dir)
            self.trainer.fixed_input.mean_ = np.load(source / "input_statistics.npz")["mean"]
            self.trainer.fixed_input.scale_ = np.load(source / "input_statistics.npz")["scale"]
            self.trainer.encoder.eval()
        if mode in {"select_gmm", "s_only", "flow"}:
            cache = source_cache
            self._test_offset = int(train_length)
            test_endpoints = np.load(Path(self.config.source_dir) / "cache" / "test_window_end_indices.npy")
        else:
            cache, endpoint_paths = self._cache_all(normalized, values, test_endpoints)
        if mode == "stage1":
            np.savez(self.output_dir / "input_statistics.npz", mean=self.trainer.fixed_input.mean_, scale=self.trainer.fixed_input.scale_)
            self._save_source_metadata("pretrained", cache, endpoint_paths, initial_state)
            return None
        np.savez(self.output_dir / "input_statistics.npz", mean=self.trainer.fixed_input.mean_, scale=self.trainer.fixed_input.scale_)
        if mode in {"select_gmm", "s_only", "flow"}:
            self._save_probe_metadata(mode, source_manifest)
        else:
            self._save_source_metadata("random" if mode == "random_gmm" else "pretrained", cache, endpoint_paths, initial_state)
        if mode == "flow":
            if tuple(self.config.selected_views) != VALID_VIEWS:
                raise ValueError("flow mode requires the complete S,E,P,R layout")
            self._fit_flow(cache, labels, test_endpoints)
        else:
            self._fit_gmm(cache, source_manifest=source_manifest)
            self._calibrate(cache)
            if labels is not None:
                self._fit_elapsed = time.perf_counter() - self._fit_started
                self._write_scores(cache, test_endpoints, labels)
        self._fitted = True
        return self

    def _fit_flow(self, cache, labels, test_endpoints):
        self.trainer.fixed_feature.fit(cache["train"], chunk_size=self.config.cache_chunk_size)
        self.trainer.stage2(cache["train"], cache["validation"], self.output_dir / "checkpoints")
        self.trainer.calibrate(cache["validation"], cache["calibration"], self.output_dir / "heads")
        nll = self.trainer.per_variable_scores(cache["test"], self.output_dir / "results" / "test_nll.npy")
        score = np.max((nll - self.trainer.calibration_center) / self.trainer.calibration_scale, axis=1)
        prediction = (score > self.trainer.threshold).astype(np.int64)
        payload = {"status": "complete", "head": "flow", "selected_views": list(VALID_VIEWS),
                   "selected_dim": self.config.layout.total_dim, "threshold": float(self.trainer.threshold),
                   "head_parameter_count": sum(parameter.numel() for parameter in self.trainer.flow.parameters())}
        if labels is not None:
            actual = np.asarray(labels).reshape(-1)[np.asarray(test_endpoints) - self._test_offset]
            from sklearn.metrics import average_precision_score, roc_auc_score
            payload["auc_roc"] = float(roc_auc_score(actual, score)) if np.unique(actual).size == 2 else None
            payload["auc_pr"] = float(average_precision_score(actual, score))
            if payload["auc_roc"] is None:
                payload["status"] = "incomplete"
                payload["auc_undefined_reason"] = "valid test labels contain one class"
            np.savez_compressed(self.output_dir / "results" / "aligned_scores.npz", scores=score,
                                predictions=prediction, time_indices=test_endpoints, actual=actual,
                                threshold=np.asarray(self.trainer.threshold))
        else:
            np.savez_compressed(self.output_dir / "results" / "aligned_scores.npz", scores=score,
                                predictions=prediction, time_indices=test_endpoints)
        _atomic_json(self.output_dir / "results" / "metrics.json", payload)

    def detect_fit(self, train_data, train_label=None):
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        self.fit(frame, len(frame), train_label)
        return self

    def detect_score(self, test_data):
        raise RuntimeError("MNRFViewProbe uses the explicit run script; score from its validated result directory")


def load_probe(path, device="cpu"):
    path = Path(path)
    manifest_path = path / "probe_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"view probe manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    config = dict(manifest["config"])
    config.pop("resolved_layout", None); config.pop("selected_indices", None); config.pop("selected_dim", None)
    config["output_dir"] = str(path); config["device"] = device
    probe = MNRFViewProbe.from_kwargs(**config)
    require_layout(manifest.get("layout"), probe.config.layout, "view probe artifact")
    if manifest.get("selected_views", list(probe.config.selected_views)) != list(probe.config.selected_views):
        raise ValueError("view probe artifact selected view order does not match its configuration")
    if manifest.get("selected_indices", probe.config.selected_indices) != probe.config.selected_indices:
        raise ValueError("view probe artifact selected indices do not match its layout")
    mode = manifest.get("mode")
    stats_path = path / "heads" / "feature_statistics.npz"
    calibration_path = path / "heads" / "calibration.npz"
    model_path = path / "heads" / "gmm_models.pkl"
    status_path = path / "heads" / "gmm_status.json"
    sample_path = path / "heads" / "gmm_fit_endpoints.npy"
    gmm_mode = mode in {"full_gmm", "select_gmm", "s_only", "random_gmm"}
    # A full GMM run uses the source-manifest labels ``pretrained``/``random``;
    # a stage-one-only export uses the same label but has no head artifacts.
    # Distinguish them by the presence of the complete head, without silently
    # accepting a partially written head.
    if mode in {"pretrained", "random"} and any(item.exists() for item in (stats_path, calibration_path, model_path, status_path, sample_path)):
        gmm_mode = True
    if gmm_mode:
        required = (stats_path, calibration_path, model_path, status_path, sample_path)
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise ValueError("view probe GMM artifact is incomplete: " + ", ".join(missing))
        status = json.loads(status_path.read_text())
        if status.get("config", {}).get("selected_indices") != probe.config.selected_indices:
            raise ValueError("view probe GMM selected indices do not match its configuration")
        with np.load(stats_path, allow_pickle=False) as stats:
            mean = np.asarray(stats["mean"], dtype=np.float32)
            scale = np.asarray(stats["scale"], dtype=np.float32)
        variables = len(manifest.get("variable_names", ()))
        expected = (variables, probe.config.layout.total_dim)
        if mean.shape != expected or scale.shape != expected or np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)):
            raise ValueError(f"view probe feature statistics shape mismatch: expected {expected}")
        sample_hash = _sha256_file(sample_path)
        if sample_hash != status.get("fit_endpoint_sha256") or sample_hash != manifest.get("gmm_fit_endpoints_sha256"):
            raise ValueError("view probe GMM fitting endpoint hash mismatch")
        with np.load(calibration_path, allow_pickle=False) as calibration:
            probe._probe_stats = {key: np.asarray(calibration[key]) for key in ("center", "scale")}
            probe._probe_stats["threshold"] = float(np.asarray(calibration["threshold"]))
        with open(model_path, "rb") as handle:
            probe._gmm = pickle.load(handle)
        if len(probe._gmm) != variables:
            raise ValueError("view probe GMM variable count does not match its manifest")
        probe._feature_mean = mean
        probe._feature_scale = scale
        probe._fitted = True
    return probe
