#!/usr/bin/env python
"""Recover missing ViewProbe feature caches without training.

The recovery path is deliberately narrower than ``run_mnrf_view_probe.py``:
it validates the frozen source and protocol, loads the saved best encoder,
and only performs the missing cache forward passes.  A generated file is
installed only when its byte SHA256 is exactly the immutable manifest hash.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import subprocess
import tempfile
import time
import traceback
import tarfile
import uuid
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mnrf_view_probe import _read_task
from ts_benchmark.baselines.mnrf_view_probe import MNRFViewProbe
from ts_benchmark.baselines.mnrf_view_probe.integrity import (
    sha256_file,
    validate_normalized_values_cache,
)
from ts_benchmark.baselines.mnrf_view_probe.probe import _strict_integer_indices
from ts_benchmark.baselines.mnrf_view_probe.probe import EndWindowDataset


SOURCES = {
    "PSM": "full_gmm-full-20261021T112121Z-2466820",
    "MSL": "full_gmm-full-20261021T112147Z-2470335",
    "Creditcard": "full_gmm-full-20261021T112159Z-2472093",
    "CICIDS": "full_gmm-full-20261021T112642Z-2510633",
}


def _json_dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _source_path(task: str) -> Path:
    return ROOT / "result" / "score" / "MNRF-ViewProbe" / task / SOURCES[task]


def _config_kwargs(manifest, output_dir, device):
    config = dict(manifest["config"])
    for key in ("resolved_layout", "selected_indices", "selected_dim", "output_dir", "resume_from"):
        config.pop(key, None)
    config["output_dir"] = str(output_dir)
    config["resume_from"] = ""
    config["device"] = device
    return config


def _expected_cache(manifest, name):
    shape = manifest.get("cache_shapes", {}).get(name)
    digest = manifest.get("cache_sha256", {}).get(name)
    if not isinstance(shape, list) or not digest:
        raise ValueError(f"manifest has no authoritative {name} cache shape/hash")
    return tuple(int(x) for x in shape), str(digest)


def _validate_endpoint(path: Path, name: str, expected_count: int, expected_hash: str):
    if not path.is_file():
        raise ValueError(f"missing {name} endpoint file: {path}")
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(f"{name} endpoint hash mismatch: expected={expected_hash} actual={actual}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    values = _strict_integer_indices(values, f"{name} endpoints")
    if len(values) != expected_count or (len(values) > 1 and np.any(np.diff(values) <= 0)):
        raise ValueError(f"{name} endpoint count/order is invalid")
    return values


def _validate_existing_feature(path: Path, name: str, manifest):
    shape, digest = _expected_cache(manifest, name)
    if not path.is_file():
        return {"status": "missing", "expected_shape": list(shape), "expected_sha256": digest}
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"existing {name}_features.npy is not a regular non-symlink file")
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError(f"existing {name}_features.npy hash mismatch: expected={digest} actual={actual}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != shape or array.dtype != np.dtype("float32"):
        raise ValueError(f"existing {name}_features.npy shape/dtype mismatch")
    for offset in range(0, len(array), 65536):
        if not np.isfinite(array[offset:offset + 65536]).all():
            raise ValueError(f"existing {name}_features.npy contains non-finite values")
    return {"status": "present", "shape": list(array.shape), "sha256": actual,
            "bytes": int(path.stat().st_size)}


def _copy_and_hash(stream, destination):
    digest = __import__("hashlib").sha256()
    destination = Path(destination)
    try:
        with destination.open("xb") as output:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def _archive_kind(path):
    name = path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
        return "tar"
    return None


def _find_exact_backup(source: Path, name: str, expected_hash: str, report_dir=None,
                       expected_payload_bytes=None):
    """Search bounded project artifact roots, including archive members.

    Archive members are inspected by name and streamed to an invocation-owned
    temporary file only when their basename matches the exact requested cache.
    No archive is extracted as a directory and no archive link is followed.
    """
    roots = tuple(path for path in (
        ROOT / "result" / "score" / "MNRF-ViewProbe",
        ROOT / "result" / "score" / "MNRF-ViewProbe-Ablation",
        ROOT / "result" / "score" / "MNRF-ViewProbe-SOnly",
        ROOT / "result" / "score" / "MNRF-ViewProbe-Components",
        ROOT / "result" / "source",
        ROOT / "result" / "backup", ROOT / "result" / "backups",
        ROOT / "result" / "review" / "backup",
        ROOT / "result" / "analysis" / "MNRF-ViewProbe",
        ROOT / "result" / "analysis" / "MNRF-ViewProbe-Ablation",
        ROOT / "result" / "review" / "MNRF-ViewProbe",
        ROOT / "result" / "review" / "MNRF-ViewProbe-Ablation",
    ) if path.exists())
    filename = f"{name}_features.npy"
    stats = {"roots": [str(p) for p in roots], "archive_count": 0,
             "candidate_count": 0, "unsupported_archives": [], "checked_files": 0,
             "checked_members": 0}
    stage_root = Path(report_dir) if report_dir else Path(tempfile.gettempdir())
    stage_root.mkdir(parents=True, exist_ok=True)
    archives = []
    files = []
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                continue
            if candidate.is_file() and candidate.name == filename:
                files.append((root, candidate))
            elif candidate.is_file() and candidate.suffix.lower() in {".zip", ".tar", ".gz", ".bz2", ".xz", ".tgz", ".tbz2", ".txz"}:
                kind = _archive_kind(candidate)
                if kind:
                    archives.append((root, candidate, kind))
                else:
                    stats["unsupported_archives"].append(str(candidate))
    stats["archive_count"] = len(archives)
    for root, candidate in files:
        stats["candidate_count"] += 1
        stats["checked_files"] += 1
        try:
            if candidate.is_symlink() or not stat.S_ISREG(candidate.lstat().st_mode):
                continue
            if (expected_payload_bytes is not None
                    and candidate.stat().st_size < expected_payload_bytes):
                continue
            resolved = candidate.resolve()
            if root.resolve() not in resolved.parents or source.resolve() in resolved.parents:
                continue
            if sha256_file(candidate) == expected_hash:
                return {"kind": "file", "path": str(candidate), "member": None,
                        "sha256": expected_hash, "stats": stats}
        except OSError:
            continue
    for root, archive, kind in archives:
        try:
            if kind == "zip":
                with zipfile.ZipFile(archive, "r") as handle:
                    for info in handle.infolist():
                        if info.is_dir() or Path(info.filename).name != filename:
                            continue
                        if expected_payload_bytes is not None and info.file_size < expected_payload_bytes:
                            continue
                        mode = (info.external_attr >> 16) & 0o170000
                        if mode not in (0, stat.S_IFREG):
                            continue
                        stats["candidate_count"] += 1
                        stats["checked_members"] += 1
                        staged = stage_root / f"archive-candidate-{os.getpid()}-{uuid.uuid4().hex}.npy"
                        if info.file_size <= 0:
                            continue
                        with handle.open(info, "r") as member:
                            digest = _copy_and_hash(member, staged)
                        if digest == expected_hash:
                            return {"kind": "archive", "path": str(archive),
                                    "member": info.filename, "staged_path": str(staged),
                                    "sha256": digest, "stats": stats}
                        staged.unlink(missing_ok=True)
            else:
                with tarfile.open(archive, "r:*") as handle:
                    for member in handle:
                        if not member.isfile() or Path(member.name).name != filename:
                            continue
                        if expected_payload_bytes is not None and member.size < expected_payload_bytes:
                            continue
                        stats["candidate_count"] += 1
                        stats["checked_members"] += 1
                        source_stream = handle.extractfile(member)
                        if source_stream is None:
                            continue
                        staged = stage_root / f"archive-candidate-{os.getpid()}-{uuid.uuid4().hex}.npy"
                        with source_stream:
                            digest = _copy_and_hash(source_stream, staged)
                        if digest == expected_hash:
                            return {"kind": "archive", "path": str(archive),
                                    "member": member.name, "staged_path": str(staged),
                                    "sha256": digest, "stats": stats}
                        staged.unlink(missing_ok=True)
        except (OSError, tarfile.TarError, zipfile.BadZipFile):
            continue
    return {"kind": None, "stats": stats}


def _load_frozen(task, source, manifest, report_dir, device, allow_missing_features=False):
    """Load the frozen encoder after strict source/protocol checks.

    During repair, one or more feature files are intentionally absent, so the
    real select_gmm cache loader cannot yet succeed. In that case this helper
    validates every immutable dependency and every present split, while the
    completed-source path later invokes the actual select_gmm loader.
    """
    if manifest.get("artifact_policy") != "full" or manifest.get("head_type") != "gmm":
        raise ValueError("fixed source is not a full GMM artifact")
    if manifest.get("stage_status") != "complete":
        raise ValueError("fixed source is not complete")
    frame, train_length, _, data_path, metadata_path, metadata_row = _read_task(
        task, manifest["config"]["data_source"]
    )
    config = _config_kwargs(manifest, report_dir / "probe", device)
    config["mode"] = "select_gmm"
    config["source_dir"] = str(source)
    probe = MNRFViewProbe.from_kwargs(**config)
    values, endpoints, _ = probe._protocol(frame, train_length)
    if probe.source_identity != manifest.get("source_identity"):
        raise ValueError("source data/time/boundary identity differs from manifest")
    validate_normalized_values_cache(source, manifest,
                                     expected_variables=list(probe.variable_names))
    checkpoint = source / "checkpoints" / "best_encoder.pt"
    checkpoint_hash = manifest.get("checkpoint_sha256", {}).get("best_encoder.pt")
    if not checkpoint.is_file() or not checkpoint_hash or sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("best_encoder.pt is missing or its immutable hash differs")
    statistics = source / "input_statistics.npz"
    statistics_hash = manifest.get("input_statistics_sha256")
    if not statistics.is_file() or not statistics_hash or sha256_file(statistics) != statistics_hash:
        raise ValueError("input_statistics.npz is missing or its immutable hash differs")
    trainer = probe._make_trainer(len(probe.variable_names))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer.encoder.load_state_dict(state.get("encoder", state))
    trainer.encoder.eval()
    with np.load(statistics, allow_pickle=False) as stats:
        trainer.fixed_input.mean_ = np.asarray(stats["mean"], dtype=np.float32)
        trainer.fixed_input.scale_ = np.asarray(stats["scale"], dtype=np.float32)
    if (np.any(~np.isfinite(trainer.fixed_input.mean_))
            or np.any(~np.isfinite(trainer.fixed_input.scale_))
            or np.any(trainer.fixed_input.scale_ <= 0)):
        raise ValueError("input statistics are non-finite or have non-positive scales")
    if trainer.fingerprint() != manifest.get("feature_fingerprint"):
        raise ValueError("frozen encoder/input-statistics fingerprint differs")
    for name in ("train", "validation", "calibration", "test"):
        shape, digest = _expected_cache(manifest, name)
        endpoint_hash = manifest.get("endpoint_sha256", {}).get(name)
        endpoint_path = source / "cache" / f"{name}_window_end_indices.npy"
        if not endpoint_hash:
            raise ValueError(f"manifest has no authoritative {name} endpoint hash")
        _validate_endpoint(endpoint_path, name, shape[0], endpoint_hash)
        cache_path = source / "cache" / f"{name}_features.npy"
        if cache_path.exists():
            _validate_existing_feature(cache_path, name, manifest)
        elif not allow_missing_features:
            raise FileNotFoundError(f"required source feature cache is missing: {cache_path}")
    probe.trainer = trainer
    expected_test = _validate_endpoint(
        source / "cache" / "test_window_end_indices.npy", "test",
        int(manifest["test_endpoint_count"]), manifest["endpoint_sha256"]["test"])
    if not np.array_equal(endpoints, expected_test):
        raise ValueError("recomputed test endpoints differ from frozen source")
    for name in ("train", "validation", "calibration"):
        shape, _ = _expected_cache(manifest, name)
        expected = EndWindowDataset(values, *probe.ranges[name], probe.config.seq_len,
                                    probe.ranges["segment_ranges"][name]).endpoints
        saved = _validate_endpoint(source / "cache" / f"{name}_window_end_indices.npy", name,
                                   shape[0], manifest["endpoint_sha256"][name])
        if not np.array_equal(expected, saved):
            raise ValueError(f"recomputed {name} endpoints differ from frozen source")
    probe._test_endpoints = np.asarray(endpoints, dtype=np.int64)
    probe.output_dir = report_dir / "probe"
    normalized = np.load(source / "cache" / "normalized_values.npy", mmap_mode="r", allow_pickle=False)
    return probe, trainer, normalized, np.asarray(endpoints, dtype=np.int64), values, {
        "data_path": str(data_path), "metadata_path": str(metadata_path),
        "metadata_row": {
            str(key): (None if (isinstance(value, float) and not np.isfinite(value)) else str(value))
            for key, value in metadata_row.items()
        },
        "train_length": int(train_length),
    }


def _feature_inventory(source, manifest):
    """Refresh all four cache/endpoint states without changing the source."""
    result = {}
    for name in ("train", "validation", "calibration", "test"):
        shape, digest = _expected_cache(manifest, name)
        cache_path = source / "cache" / f"{name}_features.npy"
        endpoint_hash = manifest.get("endpoint_sha256", {}).get(name)
        endpoint_path = source / "cache" / f"{name}_window_end_indices.npy"
        item = _validate_existing_feature(cache_path, name, manifest)
        if endpoint_hash and endpoint_path.is_file():
            endpoint = _validate_endpoint(endpoint_path, name, shape[0], endpoint_hash)
            item["endpoint"] = {"status": "present", "count": len(endpoint),
                                "sha256": endpoint_hash}
        else:
            item["endpoint"] = {"status": "missing", "expected_sha256": endpoint_hash}
        result[name] = item
    return result


def _required_artifacts(source, manifest):
    """List full-GMM requirements so partial restoration cannot look complete."""
    required = [
        "probe_manifest.json", "run_status.json", "input_statistics.npz",
        "cache/normalized_values.npy", "checkpoints/best_encoder.pt",
        "heads/gmm_fit_endpoints.npy", "heads/fixed_noise.npy",
        "heads/feature_statistics.npz", "heads/gmm_models.pkl",
        "heads/gmm_status.json", "heads/gmm_progress.json", "heads/calibration.npz",
        "heads/validation_nll.npy", "heads/calibration_nll.npy",
        "results/test_nll.npy", "results/aligned_scores.npz", "results/metrics.json",
        "checkpoints/initial_encoder.pt",
    ]
    if manifest.get("checkpoint_sha256", {}).get("initial_encoder.pt"):
        required.append("checkpoints/initial_encoder.pt")
    for name in ("train", "validation", "calibration", "test"):
        required.extend((f"cache/{name}_features.npy",
                         f"cache/{name}_window_end_indices.npy"))
    missing = [relative for relative in required if not (source / relative).is_file()]
    declared = [
        ("checkpoint_sha256.best_encoder.pt", manifest.get("checkpoint_sha256", {}).get("best_encoder.pt")),
        ("checkpoint_sha256.initial_encoder.pt", manifest.get("checkpoint_sha256", {}).get("initial_encoder.pt")),
        ("input_statistics_sha256", manifest.get("input_statistics_sha256")),
        ("normalized_cache_sha256", manifest.get("normalized_cache_sha256")),
        ("feature_statistics_sha256", manifest.get("feature_statistics_sha256")),
        ("gmm_status_sha256", manifest.get("gmm_status_sha256")),
        ("gmm_fit_endpoints_sha256", manifest.get("gmm_fit_endpoints_sha256")),
    ]
    missing.extend(f"manifest.{key}" for key, value in declared if not value)
    return required, missing


@torch.no_grad()
def _forward_split(probe, trainer, normalized, name, batch_size, out_path):
    if name == "test":
        endpoints = np.asarray(probe._current_test_endpoints, dtype=np.int64)
        segments = [tuple(item) for item in probe.source_identity["test_segments"]]
        start, end = probe.source_identity["train_length"], len(normalized)
    else:
        start, end = probe.ranges[name]
        segments = probe.ranges["segment_ranges"][name]
        endpoints = EndWindowDataset(normalized, start, end, probe.config.seq_len,
                                     segments).endpoints
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.name + f".partial-{os.getpid()}-{uuid.uuid4().hex}")
    mmap = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32,
        shape=(len(endpoints), trainer.variables, trainer.feature_dim))
    started = time.perf_counter()
    dataset = EndWindowDataset(normalized, start, end, probe.config.seq_len, segments)
    if not np.array_equal(dataset.endpoints, endpoints):
        raise ValueError(f"computed {name} endpoints differ from frozen coordinates")
    cursor = 0
    for offset in range(0, len(dataset), batch_size):
        batches = [dataset[i][0] for i in range(offset, min(len(dataset), offset + batch_size))]
        features = trainer.encoder.encode_views(torch.stack(batches).to(trainer.device))["features"]
        values = features.cpu().numpy().astype(np.float32, copy=False)
        mmap[cursor:cursor + len(values)] = values
        cursor += len(values)
    mmap.flush()
    del mmap
    return temporary, time.perf_counter() - started


def _copy_backup_to_stage(backup, staged):
    if backup["kind"] == "archive":
        source = Path(backup["staged_path"])
        with source.open("rb") as src, Path(staged).open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
    else:
        with Path(backup["path"]).open("rb") as src, Path(staged).open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())


def _npy_layout(path):
    path = Path(path)
    try:
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            offset = stream.tell()
        return {"exists": True, "bytes": path.stat().st_size, "version": list(version),
                "shape": list(shape), "dtype": str(dtype), "fortran_order": bool(fortran),
                "data_offset": int(offset), "sha256": sha256_file(path)}
    except Exception as error:
        return {"exists": path.exists(), "error": str(error)}


def _runtime_record():
    record = {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "environment_threads": {key: os.environ.get(key) for key in
                                 ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    if record["cuda_available"]:
        record["cuda"] = {"device": torch.cuda.current_device(),
                          "name": torch.cuda.get_device_name(torch.cuda.current_device()),
                          "runtime": torch.version.cuda}
    try:
        record["current_git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
        record["current_git_status"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=str(ROOT), text=True).splitlines()
    except Exception as error:
        record["git_query_error"] = str(error)
    return record


@torch.no_grad()
def diagnose_cached_batches(task, report_dir, device_name="cuda:0", batch_limit=128):
    """Replay bounded validation/calibration batches and report per-view diffs."""
    source = _source_path(task)
    manifest = json.loads((source / "probe_manifest.json").read_text())
    requested = device_name
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    probe, trainer, normalized, _, _, source_meta = _load_frozen(
        task, source, manifest, report_dir, requested, allow_missing_features=True)
    batch_size = min(int(batch_limit), int(probe.config.batch_size),
                     int(probe.config.encoding_batch_size or probe.config.batch_size))
    layout = manifest.get("layout", {}).get("slices", {})
    result = {
        "task": task, "source": str(source), "run_id": SOURCES[task],
        "source_identity": manifest.get("source_identity"),
        "feature_fingerprint": manifest.get("feature_fingerprint"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256", {}).get("best_encoder.pt"),
        "input_statistics_sha256": manifest.get("input_statistics_sha256"),
        "data_and_protocol": source_meta,
        "runtime": _runtime_record(),
        "original_config": {key: manifest.get("config", {}).get(key) for key in
                            ("device", "batch_size", "encoding_batch_size", "cpu_threads",
                             "seq_len", "hidden_dim", "pattern_len")},
        "training_receipt": {}, "npy_files": {}, "replays": [],
        "historical_missing_comparison": [name for name in ("train", "test")
                                           if not (source / "cache" / f"{name}_features.npy").exists()],
    }
    for candidate in ("training_provenance.json", "training_receipt.json", "run.log",
                      "training.jsonl", "run_request.json"):
        path = source / candidate
        result["training_receipt"][candidate] = ({"present": True, "sha256": sha256_file(path)}
                                                   if path.is_file() else {"present": False})
    provenance_path = source / "training_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        result["training_receipt_details"] = {
            "captured_before_training": provenance.get("captured_before_training"),
            "git_head": provenance.get("git_head"),
            "git_dirty": provenance.get("git_dirty"),
            "training_environment": provenance.get("training_environment"),
            "source_data": provenance.get("source_data"),
            "effective_config": provenance.get("effective_config"),
            "startup_script": provenance.get("startup_script"),
        }
        comparisons = []
        for entry in provenance.get("code_files", []):
            old_path = Path(entry.get("path", ""))
            if (not old_path.is_absolute() or not old_path.is_file()
                    or ROOT.resolve() not in old_path.resolve().parents):
                comparisons.append({"path": str(old_path), "recorded_sha256": entry.get("sha256"),
                                    "current": "unavailable"})
                continue
            current_hash = sha256_file(old_path)
            comparisons.append({"path": str(old_path), "recorded_sha256": entry.get("sha256"),
                                "current_sha256": current_hash,
                                "matches_training_receipt": current_hash == entry.get("sha256")})
        result["training_code_comparison"] = comparisons
    else:
        result["training_receipt_details"] = "unknown: no training_provenance.json"
        result["training_code_comparison"] = "unknown: no training-time code hashes"
    result["historical_runtime_fields"] = {
        "torch_num_threads": "unknown unless recorded by the training receipt",
        "torch_num_interop_threads": "unknown unless recorded by the training receipt",
        "float32_matmul_precision": "unknown unless recorded by the training receipt",
        "cuda_matmul_allow_tf32": "unknown unless recorded by the training receipt",
        "cudnn_benchmark": "unknown unless recorded by the training receipt",
        "cudnn_deterministic": "unknown unless recorded by the training receipt",
    }
    for split in ("validation", "calibration"):
        cache_path = source / "cache" / f"{split}_features.npy"
        endpoint_path = source / "cache" / f"{split}_window_end_indices.npy"
        if not cache_path.is_file() or not endpoint_path.is_file():
            result["replays"].append({"split": split, "status": "unavailable"})
            continue
        cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        endpoints = _strict_integer_indices(
            np.load(endpoint_path, mmap_mode="r", allow_pickle=False), f"{split} endpoints")
        n = len(endpoints)
        starts = {"beginning": 0}
        starts["middle"] = max(0, ((max(0, n // 2)) // batch_size) * batch_size)
        if n >= batch_size:
            starts["last_full"] = ((n - batch_size) // batch_size) * batch_size
        if n % batch_size:
            starts["tail"] = (n // batch_size) * batch_size
        used = set()
        for label, offset in starts.items():
            if offset >= n or offset in used:
                continue
            used.add(offset)
            current = endpoints[offset:min(n, offset + batch_size)]
            windows = np.stack([normalized[int(end) - probe.config.seq_len + 1:int(end) + 1]
                                for end in current])
            features = trainer.encoder.encode_views(
                torch.as_tensor(windows, device=trainer.device))["features"].cpu().numpy()
            by_view = {}
            for view in ("S", "E", "P", "R"):
                left, right = map(int, layout[view])
                actual = features[..., left:right]
                expected = cached[offset:offset + len(current), :, left:right]
                delta = np.abs(actual - expected)
                by_view[view] = {
                    "exact": bool(np.array_equal(actual, expected)),
                    "max_abs": float(delta.max(initial=0.0)),
                    "non_equal_elements": int(np.count_nonzero(actual != expected)),
                    "elements": int(actual.size),
                }
            result["replays"].append({"split": split, "batch": label,
                                      "start": int(offset), "size": int(len(current)),
                                      "global_endpoints": [int(current[0]), int(current[-1])],
                                      "views": by_view})
        result["npy_files"][split] = _npy_layout(cache_path)
        result["npy_files"][split + "_endpoints"] = _npy_layout(endpoint_path)
    out = report_dir / f"{task}-bounded-forward-diagnostic.json"
    _json_dump(out, result)
    shutil.rmtree(report_dir / "probe", ignore_errors=True)
    return result


def _validate_complete_source(source, manifest, task_report, device_name):
    from scripts.validate_mnrf_view_probe import validate

    required, missing = _required_artifacts(source, manifest)
    if missing:
        raise ValueError("full source is incomplete; missing: " + ", ".join(missing))
    native = validate(source)
    if native.get("status") != "complete":
        raise ValueError(f"native validator did not report complete: {native.get('status')}")
    frame, train_length, _, data_path, metadata_path, metadata_row = _read_task(
        task_report.name, manifest["config"]["data_source"])
    config = _config_kwargs(manifest, task_report / "select-gmm-preflight", device_name)
    config["mode"] = "select_gmm"
    config["source_dir"] = str(source)
    probe = MNRFViewProbe.from_kwargs(**config)
    _, endpoints, _ = probe._protocol(frame, train_length)
    if probe.source_identity != manifest.get("source_identity"):
        raise ValueError("select_gmm preflight data/time/boundary identity differs")
    # Use the exact read-only source loading path that select_gmm.fit() uses.
    # It validates the checkpoint, input statistics, all four cache files,
    # endpoints, noise/fitting sample and frozen feature fingerprint without
    # fitting a head or running an encoder forward.
    _, _, cache = probe._load_source(source)
    if not cache:
        raise ValueError("select_gmm preflight returned no source feature caches")
    frozen_test = cache["test"]
    if len(frozen_test) != len(endpoints):
        raise ValueError("select_gmm preflight test cache length differs from source coordinates")
    trainer = probe.trainer
    source_meta = {
        "data_path": str(data_path), "metadata_path": str(metadata_path),
        "metadata_row": {str(key): (None if (isinstance(value, float)
                                                and not np.isfinite(value)) else str(value))
                         for key, value in metadata_row.items()},
        "train_length": int(train_length),
    }
    return {
        "native_validation": native,
        "select_gmm_source_preflight": "passed",
        "required_artifact_count": len(required),
        "source_preflight_variables": len(probe.variable_names),
        "source_preflight_device": str(trainer.device),
        "source_preflight": source_meta,
    }


@torch.no_grad()
def _test_forward(trainer, normalized, endpoints, seq_len, batch_size, device, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.name + f".partial-{os.getpid()}")
    mmap = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                     shape=(len(endpoints), trainer.variables, trainer.feature_dim))
    started = time.perf_counter()
    for offset in range(0, len(endpoints), batch_size):
        batch_endpoints = endpoints[offset:offset + batch_size]
        windows = np.stack([normalized[int(e) - seq_len + 1:int(e) + 1] for e in batch_endpoints])
        features = trainer.encoder.encode_views(torch.as_tensor(windows, device=device))["features"]
        mmap[offset:offset + len(batch_endpoints)] = features.cpu().numpy().astype(np.float32, copy=False)
    mmap.flush(); del mmap
    return temporary, time.perf_counter() - started


def _install_exact(temporary: Path, target: Path, expected_hash: str):
    """Install a verified regular file atomically without replacing a target."""
    temporary, target = Path(temporary), Path(target)
    actual = sha256_file(temporary)
    if actual != expected_hash:
        raise ValueError(f"generated {target.name} hash mismatch: expected={expected_hash} actual={actual}")
    if target.parent.is_symlink():
        raise FileExistsError(f"cache target parent is a symbolic link: {target.parent}")
    target.parent.mkdir(parents=True, exist_ok=True)

    def existing_result():
        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(mode):
            raise FileExistsError(f"cache target exists as a non-regular directory entry: {target}")
        existing = sha256_file(target)
        if existing != expected_hash:
            raise FileExistsError(
                f"cache target conflict: {target}; expected={expected_hash} actual={existing}"
            )
        return "race_reused", existing

    prior = existing_result()
    if prior is not None:
        return prior

    target_stage = None
    try:
        try:
            os.link(str(temporary), str(target))
        except OSError as error:
            if error.errno == errno.EEXIST:
                prior = existing_result()
                if prior is None:
                    raise
                return prior
            if error.errno != errno.EXDEV:
                raise
            # Copy to an exclusive temporary on the destination filesystem,
            # then link with no-replace semantics. Never fall back to replace.
            fd, staged_name = tempfile.mkstemp(prefix=f".{target.name}.recover-",
                                               suffix=".tmp", dir=str(target.parent))
            target_stage = Path(staged_name)
            with os.fdopen(fd, "wb") as output, temporary.open("rb") as source_file:
                shutil.copyfileobj(source_file, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            copied_hash = sha256_file(target_stage)
            if copied_hash != expected_hash:
                raise ValueError(
                    f"cross-filesystem staged hash mismatch: expected={expected_hash} actual={copied_hash}"
                )
            try:
                os.link(str(target_stage), str(target))
            except OSError as link_error:
                if link_error.errno != errno.EEXIST:
                    raise
                prior = existing_result()
                if prior is None:
                    raise
                return prior
        return "installed", expected_hash
    finally:
        if target_stage is not None:
            target_stage.unlink(missing_ok=True)


def recover_task(task: str, report_dir: Path, device_name: str, requested_splits=None,
                 allow_forward=True):
    source = _source_path(task)
    task_report = report_dir / task
    task_report.mkdir(parents=True, exist_ok=True)
    result = {"task": task, "source": str(source), "run_id": SOURCES[task],
              "device_requested": device_name, "status": "failed", "files": {},
              "recovery_scope": "exact-cache-files-only; no training/gmm/calibration"}
    if not source.is_dir():
        raise FileNotFoundError(f"source run missing: {source}")
    manifest_path = source / "probe_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact_policy") != "full" or manifest.get("head_type") != "gmm":
        raise ValueError("fixed source is not a full GMM artifact")
    result["manifest_sha256_before"] = sha256_file(manifest_path)
    result["files"] = _feature_inventory(source, manifest)
    original_cache_hashes = {
        name: item.get("sha256") for name, item in result["files"].items()
        if item.get("status") == "present"
    }
    result["recovery_actions"] = {}
    result["backup_searches"] = {}
    all_missing = [name for name, item in result["files"].items()
                   if item.get("status") == "missing"]
    result["missing_before"] = list(all_missing)
    if requested_splits:
        missing_allowed = [name for name in all_missing if name in requested_splits]
    else:
        missing_allowed = list(all_missing)
    result["requested_missing_splits"] = missing_allowed
    private = task_report / f"temporary-cache-{os.getpid()}-{uuid.uuid4().hex}"
    private.mkdir(exist_ok=False)
    try:
        # Exact backups are independent of raw data, checkpoint availability,
        # and compute device, so try them before any model/data precondition.
        backup_search = {}
        for name in missing_allowed:
            shape, digest = _expected_cache(manifest, name)
            backup = _find_exact_backup(
                source, name, digest, report_dir=private,
                expected_payload_bytes=int(np.prod(shape)) * np.dtype("float32").itemsize,
            )
            backup_search[name] = backup.get("stats", {})
            result["backup_searches"][name] = backup_search[name]
            if backup.get("kind") is None:
                result["files"][name]["backup_search"] = backup_search[name]
                continue
            staged = private / f"{name}_backup-{uuid.uuid4().hex}.npy"
            if backup["kind"] == "archive":
                # Archive extraction already produced a private, streaming-hashed
                # staging file. Copy into a distinct install stage so cleanup
                # ownership remains explicit.
                _copy_backup_to_stage(backup, staged)
                Path(backup["staged_path"]).unlink(missing_ok=True)
            else:
                _copy_backup_to_stage(backup, staged)
            array = np.load(staged, mmap_mode="r", allow_pickle=False)
            if tuple(array.shape) != shape or array.dtype != np.dtype("float32"):
                raise ValueError(f"exact backup has invalid {name} shape/dtype")
            for offset in range(0, len(array), 65536):
                if not np.isfinite(array[offset:offset + 65536]).all():
                    raise ValueError(f"exact backup {name} contains non-finite values")
            status, installed_hash = _install_exact(
                staged, source / "cache" / f"{name}_features.npy", digest)
            result["files"][name].update({
                "status": "present" if installed_hash == digest else "missing",
                "recovery": "exact_backup", "backup_path": backup.get("path"),
                "backup_member": backup.get("member"), "backup_search": backup.get("stats", {}),
                "install": status, "sha256": installed_hash,
                "bytes": int((source / "cache" / f"{name}_features.npy").stat().st_size),
            })
            result["recovery_actions"][name] = {
                "method": "exact_backup", "install": status,
                "backup_path": backup.get("path"), "archive_member": backup.get("member"),
                "sha256": installed_hash,
            }
        # Re-enumerate after installs; stale pre-recovery missing markers must
        # never influence the final status.
        result["files"] = _feature_inventory(source, manifest)
        remaining = [name for name, item in result["files"].items()
                     if item.get("status") == "missing"]
        endpoint_missing = [name for name, item in result["files"].items()
                            if item.get("endpoint", {}).get("status") != "present"]
        result["missing_after_exact_backup"] = remaining
        if requested_splits:
            unrequested = [name for name in remaining if name not in requested_splits]
        else:
            unrequested = []
        if unrequested:
            result["status"] = "partial_unrequested_split"
            result["missing"] = remaining
            result["reason"] = "split restriction left required cache missing"
        elif remaining and endpoint_missing:
            result["status"] = "blocked"
            result["missing"] = remaining
            result["missing_endpoints"] = endpoint_missing
            result["reason"] = "cannot forward missing cache because frozen endpoint files are incomplete"
        elif remaining and not allow_forward:
            result["status"] = "blocked"
            result["missing"] = remaining
            result["reason"] = "backup-only run found no exact archive/scattered copy"
            result["backup_search_complete"] = True
        elif remaining:
            selected_device = device_name
            if selected_device == "auto":
                selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            probe, trainer, normalized, _, _, source_meta = _load_frozen(
                task, source, manifest, task_report, selected_device,
                allow_missing_features=True)
            result.update(source_meta)
            result["device"] = str(trainer.device)
            result["batch_size"] = int(min(probe.config.batch_size,
                                            probe.config.encoding_batch_size or probe.config.batch_size))
            # Bounded throughput smoke on cached validation coordinates; this
            # does not alter model state or source artifacts.
            validation_cache_path = source / "cache" / "validation_features.npy"
            validation_endpoint_path = source / "cache" / "validation_window_end_indices.npy"
            if validation_cache_path.is_file() and validation_endpoint_path.is_file():
                validation_endpoints = np.load(validation_endpoint_path, mmap_mode="r", allow_pickle=False)
                validation_cache = np.load(validation_cache_path, mmap_mode="r", allow_pickle=False)
                bench_n = min(result["batch_size"], len(validation_endpoints))
                bench_endpoints = np.asarray(validation_endpoints[:bench_n], dtype=np.int64)
                started = time.perf_counter()
                with torch.no_grad():
                    windows = np.stack([normalized[int(e) - probe.config.seq_len + 1:int(e) + 1]
                                        for e in bench_endpoints])
                    bench_features = trainer.encoder.encode_views(
                        torch.as_tensor(windows, device=trainer.device))["features"].cpu().numpy()
                elapsed = time.perf_counter() - started
                result["benchmark"] = {
                    "windows": int(bench_n), "seconds": elapsed,
                    "windows_per_second": float(bench_n / max(elapsed, 1e-12)),
                    "max_abs_vs_validation": float(np.max(np.abs(bench_features - validation_cache[:bench_n]))),
                }
            for name in remaining:
                shape, digest = _expected_cache(manifest, name)
                staged, elapsed = _forward_split(
                    probe, trainer, normalized, name, result["batch_size"],
                    private / f"{name}_features.npy")
                array = np.load(staged, mmap_mode="r", allow_pickle=False)
                if tuple(array.shape) != shape or array.dtype != np.dtype("float32"):
                    raise ValueError(f"generated {name} feature shape/dtype mismatch")
                generated_hash = sha256_file(staged)
                result["files"][name]["forward"] = {
                    "elapsed_seconds": elapsed, "generated_sha256": generated_hash,
                    "expected_sha256": digest, "expected_shape": list(shape),
                }
                status, installed_hash = _install_exact(
                    staged, source / "cache" / f"{name}_features.npy", digest)
                result["files"][name].update({"install": status, "sha256": installed_hash})
                result["recovery_actions"][name] = {
                    "method": "frozen_forward", "install": status,
                    "elapsed_seconds": elapsed, "sha256": installed_hash,
                    "generated_sha256": generated_hash,
                }
                if installed_hash != digest:
                    result["files"][name].update({"status": "recovery_failed",
                                                   "error": "installed hash differs from manifest"})
            result["files"] = _feature_inventory(source, manifest)
        # Complete means every original artifact is present, the native
        # validator accepts it, and the real select_gmm source loader accepts
        # all frozen caches/checkpoints/statistics. No fit is invoked.
        required, missing_required = _required_artifacts(source, manifest)
        result["required_artifacts"] = {"count": len(required), "missing": missing_required}
        if not missing_required and not any(item.get("status") == "missing"
                                            or item.get("endpoint", {}).get("status") != "present"
                                            for item in result["files"].values()):
            result["completion_validation"] = _validate_complete_source(
                source, manifest, task_report, device_name)
            result["status"] = "recovered" if result["recovery_actions"] else "already_complete"
        else:
            result["status"] = "blocked"
            result["missing"] = sorted(set(missing_required + [
                f"cache/{name}_features.npy" for name, item in result["files"].items()
                if item.get("status") == "missing"
            ]))
            if result.get("backup_search_complete") and not allow_forward:
                result["reason"] = "exact backup search found no matching file; frozen forward was disabled"
            else:
                result["reason"] = "source remains incomplete after exact restoration/forward attempt"
        result["manifest_sha256_after"] = sha256_file(manifest_path)
        result["preserved_existing_cache_hashes"] = {
            name: {"before": digest,
                   "after": sha256_file(source / "cache" / f"{name}_features.npy")}
            for name, digest in original_cache_hashes.items()
        }
        _json_dump(task_report / "preflight.json", result)
    except Exception as error:
        result["status"] = "blocked"
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        result["files"] = _feature_inventory(source, manifest)
        result["manifest_sha256_after"] = sha256_file(manifest_path)
        _json_dump(task_report / "preflight.json", result)
    finally:
        shutil.rmtree(private, ignore_errors=True)
        probe_temp = task_report / "probe"
        if probe_temp.exists():
            shutil.rmtree(probe_temp, ignore_errors=True)
    _json_dump(task_report / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", choices=sorted(SOURCES))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", action="append",
                        choices=("train", "validation", "calibration", "test"),
                        help="limit recovery to a missing split (repeatable)")
    parser.add_argument("--backup-only", action="store_true",
                        help="restore exact scattered/archive copies; never load or run an encoder")
    parser.add_argument("--diagnose-forward", action="store_true",
                        help="bounded validation/calibration replay diagnostics only")
    parser.add_argument("--report-dir", default="")
    args = parser.parse_args()
    tasks = args.task or (list(SOURCES) if args.all else [])
    if not tasks:
        parser.error("provide --all or at least one --task")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_dir = Path(args.report_dir).resolve() if args.report_dir else (
        ROOT / "result" / "analysis" / "MNRF-ViewProbe-Ablation" / f"cache-recovery-{stamp}-{os.getpid()}"
    )
    report_dir.mkdir(parents=True, exist_ok=False)
    if args.diagnose_forward:
        diagnostics = []
        for task in tasks:
            try:
                diagnostics.append(diagnose_cached_batches(task, report_dir / task, args.device))
            except Exception as error:
                diagnostics.append({"task": task, "status": "diagnostic_failed", "error": str(error),
                                    "traceback": traceback.format_exc()})
        _json_dump(report_dir / "forward_diagnostics.json", diagnostics)
        print(json.dumps({"report_dir": str(report_dir), "diagnostics": diagnostics},
                         indent=2, sort_keys=True))
        raise SystemExit(0 if all(item.get("replays") for item in diagnostics) else 1)
    results = []
    for task in tasks:
        try:
            results.append(recover_task(task, report_dir, args.device, args.split,
                                        allow_forward=not args.backup_only))
        except Exception as error:
            failure = {"task": task, "source": str(_source_path(task)), "status": "failed",
                       "error": str(error), "traceback": traceback.format_exc()}
            results.append(failure)
            _json_dump(report_dir / task / "result.json", failure)
            print(f"[{task}] recovery failed: {error}", file=sys.stderr)
    summary = {"schema": "mnrf-view-probe-cache-recovery-v1", "report_dir": str(report_dir),
               "device_requested": args.device, "results": results}
    _json_dump(report_dir / "recovery_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if all(item.get("status") in {"recovered", "already_complete"} for item in results) else 1)


if __name__ == "__main__":
    main()
