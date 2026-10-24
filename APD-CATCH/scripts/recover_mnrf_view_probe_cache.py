#!/usr/bin/env python
"""Recover missing ViewProbe feature caches without training.

The recovery path is deliberately narrower than ``run_mnrf_view_probe.py``:
it validates the frozen source and protocol, loads the saved best encoder,
and only performs the missing cache forward passes.  A generated file is
installed only when its byte SHA256 is exactly the immutable manifest hash.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

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


def _find_exact_backup(source: Path, name: str, expected_hash: str):
    """Search only project result/review trees for an exact byte backup."""
    roots = (ROOT / "result" / "score", ROOT / "result" / "analysis", ROOT / "result" / "review")
    filename = f"{name}_features.npy"
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in root.rglob(filename):
            try:
                resolved = candidate.resolve()
                if source in resolved.parents or resolved == source:
                    continue
                if sha256_file(candidate) == expected_hash:
                    return candidate
            except OSError:
                continue
    return None


def _load_frozen(task, source, manifest, report_dir, device):
    """Validate protocol and return a configured probe/trainer/normalized mmap."""
    if manifest.get("artifact_policy") != "full" or manifest.get("head_type") != "gmm":
        raise ValueError("fixed source is not a full GMM artifact")
    if manifest.get("stage_status") != "complete":
        raise ValueError("fixed source is not complete")
    validate_normalized_values_cache(source, manifest)
    checkpoint = source / "checkpoints" / "best_encoder.pt"
    declared_checkpoint = manifest.get("checkpoint_sha256", {}).get("best_encoder.pt")
    if not checkpoint.is_file() or not declared_checkpoint:
        raise ValueError("best_encoder.pt or its authoritative hash is missing")
    if sha256_file(checkpoint) != declared_checkpoint:
        raise ValueError("best_encoder.pt sha256 mismatch")
    statistics = source / "input_statistics.npz"
    declared_stats = manifest.get("input_statistics_sha256")
    if not statistics.is_file() or not declared_stats or sha256_file(statistics) != declared_stats:
        raise ValueError("input_statistics.npz is missing or changed")
    frame, train_length, _, data_path, metadata_path, metadata_row = _read_task(
        task, manifest["config"]["data_source"]
    )
    config = _config_kwargs(manifest, report_dir / "probe", device)
    probe = MNRFViewProbe.from_kwargs(**config)
    values, endpoints, _ = probe._protocol(frame, train_length)
    if probe.source_identity != manifest.get("source_identity"):
        raise ValueError("source data/time/boundary identity differs from manifest")
    expected_test = _validate_endpoint(
        source / "cache" / "test_window_end_indices.npy", "test",
        int(manifest["test_endpoint_count"]), manifest["endpoint_sha256"]["test"])
    if not np.array_equal(endpoints, expected_test):
        raise ValueError("recomputed test endpoints differ from frozen source")
    for name in ("train", "validation", "calibration"):
        shape, _ = _expected_cache(manifest, name)
        endpoint_path = source / "cache" / f"{name}_window_end_indices.npy"
        _validate_endpoint(endpoint_path, name, shape[0], manifest["endpoint_sha256"][name])
    trainer = probe._make_trainer(len(probe.variable_names))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer.encoder.load_state_dict(state.get("encoder", state))
    trainer.encoder.eval()
    with np.load(statistics, allow_pickle=False) as stats:
        trainer.fixed_input.mean_ = np.asarray(stats["mean"], dtype=np.float32)
        trainer.fixed_input.scale_ = np.asarray(stats["scale"], dtype=np.float32)
    if trainer.fingerprint() != manifest.get("feature_fingerprint"):
        raise ValueError("frozen encoder/input-statistics fingerprint differs")
    probe.trainer = trainer
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
    actual = sha256_file(temporary)
    if actual != expected_hash:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"generated {target.name} hash mismatch: expected={expected_hash} actual={actual}")
    if target.exists():
        existing = sha256_file(target)
        temporary.unlink(missing_ok=True)
        if existing != expected_hash:
            raise ValueError(f"target appeared with a conflicting hash: {target}")
        return "race_reused", existing
    os.replace(temporary, target)
    return "installed", actual


def recover_task(task: str, report_dir: Path, device_name: str, requested_splits=None):
    source = _source_path(task)
    manifest = json.loads((source / "probe_manifest.json").read_text())
    task_report = report_dir / task
    task_report.mkdir(parents=True, exist_ok=True)
    result = {"task": task, "source": str(source), "run_id": SOURCES[task],
              "device_requested": device_name, "status": "failed", "files": {}}
    if not source.is_dir():
        raise FileNotFoundError(f"source run missing: {source}")
    for name in ("train", "validation", "calibration", "test"):
        result["files"][name] = _validate_existing_feature(source / "cache" / f"{name}_features.npy", name, manifest)
    all_missing = [name for name in ("train", "test") if result["files"][name]["status"] == "missing"]
    missing = list(all_missing)
    if requested_splits:
        missing = [name for name in missing if name in requested_splits]
    if not all_missing:
        result["status"] = "already_complete"
        _json_dump(task_report / "result.json", result)
        return result
    if not missing:
        result["status"] = "partial_unrequested_split"
        result["missing"] = all_missing
        _json_dump(task_report / "result.json", result)
        return result
    selected_device = device_name
    if selected_device == "auto":
        selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Keep the private temporary probe/report path outside the source tree.
    probe, trainer, normalized, test_endpoints, values, source_meta = _load_frozen(
        task, source, manifest, task_report, selected_device)
    result.update(source_meta)
    result["device"] = str(trainer.device)
    result["batch_size"] = int(min(probe.config.batch_size, probe.config.encoding_batch_size or probe.config.batch_size))
    result["missing"] = missing
    # Required preflight benchmark: one original-size batch on the validation
    # cache coordinates, without writing or altering any source file.
    validation_endpoints = np.load(source / "cache" / "validation_window_end_indices.npy", mmap_mode="r")
    validation_cache = np.load(source / "cache" / "validation_features.npy", mmap_mode="r")
    bench_n = min(result["batch_size"], len(validation_endpoints))
    bench_endpoints = np.asarray(validation_endpoints[:bench_n], dtype=np.int64)
    started = time.perf_counter()
    with torch.no_grad():
        windows = np.stack([normalized[int(e) - probe.config.seq_len + 1:int(e) + 1] for e in bench_endpoints])
        bench_features = trainer.encoder.encode_views(torch.as_tensor(windows, device=trainer.device))["features"].cpu().numpy()
    elapsed = time.perf_counter() - started
    result["benchmark"] = {"windows": int(bench_n), "seconds": elapsed,
                           "windows_per_second": float(bench_n / max(elapsed, 1e-12)),
                           "max_abs_vs_validation": float(np.max(np.abs(bench_features - validation_cache[:bench_n]))) }
    _json_dump(task_report / "preflight.json", result)
    # Generate only missing splits in a private temporary directory.  The
    # endpoint files are already frozen and are never replaced.
    private = task_report / f"temporary-cache-{os.getpid()}"
    private.mkdir(exist_ok=False)
    try:
        for name in missing:
            try:
                shape, digest = _expected_cache(manifest, name)
                backup = _find_exact_backup(source, name, digest)
                if backup is not None:
                    staged = private / f"{name}_features.npy"
                    shutil.copyfile(backup, staged)
                    array = np.load(staged, mmap_mode="r", allow_pickle=False)
                    if tuple(array.shape) != shape or array.dtype != np.dtype("float32"):
                        staged.unlink(missing_ok=True)
                        raise ValueError(f"exact backup has invalid {name} shape/dtype")
                    result["files"][name].update({"recovery": "exact_backup", "backup": str(backup)})
                    status, installed_hash = _install_exact(
                        staged, source / "cache" / f"{name}_features.npy", digest)
                    result["files"][name].update({"install": status, "sha256": installed_hash,
                                                   "bytes": int((source / "cache" / f"{name}_features.npy").stat().st_size)})
                    continue
                if name == "train":
                    segment = probe.ranges["train"]
                    started = time.perf_counter()
                    temp_path, _, _ = trainer.extract_cache(
                        normalized, segment, name, private, probe.segments)
                    elapsed = time.perf_counter() - started
                else:
                    temp_path, elapsed = _test_forward(
                        trainer, normalized, test_endpoints, probe.config.seq_len,
                        result["batch_size"], trainer.device, private / "test_features.npy")
                generated = Path(temp_path)
                actual_array = np.load(generated, mmap_mode="r", allow_pickle=False)
                if tuple(actual_array.shape) != shape or actual_array.dtype != np.dtype("float32"):
                    raise ValueError(f"generated {name} feature shape/dtype mismatch")
                result["files"][name].update({"elapsed_seconds": elapsed, "generated_sha256": sha256_file(generated),
                                               "expected_shape": list(shape), "expected_sha256": digest})
                status, installed_hash = _install_exact(
                    generated, source / "cache" / f"{name}_features.npy", digest)
                result["files"][name].update({"install": status, "sha256": installed_hash,
                                               "bytes": int((source / "cache" / f"{name}_features.npy").stat().st_size)})
            except Exception as error:
                result["files"][name].update({"status": "recovery_failed", "error": str(error)})
        result["status"] = "recovered" if all(
            result["files"][name].get("status") == "present"
            or result["files"][name].get("install") in {"installed", "race_reused"}
            for name in ("train", "test")
        ) and all(
            result["files"][name].get("install") in {"installed", "race_reused"}
            for name in missing
        ) else "partial_or_failed"
    finally:
        shutil.rmtree(private, ignore_errors=True)
        # Remove only this task's private probe objects; never glob in source.
        shutil.rmtree(task_report / "probe", ignore_errors=True)
    _json_dump(task_report / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", choices=sorted(SOURCES))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", action="append", choices=("train", "test"),
                        help="limit recovery to a missing split (repeatable)")
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
    results = []
    for task in tasks:
        try:
            results.append(recover_task(task, report_dir, args.device, args.split))
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
