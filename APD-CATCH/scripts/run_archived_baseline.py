#!/usr/bin/env python
"""Run one APD-CATCH baseline with a pre-created, provenance-rich archive.

The wrapper does not alter model behavior. It delegates to run_benchmark.py and
adds opt-in point-output capture for later coordinate-aware evaluation.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import pickle
import selectors
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result"
CONFIG_ROOT = ROOT / "config"
TASKS = {"PSM", "MSL"}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_name("." + path.name + ".tmp-" + str(os.getpid()))
    with open(temporary, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _git_value(*args):
    result = subprocess.run(
        ["git", "-C", str(ROOT)] + list(args),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _unique_run_path(entry, method_path, task, override=None):
    if override:
        raw = Path(override)
        target = raw if raw.is_absolute() else RESULT_ROOT / raw
        resolved = target.resolve(strict=False)
        try:
            resolved.relative_to(RESULT_ROOT.resolve())
        except ValueError:
            raise ValueError("BASELINE_SAVE_PATH must stay under this project's result/")
        if target.exists() or target.is_symlink():
            raise FileExistsError("refusing to reuse existing archive path: {}".format(target))
        return target, str(target.relative_to(RESULT_ROOT))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    relative = Path(entry) / method_path / task / "run-{}-{}".format(stamp, os.getpid())
    target = RESULT_ROOT / relative
    if target.exists() or target.is_symlink():
        raise FileExistsError("run archive path already exists: {}".format(target))
    return target, relative.as_posix()


def _package_versions():
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python 3.7 compatibility
        import importlib_metadata as metadata
    packages = {}
    for name in ("torch", "numpy", "pandas", "scikit-learn", "salesforce-merlion"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def _data_evidence(task):
    data_path = ROOT / "dataset" / "anomaly_detect" / "data" / (task + ".csv")
    meta_path = ROOT / "dataset" / "anomaly_detect" / "DETECT_META.csv"
    evidence = {"data_path": str(data_path), "metadata_path": str(meta_path)}
    if data_path.is_file():
        evidence["data_sha256"] = sha256_file(data_path)
    else:
        evidence["data_sha256"] = None
    if meta_path.is_file():
        evidence["metadata_sha256"] = sha256_file(meta_path)
        try:
            import pandas as pd

            rows = pd.read_csv(meta_path)
            selected = rows.loc[rows["file_name"].astype(str) == task + ".csv"]
            records = selected.to_dict(orient="records")
            for record in records:
                for key, value in list(record.items()):
                    if isinstance(value, float) and math.isnan(value):
                        record[key] = None
            evidence["metadata_rows"] = records
        except Exception as exc:
            evidence["metadata_error"] = "{}: {}".format(type(exc).__name__, exc)
    else:
        evidence["metadata_sha256"] = None
    return evidence


def _snapshot_startup(path, destination):
    if path is None:
        return None
    source = Path(path).resolve(strict=True)
    content = source.read_bytes()
    target = destination / "startup_script.sh"
    with open(target, "xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return {"source_path": str(source), "snapshot_path": str(target), "sha256": sha256_bytes(content)}


def _decode_record_value(value):
    if not isinstance(value, str) or not value:
        raise ValueError("record artifact field is empty")
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def _global_artifact_indices(local_indices, payload):
    """Map serialized test-frame indices back to the official source axis."""
    indices = np.asarray(local_indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("benchmark test indices are not one-dimensional integers")
    try:
        offset = int(payload.get("coordinate_offset", 0))
    except (TypeError, ValueError):
        raise ValueError("artifact coordinate_offset is not an integer")
    return indices.astype(np.int64, copy=False) + offset


def _ratio_name(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "default"
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def export_point_artifacts(run_dir, task, entry, record_paths):
    """Export exact-length native outputs; reject ambiguous short-window alignment."""
    import pandas as pd

    from ts_benchmark.common.constant import ANOMALY_DETECT_DATASET_PATH
    from ts_benchmark.data.utils import read_data
    from ts_benchmark.recording import read_record_file

    data_path = Path(ANOMALY_DETECT_DATASET_PATH) / "data" / (task + ".csv")
    source_frame = read_data(str(data_path))
    if not isinstance(source_frame, pd.DataFrame) or "label" not in source_frame.columns:
        raise ValueError("source loader did not produce a labeled time-series frame")
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=False)
    outputs = []
    for record_path in record_paths:
        frame = read_record_file(str(record_path))
        if frame.empty:
            continue
        for row_index, row in frame.iterrows():
            log_info = row.get("log_info")
            if isinstance(log_info, str) and log_info.strip():
                raise RuntimeError("benchmark task record contains an error: {}".format(log_info[:500]))
            actual = _decode_record_value(row.get("actual_data"))
            payload = _decode_record_value(row.get("inference_data"))
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError("record lacks opt-in point artifact payload")
            if payload.get("kind") != entry:
                raise ValueError("record artifact kind does not match run entry")
            values = np.asarray(payload["raw_values"]).reshape(-1)
            if not hasattr(actual, "index"):
                raise ValueError("serialized labels have no original benchmark index")
            indices = _global_artifact_indices(np.asarray(actual.index), payload)
            if values.size != indices.size:
                raise ValueError(
                    "output length {} differs from test coordinate count {}; window alignment is not inferred".format(
                        values.size, indices.size
                    )
                )
            if indices.size == 0 or np.any(np.diff(indices) <= 0):
                raise ValueError("test coordinates are empty, duplicated, or unordered")
            if indices[0] < 0 or indices[-1] >= len(source_frame):
                raise ValueError("test coordinate lies outside loaded source axis")
            expected_labels = source_frame["label"].to_numpy()[indices]
            labels = (
                actual["label"].to_numpy(copy=False).reshape(-1)
                if hasattr(actual, "columns") and "label" in actual.columns
                else np.asarray(actual).reshape(-1)
            )
            if labels.size != indices.size or not np.array_equal(labels, expected_labels):
                raise ValueError("serialized labels do not match source labels at coordinates")
            times = np.asarray(source_frame.index.astype(str))[indices]
            payload_data = {
                "schema_version": np.asarray([1], dtype=np.int64),
                "task": np.asarray([task]),
                "entry": np.asarray([entry]),
                "coordinate_offset": np.asarray([int(indices[0] - np.asarray(actual.index)[0])], dtype=np.int64),
                "global_indices": indices,
                "time_values": times,
                "labels": labels.astype(np.int8, copy=False),
                "valid_mask": np.ones(indices.size, dtype=np.bool_),
                "score_direction": np.asarray(["higher_is_anomalous"] if entry == "score" else ["not_applicable"]),
                "source_data_sha256": np.asarray([sha256_file(data_path)]),
                "source_feature_names": np.asarray([str(c) for c in source_frame.columns if c != "label"]),
            }
            if entry == "score":
                payload_data["scores"] = values.astype(np.float64, copy=False)
            else:
                payload_data["predictions"] = values.astype(np.int8, copy=False)
            ratio = row.get("typical_anomaly_ratio")
            output = artifacts_dir / "{}_{}_{}.npz".format(entry, _ratio_name(ratio), row_index)
            with open(output, "xb") as handle:
                np.savez_compressed(handle, **payload_data)
            outputs.append({"path": str(output), "sha256": sha256_file(output), "points": int(indices.size)})
    if not outputs:
        raise RuntimeError("no point artifact was exported from benchmark records")
    return outputs


def _record_error(record_paths):
    """Return the first per-series error retained by the benchmark record."""
    from ts_benchmark.recording import read_record_file

    for record_path in record_paths:
        frame = read_record_file(str(record_path))
        if "log_info" not in frame.columns:
            continue
        for value in frame["log_info"].tolist():
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _record_paths(created):
    return [
        Path(path)
        for path in created
        if path.endswith(".tar.gz")
        or (path.endswith(".csv") and not Path(path).name.startswith("test_report"))
    ]


def _run_and_tee(command, cwd, stdout_path, stderr_path):
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        env=os.environ.copy(),
    )
    files = {
        process.stdout: (open(stdout_path, "xb", buffering=0), sys.stdout.buffer),
        process.stderr: (open(stderr_path, "xb", buffering=0), sys.stderr.buffer),
    }
    selector = selectors.DefaultSelector()
    for pipe in files:
        selector.register(pipe, selectors.EVENT_READ)
    try:
        while selector.get_map():
            for key, _ in selector.select(timeout=1):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                log_file, terminal = files[key.fileobj]
                log_file.write(chunk)
                terminal.write(chunk)
                terminal.flush()
        return process.wait()
    finally:
        selector.close()
        for log_file, _ in files.values():
            log_file.close()


def _effective_config(args, save_path):
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path) if config_path.parts[0] == "config" else (CONFIG_ROOT / config_path)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    hyper_params = json.loads(args.model_hyper_params)
    strategy_args = dict(config["evaluation_config"]["strategy_args"])
    strategy_args.update({"seed": args.seed, "save_anomaly_artifacts": True, "artifact_kind": args.entry})
    model_config = {
        "models": [{"model_name": args.model_name, "adapter": None, "model_hyper_params": hyper_params}],
        "recommend_model_hyper_params": config.get("model_config", {}).get("recommend_model_hyper_params", {}),
    }
    data_config = dict(config["data_config"])
    data_config.update({"data_set_name": ["large_detect"], "data_name_list": [args.task + ".csv"]})
    return config_path, {
        "data_config": data_config,
        "model_config": model_config,
        "evaluation_config": {"metrics": config["evaluation_config"].get("metrics", "all"), "strategy_args": strategy_args},
        "report_config": dict(config["report_config"], save_path=save_path),
        "resources": {"eval_backend": "sequential", "gpus": args.gpus, "num_workers": args.num_workers, "num_cpus": args.num_cpus, "timeout_seconds": args.timeout, "seed": args.seed},
    }


def run(args):
    if args.task not in TASKS:
        raise ValueError("this first-batch launcher supports PSM and MSL only")
    if args.entry not in ("score", "label"):
        raise ValueError("entry must be score or label")
    config_path, effective = _effective_config(args, "pending")
    if (args.entry == "score") != ("detect_score" in config_path.name):
        raise ValueError("config path does not match score/label entry")
    entry_dir, save_path = _unique_run_path(args.entry, args.method_path, args.task, os.environ.get("BASELINE_SAVE_PATH"))
    entry_dir.mkdir(parents=True, exist_ok=False)
    snapshot = _snapshot_startup(args.startup_script, entry_dir)
    config_path, effective = _effective_config(args, save_path)
    effective["report_config"]["save_path"] = save_path
    (entry_dir / "config_source.json").write_bytes(config_path.read_bytes())
    atomic_json(entry_dir / "effective_config.json", effective)
    command = [
        sys.executable, str(ROOT / "scripts" / "run_benchmark.py"),
        "--config-path", config_path.name,
        "--data-name-list", args.task + ".csv",
        "--model-name", args.model_name,
        "--model-hyper-params", args.model_hyper_params,
        "--strategy-args", json.dumps(effective["evaluation_config"]["strategy_args"], sort_keys=True),
        "--seed", str(args.seed), "--gpus", str(args.gpus),
        "--num-workers", str(args.num_workers), "--num-cpus", str(args.num_cpus),
        "--timeout", str(args.timeout), "--save-path", save_path,
    ]
    status = {
        "schema_version": 1,
        "run_id": entry_dir.name,
        "method": args.method_path,
        "model_name": args.model_name,
        "upstream_implementation": args.upstream,
        "entry": args.entry,
        "task": args.task,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "startup_snapshot": snapshot,
        "command": command,
        "cwd": str(ROOT),
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_dirty_paths": _git_value("status", "--porcelain").splitlines(),
        "python": {"executable": sys.executable, "version": sys.version},
        "packages": _package_versions(),
        "cuda_visible_devices_inherited": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_evidence": _data_evidence(args.task),
        "effective_config_path": str(entry_dir / "effective_config.json"),
        "started_utc": None,
        "ended_utc": None,
        "benchmark_returncode": None,
        "wrapper_returncode": None,
        "failure_reason": None,
        "record_files": [],
        "score_or_label_artifacts": [],
        "calibration_artifact": None,
        "native_threshold_provenance": "method native; inspect saved effective config and report; generic threshold export is unavailable",
    }
    manifest_path = entry_dir / "run_manifest.json"
    atomic_json(manifest_path, status)
    before = set(entry_dir.glob("*.tar.gz")) | set(entry_dir.glob("test_report*.csv"))
    status["started_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, status)
    try:
        status["benchmark_returncode"] = _run_and_tee(
            command, ROOT, entry_dir / "stdout.log", entry_dir / "stderr.log"
        )
        after = set(entry_dir.glob("*.tar.gz")) | set(entry_dir.glob("test_report*.csv"))
        created = sorted(str(p) for p in after.difference(before))
        status["record_files"] = created
        if status["benchmark_returncode"] != 0:
            status["status"] = "benchmark_process_failed"
            status["failure_reason"] = "run_benchmark.py exited {}".format(status["benchmark_returncode"])
        elif (record_error := _record_error(_record_paths(created))):
            status["status"] = "benchmark_task_failed"
            status["failure_reason"] = record_error[:1000]
        elif args.entry == "score":
            records = _record_paths(created)
            try:
                status["score_or_label_artifacts"] = export_point_artifacts(entry_dir, args.task, args.entry, records)
                status["status"] = "complete_native_score_artifact"
            except Exception as exc:
                status["status"] = "artifact_export_blocked"
                status["failure_reason"] = "{}: {}".format(type(exc).__name__, exc)
        else:
            status["status"] = "complete_native_label_no_threshold_export"
            status["failure_reason"] = "native labels saved; generic model threshold is not exposed by the adapter contract"
    except Exception as exc:
        status["status"] = "launcher_failed"
        status["failure_reason"] = "{}: {}".format(type(exc).__name__, exc)
        raise
    finally:
        status["ended_utc"] = datetime.now(timezone.utc).isoformat()
        status["wrapper_returncode"] = 0 if status["status"].startswith("complete_native") else 1
        atomic_json(manifest_path, status)
    return 0 if status["wrapper_returncode"] == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--entry", required=True, choices=("score", "label"))
    parser.add_argument("--method-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-hyper-params", default="{}")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--startup-script", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60000)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print("archive launcher failed: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
