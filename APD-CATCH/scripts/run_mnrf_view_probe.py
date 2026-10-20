#!/usr/bin/env python
"""Run one independent MNRF frozen-view probe task.

The command performs one task and one head only. It never writes a source
MNRF-Lite V2 run; source directories are read-only inputs.
"""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
from ts_benchmark.baselines.mnrf_view_probe import IndexedViewProbeStrategy, MNRFViewProbe
from ts_benchmark.baselines.mnrf_view_probe.integrity import canonical_segment_ids, normalize_binary_labels
from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource, LocalExternalAnomalyDetectDataSource


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture_training_provenance(args, startup_script=None, data_path=None,
                                 metadata_path=None, metadata_row=None,
                                 frame=None, train_length=None):
    """Capture run facts before training; the CLI persists them after preparation."""
    files = [Path(__file__).resolve(), ROOT / "ts_benchmark/baselines/mnrf_lite_v2/MNRFLiteV2.py",
             ROOT / "ts_benchmark/baselines/mnrf_lite_v2/windows.py",
             ROOT / "ts_benchmark/baselines/mnrf_lite_v2/training.py"]
    files.extend(sorted((ROOT / "ts_benchmark/baselines/mnrf_view_probe").glob("*.py")))
    if startup_script:
        files.append(Path(startup_script).expanduser().resolve())
    code_files = []
    for path in files:
        if path.is_file():
            code_files.append({"path": str(path), "sha256": _sha256_file(path)})
    try:
        import subprocess
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=True).stdout.strip()
        status_text = subprocess.run(["git", "status", "--short"], cwd=ROOT,
                                     text=True, capture_output=True, check=True).stdout
        dirty = bool(status_text.strip())
        changed_paths = []
        for line in status_text.splitlines():
            relative = line[3:].strip()
            candidate = ROOT / relative
            if candidate.is_file() and candidate.suffix in {".py", ".sh", ".json"}:
                changed_paths.append({"path": relative, "sha256": _sha256_file(candidate)})
    except Exception as error:
        head, dirty, status_text, changed_paths = None, None, None, []
        git_error = str(error)
    else:
        git_error = None
    import torch
    import sklearn
    startup_text = None
    if startup_script and Path(startup_script).expanduser().is_file():
        startup_text = Path(startup_script).expanduser().read_text(encoding="utf-8")
    source_info = None
    if frame is not None and train_length is not None:
        time_index = MNRFLiteV2._time_index(frame)
        boundary_columns = MNRFLiteV2._boundary_columns(frame)
        boundary_ids = MNRFLiteV2._boundary_ids(frame)
        feature_values, variable_names = MNRFLiteV2._features_from_frame(frame)
        probe_module = __import__("ts_benchmark.baselines.mnrf_view_probe.probe", fromlist=["_chunked_array_hash", "_hash_strings"])
        source_info = {"task": args.task, "data_source": args.data_source,
                       "rows": int(len(frame)), "train_length": int(train_length),
                       "variable_names": list(map(str, variable_names)),
                       "time_kind": time_index.kind, "time_source_timezone": time_index.source_timezone,
                       "time_values_sha256": probe_module._chunked_array_hash(time_index.values_ns),
                       "feature_data_sha256": probe_module._chunked_array_hash(feature_values),
                       "boundary_columns": list(map(str, boundary_columns)),
                       "boundary_values_sha256": None if boundary_ids is None else probe_module._hash_strings(boundary_ids),
                       "selected_metadata_row": {str(k): str(v) for k, v in (metadata_row or {}).items()},
                       "data_file": str(data_path), "data_file_sha256": _sha256_file(data_path),
                       "metadata_file": str(metadata_path), "metadata_file_sha256": _sha256_file(metadata_path)}
    device = str(args.device)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else None
    return {"captured_before_training": True, "git_head": head, "git_dirty": dirty,
            "git_status_porcelain": status_text, "dirty_relevant_file_hashes": changed_paths,
            "git_error": git_error, "code_files": code_files,
            "startup_script": {"path": str(Path(startup_script).resolve()) if startup_script else None,
                               "sha256": _sha256_file(startup_script) if startup_script and Path(startup_script).is_file() else None,
                               "text": startup_text},
            "source_data": source_info,
            "command": [str(item) for item in __import__("sys").argv],
            "effective_config": vars(args), "training_environment": {
                "python": platform.python_version(), "platform": platform.platform(),
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()), "requested_device": device,
                "gpu": gpu,
                "numpy": __import__("numpy").__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            }}


def _write_run_supporting_evidence_unchecked(output_path, frame, train_length, test_labels,
                                            data_path, metadata_path, provenance,
                                            logical_output_path=None):
    """Persist complete test-axis labels/coordinates and training provenance."""
    output_path = Path(output_path)
    logical_output_path = Path(logical_output_path or output_path)
    if not output_path.is_dir() or not (output_path / "probe_manifest.json").is_file():
        raise ValueError("cannot write supporting evidence before a prepared completed run")
    labels = normalize_binary_labels(test_labels, len(frame) - int(train_length), "test labels")
    times = np.asarray(MNRFLiteV2._time_index(frame).values_ns[int(train_length):], dtype=np.int64)
    indices = np.arange(int(train_length), int(train_length) + len(labels), dtype=np.int64)
    label_column = next((column for column in frame.columns if str(column).lower() == "label"), None)
    if label_column is None:
        raise ValueError("parsed source frame lost its label column")
    full_labels = normalize_binary_labels(frame[label_column].to_numpy(), len(frame), "full source labels")
    if not np.array_equal(full_labels[int(train_length):], labels):
        raise ValueError("test labels differ from the parsed full source frame")
    all_times = np.asarray(MNRFLiteV2._time_index(frame).values_ns, dtype=np.int64)
    all_indices = np.arange(len(frame), dtype=np.int64)
    if len(times) != len(labels):
        raise ValueError("complete test time axis does not match labels")
    manifest_path = output_path / "probe_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_path = output_path / "training_provenance.json"
    if receipt_path.is_file():
        original_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        requested_migrations = (provenance or {}).get("migrations")
        provenance = original_receipt
        if requested_migrations is not None:
            provenance["migrations"] = requested_migrations
    identity = manifest.get("source_identity", {})
    segments = identity.get("test_segments", [])
    all_segment_ids = canonical_segment_ids(identity)
    segment_ids = all_segment_ids[int(train_length):]
    provenance = dict(provenance)
    startup = provenance.get("startup_script", {})
    snapshot = output_path / "provenance" / "startup_script.sh"
    if startup.get("text") is not None:
        if not snapshot.is_file():
            _atomic_bytes(snapshot, startup["text"].encode("utf-8"))
        startup["snapshot"] = str(snapshot.relative_to(output_path))
        startup["snapshot_sha256"] = _sha256_file(snapshot)
        startup.pop("text", None)
    # Preserve the training-time receipt before the ancillary axis write. If
    # the latter is interrupted, the dedicated supplement command can verify
    # the original inputs and finish without rerunning the model.
    provenance_path = output_path / "training_provenance.json"
    _atomic_json(provenance_path, provenance)

    fd, temporary_name = tempfile.mkstemp(prefix=".analysis_inputs.tmp-", suffix=".npz",
                                          dir=str(output_path))
    temporary = Path(temporary_name)
    with os.fdopen(fd, "wb") as temporary_file:
        np.savez(temporary_file, test_labels=labels.astype(np.int8, copy=False),
                 test_time_values_ns=times, test_global_indices=indices,
                 test_segment_ids=segment_ids,
                 all_labels=full_labels.astype(np.int8, copy=False),
                 all_time_values_ns=all_times,
                 all_global_indices=all_indices,
                 all_segment_ids=all_segment_ids)
        temporary_file.flush()
    os.replace(temporary, output_path / "analysis_inputs.npz")
    boundary_columns = manifest.get("source_identity", {}).get("boundary_columns", [])
    boundary_values = MNRFLiteV2._boundary_ids(frame)
    boundary_identity = {"columns": list(map(str, boundary_columns)), "values": []}
    boundary_identity["test_segment_identity"] = []
    if boundary_values is not None:
        unique = list(dict.fromkeys(map(str, boundary_values[int(train_length):])))
        boundary_identity["values"] = unique
        boundary_identity["test_segment_identity"] = [
            {"segment_id": int(all_segment_ids[int(left)]), "boundary_value": str(boundary_values[int(left)])}
            for left, _ in segments
        ]
    else:
        boundary_identity["test_segment_identity"] = [
            {"segment_id": int(all_segment_ids[int(left)]), "boundary_value": None}
            for left, _ in segments
        ]
    input_sha = _sha256_file(output_path / "analysis_inputs.npz")
    provenance["analysis_inputs"] = {"npz": str(logical_output_path / "analysis_inputs.npz"),
                                      "sha256": input_sha}
    _atomic_json(provenance_path, provenance)
    manifest.setdefault("analysis_artifacts", {})["analysis_inputs_sha256"] = input_sha
    manifest["analysis_artifacts"]["test_time_values_sha256"] = __import__(
        "ts_benchmark.baselines.mnrf_view_probe.probe", fromlist=["_chunked_array_hash"])._chunked_array_hash(times)
    manifest["analysis_artifacts"]["full_axis_sha256"] = {
        "labels": __import__("ts_benchmark.baselines.mnrf_view_probe.probe", fromlist=["_chunked_array_hash"])._chunked_array_hash(full_labels.astype(np.int8, copy=False)),
        "time_values_ns": __import__("ts_benchmark.baselines.mnrf_view_probe.probe", fromlist=["_chunked_array_hash"])._chunked_array_hash(all_times),
        "global_indices": __import__("ts_benchmark.baselines.mnrf_view_probe.probe", fromlist=["_chunked_array_hash"])._chunked_array_hash(all_indices),
        "segment_ids": __import__("ts_benchmark.baselines.mnrf_view_probe.probe", fromlist=["_chunked_array_hash"])._chunked_array_hash(all_segment_ids),
    }
    manifest["analysis_artifacts"]["training_provenance_sha256"] = _sha256_file(provenance_path)
    _atomic_json(manifest_path, manifest)
    metadata = {
        "schema": "mnrf-view-probe-analysis-inputs-v1", "source_manifest_sha256": _sha256_file(manifest_path),
        "source_identity_sha256": manifest.get("source_identity_sha256"),
        "train_length": int(train_length), "test_length": int(len(labels)),
        "labels_sha256": input_sha,
        "full_rows": int(len(frame)),
        "data_file": str(data_path), "data_file_sha256": _sha256_file(data_path),
        "metadata_file": str(metadata_path), "metadata_file_sha256": _sha256_file(metadata_path),
        "variable_names": manifest.get("source_identity", {}).get("variable_names", []),
        "time_kind": manifest.get("source_identity", {}).get("time_index", {}).get("kind"),
        "test_segments": manifest.get("source_identity", {}).get("test_segments", []),
        "full_segments": identity.get("segments", []),
        "test_segment_identity": boundary_identity["test_segment_identity"],
        "boundary_columns": list(map(str, boundary_columns)), "boundary_identity": boundary_identity,
        "test_segment_identity": boundary_identity["test_segment_identity"],
        "selected_metadata_row": {str(k): str(v) for k, v in (provenance.get("source_data", {}).get("selected_metadata_row", {}) or {}).items()},
        "provenance_file": "training_provenance.json",
        "provenance_sha256": _sha256_file(provenance_path),
    }
    _atomic_json(output_path / "analysis_inputs.json", metadata)


def _write_run_supporting_evidence(output_path, frame, train_length, test_labels,
                                   data_path, metadata_path, provenance):
    """Atomically publish auxiliary evidence without touching unrelated files."""
    output_path = Path(output_path)
    _preflight_supporting_evidence(output_path, frame, train_length, test_labels,
                                   data_path, metadata_path, provenance)
    relative_paths = (
        Path("probe_manifest.json"), Path("training_provenance.json"),
        Path("analysis_inputs.npz"), Path("analysis_inputs.json"),
        Path("provenance/startup_script.sh"),
    )
    before = {}
    for relative in relative_paths:
        path = output_path / relative
        parent = path.parent
        if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
            raise ValueError("refusing auxiliary write through non-directory/symlink parent: {}".format(parent))
        if path.is_symlink():
            raise ValueError("refusing to replace symlink auxiliary artifact: {}".format(path))
        if path.exists() and not path.is_file():
            raise ValueError("refusing to replace non-file auxiliary artifact: {}".format(path))
        before[relative] = path.read_bytes() if path.is_file() else None

    transaction_dir = Path(tempfile.mkdtemp(prefix=".viewprobe-aux-txn-", dir=str(output_path)))
    attempted = []
    try:
        for relative in (Path("probe_manifest.json"), Path("training_provenance.json"),
                         Path("provenance/startup_script.sh")):
            data = before[relative]
            if data is not None:
                staged = transaction_dir / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(data)
        for relative, data in before.items():
            if data is not None:
                backup = transaction_dir / "backup" / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(data)
        _write_run_supporting_evidence_unchecked(
            transaction_dir, frame, train_length, test_labels,
            data_path, metadata_path, provenance, logical_output_path=output_path)
        for relative in relative_paths:
            staged = transaction_dir / relative
            if not staged.is_file():
                continue
            target = output_path / relative
            if target.parent.is_symlink():
                raise ValueError("refusing auxiliary publish through a symlink parent: {}".format(target.parent))
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError("auxiliary target changed type during transaction: {}".format(target))
            current = target.read_bytes() if target.is_file() else None
            if current != before[relative]:
                raise RuntimeError("auxiliary target changed during transaction; refusing to overwrite: {}".format(target))
            new_bytes = staged.read_bytes()
            attempted.append((relative, new_bytes))
            os.replace(staged, target)
    except BaseException as original_error:
        rollback_errors = []
        for relative, new_bytes in reversed(attempted):
            target = output_path / relative
            old_bytes = before[relative]
            try:
                if target.is_symlink():
                    raise OSError("target became a symlink during rollback")
                current = target.read_bytes() if target.is_file() else None
                if current == old_bytes:
                    continue
                if current != new_bytes:
                    raise OSError("target changed outside this transaction; refusing to overwrite it")
                if old_bytes is None:
                    target.unlink()
                else:
                    _atomic_bytes(target, old_bytes)
            except BaseException as rollback_error:
                rollback_errors.append("{}: {}".format(target, rollback_error))
        if rollback_errors:
            raise RuntimeError(
                "auxiliary write failed ({}); rollback failed ({}); recovery snapshots retained at {}"
                .format(original_error, "; ".join(rollback_errors), transaction_dir)
            ) from original_error
        shutil.rmtree(transaction_dir)
        raise
    shutil.rmtree(transaction_dir)


def _preflight_supporting_evidence(output_path, frame, train_length, test_labels,
                                   data_path, metadata_path, provenance):
    """Perform semantic checks before creating transaction files."""
    output_path = Path(output_path)
    manifest_path = output_path / "probe_manifest.json"
    if not output_path.is_dir() or not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("cannot write supporting evidence before a prepared completed run")
    train_length = int(train_length)
    if train_length < 0 or train_length > len(frame):
        raise ValueError("supporting evidence train boundary is outside the source axis")
    labels = normalize_binary_labels(test_labels, len(frame) - train_length, "test labels")
    label_column = next((column for column in frame.columns if str(column).lower() == "label"), None)
    if label_column is None:
        raise ValueError("parsed source frame lost its label column")
    full_labels = normalize_binary_labels(frame[label_column].to_numpy(), len(frame), "full source labels")
    if not np.array_equal(full_labels[train_length:], labels):
        raise ValueError("test labels differ from the parsed full source frame")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("source_identity") or {}
    if identity.get("input_length") != len(frame) or identity.get("train_length") != train_length:
        raise ValueError("supporting evidence source length/train boundary differs from frozen identity")
    canonical_segment_ids(identity)
    times = np.asarray(MNRFLiteV2._time_index(frame).values_ns, dtype=np.int64)
    if len(times) != len(full_labels):
        raise ValueError("complete source time axis does not match labels")
    source_data = (provenance or {}).get("source_data") or {}
    for path, digest_key, description in (
            (Path(data_path), "data_file_sha256", "task CSV"),
            (Path(metadata_path), "metadata_file_sha256", "metadata CSV")):
        if not path.is_file():
            raise ValueError("supporting evidence {} is missing: {}".format(description, path))
        declared = source_data.get(digest_key)
        if not declared or _sha256_file(path) != declared:
            raise ValueError("supporting evidence {} identity is missing or changed: {}".format(description, path))
    if source_data.get("rows") != len(frame) or source_data.get("train_length") != train_length:
        raise ValueError("supporting evidence parsed row/train identity differs")
    if source_data.get("variable_names") != identity.get("variable_names"):
        raise ValueError("supporting evidence variable order differs from frozen identity")
    if source_data.get("feature_data_sha256") != identity.get("feature_data_sha256"):
        raise ValueError("supporting evidence feature values differ from frozen identity")
    return manifest


def _atomic_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    _atomic_bytes(path, payload.encode("utf-8"))


def _atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".tmp-", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _persist_training_receipt(output_path, provenance, resume=False):
    """Persist immutable training facts after run preparation and before fit."""
    output_path = Path(output_path)
    receipt = output_path / "training_provenance.json"
    if resume:
        # The original training facts are immutable; the current invocation is
        # recorded separately and can never replace them.
        context = output_path / "provenance" / "resume_context.json"
        context.parent.mkdir(exist_ok=True)
        _atomic_json(context, {"resume_command": list(sys.argv), "resume_environment": provenance.get("training_environment")})
        return
    if receipt.exists():
        raise ValueError("refusing to overwrite an existing training provenance receipt")
    saved = dict(provenance)
    startup = dict(saved.get("startup_script") or {})
    if startup.get("text") is not None:
        snapshot_dir = output_path / "provenance"
        snapshot_dir.mkdir(exist_ok=True)
        snapshot = snapshot_dir / "startup_script.sh"
        _atomic_bytes(snapshot, startup["text"].encode("utf-8"))
        startup["snapshot"] = str(snapshot.relative_to(output_path))
        startup["snapshot_sha256"] = _sha256_file(snapshot)
        startup.pop("text", None)
    saved["startup_script"] = startup
    saved["receipt_schema"] = "mnrf-view-probe-training-receipt-v1"
    _atomic_json(receipt, saved)
    print("MNRF event=training_receipt_saved_before_fit path={}".format(receipt), flush=True)


def _read_task(task, source, data_root=None):
    if source == "large_detect":
        folder = Path(data_root).expanduser() if data_root else ROOT / "dataset" / "anomaly_detect"
        source_class = LocalAnomalyDetectDataSource
        metadata_name = "DETECT_META.csv"
    elif source == "external_detect":
        folder = Path(data_root).expanduser() if data_root else ROOT / "dataset" / "external_validation"
        source_class = LocalExternalAnomalyDetectDataSource
        metadata_name = "EXTERNAL_DETECT_META.csv"
    else:
        raise ValueError("data source must be large_detect or external_detect")
    reader = source_class.__new__(source_class)
    reader.local_data_path = str(folder / "data")
    frame = reader._load_series(f"{task}.csv")
    metadata = pd.read_csv(folder / metadata_name)
    row = metadata.loc[metadata.file_name.astype(str) == f"{task}.csv"]
    if len(row) != 1:
        raise ValueError(f"metadata has no unique row for {task}.csv")
    train_length = int(row.iloc[0].train_lens)
    label_column = next((column for column in frame.columns if str(column).lower() == "label"), None)
    if label_column is None:
        raise ValueError("task data has no label column")
    normalize_binary_labels(frame[label_column].to_numpy(), len(frame), "full source labels")
    test_labels = normalize_binary_labels(frame.iloc[train_length:][label_column].to_numpy(),
                                          len(frame) - train_length, "test labels")
    return frame, train_length, test_labels, folder / "data" / f"{task}.csv", folder / metadata_name, row.iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-source", choices=("large_detect", "external_detect"), default="large_detect")
    parser.add_argument("--mode", choices=("stage1", "full_gmm", "random_gmm", "select_gmm", "s_only", "flow"), default="full_gmm")
    parser.add_argument("--views", default="S,E,P,R")
    parser.add_argument("--source-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--pattern-len", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--feature-std-floor", type=float, default=0.05)
    parser.add_argument("--score-mad-floor", type=float, default=0.001)
    parser.add_argument("--calibration-quantile", type=float, default=0.995)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-chunk-size", type=int, default=256)
    parser.add_argument("--encoding-batch-size", type=int, default=0)
    parser.add_argument("--evolution-dim", type=int, default=None)
    parser.add_argument("--pattern-dim", type=int, default=None)
    parser.add_argument("--relation-dim", type=int, default=None)
    parser.add_argument("--view-dim", type=int, default=8)
    parser.add_argument("--gmm-components", type=int, default=4)
    parser.add_argument("--gmm-covariance-type", default="full", choices=("full", "tied", "diag", "spherical"))
    parser.add_argument("--gmm-reg-covar", type=float, default=0.001)
    parser.add_argument("--gmm-n-init", type=int, default=1)
    parser.add_argument("--gmm-max-iter", type=int, default=100)
    parser.add_argument("--gmm-fit-max-windows", type=int, default=20000)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--train-noise-std", type=float, default=0.01)
    parser.add_argument("--noise-seed", type=int, default=20261013)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--artifact-policy", choices=("full", "results"), default="full")
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--log-every-seconds", type=float, default=30.0)
    parser.add_argument("--gmm-log-every-iters", type=int, default=10)
    parser.add_argument("--startup-log", default="",
                        help="shell preflight log to include in the durable run log")
    parser.add_argument("--startup-script", default="",
                        help="invoking task script captured in training provenance")
    parser.add_argument("--data-root", default="",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    def _terminate(signum, _frame):
        raise RuntimeError("received signal {}".format(signum))

    signal.signal(signal.SIGINT, _terminate)
    signal.signal(signal.SIGTERM, _terminate)
    # Keep one durable log for each run without creating anything in the run
    # directory before resume preflight has completed.  This is deliberately a
    # CLI concern so the library API remains usable without a logging policy.
    fd, temporary_log = tempfile.mkstemp(prefix="mnrf_view_probe-", suffix=".log")
    os.close(fd)
    temporary_log = Path(temporary_log)
    log_handle = temporary_log.open("a", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr

    startup_log = Path(args.startup_log).expanduser() if args.startup_log else None
    if startup_log is not None and startup_log.is_file():
        with startup_log.open("rb") as source:
            initial = source.read()
        if initial:
            log_handle.write(initial.decode("utf-8", errors="replace"))
            log_handle.flush()
    output_path = Path(args.output_dir or args.resume_from).expanduser().resolve()

    class _Tee:
        def __init__(self, primary, secondary, target_path, attach_when, state):
            self.primary, self.secondary = primary, secondary
            self.target_path, self.attach_when = target_path, attach_when
            self.state = state

        def _attach(self):
            if self.state["target"] is not None or not self.attach_when():
                return
            self.secondary.flush()
            self.target_path.parent.mkdir(parents=True, exist_ok=True)
            exists = self.target_path.is_file() and self.target_path.stat().st_size > 0
            self.state["target"] = self.target_path.open("ab")
            if exists:
                self.state["target"].write(b"\n--- resumed live log ---\n")
            with temporary_log.open("rb") as source:
                shutil.copyfileobj(source, self.state["target"])
            self.state["target"].flush()

        def write(self, value):
            # Attach before writing the current chunk.  The temporary log is
            # copied exactly once; after that point the same chunk is written
            # to the durable target and the temporary mirror.
            self._attach()
            self.primary.write(value)
            self.secondary.write(value)
            if self.state["target"] is not None:
                encoded = value.encode("utf-8", errors="replace") if isinstance(value, str) else value
                self.state["target"].write(encoded)
            self.primary.flush()
            self.secondary.flush()
            if self.state["target"] is not None:
                self.state["target"].flush()
            return len(value)

        def flush(self):
            self.primary.flush()
            self.secondary.flush()
            if self.state["target"] is not None:
                self.state["target"].flush()

        def isatty(self):
            return self.primary.isatty()

    # ``probe._writes_started`` flips only after read-only resume preflight and
    # directory preparation.  Thus rejected resumes never modify their target
    # run.log, while accepted runs become durable as soon as the first event is
    # emitted after preparation.
    target_path = output_path / "run.log"
    live_state = {"target": None}
    sys.stdout = _Tee(original_stdout, log_handle, target_path,
                      lambda: probe is not None and getattr(probe, "_writes_started", False), live_state)
    sys.stderr = _Tee(original_stderr, log_handle, target_path,
                      lambda: probe is not None and getattr(probe, "_writes_started", False), live_state)
    probe = None
    failed = None
    failure_context = None
    try:
        print("MNRF event=startup task={} mode={} run_id={} pid={} output_dir={} log_path={} device={} "
              "cuda_visible_devices={} seed={} seq_len={} batch_size={} encoding_batch_size={} "
              "views={} dims=1/{}/{}/{} selected_dim={} artifact_policy={} cpu_threads={} "
              "model_training=uninitialized grad_enabled={} debug_callback=None pytorch={} cuda_runtime={} cudnn={} gpu={}".format(
                  args.task, args.mode, output_path.name, os.getpid(), output_path, output_path / "run.log", args.device,
                  os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"), args.seed, args.seq_len,
                  args.batch_size, args.encoding_batch_size or args.batch_size, args.views,
                  args.evolution_dim if args.evolution_dim is not None else args.view_dim,
                  args.pattern_dim if args.pattern_dim is not None else args.view_dim,
                  args.relation_dim if args.relation_dim is not None else args.view_dim,
                  1 + (args.evolution_dim if args.evolution_dim is not None else args.view_dim)
                  + (args.pattern_dim if args.pattern_dim is not None else args.view_dim)
                  + (args.relation_dim if args.relation_dim is not None else args.view_dim),
                  args.artifact_policy, args.cpu_threads, __import__("torch").is_grad_enabled(),
                  __import__("torch").__version__, getattr(__import__("torch").version, "cuda", None),
                  getattr(__import__("torch").backends.cudnn, "version", lambda: None)(),
                  __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else "cpu"),
              flush=True)
        print("MNRF event=preflight_start", flush=True)
        frame, train_length, test_labels, data_path, metadata_path, metadata_row = _read_task(
            args.task, args.data_source, args.data_root or None)
        training_provenance = _capture_training_provenance(
            args, args.startup_script, data_path, metadata_path, metadata_row, frame, train_length)
        kwargs = dict(seed=args.seed, seq_len=args.seq_len, pattern_len=args.pattern_len,
                      hidden_dim=args.hidden_dim, batch_size=args.batch_size,
                      stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
                      patience=args.patience, learning_rate=args.learning_rate,
                      weight_decay=args.weight_decay, feature_std_floor=args.feature_std_floor,
                      score_mad_floor=args.score_mad_floor,
                      calibration_quantile=args.calibration_quantile,
                      device=args.device, cache_chunk_size=args.cache_chunk_size,
                      encoding_batch_size=args.encoding_batch_size, view_dim=args.view_dim,
                      evolution_dim=args.evolution_dim, pattern_dim=args.pattern_dim,
                      relation_dim=args.relation_dim, mode=args.mode,
                      selected_views=tuple(item for item in args.views.split(",") if item),
                      source_dir=args.source_dir, output_dir=args.output_dir,
                      resume_from=args.resume_from, data_source=args.data_source,
                      task_name=f"{args.task}.csv",
                      gmm_components=args.gmm_components, gmm_covariance_type=args.gmm_covariance_type,
                      gmm_reg_covar=args.gmm_reg_covar, gmm_n_init=args.gmm_n_init,
                      gmm_max_iter=args.gmm_max_iter, gmm_fit_max_windows=args.gmm_fit_max_windows,
                      train_noise_std=args.train_noise_std, noise_seed=args.noise_seed,
                      cpu_threads=args.cpu_threads, artifact_policy=args.artifact_policy,
                      log_every_steps=args.log_every_steps, log_every_seconds=args.log_every_seconds,
                      gmm_log_every_iters=args.gmm_log_every_iters)
        probe = MNRFViewProbe(**kwargs)
        if args.mode == "full_gmm" and args.artifact_policy == "full":
            probe._on_writes_started = lambda: _persist_training_receipt(
                output_path, training_provenance, resume=bool(args.resume_from))
        strategy = IndexedViewProbeStrategy(probe)
        print("MNRF event=data_read_complete task={} train_rows={} test_rows={} device={}".format(
            args.task, train_length, len(frame) - train_length, probe.device), flush=True)
        if args.mode == "stage1":
            probe.fit(frame, train_length, None)
        else:
            strategy.run(frame.iloc[:train_length].copy(), frame.iloc[train_length:].copy(), test_labels)
        if args.mode == "full_gmm" and args.artifact_policy == "full":
            if args.resume_from and not (output_path / "training_provenance.json").is_file():
                raise ValueError("resumed run has no original training receipt; it will not be replaced with resume-time facts")
            try:
                _write_run_supporting_evidence(output_path, frame, train_length, test_labels,
                                               data_path, metadata_path, training_provenance)
            except Exception as error:
                failure_context = "auxiliary_support"
                print("MNRF event=auxiliary_support_failed model_status=complete reason={}".format(error), flush=True)
                raise
        print("MNRF event=run_complete task={} output_dir={}".format(args.task, output_path), flush=True)
        print(json.dumps({"status": "complete", "task": args.task, "mode": args.mode,
                          "output_dir": str(probe.output_dir), "selected_views": list(probe.config.selected_views),
                          "selected_dim": probe.config.selected_dim, "layout": probe.config.layout.to_dict()}, sort_keys=True))
    except BaseException as error:
        failed = error
        event = "auxiliary_support_failed" if failure_context == "auxiliary_support" else "run_failed"
        model_status = " model_status=complete" if failure_context == "auxiliary_support" else ""
        print("MNRF event={}{} error_type={} reason={}".format(
            event, model_status, type(error).__name__, error), flush=True)
        traceback.print_exc()
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log_handle.flush()
        log_handle.close()
        if live_state["target"] is not None:
            live_state["target"].flush()
            os.fsync(live_state["target"].fileno())
            live_state["target"].close()
        # The live Tee normally attached after _prepare_dirs.  Keep a
        # preflight-only log outside the run directory so a rejected resume is
        # fully read-only and still leaves an actionable diagnostic.
        attached = output_path.is_dir() and getattr(probe, "_writes_started", False)
        if attached and not (output_path / "run.log").is_file():
            with (output_path / "run.log").open("ab") as target, temporary_log.open("rb") as source:
                target.write(source.read()); target.flush(); os.fsync(target.fileno())
        if startup_log is not None and startup_log.is_file() and (failed is None or attached):
            startup_log.unlink(missing_ok=True)
        if failed is None or attached:
            temporary_log.unlink(missing_ok=True)
        else:
            if startup_log is not None and startup_log.is_file():
                print("MNRF startup_log={}".format(startup_log), file=original_stderr, flush=True)
            print("MNRF preflight_log={}".format(temporary_log), file=original_stderr, flush=True)
    if failed is not None:
        raise SystemExit(1) from failed


if __name__ == "__main__":
    main()
