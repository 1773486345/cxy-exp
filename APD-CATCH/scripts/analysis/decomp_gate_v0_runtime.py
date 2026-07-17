"""Atomic persistence and immutable-study helpers for Gate v0 orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value)!r}")


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, default=json_default).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def stage_state(
    manifest: Dict[str, Any],
    stage: str,
    *,
    status: str | None = None,
    anomaly_type: str | None = None,
    last_completed_stage: str | None = None,
    exit_reason: str | None = None,
) -> None:
    manifest["updated_at_utc"] = utc_now()
    manifest["current_stage"] = stage
    manifest["current_anomaly_type"] = anomaly_type
    if status is not None:
        manifest["status"] = status
    if last_completed_stage is not None:
        manifest["last_completed_stage"] = last_completed_stage
    if exit_reason is not None:
        manifest["exit_reason"] = exit_reason


def write_progress(shard_directory: Path, manifest: Dict[str, Any]) -> None:
    atomic_write_json(shard_directory / "manifest.json", manifest)
    atomic_write_json(
        shard_directory / "progress.json",
        {
            key: manifest.get(key)
            for key in (
                "status", "pid", "hostname", "started_at_utc", "updated_at_utc",
                "current_stage", "current_seed", "current_anomaly_type",
                "last_completed_stage", "exit_reason",
            )
        },
    )


def runtime_identity() -> Dict[str, Any]:
    return {"pid": os.getpid(), "hostname": socket.gethostname(), "started_at_utc": utc_now()}
