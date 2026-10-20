"""Shared integrity checks for reusable MNRF ViewProbe artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def normalize_binary_labels(values, expected_length=None, name="labels"):
    """Validate exact numeric 0/1 labels and return canonical int8 values.

    Boolean labels are intentionally accepted: bool is an exact two-valued
    numeric representation, not an implicit integer cast.
    """
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if expected_length is not None and len(labels) != int(expected_length):
        raise ValueError(f"{name} length {len(labels)} does not match expected {int(expected_length)}")
    if labels.dtype.kind not in "iufb":
        raise ValueError(f"{name} must have a real integer, unsigned, float, or boolean dtype")
    if labels.dtype.kind == "f" and not np.isfinite(labels).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{name} must contain exact binary values 0 or 1")
    return labels.astype(np.int8, copy=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_normalized_values_cache(run_dir, manifest, expected_variables=None,
                                     chunk_rows=65536):
    """Verify the actual normalized cache bytes and its frozen input binding."""
    run = Path(run_dir).resolve()
    identity = manifest.get("source_identity") or {}
    cache_path = run / "cache" / "normalized_values.npy"
    declared = manifest.get("normalized_cache_sha256")
    if not declared:
        raise ValueError("normalized_values.npy has no authoritative normalized_cache_sha256 binding")
    if not cache_path.is_file():
        raise ValueError("normalized_values.npy is missing from the self-contained full source")
    try:
        array = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise ValueError(f"normalized_values.npy cannot be loaded: {error}") from error
    input_length = identity.get("input_length")
    variables = expected_variables
    if variables is None:
        variables = identity.get("variable_names", manifest.get("variable_names", []))
    if not isinstance(input_length, int) or not isinstance(variables, (list, tuple)):
        raise ValueError("normalized cache is not bound to a valid frozen input length/variable order")
    expected_shape = (input_length, len(variables))
    if array.shape != expected_shape or array.dtype != np.dtype("float32"):
        raise ValueError(
            f"normalized_values.npy shape/dtype mismatch: expected {expected_shape}/float32, "
            f"got {array.shape}/{array.dtype}"
        )
    if manifest.get("variable_names", list(variables)) != list(variables):
        raise ValueError("normalized cache variable order differs from frozen source identity")
    for start in range(0, len(array), int(chunk_rows)):
        if not np.isfinite(array[start:start + int(chunk_rows)]).all():
            raise ValueError(f"normalized_values.npy contains non-finite values in rows {start}:{start + int(chunk_rows)}")
    actual = sha256_file(cache_path)
    if actual != declared:
        raise ValueError(
            f"normalized_values.npy sha256 mismatch: expected={declared} actual={actual}"
        )
    stats_path = run / "input_statistics.npz"
    declared_stats = manifest.get("raw_input_statistics_sha256", manifest.get("input_statistics_sha256"))
    if not declared_stats or not stats_path.is_file() or sha256_file(stats_path) != declared_stats:
        raise ValueError("normalized_values.npy input-statistics binding is missing or changed")
    try:
        with np.load(stats_path, allow_pickle=False) as stats:
            mean = np.asarray(stats["mean"])
            scale = np.asarray(stats["scale"])
    except Exception as error:
        raise ValueError(f"normalized cache input statistics cannot be read: {error}") from error
    if (mean.shape != (len(variables),) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError("normalized cache input statistics have invalid shape or values")
    return {"path": str(cache_path), "sha256": actual, "shape": list(array.shape),
            "dtype": str(array.dtype), "bytes": int(cache_path.stat().st_size)}


def canonical_segment_ids(identity):
    """Build complete-axis IDs from frozen segments and verify test intersections."""
    input_length = int(identity.get("input_length", -1))
    train_length = int(identity.get("train_length", -1))
    segments = identity.get("segments")
    test_segments = identity.get("test_segments")
    if input_length < 0 or train_length < 0 or train_length > input_length:
        raise ValueError("frozen segment axis has invalid length or train boundary")
    if not isinstance(segments, list) or not isinstance(test_segments, list):
        raise ValueError("frozen complete/test segment lists are missing")
    ids = np.full(input_length, -1, dtype=np.int32)
    intersections = []
    previous_right = 0
    for segment_id, interval in enumerate(segments):
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("frozen continuous segment has invalid interval")
        left, right = map(int, interval)
        if left < 0 or right <= left or right > input_length or left < previous_right:
            raise ValueError("frozen continuous segments overlap or exceed the full axis")
        if np.any(ids[left:right] != -1):
            raise ValueError("frozen continuous segments overlap")
        ids[left:right] = segment_id
        previous_right = right
        a, b = max(left, train_length), min(right, input_length)
        if b > a:
            intersections.append([a, b])
    if input_length and np.any(ids < 0):
        raise ValueError("frozen continuous segments do not cover the complete source axis")
    normalized_test = []
    previous_right = train_length
    for interval in test_segments:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("frozen test segment has invalid interval")
        left, right = map(int, interval)
        if left < train_length or right <= left or right > input_length or left < previous_right:
            raise ValueError("frozen test segments overlap or exceed the test axis")
        normalized_test.append([left, right])
        previous_right = right
    if normalized_test != intersections:
        raise ValueError("frozen test segments are not the complete-axis segment intersections")
    return ids
