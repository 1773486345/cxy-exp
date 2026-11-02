"""Verified MSL channel-to-row boundary mapping for the segmented diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = ROOT / "result/analysis/MNRF-ViewProbe-DataRepair/msl-raw-mapping-20261031.json"
EXPECTED_FEATURE_SHA256 = "5fe9f96f80e632f01640f3a59f65e295ec300e084b6700d571120a22b9074a5a"
FEATURE_COLUMNS = [f"col_{index}" for index in range(55)]


def load_mapping(path=DEFAULT_MAPPING, *, require_registry=True):
    path = Path(path).expanduser().resolve()
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("schema") != "msl-raw-mapping-v1":
        raise ValueError("unsupported MSL sequence mapping schema")
    channels = mapping.get("channels")
    counts = mapping.get("channel_row_counts")
    if not isinstance(channels, list) or not channels or len(set(channels)) != len(channels):
        raise ValueError("MSL channel mapping must contain unique ordered channels")
    if not isinstance(counts, dict) or set(counts) != set(channels):
        raise ValueError("MSL mapping row counts do not match its ordered channel list")
    for channel in channels:
        item = counts[channel]
        if set(item) != {"train", "test"} or any(int(item[key]) <= 0 for key in item):
            raise ValueError(f"invalid train/test row counts for MSL channel {channel}")
    train_rows = sum(int(counts[channel]["train"]) for channel in channels)
    test_rows = sum(int(counts[channel]["test"]) for channel in channels)
    if train_rows != int(mapping.get("train_rows", -1)) or test_rows != int(mapping.get("test_rows", -1)):
        raise ValueError("MSL mapping row totals do not match channel counts")
    if mapping.get("local_feature_names") != FEATURE_COLUMNS or int(mapping.get("columns", -1)) != 55:
        raise ValueError("MSL mapping does not preserve the verified 55-column order")
    if mapping.get("expected_feature_sha256") != EXPECTED_FEATURE_SHA256:
        raise ValueError("MSL mapping feature identity differs from the verified source")
    if require_registry and (len(channels) != 27 or channels != sorted(channels)):
        raise ValueError("formal MSL boundary protocol requires the verified sorted 27-channel registry")
    return mapping, hashlib.sha256(path.read_bytes()).hexdigest(), path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_csv(mapping, data_path):
    expected = mapping.get("local_csv_sha256")
    if not expected:
        raise ValueError("MSL boundary mapping has no bound local CSV digest")
    actual = sha256_file(data_path)
    if actual != expected:
        raise ValueError(f"MSL source CSV digest differs from sequence mapping: {actual} != {expected}")
    return actual


def _frame_feature_hash(frame, columns, chunk_rows=8192):
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": [len(frame), len(columns)], "dtype": "float32"},
                             sort_keys=True).encode())
    for start in range(0, len(frame), int(chunk_rows)):
        block = frame.iloc[start:start + int(chunk_rows)].loc[:, columns].to_numpy(dtype=np.float32)
        digest.update(np.ascontiguousarray(block).tobytes())
    return digest.hexdigest()


def annotate_frame(frame, train_length, mapping, *, require_registry=True):
    """Attach non-feature entity ids without changing values, labels or row order."""
    import pandas as pd

    channels = mapping["channels"]
    counts = mapping["channel_row_counts"]
    expected_train = sum(int(counts[channel]["train"]) for channel in channels)
    expected_test = sum(int(counts[channel]["test"]) for channel in channels)
    if int(train_length) != expected_train or len(frame) != expected_train + expected_test:
        raise ValueError(
            "MSL source rows differ from verified sequence mapping: "
            f"train={train_length}/{expected_train}, total={len(frame)}/{expected_train + expected_test}"
        )
    feature_columns = [str(column) for column in frame.columns
                       if str(column).lower() not in {
                           "label", "timestamp", "date", "time", "datetime", "index",
                           "entity_id", "session_id", "record_id", "scenario_id"}]
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(f"MSL parsed feature order differs from mapping: {feature_columns}")
    feature_hash = _frame_feature_hash(frame, feature_columns)
    if feature_hash != mapping.get("expected_feature_sha256"):
        raise ValueError("parsed MSL feature values/order differ from the verified raw mapping")
    if require_registry and (len(channels) != 27 or channels != sorted(channels)):
        raise ValueError("formal MSL boundary protocol requires the verified sorted 27-channel registry")
    ids = np.empty(len(frame), dtype=object)
    cursor = 0
    for channel in channels:
        stop = cursor + int(counts[channel]["train"])
        ids[cursor:stop] = f"train:{channel}"
        cursor = stop
    if cursor != int(train_length):
        raise ValueError("MSL training sequence map does not end at metadata train boundary")
    for channel in channels:
        stop = cursor + int(counts[channel]["test"])
        ids[cursor:stop] = f"test:{channel}"
        cursor = stop
    if cursor != len(frame):
        raise ValueError("MSL test sequence map does not cover the full source")
    result = frame.copy()
    if "entity_id" in result.columns:
        raise ValueError("MSL parsed input unexpectedly already has entity_id")
    result["entity_id"] = ids
    return result


def mapping_summary(mapping, mapping_sha256, mapping_path):
    return {
        "schema": mapping["schema"],
        "mapping_path": str(mapping_path),
        "mapping_sha256": mapping_sha256,
        "channels": list(mapping["channels"]),
        "channel_row_counts": mapping["channel_row_counts"],
        "train_rows": int(mapping["train_rows"]),
        "test_rows": int(mapping["test_rows"]),
        "feature_columns": list(FEATURE_COLUMNS),
        "expected_feature_sha256": EXPECTED_FEATURE_SHA256,
        "local_csv_sha256": mapping.get("local_csv_sha256"),
        "boundary_id_rule": "distinct train:<channel> and test:<channel>; never cross any source sequence or train/test edge",
    }
