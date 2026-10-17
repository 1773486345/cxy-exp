#!/usr/bin/env python
"""Read-only metadata and path check; does not construct caches or train."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
from ts_benchmark.baselines.mnrf_lite_v2.training import EndWindowDataset, purged_ranges
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
from ts_benchmark.baselines.mnrf_lite_v2.windows import (
    TimeIndex, continuous_segments, infer_expected_interval, intersect_segments,
)
from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource, LocalExternalAnomalyDetectDataSource

def _load_task_read_only(source, folder, task):
    source_type = LocalAnomalyDetectDataSource if source == "large_detect" else LocalExternalAnomalyDetectDataSource
    reader = source_type.__new__(source_type)
    reader.local_data_path = str(folder / "data")
    try:
        frame = reader._load_series(f"{task}.csv")
    except Exception as error:
        raise ValueError(f"task={task} source={source}: source parser failed: {error}") from error
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    return frame


def preflight(task: str, source: str, batch_size: int = 64, seq_len: int = 128,
              encoding_batch_size: int = 0, root: Path = ROOT):
    if source == "large_detect":
        folder = root / "dataset/anomaly_detect"
        meta_path = folder / "DETECT_META.csv"
    elif source == "external_detect":
        folder = root / "dataset/external_validation"
        meta_path = folder / "EXTERNAL_DETECT_META.csv"
    else:
        raise ValueError("source must be large_detect or external_detect")
    data_path = folder / "data" / f"{task}.csv"
    if not meta_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"task={task} source={source}: metadata or task CSV missing: {meta_path}, {data_path}")
    metadata = pd.read_csv(meta_path)
    if "file_name" not in metadata or "train_lens" not in metadata:
        raise ValueError(f"task={task} source={source}: metadata requires file_name and train_lens columns")
    if metadata.file_name.duplicated().any():
        duplicates = metadata.loc[metadata.file_name.duplicated(keep=False), "file_name"].astype(str).tolist()
        raise ValueError(f"task={task} source={source}: metadata file_name is not unique: {duplicates}")
    row = metadata.loc[metadata.file_name == f"{task}.csv"]
    if len(row) != 1:
        raise ValueError(f"task={task} source={source}: expected exactly one metadata row for {task}.csv; found {len(row)}")
    record = row.iloc[0]
    univariate_value = record.get("if_univariate", False)
    is_univariate = (str(univariate_value).strip().lower() in {"1", "true", "yes"}
                     if not pd.isna(univariate_value) else False)
    try:
        if pd.isna(record["train_lens"]):
            raise ValueError("train_lens is missing")
        train_length = int(record["train_lens"])
    except Exception as error:
        raise ValueError(f"task={task} source={source}: invalid train_lens metadata: {error}") from error
    try:
        config = MNRFConfig.from_kwargs(batch_size=batch_size, seq_len=seq_len,
                                        encoding_batch_size=encoding_batch_size)
    except Exception as error:
        raise ValueError(f"task={task} source={source}: invalid MNRF preflight configuration: {error}") from error
    frame = _load_task_read_only(source, folder, task)
    if "label" not in {str(column).lower() for column in frame.columns}:
        raise ValueError(f"task={task} source={source}: parsed series has no label column")
    try:
        values, actual_variables = MNRFLiteV2._features_from_frame(frame)
    except Exception as error:
        raise ValueError(f"task={task} source={source}: invalid parsed features: {error}") from error
    if train_length >= len(frame):
        raise ValueError(f"task={task} source={source}: metadata train_lens={train_length} must leave a nonempty test interval")
    train_frame = frame.iloc[:train_length]
    train_values, actual_variables = MNRFLiteV2._features_from_frame(train_frame)
    if len(actual_variables) < 2:
        raise ValueError(f"task={task} source={source}: parsed feature count D={len(actual_variables)} is below 2")
    if is_univariate:
        raise ValueError(f"task={task} source={source}: metadata marks task univariate but MNRF-Lite V2 requires D >= 2")
    try:
        split = purged_ranges(train_length, seq_len)
    except Exception as error:
        raise ValueError(f"task={task} source={source}: official training boundary cannot support purged splits: {error}") from error
    time_index = MNRFLiteV2._time_index(train_frame)
    boundary_ids = MNRFLiteV2._boundary_ids(train_frame)
    a, b = split["train"]
    training_time = TimeIndex(time_index.values_ns[a:b], time_index.kind, time_index.source_timezone)
    training_boundaries = None if boundary_ids is None else boundary_ids[a:b]
    expected_interval = infer_expected_interval(training_time, b - a, training_boundaries)
    train_segments, train_audit = continuous_segments(
        time_index, train_length, expected_interval=expected_interval,
        boundary_ids=boundary_ids,
    )
    split_counts = {}
    for name in ("train", "validation", "calibration"):
        parts = intersect_segments(train_segments, *split[name])
        split_counts[name] = len(EndWindowDataset(train_values, *split[name], seq_len, parts))
    test_frame = frame.iloc[train_length:]
    test_segments, test_audit = continuous_segments(
        MNRFLiteV2._time_index(test_frame), len(test_frame), expected_interval=expected_interval,
        boundary_ids=MNRFLiteV2._boundary_ids(test_frame),
    )
    test_values, _ = MNRFLiteV2._features_from_frame(test_frame)
    test_windows = len(EndWindowDataset(test_values, 0, len(test_values), seq_len, test_segments))
    if min(split_counts.values()) < 1 or test_windows < 1:
        raise ValueError(f"task={task} source={source}: unsupported after continuous segmentation: split windows={split_counts}, test windows={test_windows}")
    print("MNRF-Lite V2 read-only preflight")
    print(f"task={task}.csv source={source} data={data_path}")
    print(f"variables={len(actual_variables)} seq_len={config.seq_len} batch_size={config.batch_size} "
          f"encoding_batch_size={min(config.batch_size, config.encoding_batch_size or config.batch_size)}")
    print(f"external_train_boundary={train_length} total_rows={len(frame)}")
    print(f"train_segments={train_segments} continuity={train_audit} frozen_expected_interval_ns={expected_interval}")
    print(f"test_segments={test_segments} continuity={test_audit}")
    print(f"internal_purged_ranges={split} usable_windows={split_counts} test_usable_windows={test_windows}")
    if split_counts["validation"] <= 1 or split_counts["calibration"] <= 1:
        print("WARNING: validation/calibration windows are too few for robust scale or high-quantile threshold estimation")
    print("plan=stage1 -> frozen feature cache -> stage2 -> validation/calibration -> valid-window test")
    print("training/cache not started")
    return {
        "task": task, "source": source, "train_length": train_length,
        "total_rows": len(frame), "feature_names": tuple(map(str, actual_variables)),
        "expected_interval_ns": expected_interval, "train_segments": train_segments,
        "test_segments": test_segments, "split_window_counts": split_counts,
        "test_window_count": test_windows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-source", choices=("large_detect", "external_detect"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--encoding-batch-size", type=int, default=0)
    args = parser.parse_args()
    preflight(args.task, args.data_source, args.batch_size, args.seq_len, args.encoding_batch_size)


if __name__ == "__main__":
    main()
