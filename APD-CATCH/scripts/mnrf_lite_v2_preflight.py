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
from ts_benchmark.baselines.mnrf_lite_v2.windows import continuous_segments, intersect_segments
from ts_benchmark.data.utils import read_data

def preflight(task: str, source: str, batch_size: int = 64, seq_len: int = 128, encoding_batch_size: int = 0):
    if source == "large_detect":
        folder = ROOT / "dataset/anomaly_detect"
        meta_path = folder / "DETECT_META.csv"
    elif source == "external_detect":
        folder = ROOT / "dataset/external_validation"
        meta_path = folder / "EXTERNAL_DETECT_META.csv"
    else:
        raise ValueError("source must be large_detect or external_detect")
    data_path = folder / "data" / f"{task}.csv"
    if not meta_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"metadata or task CSV missing: {meta_path}, {data_path}")
    metadata = pd.read_csv(meta_path)
    row = metadata.loc[metadata.file_name == f"{task}.csv"]
    if len(row) != 1:
        raise ValueError(f"expected exactly one metadata row for {task}.csv; found {len(row)}")
    record = row.iloc[0]
    if bool(record.get("if_univariate", False)):
        raise ValueError("task is marked univariate; MNRF-Lite V2 requires D >= 2")
    train_length = int(record["train_lens"])
    if source == "large_detect":
        columns = pd.read_csv(data_path, usecols=["cols"])["cols"].drop_duplicates().astype(str).tolist()
        variables = [name for name in columns if name.lower() not in {"label", "entity_id", "session_id", "record_id", "scenario_id"}]
    else:
        columns = pd.read_csv(data_path, nrows=0).columns.astype(str).tolist()
        variables = [name for name in columns if name.lower() not in {"label", "timestamp", "date", "time", "datetime", "index", "entity_id", "session_id", "record_id", "scenario_id"}]
    if len(variables) < 2:
        raise ValueError(f"data resolves to D={len(variables)} features")
    config = MNRFConfig.from_kwargs(batch_size=batch_size, seq_len=seq_len,
                                    encoding_batch_size=encoding_batch_size)
    frame = read_data(str(data_path))
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("MNRF-Lite V2 preflight currently requires a tabular time-series DataFrame")
    feature_frame = frame.loc[:, [column for column in frame.columns if str(column).lower() != "label"]]
    values, actual_variables = MNRFLiteV2._features_from_frame(feature_frame)
    if train_length >= len(frame):
        raise ValueError(f"metadata train_lens={train_length} must leave a nonempty external test interval")
    ranges = purged_ranges(train_length, seq_len)
    if len(actual_variables) != len(variables):
        raise ValueError(f"metadata feature count {len(variables)} differs from loaded count {len(actual_variables)}")
    train_frame = feature_frame.iloc[:train_length]
    train_values, _ = MNRFLiteV2._features_from_frame(train_frame)
    train_segments, train_audit = continuous_segments(MNRFLiteV2._time_index(train_frame), len(train_frame),
                                                      boundary_ids=MNRFLiteV2._boundary_ids(train_frame))
    split = purged_ranges(train_length, seq_len)
    split_counts = {}
    for name in ("train", "validation", "calibration"):
        parts = intersect_segments(train_segments, *split[name])
        split_counts[name] = len(EndWindowDataset(train_values, *split[name], seq_len, parts))
    test_frame = feature_frame.iloc[train_length:]
    test_segments, test_audit = continuous_segments(MNRFLiteV2._time_index(test_frame), len(test_frame),
                                                     boundary_ids=MNRFLiteV2._boundary_ids(test_frame))
    test_windows = len(EndWindowDataset(values, train_length, len(values), seq_len,
                                        [(left + train_length, right + train_length) for left, right in test_segments]))
    if min(split_counts.values()) < 1 or test_windows < 1:
        raise ValueError(f"unsupported after continuous segmentation: split windows={split_counts}, test windows={test_windows}")
    print("MNRF-Lite V2 read-only preflight")
    print(f"task={task}.csv source={source} data={data_path}")
    print(f"variables={len(actual_variables)} seq_len={config.seq_len} batch_size={config.batch_size} "
          f"encoding_batch_size={min(config.batch_size, config.encoding_batch_size or config.batch_size)}")
    print(f"external_train_boundary={train_length} total_rows={len(frame)}")
    print(f"train_segments={train_segments} continuity={train_audit}")
    print(f"test_segments={test_segments} continuity={test_audit}")
    print(f"internal_purged_ranges={ranges} usable_windows={split_counts} test_usable_windows={test_windows}")
    if split_counts["validation"] <= 1 or split_counts["calibration"] <= 1:
        print("WARNING: validation/calibration windows are too few for robust scale or high-quantile threshold estimation")
    print("plan=stage1 -> frozen feature cache -> stage2 -> validation/calibration -> valid-window test")
    print("training/cache not started")


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
