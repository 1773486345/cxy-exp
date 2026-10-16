#!/usr/bin/env python
"""Read-only metadata and path check; does not construct caches or train."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
from ts_benchmark.baselines.mnrf_lite_v2.training import purged_ranges

def preflight(task: str, source: str, batch_size: int = 64, seq_len: int = 128):
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
    ranges = purged_ranges(train_length, seq_len)
    if source == "large_detect":
        columns = pd.read_csv(data_path, usecols=["cols"])["cols"].drop_duplicates().astype(str).tolist()
        variables = [name for name in columns if name.lower() != "label"]
    else:
        columns = pd.read_csv(data_path, nrows=0).columns.astype(str).tolist()
        variables = [name for name in columns if name.lower() not in {"label", "timestamp", "date", "time", "datetime", "index"}]
    if len(variables) < 2:
        raise ValueError(f"data resolves to D={len(variables)} features")
    config = MNRFConfig.from_kwargs(batch_size=batch_size, seq_len=seq_len)
    print("MNRF-Lite V2 read-only preflight")
    print(f"task={task}.csv source={source} data={data_path}")
    print(f"variables={len(variables)} seq_len={config.seq_len} batch_size={config.batch_size}")
    print(f"train_length={train_length} purged_ranges={ranges}")
    print("plan=stage1 -> frozen feature cache -> stage2 -> validation/calibration -> valid-window test")
    print("training/cache not started")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-source", choices=("large_detect", "external_detect"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()
    preflight(args.task, args.data_source, args.batch_size, args.seq_len)


if __name__ == "__main__":
    main()
