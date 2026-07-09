#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource
from ts_benchmark.utils.data_processing import split_before


DEFAULT_DATASETS = [
    "Genesis.csv",
    "Weather.csv",
    "Energy.csv",
    "SKAB.csv",
    "MSDS.csv",
    "Daphnet.csv",
    "GECCO.csv",
    "ExathlonSmall.csv",
    "Metro.csv",
]

PICKLE_PROTOCOL = 4  # InterFusion runs under Python 3.6, which cannot read protocol 5.


def _export_one(series_name: str, output_dir: Path) -> None:
    source = LocalAnomalyDetectDataSource()
    source.load_series_list([series_name])
    DataPool().set_pool(source.dataset)
    pool = DataPool().get_pool()
    data = pool.get_series(series_name).reset_index(drop=True)
    train_len = int(pool.get_series_meta_info(series_name)["train_lens"].item())
    train, test = split_before(data, train_len)

    train_values = train.loc[:, train.columns != "label"].to_numpy(dtype=np.float32, copy=True)
    test_values = test.loc[:, test.columns != "label"].to_numpy(dtype=np.float32, copy=True)
    test_label = test.loc[:, ["label"]].to_numpy(dtype=np.float32, copy=True).reshape(-1)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{series_name}_train.pkl", "wb") as f:
        pickle.dump(train_values, f, protocol=PICKLE_PROTOCOL)
    with open(output_dir / f"{series_name}_test.pkl", "wb") as f:
        pickle.dump(test_values, f, protocol=PICKLE_PROTOCOL)
    with open(output_dir / f"{series_name}_test_label.pkl", "wb") as f:
        pickle.dump(test_label, f, protocol=PICKLE_PROTOCOL)

    print(f"{series_name}: train={train_values.shape} test={test_values.shape} label={test_label.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
    )
    parser.add_argument(
        "--output-dir",
        default="/media/h3c/users/wangyueyang1/cxy/baseline_repos/InterFusion/data/processed",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for dataset in args.datasets:
        _export_one(dataset, output_dir)


if __name__ == "__main__":
    main()
