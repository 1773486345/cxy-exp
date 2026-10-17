#!/usr/bin/env python
"""Run one independent MNRF frozen-view probe task.

The command performs one task and one head only. It never writes a source
MNRF-Lite V2 run; source directories are read-only inputs.
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
from ts_benchmark.baselines.mnrf_view_probe import MNRFViewProbe
from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource, LocalExternalAnomalyDetectDataSource


def _read_task(task, source):
    if source == "large_detect":
        folder = ROOT / "dataset" / "anomaly_detect"
        source_class = LocalAnomalyDetectDataSource
        metadata_name = "DETECT_META.csv"
    elif source == "external_detect":
        folder = ROOT / "dataset" / "external_validation"
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
    test_labels = frame.iloc[train_length:][label_column].to_numpy()
    return frame, train_length, test_labels


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
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
    args = parser.parse_args()
    frame, train_length, test_labels = _read_task(args.task, args.data_source)
    kwargs = dict(seed=args.seed, seq_len=args.seq_len, batch_size=args.batch_size,
                  stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
                  patience=args.patience,
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
                  cpu_threads=args.cpu_threads)
    probe = MNRFViewProbe(**kwargs)
    probe.fit(frame, train_length, None if args.mode == "stage1" else test_labels)
    print(json.dumps({"status": "complete", "task": args.task, "mode": args.mode,
                      "output_dir": str(probe.output_dir), "selected_views": list(probe.config.selected_views),
                      "selected_dim": probe.config.selected_dim, "layout": probe.config.layout.to_dict()}, sort_keys=True))


if __name__ == "__main__":
    main()
