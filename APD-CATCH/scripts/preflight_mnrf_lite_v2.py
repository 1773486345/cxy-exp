"""Read the formal loader and calculate endpoints; never construct or fit a model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2, _hash_array
from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
from ts_benchmark.baselines.mnrf_lite_v2.training import purged_ranges
from ts_benchmark.baselines.mnrf_lite_v2.windows import (
    continuous_segments, infer_expected_interval, intersect_segments, window_endpoints, TimeIndex,
)


def preflight(task, data_root, params, expected_test=None, expected_calibration=None, expected_fields=None,
              msl_sequence_mapping=None):
    config = MNRFConfig.from_kwargs(**{k: v for k, v in params.items() if k != "score_readout"})
    source = LocalAnomalyDetectDataSource(str(data_root))
    source.load_series_list([task + ".csv"])
    frame = source.dataset.get_series(task + ".csv")
    selected = source.dataset.metadata.loc[lambda x: x.file_name == task + ".csv"]
    if len(selected) != 1:
        raise ValueError("task must have exactly one metadata registration")
    train = int(selected.iloc[0].train_lens)
    mapping_record = None
    if msl_sequence_mapping:
        from scripts.mnrf_msl_sequence_boundaries import prepare_formal_frame
        if task != "MSL" or config.score_target != "col_0":
            raise ValueError("MSL boundary preflight requires task MSL and score_target col_0")
        frame = prepare_formal_frame(frame, train, Path(data_root) / "data/MSL.csv", msl_sequence_mapping)
        mapping_record = frame.attrs["mnrf_sequence_mapping"]
    elif config.score_target:
        raise ValueError("col_0 scoring requires the verified MSL sequence mapping")
    values, columns = MNRFLiteV2._features_from_frame(frame)
    if expected_fields is not None and list(map(str, columns)) != expected_fields:
        raise ValueError("unexpected formal input field order")
    labels = frame["label"].to_numpy()
    if labels.dtype.kind not in "biuf" or not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("expected exact binary labels")
    ranges = purged_ranges(train, config.seq_len)
    time = MNRFLiteV2._time_index(frame)
    left, right = ranges["train"]
    train_ids = MNRFLiteV2._boundary_ids(frame.iloc[left:right])
    interval = infer_expected_interval(TimeIndex(time.values_ns[left:right], time.kind, time.source_timezone),
                                       right-left, train_ids)
    segments, _ = continuous_segments(time, len(frame), expected_interval=interval,
                                     tolerance=.05, boundary_ids=MNRFLiteV2._boundary_ids(frame))
    counts = {name: len(window_endpoints(intersect_segments(segments, *ranges[name]), config.seq_len))
              for name in ("train", "validation", "calibration")}
    test = frame.iloc[train:]
    test_segments, _ = continuous_segments(MNRFLiteV2._time_index(test), len(test), expected_interval=interval,
                                          tolerance=.05, boundary_ids=MNRFLiteV2._boundary_ids(test))
    endpoints = window_endpoints(test_segments, config.seq_len)
    if expected_test is not None and len(endpoints) != expected_test:
        raise ValueError(f"test endpoint count mismatch: {len(endpoints)} != {expected_test}")
    if expected_calibration is not None and counts["calibration"] != expected_calibration:
        raise ValueError(f"calibration endpoint count mismatch: {counts['calibration']} != {expected_calibration}")
    split_bounds = {name: ranges[name] for name in ("train", "validation", "calibration")}
    split_bounds["test"] = (train, len(frame))
    crossing = {}
    short_segments = {}
    ids = MNRFLiteV2._boundary_ids(frame)
    for name, (lo, hi) in split_bounds.items():
        bounded = window_endpoints(intersect_segments(segments, lo, hi), config.seq_len)
        naive = np.arange(lo + config.seq_len - 1, hi, dtype=np.int64)
        crossing[name] = dict(endpoints=len(bounded), excluded_cross_boundary_windows=len(naive)-len(bounded),
                              cross_boundary_windows_remaining=int(np.sum(ids[bounded] != ids[bounded-config.seq_len+1]))
                              if ids is not None else 0)
        short_segments[name] = [[left, right] for left, right in intersect_segments(segments, lo, hi)
                                if right-left < config.seq_len]
    return dict(status="passed-read-only-no-model-created", task=task, data_root=str(Path(data_root).resolve()),
                variable_names=list(map(str, columns)), train_length=train, official_test_rows=len(test),
                seq_len=config.seq_len, seed=config.seed, score_readout=params["score_readout"],
                batch_size=config.batch_size, encoding_batch_size=config.encoding_batch_size,
                split_ranges=ranges, window_counts=counts, test_endpoints=len(endpoints),
                full_input_sha256=_hash_array(values), full_time_sha256=_hash_array(time),
                full_labels_sha256=_hash_array(labels.astype(np.int8)), score_target=config.score_target or "all_variables",
                sequence_mapping=mapping_record, boundary_window_checks=crossing,
                short_segments_retained_without_complete_windows=short_segments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--expected-test", type=int)
    parser.add_argument("--expected-calibration", type=int)
    parser.add_argument("--expected-fields")
    parser.add_argument("--msl-sequence-mapping")
    args = parser.parse_args()
    print(json.dumps(preflight(args.task, args.data_root, json.loads(args.params), args.expected_test,
                              args.expected_calibration, json.loads(args.expected_fields) if args.expected_fields else None,
                              args.msl_sequence_mapping),
                     indent=2))
