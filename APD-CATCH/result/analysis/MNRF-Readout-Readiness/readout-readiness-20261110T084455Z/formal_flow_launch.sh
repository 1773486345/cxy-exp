#!/usr/bin/env bash
# Prepared future training command. This batch only executes --dry-run.
# One task per invocation; CPU only; no environment activation or CUDA rewrite.
set -euo pipefail
TASK="${1:?PSM or CalIt2 required}"
ACTION="${2:---dry-run}"
case "$TASK" in PSM|CalIt2) ;; *) exit 2 ;; esac
case "$ACTION" in --dry-run|--execute) ;; *) exit 2 ;; esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
cd "$ROOT"
python - "$TASK" "$ACTION" <<'PY'
import json
import os
import runpy
import sys
from dataclasses import fields, asdict
from pathlib import Path

from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
from scripts.compare_mnrf_frozen_readouts import compare
from scripts.run_mnrf_view_probe import _read_task
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2, _hash_array
from ts_benchmark.baselines.mnrf_view_probe.probe import _chunked_array_hash
from ts_benchmark.baselines.mnrf_lite_v2.training import purged_ranges
from ts_benchmark.baselines.mnrf_lite_v2.windows import continuous_segments, infer_expected_interval, intersect_segments, window_endpoints, TimeIndex
import numpy as np
import torch
from threadpoolctl import threadpool_limits

task, action = sys.argv[1:]
root = Path.cwd()
if task == "CalIt2":
    data_root = root / "result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031/data_root"
    historical = root / "result/score/MNRF-ViewProbe-RawCount/CalIt2/rawcount-full-20261031T20261031T092320Z-1093827"
    output = root / "result/score/MNRF-Lite-V2/CalIt2-RawCount/flow-readout-4c264c9"
else:
    data_root = root / "dataset/anomaly_detect"
    historical = root / "result/score/MNRF-ViewProbe/PSM/full_gmm-full-20261024T115119Z-3976714"
    output = root / "result/score/MNRF-Lite-V2/PSM/flow-readout-4c264c9"
source = json.loads((historical / "probe_manifest.json").read_text())
known = {field.name for field in fields(MNRFConfig)}
config = {key: value for key, value in source["config"].items() if key in known}
config.update(device="cpu", output_dir=str(output), resume_from="", task_name=task+".csv")
config = asdict(MNRFConfig.from_kwargs(**config))
assert config["seed"] == 2021 and config["seq_len"] == 192
assert config["calibration_quantile"] == .995
params = {**config, "score_readout": "flow"}
arguments = ["scripts/run_benchmark.py", "--config-path", "mnrf_lite_v2_unfixed_detect_score.json",
             "--data-set-name", "large_detect", "--data-name-list", task+".csv",
             "--model-name", "mnrf_lite_v2.MNRFLiteV2", "--model-hyper-params", json.dumps(params),
             "--seed", "2021", "--num-workers", "1", "--timeout", "60000",
             "--save-path", str(output.relative_to(root / "result"))]
print(json.dumps(dict(action=action, task=task, data_root=str(data_root), config=params,
                     arguments=arguments, historical_gmm_reference=str(historical)), indent=2))
torch.set_num_threads(1)
with threadpool_limits(limits=1):
    frame, train, _, csv_path, metadata_path, _ = _read_task(task, "large_detect", data_root)
    features = frame.drop(columns="label")
    values, columns = MNRFLiteV2._features_from_frame(features)
    assert list(map(str, columns)) == source["variable_names"]
    assert _chunked_array_hash(values) == source["source_identity"]["feature_data_sha256"]
    ranges = purged_ranges(train, config["seq_len"])
    time = MNRFLiteV2._time_index(features)
    left, right = ranges["train"]
    cadence = infer_expected_interval(TimeIndex(time.values_ns[left:right], time.kind, time.source_timezone), right-left)
    segments, _ = continuous_segments(time, len(frame), expected_interval=cadence, tolerance=.05)
    counts = {name:len(window_endpoints(intersect_segments(segments,*ranges[name]),config["seq_len"]))
              for name in ("train","validation","calibration")}
    test_time = MNRFLiteV2._time_index(features.iloc[train:])
    test_segments, _ = continuous_segments(test_time,len(frame)-train,expected_interval=cadence,tolerance=.05)
    test_endpoints = window_endpoints(test_segments,config["seq_len"])
    if task == "CalIt2":
        assert columns == ["9", "7"]
        assert len(test_endpoints)==2329 and counts["calibration"]==24
    print(json.dumps(dict(preflight="passed-no-model-created",data_csv=str(csv_path),metadata=str(metadata_path),
        variable_names=list(map(str,columns)),official_train_rows=train,official_test_rows=len(frame)-train,
        split_ranges={name:ranges[name] for name in counts},window_counts=counts,
        test_windows=len(test_endpoints),test_first_local=int(test_endpoints[0]),
        full_input_sha256=_hash_array(values),head_seed=config["seed"]),indent=2))
    data_contract = dict(source_run_id=output.name, task=task, train_length=train,
        variable_names=list(map(str,columns)), full_input_sha256=_hash_array(values),
        full_time_sha256=_hash_array(time), full_labels_sha256=_hash_array(frame["label"].to_numpy().astype(np.int8)),
        data_version="CalIt2-RawCount-20261031" if task=="CalIt2" else "PSM-benchmark-original",
        data_root=str(data_root), data_csv=str(csv_path), official_test_rows=len(frame)-train,
        captured="startup, before any training; not reread at completion")
if action == "--dry-run":
    # Only actual loading and coordinate calculations; no model or output run.
    raise SystemExit(0)
if output.exists():
    raise SystemExit("Target already exists: use its saved comparison; no automatic retraining")
import ts_benchmark.data.data_source as data_source
from scripts.validate_mnrf_lite_v2_run import main as validate_run
# Override only this process's source root. Shared DETECT_META is never edited.
data_source.ANOMALY_DETECT_DATASET_PATH = str(data_root)
os.environ["TS_BENCHMARK_PRESERVE_CUDA_VISIBLE_DEVICES"] = "1"
torch.set_num_threads(1)
with threadpool_limits(limits=1):
    frame, train, *_ = _read_task(task, "large_detect", data_root)
    if task == "CalIt2":
        assert frame.drop(columns="label").columns.tolist() == ["9", "7"]
        assert len(frame)-train-191 == 2329
    sys.argv = arguments
    runpy.run_path("scripts/run_benchmark.py", run_name="__main__")
    torch.set_num_threads(1)
    with (output / "readout_data_contract.json").open("x") as handle:
        json.dump(data_contract, handle, indent=2)
        handle.write("\n")
    compare(output, task, frame, train, output.parent / (output.name+"-readout-comparison"))
PY
