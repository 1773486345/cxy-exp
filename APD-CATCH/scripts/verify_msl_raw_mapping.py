#!/usr/bin/env python
"""Verify the 27-channel MSL raw mapping without changing the benchmark files."""
import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path
import tempfile
import sys
import concurrent.futures

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_SHA256 = "5fe9f96f80e632f01640f3a59f65e295ec300e084b6700d571120a22b9074a5a"
HF_BASE = "https://huggingface.co/datasets/appleparan/telemanom/resolve/main/data/data"


def _sha256_array(array):
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True).encode())
    flat = np.asarray(array).reshape(-1)
    for start in range(0, flat.size, 65536):
        digest.update(np.ascontiguousarray(flat[start:start + 65536]).tobytes())
    return digest.hexdigest()


def _load_ids():
    url = "https://huggingface.co/datasets/appleparan/telemanom/resolve/main/labeled_anomalies.csv"
    frame = pd.read_csv(io.BytesIO(urllib.request.urlopen(url, timeout=60).read()))
    ids = sorted(frame.loc[frame["spacecraft"] == "MSL", "chan_id"].astype(str).tolist())
    if len(ids) != 27:
        raise ValueError("expected 27 MSL channel records, found {}".format(len(ids)))
    return ids


def _load_array(split, channel):
    url = "{}/{}/{}.npy".format(HF_BASE, split, channel)
    payload = urllib.request.urlopen(url, timeout=120).read()
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != 55 or not np.isfinite(array).all():
        raise ValueError("{} {} has invalid shape or non-finite values".format(split, channel))
    return array


def verify(local_csv=None, output=None):
    local_csv = Path(local_csv or ROOT / "dataset/anomaly_detect/data/MSL.csv").resolve()
    ids = _load_ids()
    requests = [(split, channel) for split in ("train", "test") for channel in ids]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        fetched = list(pool.map(lambda item: (item[0], item[1], _load_array(*item)), requests))
    train = [array for split, channel, array in fetched if split == "train"]
    test = [array for split, channel, array in fetched if split == "test"]
    values = np.concatenate(train + test, axis=0).astype(np.float32, copy=False)
    digest = _sha256_array(values)
    local = pd.read_csv(local_csv)
    # The benchmark long table is column-major by variable; use its loader
    # convention rather than comparing the CSV row encoding directly.
    from scripts.mnrf_common_data import load_task_frame
    frame, label_column = load_task_frame(local_csv)
    from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
    local_values, names = MNRFLiteV2._features_from_frame(frame)
    local_values = np.asarray(local_values, dtype=np.float32)
    if local_values.shape != values.shape:
        raise ValueError("local MSL shape {} differs from raw {}".format(local_values.shape, values.shape))
    difference = np.asarray(local_values, dtype=np.float64) - values.astype(np.float64)
    exact = bool(np.array_equal(local_values, values))
    result = {
        "schema": "msl-raw-mapping-v1", "source": "Telemanom Kaggle-compatible public mirror",
        "kaggle_dataset": "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
        "mirror": "appleparan/telemanom on Hugging Face",
        "kaggle_cli_available": True, "kaggle_credentials_available": False,
        "channels": ids, "channel_order": "lexicographic channel-name order",
        "channel_row_counts": {channel: {"train": int(train[index].shape[0]), "test": int(test[index].shape[0])}
                               for index, channel in enumerate(ids)},
        "train_rows": int(sum(x.shape[0] for x in train)),
        "test_rows": int(sum(x.shape[0] for x in test)),
        "columns": 55, "dtype": "float32", "raw_feature_sha256": digest,
        "expected_feature_sha256": EXPECTED_SHA256, "expected_sha256_match": digest == EXPECTED_SHA256,
        "local_csv": str(local_csv), "local_csv_sha256": hashlib.sha256(local_csv.read_bytes()).hexdigest(),
        "local_feature_names": list(map(str, names)), "local_shape": list(local_values.shape),
        "local_matches_raw_float32": exact,
        "max_abs_difference": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "non_equal_values": int(np.count_nonzero(difference)),
        "boundary_protocol": "concatenate sorted 27 train arrays, then sorted 27 test arrays; no short sequence dropped",
        "scoring_roles": {"col_0": "telemetry target candidate", "col_1_to_54": "command/context conditions; retain as inputs"},
        "binary_condition_normalization": "not applied in this audit; source arrays and frozen CSV remain unchanged",
    }
    if output:
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-csv", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.local_csv or None, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
