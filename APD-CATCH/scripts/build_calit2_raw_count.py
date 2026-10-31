#!/usr/bin/env python
"""Build an isolated, lossless CalIt2 long-table data contract.

The checked-in CalIt2 benchmark file contains the historical modulo-10
projection of the UCI counts.  This tool deliberately writes a new data root
and never edits that file or the shared metadata table.
"""
import argparse
import hashlib
import io
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UCI_URL = "https://archive.ics.uci.edu/static/public/156/calit2+building+people+counts.zip"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_uci_data(raw_bytes):
    rows = pd.read_csv(io.BytesIO(raw_bytes), header=None,
                       names=["flow_id", "date", "time", "count"])
    if rows.empty or rows[["flow_id", "date", "time", "count"]].isna().any().any():
        raise ValueError("UCI CalIt2.data contains missing fields")
    rows["flow_id"] = rows["flow_id"].astype(int)
    rows["count"] = rows["count"].astype(np.int64)
    if set(rows["flow_id"].unique()) != {7, 9}:
        raise ValueError("expected exactly UCI flow ids 7 and 9")
    rows["timestamp"] = pd.to_datetime(rows["date"] + " " + rows["time"],
                                        format="%m/%d/%y %H:%M:%S")
    pivot = rows.pivot(index="timestamp", columns="flow_id", values="count").sort_index()
    if len(pivot) != 5040 or pivot.isna().any().any():
        raise ValueError("UCI CalIt2 data did not form 5040 complete timestamps")
    deltas = pivot.index.to_series().diff().dropna()
    if not (deltas == pd.Timedelta(minutes=30)).all():
        raise ValueError("UCI CalIt2 timestamps are not a continuous 30-minute axis")
    return pivot[[9, 7]].astype(np.int64)


def load_uci(url=UCI_URL):
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        try:
            raw = archive.read("CalIt2.data")
            events = archive.read("CalIt2.events")
        except KeyError as error:
            raise ValueError("UCI archive lacks CalIt2.data/CalIt2.events") from error
    return parse_uci_data(raw), hashlib.sha256(raw).hexdigest(), hashlib.sha256(events).hexdigest()


def build(output_root, old_csv=None, old_meta=None, url=UCI_URL):
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("RawCount output root must be new and empty: {}".format(output_root))
    old_csv = Path(old_csv or ROOT / "dataset/anomaly_detect/data/CalIt2.csv").resolve()
    old_meta = Path(old_meta or ROOT / "dataset/anomaly_detect/DETECT_META.csv").resolve()
    old = pd.read_csv(old_csv)
    if list(old.columns) != ["date", "data", "cols"]:
        raise ValueError("historical CalIt2 CSV does not have the expected long schema")
    labels = old.loc[old["cols"].astype(str) == "label", "data"].to_numpy()
    if len(labels) != 5040 or not np.isin(labels, [0, 1]).all():
        raise ValueError("historical labels are not a binary 5040-point axis")
    pivot, data_sha, events_sha = load_uci(url)
    timestamps = pd.to_datetime(old.loc[old["cols"].astype(str) == "label", "date"])
    if not np.array_equal(timestamps.to_numpy(), pivot.index.to_numpy()):
        raise ValueError("UCI timestamp axis differs from the frozen CalIt2 label axis")
    for name in ("9", "7"):
        historical = old.loc[old["cols"].astype(str) == name, "data"].to_numpy(dtype=np.int64)
        if not np.array_equal(historical, pivot[int(name)].to_numpy() % 10):
            raise ValueError("historical {} channel is not the documented modulo-10 projection".format(name))
    output_root.mkdir(parents=True)
    (output_root / "data").mkdir()
    # Preserve the benchmark loader's long-table ordering exactly: all rows of
    # each variable, then the full label axis.
    output = pd.concat([
        pd.DataFrame({"date": pivot.index.strftime("%Y/%-m/%-d %-H:%M"), "data": pivot[9].to_numpy(), "cols": "9"}),
        pd.DataFrame({"date": pivot.index.strftime("%Y/%-m/%-d %-H:%M"), "data": pivot[7].to_numpy(), "cols": "7"}),
        pd.DataFrame({"date": pivot.index.strftime("%Y/%-m/%-d %-H:%M"), "data": labels.astype(np.int8), "cols": "label"}),
    ], ignore_index=True)
    data_path = output_root / "data" / "CalIt2.csv"
    output.to_csv(data_path, index=False)
    meta = pd.read_csv(old_meta)
    row = meta.loc[meta["file_name"].astype(str) == "CalIt2.csv"]
    if len(row) != 1:
        raise ValueError("shared metadata has no unique CalIt2 row")
    metadata_path = output_root / "DETECT_META.csv"
    row.to_csv(metadata_path, index=False)
    contract = {
        "schema": "calit2-raw-count-v1", "source_url": url,
        "uci_data_sha256": data_sha, "uci_events_sha256": events_sha,
        "historical_csv_sha256": sha256(old_csv), "historical_metadata_sha256": sha256(old_meta),
        "data_path": str(data_path), "metadata_path": str(metadata_path),
        "rows": 5040, "train_length": int(row.iloc[0]["train_lens"]),
        "variables": ["9", "7"], "label_source": "historical benchmark binary label axis",
        "conversion": "UCI integer counts retained; historical CSV equals count modulo 10",
        "max_counts": {"9": int(pivot[9].max()), "7": int(pivot[7].max())},
        "timestamp_rule": "UCI date/time parsed as 30-minute DatetimeIndex",
    }
    (output_root / "RAWCOUNT_PROTOCOL.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--url", default=UCI_URL)
    args = parser.parse_args()
    print(json.dumps(build(args.output_root, url=args.url), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
