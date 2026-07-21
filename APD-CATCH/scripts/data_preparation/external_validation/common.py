"""Shared, non-model utilities for the fixed external-validation datasets.

The prepared files intentionally use a compact wide format.  They are read only by
``LocalExternalAnomalyDetectDataSource`` and never alter the legacy anomaly-data
loader or any frozen baseline implementation.
"""

from __future__ import annotations

import hashlib
import json
import csv
import shutil
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import fcntl
except ImportError:  # pragma: no cover - the supported execution host is Linux.
    fcntl = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT_ROOT / "dataset" / "external_validation" / "raw"
PREPARED_ROOT = PROJECT_ROOT / "dataset" / "external_validation"
DATA_ROOT = PREPARED_ROOT / "data"
METADATA_PATH = PREPARED_ROOT / "EXTERNAL_DETECT_META.csv"
RESULT_ROOT = PROJECT_ROOT / "result" / "external_decomposition_validation"
REGISTRY_PATH = RESULT_ROOT / "external_dataset_registry.csv"

TASK_ORDER = [
    "HAI20_07",
    "BATADAL",
    "MetroPT3",
    *[f"MTSB_OPPORTUNITY_{index:02d}" for index in range(1, 14)],
    "MTSB_OCCUPANCY_01",
    "MTSB_OCCUPANCY_02",
    "MTSB_METRO",
    "MTSB_SWAN_SF",
]

PAPER_DATASET = {
    "HAI20_07": "HAI20_07",
    "BATADAL": "BATADAL",
    "MetroPT3": "MetroPT3",
    **{f"MTSB_OPPORTUNITY_{index:02d}": "OPPORTUNITY" for index in range(1, 14)},
    "MTSB_OCCUPANCY_01": "Occupancy",
    "MTSB_OCCUPANCY_02": "Occupancy",
    "MTSB_METRO": "Metro",
    "MTSB_SWAN_SF": "SWAN-SF",
}

METRO_FAULT_INTERVALS = [
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00"),
]
BATADAL_ATTACK_INTERVALS = [
    ("16/01/2017 09", "19/01/2017 06"),
    ("30/01/2017 08", "02/02/2017 00"),
    ("09/02/2017 03", "10/02/2017 09"),
    ("12/02/2017 01", "13/02/2017 07"),
    ("24/02/2017 05", "28/02/2017 08"),
    ("10/03/2017 14", "13/03/2017 21"),
    ("25/03/2017 20", "27/03/2017 01"),
]

MTSBENCH_PAIRS = [
    ("OPPORTUNITY", "OPPORTUNITY_S1-ADL2"),
    ("OPPORTUNITY", "OPPORTUNITY_S1-ADL3"),
    ("OPPORTUNITY", "OPPORTUNITY_S1-ADL4"),
    ("OPPORTUNITY", "OPPORTUNITY_S1-ADL5"),
    ("OPPORTUNITY", "OPPORTUNITY_S2-ADL1"),
    ("OPPORTUNITY", "OPPORTUNITY_S2-ADL2"),
    ("OPPORTUNITY", "OPPORTUNITY_S3-ADL3"),
    ("OPPORTUNITY", "OPPORTUNITY_S3-ADL4"),
    ("OPPORTUNITY", "OPPORTUNITY_S3-ADL5"),
    ("OPPORTUNITY", "OPPORTUNITY_S4-ADL2"),
    ("OPPORTUNITY", "OPPORTUNITY_S4-ADL3"),
    ("OPPORTUNITY", "OPPORTUNITY_S4-ADL4"),
    ("OPPORTUNITY", "OPPORTUNITY_S4-ADL5"),
    ("room-occupancy", "room-occupancy"),
    ("room-occupancy", "room-occupancy_1"),
    ("metro", "metro_traffic-volume"),
    ("swan", "swan_sf"),
]
MTSBENCH_RELATIVE_PATHS = tuple(
    f"{directory}/{stem}_{split}.csv"
    for directory, stem in MTSBENCH_PAIRS
    for split in ("train", "test")
)
MTSBENCH_MISSING_PATHS = (
    "room-occupancy/room-occupancy_test.csv",
    "room-occupancy/room-occupancy_1_train.csv",
    "room-occupancy/room-occupancy_1_test.csv",
    "metro/metro_traffic-volume_train.csv",
    "metro/metro_traffic-volume_test.csv",
    "swan/swan_sf_train.csv",
    "swan/swan_sf_test.csv",
)


def ensure_directories() -> None:
    for path in (RAW_ROOT, DATA_ROOT, RESULT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


@contextmanager
def download_session_lock():
    """Fail fast when another external-source downloader owns the raw directory."""
    ensure_directories()
    lock_path = RAW_ROOT / ".download.lock"
    with lock_path.open("a+") as handle:
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "another external-validation downloader is already running; "
                    "wait for it to finish, then rerun this command"
                ) from error
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def download(url: str, destination: Path, force: bool = False) -> Path:
    """Download one official file with resumable, terminal-visible progress."""
    if destination.exists() and not force:
        print(f"[cached] {destination.name} ({destination.stat().st_size / 2**20:.1f} MiB)", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    offset = temporary.stat().st_size if temporary.exists() and not force else 0
    headers = {"User-Agent": "APD-CATCH-external-validation/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        append = offset and response.status == 206
        mode = "ab" if append else "wb"
        if not append:
            offset = 0
        remaining = response.headers.get("Content-Length")
        total = offset + int(remaining) if remaining and remaining.isdigit() else None
        action = "resuming" if offset else "downloading"
        total_text = f"/{total / 2**20:.1f} MiB" if total else ""
        print(f"[{action}] {destination.name}: {offset / 2**20:.1f} MiB{total_text}", flush=True)
        downloaded = offset
        last_report_bytes = downloaded
        last_report_time = time.monotonic()
        with temporary.open(mode) as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                now = time.monotonic()
                if downloaded - last_report_bytes >= 8 * 2**20 or now - last_report_time >= 5:
                    percent = f" ({100.0 * downloaded / total:.1f}%)" if total else ""
                    print(f"  {destination.name}: {downloaded / 2**20:.1f} MiB{total_text}{percent}", flush=True)
                    last_report_bytes = downloaded
                    last_report_time = now
    if temporary.exists():
        temporary.replace(destination)
    elif destination.exists():
        print(f"[completed by another process] {destination.name}: reusing verified target", flush=True)
    else:
        raise FileNotFoundError(f"download temporary file disappeared: {temporary}")
    print(f"[complete] {destination.name}: {destination.stat().st_size / 2**20:.1f} MiB sha256={sha256(destination)}", flush=True)
    return destination


def mbench_url(relative_path: str) -> str:
    return "https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main/" + relative_path


def validate_mtsbench_source_dir(source_root: Path) -> list[dict[str, str]]:
    """Validate a downloaded relay artifact before any file enters APD-CATCH."""
    source_root = source_root.resolve()
    manifest_path = source_root / "mtsbench_missing_sha256.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing artifact checksum manifest: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fields = {"repo_relative_path", "bytes", "sha256"}
    if not rows or set(rows[0]) != fields:
        raise ValueError("artifact manifest must contain repo_relative_path, bytes, sha256")
    paths = [row["repo_relative_path"] for row in rows]
    if tuple(paths) != MTSBENCH_MISSING_PATHS:
        raise ValueError("artifact manifest does not contain exactly the frozen seven missing paths")
    validated = []
    for row in rows:
        relative = Path(row["repo_relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix().endswith("_val.csv"):
            raise ValueError(f"invalid artifact repository path: {relative}")
        source_path = source_root / relative
        if not source_path.is_file():
            raise FileNotFoundError(f"artifact source file is absent: {relative}")
        byte_count = int(row["bytes"])
        if source_path.stat().st_size != byte_count:
            raise ValueError(f"artifact byte count mismatch: {relative}")
        actual_sha256 = sha256(source_path)
        if actual_sha256 != row["sha256"]:
            raise ValueError(f"artifact SHA-256 mismatch: {relative}")
        validated.append({"repo_relative_path": relative.as_posix(), "bytes": str(byte_count), "sha256": actual_sha256})
    return validated


def import_mtsbench_source_dir(source_root: Path) -> int:
    """Copy only checksum-validated relay files into the existing raw source tree."""
    rows = validate_mtsbench_source_dir(source_root)
    destination_root = RAW_ROOT / "mTSBench"
    copied = 0
    for row in rows:
        relative = Path(row["repo_relative_path"])
        source_path = source_root / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) == row["sha256"]:
            continue
        shutil.copy2(source_path, destination)
        if sha256(destination) != row["sha256"]:
            raise RuntimeError(f"local copy SHA-256 mismatch: {relative}")
        copied += 1
    absent = [path for path in MTSBENCH_RELATIVE_PATHS if not (destination_root / path).is_file()]
    if absent:
        raise RuntimeError(f"mTSBench raw tree is still incomplete after import: {absent}")
    return copied


def download_all_sources(force: bool = False) -> dict[str, list[Path]]:
    """Fetch exactly the four fixed official sources and no benchmark outputs."""
    with download_session_lock():
        source_files: dict[str, list[Path]] = {}
        hai_dir = RAW_ROOT / "hai-20.07"
        source_files["HAI20_07"] = [
            download(
                "https://raw.githubusercontent.com/icsdataset/hai/master/hai-20.07/" + filename,
                hai_dir / filename,
                force,
            )
            for filename in ("train1.csv.gz", "train2.csv.gz", "test1.csv.gz", "test2.csv.gz")
        ]
        batadal_dir = RAW_ROOT / "batadal"
        source_files["BATADAL"] = [
            download("https://www.batadal.net/data/BATADAL_dataset03.csv", batadal_dir / "BATADAL_dataset03.csv", force),
            download("https://www.batadal.net/data/BATADAL_test_dataset.zip", batadal_dir / "BATADAL_test_dataset.zip", force),
            download("https://www.batadal.net/images/Attacks_TestDataset.png", batadal_dir / "Attacks_TestDataset.png", force),
        ]
        metro_dir = RAW_ROOT / "metropt3"
        source_files["MetroPT3"] = [
            download(
                "https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip",
                metro_dir / "metropt+3+dataset.zip",
                force,
            )
        ]
        mbench_files: list[Path] = []
        for directory, stem in MTSBENCH_PAIRS:
            for split in ("train", "test"):
                relative = f"{directory}/{stem}_{split}.csv"
                mbench_files.append(download(mbench_url(relative), RAW_ROOT / "mTSBench" / relative, force))
        source_files["mTSBench"] = mbench_files
        return source_files


def _timestamp(frame: pd.DataFrame, column: str, *, fmt: str | None = None) -> pd.Series:
    values = pd.to_datetime(frame[column], format=fmt, errors="raise")
    if values.isna().any():
        raise ValueError(f"timestamp column {column!r} contains missing values")
    return values


def _numeric_frame(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int]:
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    non_numeric = frame.loc[:, columns].notna() & numeric.isna()
    if bool(non_numeric.to_numpy().any()):
        offenders = list(non_numeric.columns[non_numeric.any()])
        raise ValueError(f"unparseable numeric feature values in columns {offenders}")
    infinite = np.isinf(numeric.to_numpy(dtype=np.float64, copy=False))
    invalid_count = int(numeric.isna().sum().sum()) + int(infinite.sum())
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric, invalid_count


def _fill_missing(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    missing_before = int(train.isna().sum().sum() + test.isna().sum().sum())
    train = train.ffill()
    medians = train.median(axis=0, skipna=True).fillna(0.0)
    train = train.fillna(medians)
    test = test.ffill().fillna(medians)
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("fixed missing-value policy left NaN values")
    return train, test, missing_before


def _binary(values: pd.Series | np.ndarray, name: str) -> np.ndarray:
    values = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite labels")
    return (values > 0).astype(np.int8, copy=False)


def _sort(timestamp: pd.Series, frame: pd.DataFrame, labels: np.ndarray) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    order = np.argsort(timestamp.to_numpy(), kind="stable")
    return timestamp.iloc[order].reset_index(drop=True), frame.iloc[order].reset_index(drop=True), labels[order]


def _upsert_csv(path: Path, row: dict, key: str) -> None:
    if path.exists():
        current = pd.read_csv(path)
        current = current[current[key] != row[key]]
        output = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    else:
        output = pd.DataFrame([row])
    output.sort_values(key, inplace=True)
    output.to_csv(path, index=False)


def write_prepared_task(
    task: str,
    timestamps_train: pd.Series,
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    timestamps_test: pd.Series,
    test_features: pd.DataFrame,
    test_labels: np.ndarray,
    source_files: Iterable[Path],
    source_url: str,
    label_policy: str,
    extra: dict | None = None,
) -> dict:
    """Apply the fixed cleaning policy, write one compact task, and register it."""
    if task not in TASK_ORDER:
        raise ValueError(f"unexpected fixed external task {task}")
    if list(train_features.columns) != list(test_features.columns):
        raise ValueError(f"{task}: train/test feature columns differ")
    if train_features.shape[1] < 2:
        raise ValueError(f"{task}: fewer than two feature columns")
    if len(test_labels) != len(test_features) or len(train_labels) != len(train_features):
        raise ValueError(f"{task}: labels do not align with features")
    if not ({0, 1}.issubset(set(np.unique(test_labels).tolist()))):
        raise ValueError(f"{task}: test labels must contain normal and anomaly points")

    train_features, test_features, missing_count = _fill_missing(train_features, test_features)
    timestamps_train, train_features, train_labels = _sort(timestamps_train, train_features, train_labels)
    timestamps_test, test_features, test_labels = _sort(timestamps_test, test_features, test_labels)
    if not timestamps_train.is_monotonic_increasing or not timestamps_test.is_monotonic_increasing:
        raise ValueError(f"{task}: timestamp stable sorting failed")

    train_values = np.ascontiguousarray(train_features.to_numpy(dtype=np.float32, copy=True))
    test_values = np.ascontiguousarray(test_features.to_numpy(dtype=np.float32, copy=True))
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        raise ValueError(f"{task}: prepared values are not finite")
    feature_names = list(train_features.columns)
    combined = pd.concat(
        [
            pd.DataFrame(train_values, columns=feature_names),
            pd.DataFrame(test_values, columns=feature_names),
        ],
        ignore_index=True,
    )
    combined.insert(0, "timestamp", pd.concat([timestamps_train, timestamps_test], ignore_index=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%f"))
    combined["label"] = np.concatenate([train_labels, test_labels]).astype(np.int8, copy=False)
    ensure_directories()
    output_path = DATA_ROOT / f"{task}.csv"
    combined.to_csv(output_path, index=False)

    metadata_row = {
        "file_name": output_path.name,
        "freq": "external",
        "if_univariate": False,
        "size": "external",
        "length": len(combined),
        "trend": "",
        "seasonal": "",
        "stationary": "",
        "transition": "",
        "shifting": "",
        "correlation": "",
        "train_lens": len(train_values),
        "test_lens": len(test_values),
        "paper_dataset": PAPER_DATASET[task],
    }
    _upsert_csv(METADATA_PATH, metadata_row, "file_name")
    files = list(source_files)
    registry_row = {
        "task": task,
        "paper_dataset": PAPER_DATASET[task],
        "prepared_file": str(output_path.relative_to(PROJECT_ROOT)),
        "prepared_sha256": sha256(output_path),
        "source_url": source_url,
        "source_files": json.dumps([str(path.relative_to(RAW_ROOT)) for path in files]),
        "source_sha256": json.dumps({str(path.relative_to(RAW_ROOT)): sha256(path) for path in files}),
        "train_length": len(train_values),
        "test_length": len(test_values),
        "channel_count": len(feature_names),
        "train_anomaly_count": int(train_labels.sum()),
        "train_anomaly_ratio": float(train_labels.mean()),
        "test_anomaly_count": int(test_labels.sum()),
        "test_anomaly_ratio": float(test_labels.mean()),
        "train_timestamp_monotonic": True,
        "test_timestamp_monotonic": True,
        "feature_columns": json.dumps(feature_names),
        "missing_value_policy": "forward-fill per split; leading values use training-column median; train medians also fill test",
        "label_policy": label_policy,
        "constant_column_count": int(np.sum(np.nanstd(train_values, axis=0) == 0.0)),
        "missing_value_count_before_fill": missing_count,
        "prepared_at": utc_now(),
        **(extra or {}),
    }
    _upsert_csv(REGISTRY_PATH, registry_row, "task")
    return registry_row


def prepare_hai() -> dict:
    directory = RAW_ROOT / "hai-20.07"
    train_paths = [directory / name for name in ("train1.csv.gz", "train2.csv.gz")]
    test_paths = [directory / name for name in ("test1.csv.gz", "test2.csv.gz")]
    if not all(path.exists() for path in train_paths + test_paths):
        raise FileNotFoundError("HAI 20.07 source files missing; run download_external_validation_data.py")
    train = pd.concat([pd.read_csv(path, sep=";") for path in train_paths], ignore_index=True)
    test = pd.concat([pd.read_csv(path, sep=";") for path in test_paths], ignore_index=True)
    timestamp_column = "time"
    label_column = "attack"
    forbidden = [column for column in train.columns if column.lower().startswith("attack")]
    features = [column for column in train.columns if column not in forbidden + [timestamp_column]]
    if set(features) != set(test.columns).difference(forbidden + [timestamp_column]):
        raise ValueError("HAI 20.07 train/test feature names differ")
    train_features, train_invalid = _numeric_frame(train, features)
    test_features, test_invalid = _numeric_frame(test, features)
    return write_prepared_task(
        "HAI20_07",
        _timestamp(train, timestamp_column),
        train_features,
        _binary(train[label_column], "HAI train attack"),
        _timestamp(test, timestamp_column),
        test_features,
        _binary(test[label_column], "HAI test attack"),
        train_paths + test_paths,
        "https://github.com/icsdataset/hai/tree/master/hai-20.07",
        "official HAI 20.07 attack column thresholded to normal=0/anomaly=1; attack subtype columns excluded from features",
        {"timestamp_column": timestamp_column, "dropped_label_columns": json.dumps(forbidden), "invalid_value_count_before_fill": train_invalid + test_invalid},
    )


def prepare_batadal() -> dict:
    directory = RAW_ROOT / "batadal"
    train_path = directory / "BATADAL_dataset03.csv"
    test_zip = directory / "BATADAL_test_dataset.zip"
    image_path = directory / "Attacks_TestDataset.png"
    if not all(path.exists() for path in (train_path, test_zip, image_path)):
        raise FileNotFoundError("BATADAL source files missing; run download_external_validation_data.py")
    with zipfile.ZipFile(test_zip) as archive:
        member = "BATADAL_test_dataset.csv"
        with archive.open(member) as handle:
            test = pd.read_csv(handle)
    train = pd.read_csv(train_path)
    timestamp_column = "DATETIME"
    label_column = "ATT_FLAG"
    features = [column for column in train.columns if column not in {timestamp_column, label_column}]
    if set(features) != set(test.columns).difference({timestamp_column, label_column}):
        raise ValueError("BATADAL train/test feature names differ")
    train_features, train_invalid = _numeric_frame(train, features)
    test_features, test_invalid = _numeric_frame(test, features)
    train_time = _timestamp(train, timestamp_column, fmt="%d/%m/%y %H")
    test_time = _timestamp(test, timestamp_column, fmt="%d/%m/%y %H")
    labels = np.zeros(len(test), dtype=np.int8)
    interval_counts = []
    for start_text, end_text in BATADAL_ATTACK_INTERVALS:
        start = pd.to_datetime(start_text, format="%d/%m/%Y %H")
        end = pd.to_datetime(end_text, format="%d/%m/%Y %H")
        selected = (test_time >= start) & (test_time <= end)
        labels[selected.to_numpy()] = 1
        interval_counts.append(int(selected.sum()))
    return write_prepared_task(
        "BATADAL",
        train_time,
        train_features,
        np.zeros(len(train), dtype=np.int8),
        test_time,
        test_features,
        labels,
        [train_path, test_zip, image_path],
        "https://www.batadal.net/data.html",
        "official List of attacks in Test Dataset; both timestamp endpoints are inclusive at the hourly sampling resolution",
        {
            "timestamp_column": timestamp_column,
            "official_attack_interval_count": len(BATADAL_ATTACK_INTERVALS),
            "official_attack_interval_counts": json.dumps(interval_counts),
            "train_att_flag_present": label_column in train.columns,
            "test_att_flag_present": label_column in test.columns,
            "invalid_value_count_before_fill": train_invalid + test_invalid,
        },
    )


def prepare_metropt3() -> dict:
    directory = RAW_ROOT / "metropt3"
    archive_path = directory / "metropt+3+dataset.zip"
    if not archive_path.exists():
        raise FileNotFoundError("MetroPT-3 archive missing; run download_external_validation_data.py")
    with zipfile.ZipFile(archive_path) as archive:
        member = "MetroPT3(AirCompressor).csv"
        with archive.open(member) as handle:
            source = pd.read_csv(handle)
    timestamp_column = "timestamp"
    timestamps = _timestamp(source, timestamp_column)
    source = source.assign(**{timestamp_column: timestamps}).sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    timestamps = source[timestamp_column]
    months = timestamps.dt.to_period("M")
    first_month = months.iloc[0]
    month_values = timestamps.loc[months == first_month]
    month_end = (first_month + 1).start_time - pd.Timedelta(seconds=1)
    if month_values.iloc[0].day != 1 or month_values.iloc[-1] < month_end:
        raise ValueError(f"MetroPT-3 first observed month {first_month} is not a complete calendar month")
    train_mask = months == first_month
    test_mask = ~train_mask
    dropped = [column for column in source.columns if column.lower() in {"index", "unnamed: 0", timestamp_column}]
    features = [column for column in source.columns if column not in dropped]
    train_raw = source.loc[train_mask].reset_index(drop=True)
    test_raw = source.loc[test_mask].reset_index(drop=True)
    train_features, train_invalid = _numeric_frame(train_raw, features)
    test_features, test_invalid = _numeric_frame(test_raw, features)
    labels = np.zeros(len(test_raw), dtype=np.int8)
    interval_counts = []
    test_time = test_raw[timestamp_column]
    for start_text, end_text in METRO_FAULT_INTERVALS:
        start, end = pd.to_datetime(start_text), pd.to_datetime(end_text)
        selected = (test_time >= start) & (test_time <= end)
        labels[selected.to_numpy()] = 1
        interval_counts.append(int(selected.sum()))
    return write_prepared_task(
        "MetroPT3",
        train_raw[timestamp_column],
        train_features,
        np.zeros(len(train_raw), dtype=np.int8),
        test_time,
        test_features,
        labels,
        [archive_path],
        "https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset",
        "four UCI fault intervals with both supplied timestamp endpoints inclusive",
        {
            "timestamp_column": timestamp_column,
            "first_complete_calendar_month": str(first_month),
            "dropped_id_timestamp_columns": json.dumps(dropped),
            "official_fault_interval_count": len(METRO_FAULT_INTERVALS),
            "official_fault_interval_counts": json.dumps(interval_counts),
            "invalid_value_count_before_fill": train_invalid + test_invalid,
        },
    )


def _mtsbench_frame(path: Path) -> tuple[pd.Series, pd.DataFrame, np.ndarray, str, str]:
    frame = pd.read_csv(path)
    labels = [column for column in frame.columns if column.lower() == "is_anomaly"]
    if len(labels) != 1:
        raise ValueError(f"{path.name}: expected exactly one is_anomaly field")
    label_column = labels[0]
    timestamp_candidates = [
        column for column in frame.columns if column.lower() in {"timestamp", "time", "date", "datetime", "index"}
    ]
    if len(timestamp_candidates) != 1:
        raise ValueError(f"{path.name}: cannot identify timestamp/index field by name")
    timestamp_column = timestamp_candidates[0]
    features = [column for column in frame.columns if column not in {timestamp_column, label_column}]
    values, _ = _numeric_frame(frame, features)
    return _timestamp(frame, timestamp_column), values, _binary(frame[label_column], f"{path.name} is_anomaly"), timestamp_column, label_column


def prepare_mtsbench() -> list[dict]:
    results: list[dict] = []
    for index, (directory, stem) in enumerate(MTSBENCH_PAIRS):
        if directory == "OPPORTUNITY":
            task = f"MTSB_OPPORTUNITY_{index + 1:02d}"
        elif directory == "room-occupancy":
            task = f"MTSB_OCCUPANCY_{index - 12:02d}"
        elif directory == "metro":
            task = "MTSB_METRO"
        else:
            task = "MTSB_SWAN_SF"
        base = RAW_ROOT / "mTSBench" / directory
        train_path, test_path = base / f"{stem}_train.csv", base / f"{stem}_test.csv"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"mTSBench source pair missing for {task}; run download_external_validation_data.py")
        train_time, train_features, train_labels, train_time_column, train_label_column = _mtsbench_frame(train_path)
        test_time, test_features, test_labels, test_time_column, test_label_column = _mtsbench_frame(test_path)
        results.append(
            write_prepared_task(
                task,
                train_time,
                train_features,
                train_labels,
                test_time,
                test_features,
                test_labels,
                [train_path, test_path],
                "https://huggingface.co/datasets/PLAN-Lab/mTSBench",
                "provided is_anomaly field, thresholded to normal=0/anomaly=1; provided train/test split only, validation file excluded",
                {
                    "source_pair": stem,
                    "train_timestamp_column": train_time_column,
                    "test_timestamp_column": test_time_column,
                    "train_label_column": train_label_column,
                    "test_label_column": test_label_column,
                },
            )
        )
    if len(results) != 17:
        raise AssertionError("mTSBench fixed task count is not 17")
    return results


def prepared_task_path(task: str) -> Path:
    return DATA_ROOT / f"{task}.csv"


def load_prepared_task(task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    metadata = pd.read_csv(METADATA_PATH).set_index("file_name")
    filename = f"{task}.csv"
    if filename not in metadata.index:
        raise KeyError(f"task {task} is not registered")
    row = metadata.loc[filename]
    frame = pd.read_csv(prepared_task_path(task))
    features = [column for column in frame.columns if column not in {"timestamp", "label"}]
    values = np.ascontiguousarray(frame.loc[:, features].to_numpy(dtype=np.float32, copy=True))
    labels = frame["label"].to_numpy(dtype=np.int8, copy=True)
    train_length = int(row["train_lens"])
    return values[:train_length], values[train_length:], labels[:train_length], labels[train_length:], frame
