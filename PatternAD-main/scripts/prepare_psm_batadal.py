#!/usr/bin/env python3
"""Download and convert the PSM and BATADAL anomaly datasets for ts_benchmark.

The loader accepts both wide and long CSV data.  These datasets are saved in
wide form (``date, feature..., label``) to avoid a 25--43-fold storage and I/O
expansion; the final ``label`` column remains the benchmark contract.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PSM_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1kohMqejb7f787XtpM4b5HR7G22nH-rEF&export=download&confirm=t"
)
BATADAL_URLS = {
    "batadal_dataset03.csv": "https://raw.githubusercontent.com/rtaormina/aeed/master/data/dataset03.csv",
    "batadal_dataset04.csv": "https://raw.githubusercontent.com/rtaormina/aeed/master/data/dataset04.csv",
    "batadal_test_dataset.csv": "https://raw.githubusercontent.com/rtaormina/aeed/master/data/test_dataset.csv",
}


def _download(url: str, destination: Path) -> None:
    print(f"Downloading {destination.name}...", flush=True)
    with urllib.request.urlopen(url, timeout=180) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _prepare_sources(
    source_root: Path | None, *, include_psm: bool
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if source_root is not None:
        return source_root, None

    temporary_dir = tempfile.TemporaryDirectory(prefix="patternad_psm_batadal_")
    root = Path(temporary_dir.name)
    if include_psm:
        psm_archive = root / "PSM.zip"
        _download(PSM_URL, psm_archive)
        with zipfile.ZipFile(psm_archive) as archive:
            archive.extractall(root)
    for filename, url in BATADAL_URLS.items():
        _download(url, root / filename)
    return root, temporary_dir


def _write_wide_form(features: pd.DataFrame, labels: pd.Series, destination: Path) -> None:
    if len(features) != len(labels):
        raise ValueError(f"Feature/label length mismatch: {len(features)} != {len(labels)}")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Features contain missing or non-finite values.")
    if not labels.isin([0, 1]).all():
        raise ValueError("Labels must be binary.")

    output = features.copy()
    output.insert(0, "date", np.arange(1, len(features) + 1, dtype=np.int64))
    output["label"] = labels.to_numpy(dtype=np.int64)
    temporary_destination = destination.with_name(f"{destination.name}.tmp")
    temporary_destination.unlink(missing_ok=True)
    try:
        output.to_csv(temporary_destination, index=False, float_format="%.10g")
        temporary_destination.replace(destination)
    except BaseException:
        temporary_destination.unlink(missing_ok=True)
        raise


def _write_text(length: int, message: str, destination: Path) -> None:
    pd.DataFrame(
        {"date": np.arange(1, length + 1, dtype=np.int64), "data": message, "cols": "channel1"}
    ).to_csv(destination, index=False)


def _write_psm(source_root: Path, destination: Path) -> tuple[int, int, int]:
    psm_root = source_root / "PSM"
    timestamp_column = "timestamp_(min)"
    temporary_destination = destination.with_name(f"{destination.name}.tmp")
    temporary_destination.unlink(missing_ok=True)
    try:
        with (psm_root / "train.csv").open(newline="") as train_file, \
            (psm_root / "test.csv").open(newline="") as test_file, \
            (psm_root / "test_label.csv").open(newline="") as label_file, \
            temporary_destination.open("w", newline="") as output_file:
            train_reader = csv.reader(train_file)
            test_reader = csv.reader(test_file)
            label_reader = csv.reader(label_file)
            writer = csv.writer(output_file)
            train_header = next(train_reader)
            test_header = next(test_reader)
            label_header = next(label_reader)
            if (
                not train_header
                or train_header[0] != timestamp_column
                or test_header != train_header
                or label_header != [timestamp_column, "label"]
            ):
                raise ValueError("Unexpected PSM CSV headers.")
            feature_names = train_header[1:]
            writer.writerow(["date", *feature_names, "label"])

            timestamp = 1
            previous_values: list[str | None] = [None] * len(feature_names)
            for row in train_reader:
                if len(row) != len(train_header):
                    raise ValueError("Malformed PSM training row.")
                values = row[1:]
                for index, value in enumerate(values):
                    if value == "":
                        if previous_values[index] is None:
                            raise ValueError("PSM has a leading missing training value.")
                        values[index] = previous_values[index]
                    else:
                        previous_values[index] = value
                writer.writerow([timestamp, *values, 0])
                timestamp += 1
            train_length = timestamp - 1

            for test_row, label_row in zip(test_reader, label_reader):
                if len(test_row) != len(test_header) or len(label_row) != 2:
                    raise ValueError("Malformed PSM test row.")
                if test_row[0] != label_row[0] or label_row[1] not in {"0", "1"}:
                    raise ValueError("PSM test timestamps or labels do not align.")
                if any(value == "" for value in test_row[1:]):
                    raise ValueError("PSM test data unexpectedly contains missing values.")
                writer.writerow([timestamp, *test_row[1:], label_row[1]])
                timestamp += 1
            if next(test_reader, None) is not None or next(label_reader, None) is not None:
                raise ValueError("PSM test and label lengths differ.")
        temporary_destination.replace(destination)
    except BaseException:
        temporary_destination.unlink(missing_ok=True)
        raise
    return timestamp - 1, len(feature_names), train_length


def _read_batadal(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    if "DATETIME" not in data.columns or "ATT_FLAG" not in data.columns:
        raise ValueError(f"Unexpected BATADAL columns in {path.name}.")
    return data.drop(columns=["DATETIME", "ATT_FLAG"]), data["ATT_FLAG"].astype(np.int64)


def _build_batadal(
    source_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_features, train_labels = _read_batadal(source_root / "batadal_dataset03.csv")
    dataset04_features, dataset04_labels = _read_batadal(source_root / "batadal_dataset04.csv")
    test_features, test_labels = _read_batadal(source_root / "batadal_test_dataset.csv")
    if train_labels.any():
        raise ValueError("BATADAL dataset03 must be the normal-only training segment.")
    if not (list(train_features.columns) == list(dataset04_features.columns) == list(test_features.columns)):
        raise ValueError("BATADAL feature columns differ across source files.")
    return train_features, dataset04_features, dataset04_labels, test_features, test_labels


def _update_metadata(metadata_path: Path, rows: list[dict[str, object]]) -> None:
    metadata = pd.read_csv(metadata_path)
    filenames = {str(row["file_name"]) for row in rows}
    metadata = metadata[
        ~metadata["file_name"].isin(
            filenames
            | {
                "PSM.csv",
                "PSM.csv.gz",
                "BATADAL.csv",
                "BATADAL.csv.gz",
                "BATADAL_dataset04.csv",
                "BATADAL_dataset04.csv.gz",
                "BATADAL_test.csv",
                "BATADAL_test.csv.gz",
            }
        )
    ]
    metadata = pd.concat([metadata, pd.DataFrame(rows)], ignore_index=True)
    metadata.to_csv(metadata_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/anomaly_detect/data"),
        help="ts_benchmark anomaly-detection data directory.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("dataset/anomaly_detect/DETECT_META.csv"),
        help="Metadata CSV to update.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Existing directory containing PSM/ and the three BATADAL CSV files.",
    )
    parser.add_argument(
        "--reuse-existing-psm",
        action="store_true",
        help="Reuse an already converted PSM.csv and PSM_text.csv in output-dir.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata_path.resolve()
    source_root, temporary_dir = _prepare_sources(
        args.source_root.resolve() if args.source_root else None,
        include_psm=not args.reuse_existing_psm,
    )
    try:
        if args.reuse_existing_psm:
            psm_path = output_dir / "PSM.csv"
            psm_text_path = output_dir / "PSM_text.csv"
            if not psm_path.is_file() or not psm_text_path.is_file():
                raise FileNotFoundError("--reuse-existing-psm requires PSM.csv and PSM_text.csv.")
            psm_length, psm_dimensions, psm_train_length = 220322, 25, 132481
        else:
            psm_length, psm_dimensions, psm_train_length = _write_psm(
                source_root, output_dir / "PSM.csv"
            )
            _write_text(
                psm_length,
                "PSM process-monitoring telemetry with 25 numeric variables.",
                output_dir / "PSM_text.csv",
            )
        (
            batadal_train_features,
            batadal_dataset04_features,
            batadal_dataset04_labels,
            batadal_test_features,
            batadal_test_labels,
        ) = _build_batadal(source_root)
        batadal_train_labels = pd.Series(
            np.zeros(len(batadal_train_features), dtype=np.int64), name="ATT_FLAG"
        )
        batadal_dataset04 = pd.concat(
            [batadal_train_features, batadal_dataset04_features], ignore_index=True
        )
        batadal_dataset04_all_labels = pd.concat(
            [batadal_train_labels, batadal_dataset04_labels], ignore_index=True
        )
        batadal_test = pd.concat([batadal_train_features, batadal_test_features], ignore_index=True)
        batadal_test_all_labels = pd.concat(
            [batadal_train_labels, batadal_test_labels], ignore_index=True
        )
        batadal_message = (
            "BATADAL water-distribution telemetry with tank, pump, flow, and pressure variables."
        )
        _write_wide_form(
            batadal_dataset04,
            batadal_dataset04_all_labels,
            output_dir / "BATADAL_dataset04.csv",
        )
        _write_text(
            len(batadal_dataset04),
            batadal_message,
            output_dir / "BATADAL_dataset04_text.csv",
        )
        _write_wide_form(batadal_test, batadal_test_all_labels, output_dir / "BATADAL_test.csv")
        _write_text(
            len(batadal_test), batadal_message, output_dir / "BATADAL_test_text.csv"
        )

        _update_metadata(
            metadata_path,
            [
                {
                    "file_name": "PSM.csv",
                    "trend": False,
                    "seasonal": False,
                    "stationary": False,
                    "pattern": False,
                    "shifting": True,
                    "dataset_name": "PSM",
                    "train_lens": psm_train_length,
                    "time_steps": psm_length,
                    "if_univariate": False,
                    "size": "large",
                    "type_value": "mult_new_new",
                    "total_len": psm_length,
                    "train/total": psm_train_length / psm_length,
                },
                {
                    "file_name": "BATADAL_dataset04.csv",
                    "trend": False,
                    "seasonal": False,
                    "stationary": False,
                    "pattern": False,
                    "shifting": True,
                    "dataset_name": "BATADAL",
                    "train_lens": len(batadal_train_features),
                    "time_steps": len(batadal_dataset04),
                    "if_univariate": False,
                    "size": "large",
                    "type_value": "mult_new_new",
                    "total_len": len(batadal_dataset04),
                    "train/total": len(batadal_train_features) / len(batadal_dataset04),
                },
                {
                    "file_name": "BATADAL_test.csv",
                    "trend": False,
                    "seasonal": False,
                    "stationary": False,
                    "pattern": False,
                    "shifting": True,
                    "dataset_name": "BATADAL",
                    "train_lens": len(batadal_train_features),
                    "time_steps": len(batadal_test),
                    "if_univariate": False,
                    "size": "large",
                    "type_value": "mult_new_new",
                    "total_len": len(batadal_test),
                    "train/total": len(batadal_train_features) / len(batadal_test),
                },
            ],
        )
        print(
            "Prepared PSM "
            f"({psm_length} points, {psm_dimensions} variables) and BATADAL "
            f"dataset04/test ({len(batadal_dataset04)}/{len(batadal_test)} points, "
            f"{batadal_train_features.shape[1]} variables).",
            flush=True,
        )
    finally:
        if temporary_dir is not None:
            temporary_dir.cleanup()


if __name__ == "__main__":
    main()
