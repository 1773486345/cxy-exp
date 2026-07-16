#!/usr/bin/env python3
"""Summarize archived local original-CATCH score and label reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def report_record(path: Path, protocol: str, dataset_file: str) -> dict | None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or len(rows[1]) < 3:
        return None
    try:
        value = float(rows[1][-1])
    except ValueError:
        return None
    return {
        "dataset_file": dataset_file,
        "protocol": protocol,
        "metric": rows[1][1],
        "value": value,
        "report_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-root",
        type=Path,
        default=REPO_ROOT.parent / "CATCH-master",
        help="Local original CATCH checkout containing result/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "result" / "causal_state_catch_v2" / "original_catch_local_reference.csv",
    )
    args = parser.parse_args()

    records = []
    for protocol in ("score", "label"):
        root = args.original_root / "result" / protocol / "CATCH"
        for path in sorted(root.glob("*/run-*/test_report.*.csv")):
            dataset_file = path.relative_to(root).parts[0]
            record = report_record(path, protocol, dataset_file)
            if record is not None:
                records.append(record)
    records.sort(key=lambda record: (record["dataset_file"], record["protocol"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)
    print(f"wrote {len(records)} archived original-CATCH records to {args.output}")


if __name__ == "__main__":
    main()
