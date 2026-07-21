"""Prepare the fixed 17-task mTSBench external-validation suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import import_mtsbench_source_dir, prepare_mtsbench


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mtsbench-source-dir",
        type=Path,
        help="local GitHub Actions artifact root containing the seven-file checksum manifest",
    )
    args = parser.parse_args()
    if args.mtsbench_source_dir is not None:
        copied = import_mtsbench_source_dir(args.mtsbench_source_dir)
        print(f"imported {copied} checksum-validated mTSBench artifact files")
    for row in prepare_mtsbench():
        print(row)


if __name__ == "__main__":
    main()
