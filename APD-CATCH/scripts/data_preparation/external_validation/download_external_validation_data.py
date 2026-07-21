"""Download only the predeclared external-validation source files."""

from __future__ import annotations

import argparse

from common import download_all_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="redownload an existing raw source file")
    args = parser.parse_args()
    downloaded = download_all_sources(force=args.force)
    for source, files in downloaded.items():
        print(f"{source}: {len(files)} source files ready")


if __name__ == "__main__":
    main()
