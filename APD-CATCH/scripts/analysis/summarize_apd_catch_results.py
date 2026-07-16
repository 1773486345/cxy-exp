#!/usr/bin/env python3
"""Merge per-worker APD-CATCH JSON results into root-level CSV summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_apd_catch_paper import write_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_root",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "result" / "causal_state_catch_v2_screen",
        help="Experiment root containing worker subdirectories.",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    write_summaries(output_root)
    print(f"wrote summaries under {output_root}")


if __name__ == "__main__":
    main()
