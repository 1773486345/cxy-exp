#!/usr/bin/env python3
"""Run Direction B1's evidence-conditioned reliability calibration protocol."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.multi_evidence.run_b0 import main


if __name__ == "__main__":
    raise SystemExit(main())
