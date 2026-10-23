#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MODE="${1:-execute}"
exec python "${ROOT}/scripts/run_mnrf_branch_ablation.py" --task Genesis --mode "${MODE}"
