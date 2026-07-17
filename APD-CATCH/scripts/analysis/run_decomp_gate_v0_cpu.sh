#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
exec python -u scripts/analysis/run_decomp_gate_v0_seed.py "$@"
