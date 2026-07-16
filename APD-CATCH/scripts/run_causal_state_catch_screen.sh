#!/usr/bin/env bash
# Single-seed screen with the four diagnostics and the acceptable PSM long task.
set -euo pipefail

gpu="${1:-0}"
output_dir="${2:-result/causal_state_catch_v2_screen}"
seed="${3:-2021}"

python scripts/run_apd_catch_paper.py \
  --datasets CalIt2 GECCO Genesis NYC PSM \
  --variants causal_catch state state_scale \
  --seed "${seed}" \
  --output-dir "${output_dir}" \
  --metrics full \
  --gpu "${gpu}"
