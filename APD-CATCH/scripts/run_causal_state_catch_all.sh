#!/usr/bin/env bash
# Run the complete target-blind Causal-State-CATCH matrix with resume support.
set -euo pipefail

gpu="${1:-0}"
output_dir="${2:-result/causal_state_catch_v2}"
seed_spec="${3:-2021}"
read -r -a seeds <<< "${seed_spec}"

for seed in "${seeds[@]}"; do
  python scripts/run_apd_catch_paper.py \
    --datasets all \
    --variants causal_catch state state_scale \
    --seed "${seed}" \
    --output-dir "${output_dir}" \
    --metrics full \
    --gpu "${gpu}"
done
