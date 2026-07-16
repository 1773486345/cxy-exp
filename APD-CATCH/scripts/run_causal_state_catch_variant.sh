#!/usr/bin/env bash
# Run exactly one Causal-State-CATCH variant in one terminal.
set -euo pipefail

gpu="${1:-0}"
dataset="${2:?usage: bash scripts/run_causal_state_catch_variant.sh GPU DATASET VARIANT [OUTPUT_ROOT] [SEED]}"
variant="${3:?usage: bash scripts/run_causal_state_catch_variant.sh GPU DATASET VARIANT [OUTPUT_ROOT] [SEED]}"
output_root="${4:-result/causal_state_catch_v2_screen}"
seed="${5:-2021}"

case "${variant}" in
  causal_catch|state|state_scale) ;;
  *)
    echo "variant must be causal_catch, state, or state_scale" >&2
    exit 2
    ;;
esac

worker_dir="${output_root%/}/workers/${dataset}_${variant}"

python scripts/run_apd_catch_paper.py \
  --datasets "${dataset}" \
  --variants "${variant}" \
  --seed "${seed}" \
  --output-dir "${worker_dir}" \
  --metrics full \
  --save-diagnostics \
  --gpu "${gpu}"
