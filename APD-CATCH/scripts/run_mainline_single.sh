#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: run_mainline_single.sh {msd|bhd} DATASET" >&2
  exit 2
fi

model="$1"
dataset="$2"
python_bin="${PYTHON_BIN:-/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python}"
output_dir="${MAINLINE_OUTPUT_DIR:-result/mainline_score_real}"

case "$model" in
  msd|bhd) ;;
  catch)
    echo "CATCH score baselines are already complete; reuse result/score/CATCH instead." >&2
    exit 2
    ;;
  *)
    echo "model must be one of: msd, bhd" >&2
    exit 2
    ;;
esac

config_script="../CATCH-master/scripts/multivariate_detection/detect_score/${dataset}_script/CATCH.sh"
if [[ ! -f "$config_script" ]]; then
  echo "No verifiable CATCH config for dataset: $dataset" >&2
  exit 2
fi

params="$(sed -n "s/.*--model-hyper-params '\({.*}\)' --gpus.*/\1/p" "$config_script")"
if [[ -z "$params" ]]; then
  echo "Could not extract model hyperparameters from: $config_script" >&2
  exit 2
fi

"$python_bin" scripts/profile_mainline_models.py \
  --dataset "$dataset" \
  --model "$model" \
  --params "$params" \
  --seed 2021 \
  --full-run \
  --result-dir "$output_dir"
