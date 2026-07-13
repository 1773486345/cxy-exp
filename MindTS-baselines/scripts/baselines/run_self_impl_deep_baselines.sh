#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/run_baseline_python.sh}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

cd "${PROJECT_ROOT}"

DATASETS=("$@")
if [ "${#DATASETS[@]}" -eq 0 ]; then
  DATASETS=(
    "Genesis.csv"
    "Weather.csv"
    "Energy.csv"
    "SKAB.csv"
    "MSDS.csv"
    "Daphnet.csv"
    "GECCO.csv"
    "ExathlonSmall.csv"
    "Metro.csv"
  )
fi

MODEL_TAGS=(
  "TranAD"
  "AnomalyTransformer"
  "USAD"
)

MODEL_NAMES=(
  "self_impl.TranAD"
  "self_impl.AnomalyTransformer"
  "self_impl.USAD"
)

MODEL_PARAMS=(
  '{}'
  '{}'
  '{}'
)

for dataset in "${DATASETS[@]}"; do
  dataset_tag="${dataset%.csv}"
  for i in "${!MODEL_TAGS[@]}"; do
    model_tag="${MODEL_TAGS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    model_params="${MODEL_PARAMS[$i]}"
    save_path="label/baselines_${dataset_tag}_${model_tag}"
    result_dir="${PROJECT_ROOT}/result/${save_path}"
    if [ "${SKIP_EXISTING}" = "1" ] && compgen -G "${result_dir}/test_report.*.csv" > /dev/null; then
      echo "[MindTS self_impl baseline] skip existing dataset=${dataset} model=${model_tag} save_path=${save_path}"
      continue
    fi

    echo "[MindTS self_impl baseline] dataset=${dataset} model=${model_tag} save_path=${save_path}"
    "${PYTHON_BIN}" -u ./scripts/run_benchmark.py \
      --config-path "unfixed_detect_label_multi_config.json" \
      --data-name-list "${dataset}" \
      --model-name "${model_name}" \
      --model-hyper-params "${model_params}" \
      --num-workers 1 \
      --timeout 60000 \
      --save-path "${save_path}"
  done
done
