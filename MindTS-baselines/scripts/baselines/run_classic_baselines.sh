#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/run_baseline_python.sh}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-unfixed_detect_label_multi_config.json}"
RESULT_NAMESPACE="${RESULT_NAMESPACE:-baselines}"

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

MODELS=(
  "PCA:classic_ad.PCA:{}"
  "IsolationForest:classic_ad.IsolationForest:{}"
  "LOF:classic_ad.LOF:{}"
  "OCSVM:classic_ad.OCSVM:{}"
)

for dataset in "${DATASETS[@]}"; do
  dataset_tag="${dataset%.csv}"
  for spec in "${MODELS[@]}"; do
    IFS=":" read -r model_tag model_name model_params <<< "${spec}"
    save_path="label/${RESULT_NAMESPACE}_${dataset_tag}_${model_tag}"
    result_dir="${PROJECT_ROOT}/result/${save_path}"
    if [ "${SKIP_EXISTING}" = "1" ] && compgen -G "${result_dir}/test_report.*.csv" > /dev/null; then
      echo "[MindTS baseline] skip existing dataset=${dataset} model=${model_tag} save_path=${save_path}"
      continue
    fi

    echo "[MindTS baseline] dataset=${dataset} model=${model_tag} save_path=${save_path}"
    "${PYTHON_BIN}" -u ./scripts/run_benchmark.py \
      --config-path "${BENCHMARK_CONFIG}" \
      --data-name-list "${dataset}" \
      --model-name "${model_name}" \
      --model-hyper-params "${model_params}" \
      --num-workers 1 \
      --timeout 60000 \
      --save-path "${save_path}"
  done
done
