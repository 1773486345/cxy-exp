#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/run_baseline_python.sh}"
LOG_DIR="${PROJECT_ROOT}/result/label/_baseline_logs"
LOG_FILE="${LOG_DIR}/gdn_sweep.log"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-unfixed_detect_label_multi_config.json}"
RESULT_NAMESPACE="${RESULT_NAMESPACE:-baselines}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

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

for dataset in "${DATASETS[@]}"; do
  dataset_tag="${dataset%.csv}"
  save_path="label/${RESULT_NAMESPACE}_${dataset_tag}_GDN"
  result_dir="${PROJECT_ROOT}/result/${save_path}"
  if [ "${SKIP_EXISTING}" = "1" ] && compgen -G "${result_dir}/test_report.*.csv" > /dev/null; then
    echo "[GDN baseline] skip existing dataset=${dataset} save_path=${save_path}" | tee -a "${LOG_FILE}"
    continue
  fi

  echo "[GDN baseline] dataset=${dataset} save_path=${save_path}" | tee -a "${LOG_FILE}"
  "${PYTHON_BIN}" -u ./scripts/run_benchmark.py \
    --config-path "${BENCHMARK_CONFIG}" \
    --data-name-list "${dataset}" \
    --model-name "self_impl.GDN" \
    --model-hyper-params '{}' \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "${save_path}" 2>&1 | tee -a "${LOG_FILE}"
done
