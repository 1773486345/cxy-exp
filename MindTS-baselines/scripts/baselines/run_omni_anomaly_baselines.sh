#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/media/h3c/users/wangyueyang1/cxy/.env/envs/mindts_env/bin/python}"
LOG_DIR="${PROJECT_ROOT}/result/label/_baseline_logs"
LOG_FILE="${LOG_DIR}/omni_anomaly_sweep.log"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OMNI_HYPER_PARAMS="${OMNI_HYPER_PARAMS:-{\"rnn_hidden\":100,\"dense_dim\":100,\"nf_layers\":4,\"max_epoch\":10,\"batch_size\":128,\"test_batch_size\":256,\"valid_step_freq\":50}}"

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
  save_path="label/baselines_${dataset_tag}_OmniAnomaly"
  result_dir="${PROJECT_ROOT}/result/${save_path}"
  if [ "${SKIP_EXISTING}" = "1" ] && compgen -G "${result_dir}/test_report.*.csv" > /dev/null; then
    echo "[OmniAnomaly baseline] skip existing dataset=${dataset} save_path=${save_path}" | tee -a "${LOG_FILE}"
    continue
  fi

  echo "[OmniAnomaly baseline] dataset=${dataset} save_path=${save_path} hyper_params=${OMNI_HYPER_PARAMS}" | tee -a "${LOG_FILE}"
  "${PYTHON_BIN}" -u ./scripts/run_benchmark.py \
    --config-path "unfixed_detect_label_multi_config.json" \
    --data-name-list "${dataset}" \
    --model-name "self_impl.OmniAnomaly" \
    --model-hyper-params "${OMNI_HYPER_PARAMS}" \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "${save_path}" 2>&1 | tee -a "${LOG_FILE}"
done
