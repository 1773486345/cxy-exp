#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAB_ROOT="${TAB_ROOT:-/media/h3c/users/wangyueyang1/cxy/TAB}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/run_baseline_python.sh}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_DIR="${PROJECT_ROOT}/result/label/_baseline_logs"

mkdir -p "${LOG_DIR}"

while true; do
  free_mib="$(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
      head -1 |
      tr -d ' '
  )"
  if [ "${free_mib}" -ge "${MIN_FREE_MIB}" ]; then
    break
  fi
  echo "[DADA retry] waiting free_mib=${free_mib} threshold=${MIN_FREE_MIB}"
  sleep "${POLL_SECONDS}"
done

run_dada() {
  local dataset="$1"
  local dataset_tag="${dataset%.csv}"
  local log_file="${LOG_DIR}/dada_${dataset_tag}_retry_20260614.log"

  cd "${TAB_ROOT}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN}" -u ./scripts/run_benchmark.py \
      --config-path "unfixed_detect_label_multi_config.json" \
      --data-name-list "${dataset}" \
      --model-name "pre_train.DadaModel" \
      --model-hyper-params '{"horizon": 1, "is_train": 1, "lr": 0.005, "norm": true, "sampling_rate": 1, "seq_len": 100}' \
      --adapter "PreTrain_adapter" \
      --gpus 0 \
      --num-workers 1 \
      --timeout 60000 \
      --save-path "label/baselines_${dataset_tag}_DADA" \
      > "${log_file}" 2>&1
}

has_valid_report() {
  local dataset_tag="$1"
  local result_dir="${PROJECT_ROOT}/result/label/baselines_${dataset_tag}_DADA"
  local report
  report="$(find "${result_dir}" -maxdepth 1 -type f -name 'test_report.*.csv' 2>/dev/null | sort | tail -1)"
  [ -n "${report}" ] &&
    awk -F, '
      NR > 1 && $NF != "" { count++ }
      END { exit count == 3 ? 0 : 1 }
    ' "${report}"
}

pids=()
if ! has_valid_report "ExathlonSmall"; then
  run_dada "ExathlonSmall.csv" &
  pids+=("$!")
fi
if ! has_valid_report "Metro"; then
  run_dada "Metro.csv" &
  pids+=("$!")
fi

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "[DADA retry] completed"
