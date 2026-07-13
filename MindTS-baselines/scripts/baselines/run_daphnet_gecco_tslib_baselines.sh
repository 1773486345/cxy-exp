#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/run_baseline_python.sh}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
GPU="${GPU:-0}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"

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

MODEL_TAGS=("DLinear" "PatchTST" "iTransformer" "TimesNet")
MODEL_NAMES=(
  "time_series_library.DLinear"
  "time_series_library.PatchTST"
  "time_series_library.iTransformer"
  "time_series_library.TimesNet"
)

LOG_DIR="${PROJECT_ROOT}/result/label/_baseline_logs"
mkdir -p "${LOG_DIR}"

has_three_metric_report() {
  local result_dir="$1"
  local report
  report="$(find "${result_dir}" -maxdepth 1 -type f -name 'test_report.*.csv' 2>/dev/null | sort | tail -1)"
  [ -n "${report}" ] &&
    grep -q ',affiliation_f,' "${report}" &&
    grep -q ',VUS_ROC,' "${report}" &&
    grep -q ',VUS_PR,' "${report}"
}

cd "${PROJECT_ROOT}"

for dataset in "${DATASETS[@]}"; do
  dataset_tag="${dataset%.csv}"
  case "${dataset_tag}" in
    Genesis)
      feature_count=18
      ;;
    Weather)
      feature_count=4
      ;;
    Energy|Daphnet|GECCO)
      feature_count=9
      ;;
    SKAB)
      feature_count=8
      ;;
    MSDS)
      feature_count=10
      ;;
    ExathlonSmall)
      feature_count=12
      ;;
    Metro)
      feature_count=5
      ;;
    *)
      echo "unsupported dataset: ${dataset}; expected one of the nine requested anomaly datasets" >&2
      exit 2
      ;;
  esac

  log_file="${LOG_DIR}/tslib_${dataset_tag}_${RUN_TAG}.log"
  echo "[tslib baseline sweep] dataset=${dataset} started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "${log_file}"

  for i in "${!MODEL_TAGS[@]}"; do
    model_tag="${MODEL_TAGS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    if [ "${model_tag}" = "TimesNet" ]; then
      save_path="label/${dataset_tag}_${model_tag}_baseline_h0"
    else
      save_path="label/${dataset_tag}_${model_tag}_baseline"
    fi
    result_dir="${PROJECT_ROOT}/result/${save_path}"

    if [ "${SKIP_EXISTING}" = "1" ] && has_three_metric_report "${result_dir}"; then
      echo "[tslib baseline] skip complete dataset=${dataset} model=${model_tag}" | tee -a "${log_file}"
      continue
    fi

    model_params="$(
      printf '{"batch_size":16,"c_out":%d,"d_ff":64,"d_layers":1,"d_model":256,"dec_in":%d,"e_layers":1,"enc_in":%d,"horizon":0,"lr":0.0001,"norm":true,"num_epochs":100,"patch_len":10,"patience":3,"seq_len":100,"stride":10,"task_name":"anomaly_detection"}' \
        "${feature_count}" "${feature_count}" "${feature_count}"
    )"

    echo "[tslib baseline] dataset=${dataset} model=${model_tag} save_path=${save_path}" | tee -a "${log_file}"
    "${PYTHON_BIN}" -u ./scripts/run_benchmark.py \
      --config-path "unfixed_detect_label_multi_config.json" \
      --data-name-list "${dataset}" \
      --model-name "${model_name}" \
      --model-hyper-params "${model_params}" \
      --adapter "transformer_adapter" \
      --gpus "${GPU}" \
      --num-workers 1 \
      --timeout 60000 \
      --save-path "${save_path}" \
      2>&1 | tee -a "${log_file}"
  done

  echo "[tslib baseline sweep] dataset=${dataset} finished_at=$(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "${log_file}"
done
