#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TAB_ROOT="${TAB_ROOT:-/media/h3c/users/wangyueyang1/cxy/TAB}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/run_baseline_python.sh}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-unfixed_detect_label_multi_config.json}"
RESULT_NAMESPACE="${RESULT_NAMESPACE:-baselines}"

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
  "DAGMM"
  "DADA"
  "UniTS"
  "Timer"
  "LMixer"
)

MODEL_NAMES=(
  "merlion.DAGMM"
  "pre_train.DadaModel"
  "pre_train.UniTS"
  "pre_train.TimerModel"
  "LLM.LLMMixerModel"
)

MODEL_PARAMS=(
  '{}'
  '{"horizon": 1, "is_train": 1, "lr": 0.005, "norm": true, "sampling_rate": 1, "seq_len": 100}'
  '{"horizon": 1, "is_train": 1, "norm": true, "num_epochs": 3, "sampling_rate": 1, "seq_len": 96}'
  '{"horizon": 1, "is_train": 1, "norm": true, "num_epochs": 3, "sampling_rate": 1, "seq_len": 96}'
  '{"d_model": 32, "horizon": 1, "lr": 0.001, "n_heads": 4, "norm": true, "sampling_rate": 1, "seq_len": 96, "use_norm": 1}'
)

MODEL_ADAPTERS=(
  "None"
  "PreTrain_adapter"
  "PreTrain_adapter"
  "PreTrain_adapter"
  "llm_adapter"
)

MODEL_GPUS=(
  "None"
  "0"
  "0"
  "0"
  "0"
)

LOG_DIR="${PROJECT_ROOT}/result/label/_baseline_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/tab_supported_sweep.log"

has_three_metric_report() {
  local result_dir="$1"
  local report
  report="$(find "${result_dir}" -maxdepth 1 -type f -name 'test_report.*.csv' 2>/dev/null | sort | tail -1)"
  [ -n "${report}" ] &&
    grep -q ',affiliation_f,' "${report}" &&
    grep -q ',VUS_ROC,' "${report}" &&
    grep -q ',VUS_PR,' "${report}"
}

cd "${TAB_ROOT}"

echo "[TAB supported baseline sweep] started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[TAB supported baseline sweep] tab_root=${TAB_ROOT}"
echo "[TAB supported baseline sweep] result_root=${PROJECT_ROOT}/result/label"
echo "[TAB supported baseline sweep] skip_existing=${SKIP_EXISTING}"

for dataset in "${DATASETS[@]}"; do
  dataset_tag="${dataset%.csv}"
  for i in "${!MODEL_TAGS[@]}"; do
    model_tag="${MODEL_TAGS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    model_params="${MODEL_PARAMS[$i]}"
    adapter="${MODEL_ADAPTERS[$i]}"
    gpu="${MODEL_GPUS[$i]}"
    save_path="label/${RESULT_NAMESPACE}_${dataset_tag}_${model_tag}"
    result_dir="${PROJECT_ROOT}/result/${save_path}"

    if [ "${SKIP_EXISTING}" = "1" ] && has_three_metric_report "${result_dir}"; then
      echo "[TAB supported baseline] skip complete dataset=${dataset} model=${model_tag} save_path=${save_path}"
      continue
    fi

    echo "[TAB supported baseline] dataset=${dataset} model=${model_tag} save_path=${save_path}"
    cmd=(
      "${PYTHON_BIN}" -u ./scripts/run_benchmark.py
      --config-path "${BENCHMARK_CONFIG}"
      --data-name-list "${dataset}"
      --model-name "${model_name}"
      --model-hyper-params "${model_params}"
      --num-workers 1
      --timeout 60000
      --save-path "${save_path}"
    )
    if [ "${adapter}" != "None" ]; then
      cmd+=(--adapter "${adapter}")
    fi
    if [ "${gpu}" != "None" ]; then
      cmd+=(--gpus "${gpu}")
    fi

    "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
  done
done

echo "[TAB supported baseline sweep] finished_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
