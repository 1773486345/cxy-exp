#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/run_baseline_python.sh}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-unfixed_detect_label_multi_config.json}"
RESULT_NAMESPACE="${RESULT_NAMESPACE:-baselines}"
MODEL_FILTER="${MODEL_FILTER:-}"
export BENCHMARK_CONFIG RESULT_NAMESPACE

DATASETS=("$@")

run_group() {
  local script_name="$1"
  shift
  echo "[requested baselines] running ${script_name} skip_existing=${SKIP_EXISTING}"
  SKIP_EXISTING="${SKIP_EXISTING}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    BENCHMARK_CONFIG="${BENCHMARK_CONFIG}" \
    RESULT_NAMESPACE="${RESULT_NAMESPACE}" \
    MODEL_FILTER="${MODEL_FILTER}" \
    bash "${SCRIPT_DIR}/${script_name}" "${DATASETS[@]}"
}

run_group "run_classic_baselines.sh"
run_group "run_self_impl_deep_baselines.sh"
run_group "run_gdn_baselines.sh"
run_group "run_omni_anomaly_baselines.sh"
run_group "run_interfusion_baselines.sh"
run_group "run_mtad_gat_baselines.sh"
run_group "run_tab_supported_baselines.sh"

TSLIB_DATASETS=()
if [ "${#DATASETS[@]}" -eq 0 ]; then
  TSLIB_DATASETS=(
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
else
  for dataset in "${DATASETS[@]}"; do
    case "${dataset}" in
      Genesis.csv|Weather.csv|Energy.csv|SKAB.csv|MSDS.csv|Daphnet.csv|GECCO.csv|ExathlonSmall.csv|Metro.csv)
        TSLIB_DATASETS+=("${dataset}")
        ;;
    esac
  done
fi

if [ "${#TSLIB_DATASETS[@]}" -gt 0 ]; then
  SKIP_EXISTING="${SKIP_EXISTING}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    BENCHMARK_CONFIG="${BENCHMARK_CONFIG}" \
    RESULT_NAMESPACE="${RESULT_NAMESPACE}" \
    bash "${SCRIPT_DIR}/run_daphnet_gecco_tslib_baselines.sh" "${TSLIB_DATASETS[@]}"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_requested_baselines.py"

echo "[requested baselines] complete result_root=${PROJECT_ROOT}/result/label namespace=${RESULT_NAMESPACE} config=${BENCHMARK_CONFIG}"
