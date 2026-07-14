#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/run_baseline_python.sh}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MODEL_FILTER="${MODEL_FILTER:-}"

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
  )
fi

BENCHMARK_CONFIG="unfixed_detect_label_train_calibration_config.json" \
RESULT_NAMESPACE="strict_baselines" \
SKIP_EXISTING="${SKIP_EXISTING}" \
PYTHON_BIN="${PYTHON_BIN}" \
MODEL_FILTER="${MODEL_FILTER}" \
bash "${SCRIPT_DIR}/run_all_requested_baselines.sh" "${DATASETS[@]}"
