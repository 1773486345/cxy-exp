#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/run_baseline_python.sh}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# OmniAnomaly and InterFusion use the CPU-only TF1 environment; opt in explicitly.
MODEL_FILTER="${MODEL_FILTER:-PCA,IsolationForest,LOF,OCSVM,TranAD,USAD,GDN,DAGMM,DADA,UniTS,Timer,LMixer,DLinear,PatchTST,iTransformer,TimesNet}"

BENCHMARK_CONFIG="unfixed_detect_label_train_calibration_config.json" \
RESULT_NAMESPACE="strict_patternad_baselines" \
SKIP_EXISTING="${SKIP_EXISTING}" \
PYTHON_BIN="${PYTHON_BIN}" \
MODEL_FILTER="${MODEL_FILTER}" \
bash "${SCRIPT_DIR}/run_patternad_dataset_baselines.sh" "$@"
