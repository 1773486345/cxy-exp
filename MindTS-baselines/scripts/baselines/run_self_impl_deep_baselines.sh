#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/run_baseline_python.sh}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-unfixed_detect_label_multi_config.json}"
RESULT_NAMESPACE="${RESULT_NAMESPACE:-baselines}"
MODEL_FILTER="${MODEL_FILTER:-}"

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

should_run_model() {
  local model_tag="$1"
  if [ -z "${MODEL_FILTER}" ]; then
    return 0
  fi
  case ",${MODEL_FILTER}," in
    *",${model_tag},"*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  dataset_tag="${dataset%.csv}"
  for i in "${!MODEL_TAGS[@]}"; do
    model_tag="${MODEL_TAGS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    model_params="${MODEL_PARAMS[$i]}"
    if ! should_run_model "${model_tag}"; then
      echo "[MindTS self_impl baseline] skip by model filter dataset=${dataset} model=${model_tag}"
      continue
    fi
    save_path="label/${RESULT_NAMESPACE}_${dataset_tag}_${model_tag}"
    result_dir="${PROJECT_ROOT}/result/${save_path}"
    if [ "${SKIP_EXISTING}" = "1" ] && compgen -G "${result_dir}/test_report.*.csv" > /dev/null; then
      echo "[MindTS self_impl baseline] skip existing dataset=${dataset} model=${model_tag} save_path=${save_path}"
      continue
    fi

    echo "[MindTS self_impl baseline] dataset=${dataset} model=${model_tag} save_path=${save_path}"
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
