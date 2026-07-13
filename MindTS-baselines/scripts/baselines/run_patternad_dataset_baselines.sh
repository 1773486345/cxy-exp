#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TAB_ROOT="${TAB_ROOT:-${PROJECT_ROOT}/../TAB}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/run_baseline_python.sh}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
GPU="${GPU:-0}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
MODEL_FILTER="${MODEL_FILTER:-}"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "missing baseline Python runner: ${PYTHON_BIN}" >&2
  exit 2
fi

LOG_DIR="${PROJECT_ROOT}/result/label/_baseline_logs"
LOG_FILE="${LOG_DIR}/patternad_dataset_baselines_${RUN_TAG}.log"
mkdir -p "${LOG_DIR}"

DATASET_TAGS=("$@")
if [ "${#DATASET_TAGS[@]}" -eq 0 ]; then
  DATASET_TAGS=("MetroPT3" "HAI21" "SMD")
fi

normalize_dataset_tag() {
  case "$1" in
    MetroPT3|MetroPT3.csv.gz)
      printf 'MetroPT3\n'
      ;;
    HAI21|HAI21_full|HAI21-full)
      printf 'HAI21\n'
      ;;
    SMD|SMD_full|SMD-full)
      printf 'SMD\n'
      ;;
    *)
      echo "unsupported PatternAD dataset: $1; expected MetroPT3, HAI21, or SMD" >&2
      return 2
      ;;
  esac
}

dataset_files() {
  case "$1" in
    MetroPT3)
      printf '%s\n' "MetroPT3.csv.gz"
      ;;
    HAI21)
      printf '%s\n' "HAI21_part1.csv.gz HAI21_part2.csv.gz HAI21_part3.csv.gz"
      ;;
    SMD)
      printf '%s\n' "SMD_machine-1-1.csv.gz SMD_machine-1-2.csv.gz SMD_machine-1-3.csv.gz SMD_machine-1-4.csv.gz SMD_machine-1-5.csv.gz SMD_machine-1-6.csv.gz SMD_machine-1-7.csv.gz SMD_machine-1-8.csv.gz SMD_machine-2-1.csv.gz SMD_machine-2-2.csv.gz SMD_machine-2-3.csv.gz SMD_machine-2-4.csv.gz SMD_machine-2-5.csv.gz SMD_machine-2-6.csv.gz SMD_machine-2-7.csv.gz SMD_machine-2-8.csv.gz SMD_machine-2-9.csv.gz SMD_machine-3-1.csv.gz SMD_machine-3-10.csv.gz SMD_machine-3-11.csv.gz SMD_machine-3-2.csv.gz SMD_machine-3-3.csv.gz SMD_machine-3-4.csv.gz SMD_machine-3-5.csv.gz SMD_machine-3-6.csv.gz SMD_machine-3-7.csv.gz SMD_machine-3-8.csv.gz SMD_machine-3-9.csv.gz"
      ;;
    *)
      return 2
      ;;
  esac
}

feature_count() {
  case "$1" in
    MetroPT3)
      printf '15\n'
      ;;
    HAI21)
      printf '79\n'
      ;;
    SMD)
      printf '38\n'
      ;;
    *)
      return 2
      ;;
  esac
}

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

has_result_artifacts() {
  local result_dir="$1"
  local report
  local archive
  report="$(find "${result_dir}" -maxdepth 1 -type f -name 'test_report.*.csv' 2>/dev/null | sort | tail -1 || true)"
  archive="$(find "${result_dir}" -maxdepth 1 -type f -name '*.csv.tar.gz' 2>/dev/null | sort | tail -1 || true)"
  [ -n "${report}" ] && [ -n "${archive}" ]
}

run_command() {
  local root_dir="$1"
  shift
  set +e
  (
    cd "${root_dir}"
    "$@"
  ) 2>&1 | tee -a "${LOG_FILE}"
  local status="${PIPESTATUS[0]}"
  set -e
  if [ "${status}" -ne 0 ]; then
    echo "[PatternAD baseline] failed status=${status} cmd=$*" | tee -a "${LOG_FILE}"
    if [ "${CONTINUE_ON_ERROR}" != "1" ]; then
      exit "${status}"
    fi
  fi
}

run_benchmark_model() {
  local root_dir="$1"
  local dataset_tag="$2"
  local model_tag="$3"
  local model_name="$4"
  local model_params="$5"
  local adapter="$6"
  local gpu="$7"
  local save_path
  local legacy_save_path=""
  case "${model_tag}" in
    DLinear|PatchTST|iTransformer)
      save_path="label/${dataset_tag}_${model_tag}_baseline"
      legacy_save_path="label/baselines_${dataset_tag}_${model_tag}"
      ;;
    TimesNet)
      save_path="label/${dataset_tag}_${model_tag}_baseline_h0"
      legacy_save_path="label/baselines_${dataset_tag}_${model_tag}"
      ;;
    *)
      save_path="label/baselines_${dataset_tag}_${model_tag}"
      ;;
  esac
  local result_dir="${PROJECT_ROOT}/result/${save_path}"
  local files
  read -r -a files <<< "$(dataset_files "${dataset_tag}")"

  if ! should_run_model "${model_tag}"; then
    return 0
  fi

  if [ -n "${legacy_save_path}" ]; then
    local legacy_result_dir="${PROJECT_ROOT}/result/${legacy_save_path}"
    if [ ! -e "${result_dir}" ] && has_result_artifacts "${legacy_result_dir}"; then
      mv "${legacy_result_dir}" "${result_dir}"
      echo "[PatternAD baseline] normalized completed result ${legacy_save_path} -> ${save_path}" | tee -a "${LOG_FILE}"
    fi
  fi

  if [ "${SKIP_EXISTING}" = "1" ] && has_result_artifacts "${result_dir}"; then
    echo "[PatternAD baseline] skip complete dataset=${dataset_tag} model=${model_tag} save_path=${save_path}" | tee -a "${LOG_FILE}"
    return 0
  fi

  local benchmark_entry="./scripts/run_benchmark.py"
  if [ "${root_dir}" = "${TAB_ROOT}" ]; then
    benchmark_entry="${PROJECT_ROOT}/scripts/baselines/run_tab_benchmark.py"
  fi

  local cmd=(
    "${PYTHON_BIN}" -u "${benchmark_entry}"
    --config-path "unfixed_detect_label_multi_config.json"
    --data-name-list "${files[@]}"
    --model-name "${model_name}"
    --model-hyper-params "${model_params}"
    --metrics '{"name":"affiliation_f"}' '{"name":"VUS_ROC"}' '{"name":"VUS_PR"}'
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

  echo "[PatternAD baseline] dataset=${dataset_tag} model=${model_tag} save_path=${save_path}" | tee -a "${LOG_FILE}"
  run_command "${root_dir}" "${cmd[@]}"
}

run_tslib_model() {
  local dataset_tag="$1"
  local model_tag="$2"
  local model_name="$3"
  local n_features
  n_features="$(feature_count "${dataset_tag}")"
  local model_params
  model_params="$(
    printf '{"batch_size":16,"c_out":%d,"d_ff":64,"d_layers":1,"d_model":256,"dec_in":%d,"e_layers":1,"enc_in":%d,"horizon":0,"lr":0.0001,"norm":true,"num_epochs":100,"patch_len":10,"patience":3,"seq_len":100,"stride":10,"task_name":"anomaly_detection"}' \
      "${n_features}" "${n_features}" "${n_features}"
  )"
  run_benchmark_model "${PROJECT_ROOT}" "${dataset_tag}" "${model_tag}" "${model_name}" "${model_params}" "transformer_adapter" "${GPU}"
}

echo "[PatternAD baseline sweep] started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "${LOG_FILE}"
echo "[PatternAD baseline sweep] project_root=${PROJECT_ROOT}" | tee -a "${LOG_FILE}"
echo "[PatternAD baseline sweep] tab_root=${TAB_ROOT}" | tee -a "${LOG_FILE}"
echo "[PatternAD baseline sweep] python_bin=${PYTHON_BIN}" | tee -a "${LOG_FILE}"
echo "[PatternAD baseline sweep] skip_existing=${SKIP_EXISTING} continue_on_error=${CONTINUE_ON_ERROR} gpu=${GPU}" | tee -a "${LOG_FILE}"
if [ -n "${MODEL_FILTER}" ]; then
  echo "[PatternAD baseline sweep] model_filter=${MODEL_FILTER}" | tee -a "${LOG_FILE}"
fi

for raw_dataset_tag in "${DATASET_TAGS[@]}"; do
  dataset_tag="$(normalize_dataset_tag "${raw_dataset_tag}")"
  read -r -a files <<< "$(dataset_files "${dataset_tag}")"
  echo "[PatternAD baseline sweep] dataset=${dataset_tag} files=${files[*]}" | tee -a "${LOG_FILE}"

  run_benchmark_model "${PROJECT_ROOT}" "${dataset_tag}" "PCA" "classic_ad.PCA" '{}' "None" "None"
  run_benchmark_model "${PROJECT_ROOT}" "${dataset_tag}" "IsolationForest" "classic_ad.IsolationForest" '{}' "None" "None"
  run_benchmark_model "${PROJECT_ROOT}" "${dataset_tag}" "LOF" "classic_ad.LOF" '{}' "None" "None"
  run_benchmark_model "${PROJECT_ROOT}" "${dataset_tag}" "OCSVM" "classic_ad.OCSVM" '{}' "None" "None"

  run_benchmark_model "${PROJECT_ROOT}" "${dataset_tag}" "TranAD" "self_impl.TranAD" '{}' "None" "None"
  run_benchmark_model "${PROJECT_ROOT}" "${dataset_tag}" "AnomalyTransformer" "self_impl.AnomalyTransformer" '{}' "None" "None"
  run_benchmark_model "${PROJECT_ROOT}" "${dataset_tag}" "USAD" "self_impl.USAD" '{}' "None" "None"
  run_benchmark_model "${PROJECT_ROOT}" "${dataset_tag}" "GDN" "self_impl.GDN" '{}' "None" "None"
  run_benchmark_model "${PROJECT_ROOT}" "${dataset_tag}" "OmniAnomaly" "self_impl.OmniAnomaly" '{"rnn_hidden":100,"dense_dim":100,"nf_layers":4,"max_epoch":10,"batch_size":128,"test_batch_size":256,"valid_step_freq":50}' "None" "None"
  run_benchmark_model "${PROJECT_ROOT}" "${dataset_tag}" "InterFusion" "self_impl.InterFusion" '{"rnn_hidden":64,"dense_hidden":64,"arnn_hidden":64,"posterior_flow_type":"None","posterior_flow_layers":0,"pretrain_max_epoch":10,"max_epoch":10,"batch_size":128,"test_batch_size":256,"test_n_z":1}' "None" "None"
  run_benchmark_model "${PROJECT_ROOT}" "${dataset_tag}" "MTAD-GAT" "self_impl.MTADGAT" '{"window_length":100,"run_mode":"FORECASTING","num_train_steps":100,"batch_size":128,"gru_hidden":64,"fc_hidden":64,"vae_latent":18,"conv1d_filter_width":7,"log_step_count_steps":20,"save_checkpoints_steps":100,"keep_checkpoint_max":2}' "None" "None"

  run_benchmark_model "${TAB_ROOT}" "${dataset_tag}" "DAGMM" "merlion.DAGMM" '{}' "None" "None"
  run_benchmark_model "${TAB_ROOT}" "${dataset_tag}" "DADA" "pre_train.DadaModel" '{"horizon":1,"is_train":1,"lr":0.005,"norm":true,"sampling_rate":1,"seq_len":100}' "PreTrain_adapter" "${GPU}"
  run_benchmark_model "${TAB_ROOT}" "${dataset_tag}" "UniTS" "pre_train.UniTS" '{"horizon":1,"is_train":1,"norm":true,"num_epochs":3,"sampling_rate":1,"seq_len":96}' "PreTrain_adapter" "${GPU}"
  run_benchmark_model "${TAB_ROOT}" "${dataset_tag}" "Timer" "pre_train.TimerModel" '{"horizon":1,"is_train":1,"norm":true,"num_epochs":3,"sampling_rate":1,"seq_len":96}' "PreTrain_adapter" "${GPU}"
  run_benchmark_model "${TAB_ROOT}" "${dataset_tag}" "LMixer" "LLM.LLMMixerModel" '{"d_model":32,"horizon":1,"lr":0.001,"n_heads":4,"norm":true,"sampling_rate":1,"seq_len":96,"use_norm":1}' "llm_adapter" "${GPU}"

  run_tslib_model "${dataset_tag}" "DLinear" "time_series_library.DLinear"
  run_tslib_model "${dataset_tag}" "PatchTST" "time_series_library.PatchTST"
  run_tslib_model "${dataset_tag}" "iTransformer" "time_series_library.iTransformer"
  run_tslib_model "${dataset_tag}" "TimesNet" "time_series_library.TimesNet"
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_patternad_baselines.py" | tee -a "${LOG_FILE}"
echo "[PatternAD baseline sweep] finished_at=$(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "${LOG_FILE}"
