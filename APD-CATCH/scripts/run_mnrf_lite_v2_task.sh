#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?task name required}"
DATA_SOURCE="${2:?data source required}"
ACTION="${3:---execute}"
case "${ACTION}" in --execute|--preflight) ;; *) printf 'Unknown action: %s\n' "${ACTION}" >&2; exit 2 ;; esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

case "${DATA_SOURCE}" in
  large_detect|external_detect) ;;
  *) printf 'Unsupported data source: %s\n' "${DATA_SOURCE}" >&2; exit 2 ;;
esac

GPU_ID="${GPU_ID:-0}"
SEQ_LEN="${MNRF_SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-2021}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
CACHE_CHUNK_SIZE="${CACHE_CHUNK_SIZE:-256}"
ENCODING_BATCH_SIZE="${ENCODING_BATCH_SIZE:-${BATCH_SIZE}}"
EVOLUTION_DIM="${EVOLUTION_DIM:-8}"
PATTERN_DIM="${PATTERN_DIM:-8}"
RELATION_DIM="${RELATION_DIM:-8}"
SCORE_READOUT="${MNRF_SCORE_READOUT:-flow}"
DATA_ROOT_ARGS=()
if [[ -n "${MNRF_DATA_ROOT:-}" ]]; then
  [[ "${DATA_SOURCE}" == "large_detect" ]] || { printf 'MNRF_DATA_ROOT requires large_detect.\n' >&2; exit 2; }
  DATA_ROOT="$(cd -- "${MNRF_DATA_ROOT}" && pwd -P)"
  [[ -r "${DATA_ROOT}/DETECT_META.csv" && -r "${DATA_ROOT}/data/${TASK}.csv" ]] || {
    printf 'Missing task data or metadata under %s\n' "${DATA_ROOT}" >&2; exit 2;
  }
  printf 'MNRF-Lite V2 isolated data root=%s\n' "${DATA_ROOT}"
  DATA_ROOT_ARGS=(--data-root "${DATA_ROOT}")
fi
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_REL="${MNRF_RESUME_FROM:-${MNRF_SAVE_PATH:-score/MNRF-Lite-V2/${TASK}/run-${RUN_ID}}}"

case "/${RUN_REL}/" in
  //*|*/../*|*/./*|*/.mnrf-work/*) printf 'Invalid benchmark result path: %s\n' "${RUN_REL}" >&2; exit 2 ;;
esac
ARTIFACT_POLICY="${MNRF_ARTIFACT_POLICY:-}"
if [[ -z "${ARTIFACT_POLICY}" ]]; then
  ARTIFACT_POLICY=benchmark
  if [[ -n "${MNRF_RESUME_FROM:-}" && -f "result/${RUN_REL}/run_identity.json" ]]; then
    ARTIFACT_POLICY=full
  fi
fi
case "${ARTIFACT_POLICY}" in
  benchmark|readout)
    if [[ -e "result/${RUN_REL}" || -L "result/${RUN_REL}" ]]; then
      printf 'Benchmark destination already exists: result/%s\n' "${RUN_REL}" >&2; exit 2
    fi
    WORK_REL=".mnrf-work/${RUN_REL}"
    ;;
  full) WORK_REL="${RUN_REL}" ;;
  *) printf 'Unsupported MNRF_ARTIFACT_POLICY: %s\n' "${ARTIFACT_POLICY}" >&2; exit 2 ;;
esac
case "${SCORE_READOUT}" in
  flow|state_amplitude|flow_plus_amplitude) ;;
  *) printf 'Unsupported MNRF_SCORE_READOUT: %s\n' "${SCORE_READOUT}" >&2; exit 2 ;;
esac
for value in "${SEQ_LEN}" "${BATCH_SIZE}" "${SEED}" "${STAGE1_EPOCHS}" "${STAGE2_EPOCHS}" "${CACHE_CHUNK_SIZE}" "${ENCODING_BATCH_SIZE}" "${EVOLUTION_DIM}" "${PATTERN_DIM}" "${RELATION_DIM}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || { printf 'Expected an integer override, got %s\n' "${value}" >&2; exit 2; }
done
(( SEQ_LEN >= 32 && BATCH_SIZE > 0 && STAGE1_EPOCHS > 0 && STAGE2_EPOCHS > 0 && CACHE_CHUNK_SIZE > 0 &&
   EVOLUTION_DIM > 0 && PATTERN_DIM > 0 && RELATION_DIM > 0 )) || {
  printf 'Window must be at least 32 and batch, epochs, cache chunk size, and E/P/R dimensions must be positive.\n' >&2
  exit 2
}

if [[ "${ACTION}" == "--preflight" ]]; then
  DEVICE="cpu"
elif [[ "${GPU_ID}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  DEVICE="cpu"
  export TS_BENCHMARK_PRESERVE_CUDA_VISIBLE_DEVICES=1
else
  [[ "${GPU_ID}" == "0" ]] || { printf 'This host profile supports GPU_ID=0 or GPU_ID=cpu.\n' >&2; exit 2; }
  if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    export CUDA_VISIBLE_DEVICES="0"
  fi
  DEVICE="cuda:0"
  export TS_BENCHMARK_PRESERVE_CUDA_VISIBLE_DEVICES=1
fi
printf 'MNRF-Lite V2 device=%s CUDA_VISIBLE_DEVICES=%s backend=sequential\n' "${DEVICE}" "${CUDA_VISIBLE_DEVICES-<unset>}"

RUN_ABS="${ROOT}/result/${WORK_REL}"
RESUME_FROM=""
if [[ -n "${MNRF_RESUME_FROM:-}" ]]; then
  [[ -f "${RUN_ABS}/run_identity.json" ]] || { printf 'No resumable run at %s\n' "${RUN_ABS}" >&2; exit 2; }
  RESUME_FROM="${WORK_REL}"
elif [[ -e "${RUN_ABS}" || -L "${RUN_ABS}" ]]; then
  printf 'MNRF work directory already exists: %s\n' "${RUN_ABS}" >&2; exit 2
fi
printf 'MNRF-Lite V2 artifact policy=%s work directory=%s\n' "${ARTIFACT_POLICY}" "${RUN_ABS}"
MODEL_HYPER_PARAMS="{\"seed\":${SEED},\"task_name\":\"${TASK}.csv\",\"seq_len\":${SEQ_LEN},\"pattern_len\":32,\"hidden_dim\":32,\"view_dim\":8,\"evolution_dim\":${EVOLUTION_DIM},\"pattern_dim\":${PATTERN_DIM},\"relation_dim\":${RELATION_DIM},\"batch_size\":${BATCH_SIZE},\"stage1_epochs\":${STAGE1_EPOCHS},\"stage2_epochs\":${STAGE2_EPOCHS},\"patience\":5,\"learning_rate\":0.001,\"weight_decay\":0.0001,\"feature_std_floor\":0.05,\"train_noise_std\":0.01,\"noise_seed\":20261013,\"calibration_quantile\":0.995,\"cache_chunk_size\":${CACHE_CHUNK_SIZE},\"encoding_batch_size\":${ENCODING_BATCH_SIZE},\"device\":\"${DEVICE}\",\"output_dir\":\"${RUN_ABS}\",\"resume_from\":\"${RESUME_FROM}\",\"score_readout\":\"${SCORE_READOUT}\"}"

if [[ "${ACTION}" == "--preflight" ]]; then
  [[ "${DATA_SOURCE}" == "large_detect" ]] || { printf 'Preflight requires large_detect.\n' >&2; exit 2; }
  CHECK_ARGS=()
  [[ -z "${MNRF_EXPECTED_TEST:-}" ]] || CHECK_ARGS+=(--expected-test "${MNRF_EXPECTED_TEST}")
  [[ -z "${MNRF_EXPECTED_CALIBRATION:-}" ]] || CHECK_ARGS+=(--expected-calibration "${MNRF_EXPECTED_CALIBRATION}")
  [[ -z "${MNRF_EXPECTED_FIELDS:-}" ]] || CHECK_ARGS+=(--expected-fields "${MNRF_EXPECTED_FIELDS}")
  python ./scripts/preflight_mnrf_lite_v2.py --task "${TASK}" \
    --data-root "${DATA_ROOT:-${ROOT}/dataset/anomaly_detect}" \
    --params "${MODEL_HYPER_PARAMS}" "${CHECK_ARGS[@]}"
  exit
fi

STRATEGY_ARGS="$(python - "${MNRF_DATA_VERSION:-${TASK}-benchmark-original}" <<'PY'
import json, sys
print(json.dumps({"readout_data_version": sys.argv[1]}))
PY
)"
python ./scripts/run_benchmark.py \
  --config-path "mnrf_lite_v2_unfixed_detect_score.json" \
  --data-set-name "${DATA_SOURCE}" \
  "${DATA_ROOT_ARGS[@]}" \
  --data-name-list "${TASK}.csv" \
  --model-name "mnrf_lite_v2.MNRFLiteV2" \
  --model-hyper-params "${MODEL_HYPER_PARAMS}" \
  --strategy-args "${STRATEGY_ARGS}" \
  --seed "${SEED}" \
  --num-workers 1 \
  --timeout 60000 \
  --save-path "${WORK_REL}"

python ./scripts/validate_mnrf_lite_v2_run.py "${RUN_ABS}" "${TASK}"
case "${ARTIFACT_POLICY}" in
  benchmark) python ./scripts/publish_mnrf_benchmark_outputs.py "${RUN_REL}" ;;
  readout)
    python ./scripts/publish_mnrf_benchmark_outputs.py "${RUN_REL}" --keep-work
    printf 'Validated readout source: %s\n' "${RUN_ABS}"
    ;;
esac
