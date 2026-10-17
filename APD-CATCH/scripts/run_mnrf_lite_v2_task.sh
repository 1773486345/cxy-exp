#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?task name required}"
DATA_SOURCE="${2:?data source required}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

case "${DATA_SOURCE}" in
  large_detect|external_detect) ;;
  *) printf 'Unsupported data source: %s\n' "${DATA_SOURCE}" >&2; exit 2 ;;
esac

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-2021}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
CACHE_CHUNK_SIZE="${CACHE_CHUNK_SIZE:-256}"
ENCODING_BATCH_SIZE="${ENCODING_BATCH_SIZE:-${BATCH_SIZE}}"
EVOLUTION_DIM="${EVOLUTION_DIM:-8}"
PATTERN_DIM="${PATTERN_DIM:-8}"
RELATION_DIM="${RELATION_DIM:-8}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_REL="${MNRF_RESUME_FROM:-${MNRF_SAVE_PATH:-score/MNRF-Lite-V2/${TASK}/run-${RUN_ID}}}"

case "${RUN_REL}" in
  /*|*../*) printf 'MNRF_SAVE_PATH must stay inside the repository: %s\n' "${RUN_REL}" >&2; exit 2 ;;
esac
for value in "${BATCH_SIZE}" "${SEED}" "${STAGE1_EPOCHS}" "${STAGE2_EPOCHS}" "${CACHE_CHUNK_SIZE}" "${ENCODING_BATCH_SIZE}" "${EVOLUTION_DIM}" "${PATTERN_DIM}" "${RELATION_DIM}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || { printf 'Expected an integer override, got %s\n' "${value}" >&2; exit 2; }
done
(( BATCH_SIZE > 0 && STAGE1_EPOCHS > 0 && STAGE2_EPOCHS > 0 && CACHE_CHUNK_SIZE > 0 &&
   EVOLUTION_DIM > 0 && PATTERN_DIM > 0 && RELATION_DIM > 0 )) || {
  printf 'Batch, epochs, cache chunk size, and E/P/R dimensions must be positive.\n' >&2
  exit 2
}

if [[ "${GPU_ID}" == "cpu" ]]; then
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

RUN_ABS="${ROOT}/result/${RUN_REL}"
mkdir -p "$(dirname -- "${RUN_ABS}")"
MODEL_HYPER_PARAMS="{\"seed\":${SEED},\"task_name\":\"${TASK}.csv\",\"seq_len\":128,\"pattern_len\":32,\"hidden_dim\":32,\"view_dim\":8,\"evolution_dim\":${EVOLUTION_DIM},\"pattern_dim\":${PATTERN_DIM},\"relation_dim\":${RELATION_DIM},\"batch_size\":${BATCH_SIZE},\"stage1_epochs\":${STAGE1_EPOCHS},\"stage2_epochs\":${STAGE2_EPOCHS},\"patience\":5,\"learning_rate\":0.001,\"weight_decay\":0.0001,\"feature_std_floor\":0.05,\"train_noise_std\":0.01,\"noise_seed\":20261013,\"calibration_quantile\":0.995,\"cache_chunk_size\":${CACHE_CHUNK_SIZE},\"encoding_batch_size\":${ENCODING_BATCH_SIZE},\"device\":\"${DEVICE}\",\"output_dir\":\"${RUN_ABS}\",\"resume_from\":\"${MNRF_RESUME_FROM:-}\"}"

python ./scripts/run_benchmark.py \
  --config-path "mnrf_lite_v2_unfixed_detect_score.json" \
  --data-set-name "${DATA_SOURCE}" \
  --data-name-list "${TASK}.csv" \
  --model-name "mnrf_lite_v2.MNRFLiteV2" \
  --model-hyper-params "${MODEL_HYPER_PARAMS}" \
  --seed "${SEED}" \
  --num-workers 1 \
  --timeout 60000 \
  --save-path "${RUN_REL}"

python ./scripts/validate_mnrf_lite_v2_run.py "${RUN_ABS}" "${TASK}"
