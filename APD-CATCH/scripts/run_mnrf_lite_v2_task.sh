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
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_REL="${MNRF_SAVE_PATH:-score/MNRF-Lite-V2/${TASK}/run-${RUN_ID}}"

case "${RUN_REL}" in
  /*|*../*) printf 'MNRF_SAVE_PATH must stay inside the repository: %s\n' "${RUN_REL}" >&2; exit 2 ;;
esac
for value in "${BATCH_SIZE}" "${SEED}" "${STAGE1_EPOCHS}" "${STAGE2_EPOCHS}" "${CACHE_CHUNK_SIZE}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || { printf 'Expected an integer override, got %s\n' "${value}" >&2; exit 2; }
done
(( BATCH_SIZE > 0 && STAGE1_EPOCHS > 0 && STAGE2_EPOCHS > 0 && CACHE_CHUNK_SIZE > 0 )) || {
  printf 'Batch, epochs, and cache chunk size must be positive.\n' >&2
  exit 2
}

if [[ "${GPU_ID}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  DEVICE="cpu"
  GPU_ARGS=()
else
  [[ "${GPU_ID}" =~ ^[0-9]+$ ]] || { printf 'GPU_ID must be an integer or cpu.\n' >&2; exit 2; }
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  DEVICE="cuda:0"
  GPU_ARGS=(--gpus 0)
fi

RUN_ABS="${ROOT}/result/${RUN_REL}"
mkdir -p "$(dirname -- "${RUN_ABS}")"
MODEL_HYPER_PARAMS="{\"seed\":${SEED},\"seq_len\":128,\"pattern_len\":32,\"hidden_dim\":32,\"view_dim\":8,\"batch_size\":${BATCH_SIZE},\"stage1_epochs\":${STAGE1_EPOCHS},\"stage2_epochs\":${STAGE2_EPOCHS},\"patience\":5,\"learning_rate\":0.001,\"weight_decay\":0.0001,\"feature_std_floor\":0.05,\"train_noise_std\":0.01,\"noise_seed\":20261013,\"calibration_quantile\":0.995,\"cache_chunk_size\":${CACHE_CHUNK_SIZE},\"device\":\"${DEVICE}\",\"output_dir\":\"${RUN_ABS}\"}"

python ./scripts/run_benchmark.py \
  --config-path "mnrf_lite_v2_unfixed_detect_score.json" \
  --data-set-name "${DATA_SOURCE}" \
  --data-name-list "${TASK}.csv" \
  --model-name "mnrf_lite_v2.MNRFLiteV2" \
  --model-hyper-params "${MODEL_HYPER_PARAMS}" \
  --seed "${SEED}" \
  "${GPU_ARGS[@]}" \
  --num-workers 1 \
  --timeout 60000 \
  --save-path "${RUN_REL}"

python ./scripts/validate_mnrf_lite_v2_run.py "${RUN_ABS}" "${TASK}"
