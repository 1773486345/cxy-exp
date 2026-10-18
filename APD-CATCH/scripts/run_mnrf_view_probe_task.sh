#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?task name required}"
SOURCE="${2:-large_detect}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

MODE="${VIEW_PROBE_MODE:-full_gmm}"
SEED="${SEED:-2021}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ENCODING_BATCH_SIZE="${ENCODING_BATCH_SIZE:-${BATCH_SIZE}}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
PATIENCE="${PATIENCE:-5}"
GPU_ID="${GPU_ID:-0}"
VIEW_PROBE_OUTPUT="${VIEW_PROBE_OUTPUT:-result/score/MNRF-ViewProbe/${TASK}/${MODE}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
SOURCE_DIR="${VIEW_PROBE_SOURCE_DIR:-}"
VIEWS="${VIEW_PROBE_VIEWS:-S,E,P,R}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
RESUME_FROM="${VIEW_PROBE_RESUME_FROM:-}"

if [[ "${GPU_ID}" == "cpu" ]]; then
  DEVICE=cpu
  export CUDA_VISIBLE_DEVICES=""
else
  [[ "${GPU_ID}" == "0" ]] || { printf 'GPU_ID must be 0 or cpu on this host\n' >&2; exit 2; }
  DEVICE=cuda:0
fi

if [[ "${VIEW_PROBE_OUTPUT}" = /* ]]; then
  OUTPUT_PATH="${VIEW_PROBE_OUTPUT}"
else
  OUTPUT_PATH="${ROOT}/${VIEW_PROBE_OUTPUT}"
fi
if [[ -n "${SOURCE_DIR}" && "${SOURCE_DIR}" != /* ]]; then
  SOURCE_PATH="${ROOT}/${SOURCE_DIR}"
else
  SOURCE_PATH="${SOURCE_DIR}"
fi
if [[ -n "${RESUME_FROM}" && "${RESUME_FROM}" != /* ]]; then
  RESUME_PATH="${ROOT}/${RESUME_FROM}"
else
  RESUME_PATH="${RESUME_FROM}"
fi

exec python "${ROOT}/scripts/run_mnrf_view_probe.py" \
  --task "${TASK}" --data-source "${SOURCE}" --mode "${MODE}" \
  --views "${VIEWS}" --source-dir "${SOURCE_PATH}" --output-dir "${OUTPUT_PATH}" \
  --seed "${SEED}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" --stage1-epochs "${STAGE1_EPOCHS}" \
  --pattern-len "${PATTERN_LEN:-32}" --hidden-dim "${HIDDEN_DIM:-32}" \
  --stage2-epochs "${STAGE2_EPOCHS}" \
  --patience "${PATIENCE}" --device "${DEVICE}" \
  --evolution-dim "${EVOLUTION_DIM:-8}" --pattern-dim "${PATTERN_DIM:-8}" \
  --relation-dim "${RELATION_DIM:-8}" --cache-chunk-size "${CACHE_CHUNK_SIZE:-256}" \
  --encoding-batch-size "${ENCODING_BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE:-0.001}" --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --feature-std-floor "${FEATURE_STD_FLOOR:-0.05}" --score-mad-floor "${SCORE_MAD_FLOOR:-0.001}" \
  --calibration-quantile "${CALIBRATION_QUANTILE:-0.995}" \
  --gmm-components "${GMM_COMPONENTS:-4}" --gmm-covariance-type "${GMM_COVARIANCE_TYPE:-full}" \
  --gmm-reg-covar "${GMM_REG_COVAR:-0.001}" --gmm-n-init "${GMM_N_INIT:-1}" \
  --gmm-max-iter "${GMM_MAX_ITER:-100}" --gmm-fit-max-windows "${GMM_FIT_MAX_WINDOWS:-20000}" \
  --train-noise-std "${TRAIN_NOISE_STD:-0.01}" --noise-seed "${NOISE_SEED:-20261013}" \
  --cpu-threads "${CPU_THREADS:-1}" --resume-from "${RESUME_PATH}"
