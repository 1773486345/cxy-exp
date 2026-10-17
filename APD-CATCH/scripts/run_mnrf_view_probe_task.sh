#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?task name required}"
SOURCE="${2:-large_detect}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

MODE="${VIEW_PROBE_MODE:-full_gmm}"
SEED="${SEED:-2021}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
PATIENCE="${PATIENCE:-5}"
GPU_ID="${GPU_ID:-0}"
VIEW_PROBE_OUTPUT="${VIEW_PROBE_OUTPUT:-result/score/MNRF-ViewProbe/${TASK}/${MODE}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
SOURCE_DIR="${VIEW_PROBE_SOURCE_DIR:-}"
VIEWS="${VIEW_PROBE_VIEWS:-S,E,P,R}"

if [[ "${GPU_ID}" == "cpu" ]]; then
  DEVICE=cpu
  export CUDA_VISIBLE_DEVICES=""
else
  [[ "${GPU_ID}" == "0" ]] || { printf 'GPU_ID must be 0 or cpu on this host\n' >&2; exit 2; }
  DEVICE=cuda:0
fi

exec python "${ROOT}/scripts/run_mnrf_view_probe.py" \
  --task "${TASK}" --data-source "${SOURCE}" --mode "${MODE}" \
  --views "${VIEWS}" --source-dir "${SOURCE_DIR}" --output-dir "${ROOT}/${VIEW_PROBE_OUTPUT}" \
  --seed "${SEED}" --batch-size "${BATCH_SIZE}" --stage1-epochs "${STAGE1_EPOCHS}" \
  --patience "${PATIENCE}" --device "${DEVICE}" \
  --evolution-dim "${EVOLUTION_DIM:-8}" --pattern-dim "${PATTERN_DIM:-8}" \
  --relation-dim "${RELATION_DIM:-8}" --cache-chunk-size "${CACHE_CHUNK_SIZE:-256}" \
  --encoding-batch-size "${ENCODING_BATCH_SIZE:-${BATCH_SIZE}}" \
  --gmm-components "${GMM_COMPONENTS:-4}" --gmm-fit-max-windows "${GMM_FIT_MAX_WINDOWS:-20000}" \
  --cpu-threads "${CPU_THREADS:-1}"
