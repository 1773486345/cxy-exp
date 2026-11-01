#!/usr/bin/env bash
set -euo pipefail

# RawCount CalIt2 S-only diagnostic entry.  The fixed Full source and data
# root are intentionally explicit; this wrapper never trains an encoder.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${ROOT}"

TASK="CalIt2"
DATA_SOURCE="large_detect"
MODE="select_gmm"
VIEWS="S"
SEED="2021"
SEQ_LEN="192"
PATTERN_LEN="32"
HIDDEN_DIM="32"
BATCH_SIZE="128"
ENCODING_BATCH_SIZE="128"
STAGE1_EPOCHS="50"
STAGE2_EPOCHS="30"
PATIENCE="5"
LEARNING_RATE="0.001"
WEIGHT_DECAY="0.0001"
FEATURE_STD_FLOOR="0.05"
SCORE_MAD_FLOOR="0.001"
CALIBRATION_QUANTILE="0.995"
DEVICE="cpu"
CPU_THREADS="1"
CACHE_CHUNK_SIZE="256"
VIEW_DIM="8"
EVOLUTION_DIM="8"
PATTERN_DIM="8"
RELATION_DIM="8"
GMM_COMPONENTS="4"
GMM_COVARIANCE_TYPE="full"
GMM_REG_COVAR="0.001"
GMM_N_INIT="1"
GMM_MAX_ITER="100"
GMM_FIT_MAX_WINDOWS="20000"
TRAIN_NOISE_STD="0.01"
NOISE_SEED="20261013"
LOG_EVERY_STEPS="100"
LOG_EVERY_SECONDS="30"
GMM_LOG_EVERY_ITERS="10"
ARTIFACT_POLICY="full"

DATA_ROOT="${ROOT}/result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031/data_root"
SOURCE_DIR="${ROOT}/result/score/MNRF-ViewProbe-RawCount/CalIt2/rawcount-full-20261031T20261031T092320Z-1093827"
RUN_ID="rawcount-s-only-$(date -u +%Y%m%dT%H%M%SZ)-$$"
OUTPUT_DIR="${ROOT}/result/score/MNRF-ViewProbe-RawCount/CalIt2/${RUN_ID}"

python "${ROOT}/scripts/run_mnrf_view_probe.py" \
  --task "${TASK}" --data-source "${DATA_SOURCE}" --mode "${MODE}" --views "${VIEWS}" \
  --source-dir "${SOURCE_DIR}" --data-root "${DATA_ROOT}" --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" --seq-len "${SEQ_LEN}" --pattern-len "${PATTERN_LEN}" \
  --hidden-dim "${HIDDEN_DIM}" --batch-size "${BATCH_SIZE}" \
  --encoding-batch-size "${ENCODING_BATCH_SIZE}" --stage1-epochs "${STAGE1_EPOCHS}" \
  --stage2-epochs "${STAGE2_EPOCHS}" --patience "${PATIENCE}" \
  --learning-rate "${LEARNING_RATE}" --weight-decay "${WEIGHT_DECAY}" \
  --feature-std-floor "${FEATURE_STD_FLOOR}" --score-mad-floor "${SCORE_MAD_FLOOR}" \
  --calibration-quantile "${CALIBRATION_QUANTILE}" --device "${DEVICE}" \
  --cpu-threads "${CPU_THREADS}" --cache-chunk-size "${CACHE_CHUNK_SIZE}" \
  --view-dim "${VIEW_DIM}" --evolution-dim "${EVOLUTION_DIM}" \
  --pattern-dim "${PATTERN_DIM}" --relation-dim "${RELATION_DIM}" \
  --gmm-components "${GMM_COMPONENTS}" --gmm-covariance-type "${GMM_COVARIANCE_TYPE}" \
  --gmm-reg-covar "${GMM_REG_COVAR}" --gmm-n-init "${GMM_N_INIT}" \
  --gmm-max-iter "${GMM_MAX_ITER}" --gmm-fit-max-windows "${GMM_FIT_MAX_WINDOWS}" \
  --train-noise-std "${TRAIN_NOISE_STD}" --noise-seed "${NOISE_SEED}" \
  --log-every-steps "${LOG_EVERY_STEPS}" --log-every-seconds "${LOG_EVERY_SECONDS}" \
  --gmm-log-every-iters "${GMM_LOG_EVERY_ITERS}" --artifact-policy "${ARTIFACT_POLICY}" \
  --startup-script "${BASH_SOURCE[0]}"

printf 'output_dir=%s\n' "${OUTPUT_DIR}"
