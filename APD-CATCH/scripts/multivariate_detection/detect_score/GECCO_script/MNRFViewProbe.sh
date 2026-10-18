#!/usr/bin/env bash
set -euo pipefail

# Independent MNRF ViewProbe entry. CATCH reference: seq_len=96, batch_size=128,
# num_epochs=1, lr=0.0001, Mlr=0.0001. This is a new MNRF identity.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${ROOT}"
TASK="GECCO"; DATA_SOURCE="large_detect"; MODE="full_gmm"; VIEWS="S,E,P,R"; SEED="2021"
SEQ_LEN="96"; BATCH_SIZE="128"; ENCODING_BATCH_SIZE="128"; STAGE1_EPOCHS="50"; STAGE2_EPOCHS="30"; PATIENCE="5"
LEARNING_RATE="0.001"; WEIGHT_DECAY="0.0001"; PATTERN_LEN="32"; HIDDEN_DIM="32"; VIEW_DIM="8"; EVOLUTION_DIM="8"; PATTERN_DIM="8"; RELATION_DIM="8"
CACHE_CHUNK_SIZE="256"; GMM_COMPONENTS="4"; GMM_COVARIANCE_TYPE="full"; GMM_REG_COVAR="0.001"; GMM_N_INIT="1"; GMM_MAX_ITER="100"; GMM_FIT_MAX_WINDOWS="20000"
TRAIN_NOISE_STD="0.01"; NOISE_SEED="20261013"; FEATURE_STD_FLOOR="0.05"; SCORE_MAD_FLOOR="0.001"; CALIBRATION_QUANTILE="0.995"; CPU_THREADS="1"
DEVICE="cuda:0"; SOURCE_DIR=""; RESUME_FROM=""; OUTPUT_DIR="${ROOT}/result/score/MNRF-ViewProbe/${TASK}/${MODE}-$(date -u +%Y%m%dT%H%M%SZ)-$$"

python "${ROOT}/scripts/mnrf_lite_v2_preflight.py" --task "${TASK}" --data-source "${DATA_SOURCE}" --batch-size "${BATCH_SIZE}" --seq-len "${SEQ_LEN}" --encoding-batch-size "${ENCODING_BATCH_SIZE}" --view-dim "${VIEW_DIM}" --evolution-dim "${EVOLUTION_DIM}" --pattern-dim "${PATTERN_DIM}" --relation-dim "${RELATION_DIM}"
python "${ROOT}/scripts/run_mnrf_view_probe.py" --task "${TASK}" --data-source "${DATA_SOURCE}" --mode "${MODE}" --views "${VIEWS}" --source-dir "${SOURCE_DIR}" --resume-from "${RESUME_FROM}" --output-dir "${OUTPUT_DIR}" --seed "${SEED}" --seq-len "${SEQ_LEN}" --pattern-len "${PATTERN_LEN}" --hidden-dim "${HIDDEN_DIM}" --batch-size "${BATCH_SIZE}" --stage1-epochs "${STAGE1_EPOCHS}" --stage2-epochs "${STAGE2_EPOCHS}" --patience "${PATIENCE}" --learning-rate "${LEARNING_RATE}" --weight-decay "${WEIGHT_DECAY}" --device "${DEVICE}" --cache-chunk-size "${CACHE_CHUNK_SIZE}" --encoding-batch-size "${ENCODING_BATCH_SIZE}" --view-dim "${VIEW_DIM}" --evolution-dim "${EVOLUTION_DIM}" --pattern-dim "${PATTERN_DIM}" --relation-dim "${RELATION_DIM}" --gmm-components "${GMM_COMPONENTS}" --gmm-covariance-type "${GMM_COVARIANCE_TYPE}" --gmm-reg-covar "${GMM_REG_COVAR}" --gmm-n-init "${GMM_N_INIT}" --gmm-max-iter "${GMM_MAX_ITER}" --gmm-fit-max-windows "${GMM_FIT_MAX_WINDOWS}" --train-noise-std "${TRAIN_NOISE_STD}" --noise-seed "${NOISE_SEED}" --feature-std-floor "${FEATURE_STD_FLOOR}" --score-mad-floor "${SCORE_MAD_FLOOR}" --calibration-quantile "${CALIBRATION_QUANTILE}" --cpu-threads "${CPU_THREADS}"
