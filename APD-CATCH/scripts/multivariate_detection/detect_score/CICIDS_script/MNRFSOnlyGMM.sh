#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${ROOT}"
TASK=CICIDS; SOURCE_RUN=full_gmm-full-20261021T112642Z-2510633; SEQ_LEN=192; BATCH_SIZE=128
SOURCE_DIR="${ROOT}/result/score/MNRF-ViewProbe/${TASK}/${SOURCE_RUN}"
RUN_ID="s_only-gmm-$(date -u +%Y%m%dT%H%M%SZ)-$$"
OUTPUT_DIR="${ROOT}/result/score/MNRF-ViewProbe-SOnly/${TASK}/${RUN_ID}"
mkdir -p "$(dirname -- "${OUTPUT_DIR}")"
[[ ! -e "${OUTPUT_DIR}" ]] || { printf 'refusing existing output: %s\n' "${OUTPUT_DIR}" >&2; exit 2; }
python "${ROOT}/scripts/run_mnrf_view_probe.py" --task "${TASK}" --data-source large_detect --mode s_only --views S --source-dir "${SOURCE_DIR}" --output-dir "${OUTPUT_DIR}" --seed 2021 --seq-len "${SEQ_LEN}" --pattern-len 32 --hidden-dim 32 --batch-size "${BATCH_SIZE}" --stage1-epochs 50 --stage2-epochs 30 --patience 5 --learning-rate 0.001 --weight-decay 0.0001 --device cpu --cache-chunk-size 256 --encoding-batch-size "${BATCH_SIZE}" --view-dim 8 --evolution-dim 8 --pattern-dim 8 --relation-dim 8 --gmm-components 4 --gmm-covariance-type full --gmm-reg-covar 0.001 --gmm-n-init 1 --gmm-max-iter 100 --gmm-fit-max-windows 20000 --train-noise-std 0.01 --noise-seed 20261013 --feature-std-floor 0.05 --score-mad-floor 0.001 --calibration-quantile 0.995 --cpu-threads 1 --artifact-policy full --log-every-steps 100 --log-every-seconds 30 --gmm-log-every-iters 10 --startup-script "${BASH_SOURCE[0]}"
