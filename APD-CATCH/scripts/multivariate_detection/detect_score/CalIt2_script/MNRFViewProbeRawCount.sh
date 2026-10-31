#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TASK="CalIt2"
DATA_ROOT="${ROOT}/result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031/data_root"
RUN_ID="rawcount-full-$(date -u +%Y%m%dT%H%M%SZ)-$$"
OUTPUT_DIR="${ROOT}/result/score/MNRF-ViewProbe-RawCount/${TASK}/${RUN_ID}"
ANALYSIS_DIR="${ROOT}/result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-$(date -u +%Y%m%dT%H%M%SZ)-$$"
STARTUP_LOG="$(mktemp "${TMPDIR:-/tmp}/calit2-rawcount-preflight.XXXXXX.log")"
trap 'rm -f "${STARTUP_LOG}"' EXIT

set +e
python "${ROOT}/scripts/mnrf_lite_v2_preflight.py" --task "${TASK}" --data-source large_detect \
  --data-root "${DATA_ROOT}" --batch-size 128 --seq-len 192 --encoding-batch-size 128 \
  --evolution-dim 8 --pattern-dim 8 --relation-dim 8 --view-dim 8 \
  --output-dir "${OUTPUT_DIR}" --mode full_gmm 2>&1 | tee "${STARTUP_LOG}"
PIPE_STATUS=(${PIPESTATUS[@]})
set -e
if [[ "${PIPE_STATUS[0]}" -ne 0 || "${PIPE_STATUS[1]}" -ne 0 ]]; then
  printf 'preflight failed: python=%s tee=%s\n' "${PIPE_STATUS[0]}" "${PIPE_STATUS[1]}" >&2
  exit "${PIPE_STATUS[0]}"
fi

python "${ROOT}/scripts/run_mnrf_view_probe.py" --task "${TASK}" --data-source large_detect \
  --data-root "${DATA_ROOT}" --mode full_gmm --views S,E,P,R --output-dir "${OUTPUT_DIR}" \
  --seed 2021 --seq-len 192 --pattern-len 32 --hidden-dim 32 --batch-size 128 \
  --stage1-epochs 50 --stage2-epochs 30 --patience 5 --learning-rate 0.001 --weight-decay 0.0001 \
  --feature-std-floor 0.05 --score-mad-floor 0.001 --calibration-quantile 0.995 --device cpu \
  --cache-chunk-size 256 --encoding-batch-size 128 --evolution-dim 8 --pattern-dim 8 --relation-dim 8 \
  --view-dim 8 --gmm-components 4 --gmm-covariance-type full --gmm-reg-covar 0.001 \
  --gmm-n-init 1 --gmm-max-iter 100 --gmm-fit-max-windows 20000 --cpu-threads 1 \
  --train-noise-std 0.01 --noise-seed 20261013 --artifact-policy full \
  --startup-log "${STARTUP_LOG}" --startup-script "$0"

python "${ROOT}/scripts/validate_mnrf_view_probe.py" "${OUTPUT_DIR}"
python "${ROOT}/scripts/export_mnrf_view_probe_analysis.py" --run-dir "${OUTPUT_DIR}" --output-dir "${ANALYSIS_DIR}"
printf 'output_dir=%s\nanalysis_dir=%s\n' "${OUTPUT_DIR}" "${ANALYSIS_DIR}"
