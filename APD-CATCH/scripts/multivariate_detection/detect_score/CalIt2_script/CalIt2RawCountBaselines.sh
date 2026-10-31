#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DATA_ROOT="${ROOT}/result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031/data_root"
SOURCE_DIR="${ROOT}/result/score/MNRF-ViewProbe-RawCount/CalIt2/rawcount-full-20261031T20261031T092320Z-1093827"
OUTPUT_DIR="${ROOT}/result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-$(date -u +%Y%m%dT%H%M%SZ)-$$/baselines"
python "${ROOT}/scripts/run_calit2_rawcount_baselines.py" --source-dir "${SOURCE_DIR}" --data-root "${DATA_ROOT}" --output-dir "${OUTPUT_DIR}"
printf 'baseline_analysis_dir=%s\n' "${OUTPUT_DIR}"
