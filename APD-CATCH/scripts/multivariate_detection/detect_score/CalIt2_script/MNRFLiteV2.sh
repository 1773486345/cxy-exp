#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
export MNRF_SEQ_LEN=192 SEED=2021 BATCH_SIZE=128 ENCODING_BATCH_SIZE=128
export MNRF_SCORE_READOUT=flow MNRF_ARTIFACT_POLICY=readout
export MNRF_DATA_ROOT="${ROOT}/result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031/data_root"
export MNRF_DATA_VERSION=CalIt2-RawCount-20261031
export MNRF_EXPECTED_FIELDS='["9","7"]' MNRF_EXPECTED_TEST=2329 MNRF_EXPECTED_CALIBRATION=24
exec bash "${ROOT}/scripts/run_mnrf_lite_v2_task.sh" CalIt2 large_detect "${@}"
