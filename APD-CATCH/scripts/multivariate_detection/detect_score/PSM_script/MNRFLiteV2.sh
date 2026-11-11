#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
export MNRF_SEQ_LEN=192 SEED=2021 BATCH_SIZE=128 ENCODING_BATCH_SIZE=128
export MNRF_SCORE_READOUT=flow
export MNRF_ARTIFACT_POLICY="${MNRF_ARTIFACT_POLICY:-benchmark}"
export MNRF_DATA_ROOT="${ROOT}/dataset/anomaly_detect"
export MNRF_DATA_VERSION=PSM-benchmark-original
export MNRF_EXPECTED_TEST=87650 MNRF_EXPECTED_CALIBRATION=13020
exec bash "${ROOT}/scripts/run_mnrf_lite_v2_task.sh" PSM large_detect "${@}"
