#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
export MNRF_MSL_SEQUENCE_MAPPING="${ROOT}/result/analysis/MNRF-ViewProbe-DataRepair/msl-raw-mapping-20261031.json"
export MNRF_SCORE_TARGET="col_0"
export MNRF_DATA_VERSION="MSL-sequence-bounded-col0"
export MNRF_ARTIFACT_POLICY="${MNRF_ARTIFACT_POLICY:-readout}"
exec bash "${ROOT}/scripts/run_mnrf_lite_v2_task.sh" MSL large_detect "$@"
