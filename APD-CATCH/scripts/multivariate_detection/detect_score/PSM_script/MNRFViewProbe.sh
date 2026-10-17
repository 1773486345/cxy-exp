#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
exec bash "${ROOT}/scripts/run_mnrf_view_probe_task.sh" PSM large_detect
