#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_daphnet_gecco_tslib_baselines.sh"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
GPU="${GPU:-0}"
SESSION_PREFIX="${SESSION_PREFIX:-tslib_missing}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"

launch_dataset() {
  local dataset="$1"
  local dataset_tag="${dataset%.csv}"
  local session="${SESSION_PREFIX}_${dataset_tag,,}"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    return 1
  fi

  tmux new-session -d -s "${session}" \
    "cd '${PROJECT_ROOT}' && SKIP_EXISTING='${SKIP_EXISTING}' GPU='${GPU}' RUN_TAG='${RUN_TAG}' bash '${RUNNER}' '${dataset}'"
  echo "started ${session}: ${dataset}"
}

launch_dataset "Daphnet.csv"
launch_dataset "GECCO.csv"

echo "check sessions: tmux ls | grep '${SESSION_PREFIX}'"
echo "attach: tmux attach -t ${SESSION_PREFIX}_daphnet"
