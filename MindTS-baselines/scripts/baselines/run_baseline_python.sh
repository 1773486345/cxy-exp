#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/media/h3c/users/shared_app/miniconda3/bin/conda}"
BASELINE_ENV_PREFIX="${BASELINE_ENV_PREFIX:-/media/h3c/users/wangyueyang1/cxy/.env/envs/baseline_env}"

if [ ! -x "${CONDA_BIN}" ]; then
  echo "missing conda executable: ${CONDA_BIN}" >&2
  exit 2
fi
if [ ! -x "${BASELINE_ENV_PREFIX}/bin/python" ]; then
  echo "missing baseline environment: ${BASELINE_ENV_PREFIX}" >&2
  exit 2
fi

exec "${CONDA_BIN}" run -p "${BASELINE_ENV_PREFIX}" --no-capture-output python "$@"
