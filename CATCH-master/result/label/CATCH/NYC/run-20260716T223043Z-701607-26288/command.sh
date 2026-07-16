#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/NYC/run-20260716T223043Z-701607-26288
exec bash scripts/multivariate_detection/detect_label/NYC_script/CATCH.sh
