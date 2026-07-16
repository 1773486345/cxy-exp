#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/PSM/run-20260716T223734Z-776688-6344
exec bash scripts/multivariate_detection/detect_label/PSM_script/CATCH.sh
