#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/Genesis/run-20260716T221132Z-484571-23537
exec bash scripts/multivariate_detection/detect_label/Genesis_script/CATCH.sh
