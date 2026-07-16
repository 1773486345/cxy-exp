#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/MSL/run-20260716T232646Z-1365820-1457
exec bash scripts/multivariate_detection/detect_label/MSL_script/CATCH.sh
