#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/MSL/run-20260716T213635Z-58595-28964
exec bash scripts/multivariate_detection/detect_score/MSL_script/CATCH.sh
