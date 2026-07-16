#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/NYC/run-20260716T222643Z-657156-32728
exec bash scripts/multivariate_detection/detect_score/NYC_script/CATCH.sh
