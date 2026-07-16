#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/PSM/run-20260716T222700Z-660928-22488
exec bash scripts/multivariate_detection/detect_score/PSM_script/CATCH.sh
