#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/Genesis/run-20260716T213208Z-6088-16524
exec bash scripts/multivariate_detection/detect_score/Genesis_script/CATCH.sh
