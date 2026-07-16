#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/synthetic_sub_mix0.089/run-20260715T164351Z-1135210-3936
exec bash scripts/multivariate_detection/detect_score/synthetic_sub_mix0.089_script/CATCH.sh
