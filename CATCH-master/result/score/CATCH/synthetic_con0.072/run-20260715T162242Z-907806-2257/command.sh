#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/synthetic_con0.072/run-20260715T162242Z-907806-2257
exec bash scripts/multivariate_detection/detect_score/synthetic_con0.072_script/CATCH.sh
