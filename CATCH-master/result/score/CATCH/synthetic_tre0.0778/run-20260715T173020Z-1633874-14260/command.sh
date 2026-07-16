#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/synthetic_tre0.0778/run-20260715T173020Z-1633874-14260
exec bash scripts/multivariate_detection/detect_score/synthetic_tre0.0778_script/CATCH.sh
