#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/CalIt2/run-20260715T183704Z-2374669-5874
exec bash scripts/multivariate_detection/detect_score/CalIt2_script/CATCH.sh
