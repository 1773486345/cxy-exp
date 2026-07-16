#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/synthetic_glo0.048/run-20260715T121140Z-1409853-4714
exec bash scripts/multivariate_detection/detect_score/synthetic_glo0.048_script/CATCH.sh
