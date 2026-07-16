#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/SMD/run-20260716T225633Z-1008371-29270
exec bash scripts/multivariate_detection/detect_score/SMD_script/CATCH.sh
