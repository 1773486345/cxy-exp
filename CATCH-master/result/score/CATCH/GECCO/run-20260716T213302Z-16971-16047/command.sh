#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/GECCO/run-20260716T213302Z-16971-16047
exec bash scripts/multivariate_detection/detect_score/GECCO_script/CATCH.sh
