#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/CalIt2/run-20260715T184135Z-2437260-7989
exec bash scripts/multivariate_detection/detect_label/CalIt2_script/CATCH.sh
