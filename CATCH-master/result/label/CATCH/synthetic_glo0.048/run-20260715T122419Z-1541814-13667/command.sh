#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/synthetic_glo0.048/run-20260715T122419Z-1541814-13667
exec bash scripts/multivariate_detection/detect_label/synthetic_glo0.048_script/CATCH.sh
