#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/synthetic_sub_mix0.089/run-20260715T165538Z-1259802-14879
exec bash scripts/multivariate_detection/detect_label/synthetic_sub_mix0.089_script/CATCH.sh
