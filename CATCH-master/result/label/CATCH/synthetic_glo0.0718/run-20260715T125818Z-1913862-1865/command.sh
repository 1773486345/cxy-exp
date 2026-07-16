#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/synthetic_glo0.0718/run-20260715T125818Z-1913862-1865
exec bash scripts/multivariate_detection/detect_label/synthetic_glo0.0718_script/CATCH.sh
