#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/synthetic_con0.072/run-20260715T170813Z-1387983-784
exec bash scripts/multivariate_detection/detect_label/synthetic_con0.072_script/CATCH.sh
