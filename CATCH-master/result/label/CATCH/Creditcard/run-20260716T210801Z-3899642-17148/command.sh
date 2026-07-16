#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/Creditcard/run-20260716T210801Z-3899642-17148
exec bash scripts/multivariate_detection/detect_label/Creditcard_script/CATCH.sh
