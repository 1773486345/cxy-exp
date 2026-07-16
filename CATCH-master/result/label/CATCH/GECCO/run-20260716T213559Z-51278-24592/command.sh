#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/GECCO/run-20260716T213559Z-51278-24592
exec bash scripts/multivariate_detection/detect_label/GECCO_script/CATCH.sh
