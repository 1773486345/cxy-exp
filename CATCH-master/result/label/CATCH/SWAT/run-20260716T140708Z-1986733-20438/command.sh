#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/SWAT/run-20260716T140708Z-1986733-20438
exec bash scripts/multivariate_detection/detect_label/SWAT_script/CATCH.sh
