#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/SMD/run-20260717T151403Z-1135547-9007
exec bash scripts/multivariate_detection/detect_label/SMD_script/CATCH.sh
