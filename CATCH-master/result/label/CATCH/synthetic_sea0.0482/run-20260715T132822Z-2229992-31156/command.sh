#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/synthetic_sea0.0482/run-20260715T132822Z-2229992-31156
exec bash scripts/multivariate_detection/detect_label/synthetic_sea0.0482_script/CATCH.sh
