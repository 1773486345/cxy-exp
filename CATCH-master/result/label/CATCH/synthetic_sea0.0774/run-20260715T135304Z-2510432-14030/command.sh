#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/synthetic_sea0.0774/run-20260715T135304Z-2510432-14030
exec bash scripts/multivariate_detection/detect_label/synthetic_sea0.0774_script/CATCH.sh
