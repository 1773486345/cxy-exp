#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/CICIDS/run-20260715T133707Z-2321802-10349
exec bash scripts/multivariate_detection/detect_label/CICIDS_script/CATCH.sh
