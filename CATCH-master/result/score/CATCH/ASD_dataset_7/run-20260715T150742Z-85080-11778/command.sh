#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/ASD_dataset_7/run-20260715T150742Z-85080-11778
exec bash scripts/multivariate_detection/detect_score/ASD_dataset_7_script/CATCH.sh
