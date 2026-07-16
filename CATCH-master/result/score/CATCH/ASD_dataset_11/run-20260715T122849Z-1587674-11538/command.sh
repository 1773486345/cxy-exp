#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/ASD_dataset_11/run-20260715T122849Z-1587674-11538
exec bash scripts/multivariate_detection/detect_score/ASD_dataset_11_script/CATCH.sh
