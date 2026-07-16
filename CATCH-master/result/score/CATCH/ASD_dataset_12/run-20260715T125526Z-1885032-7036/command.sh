#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/ASD_dataset_12/run-20260715T125526Z-1885032-7036
exec bash scripts/multivariate_detection/detect_score/ASD_dataset_12_script/CATCH.sh
