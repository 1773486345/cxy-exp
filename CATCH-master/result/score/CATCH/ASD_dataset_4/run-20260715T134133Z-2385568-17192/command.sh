#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/ASD_dataset_4/run-20260715T134133Z-2385568-17192
exec bash scripts/multivariate_detection/detect_score/ASD_dataset_4_script/CATCH.sh
