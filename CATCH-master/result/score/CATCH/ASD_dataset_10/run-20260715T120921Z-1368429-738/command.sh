#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/ASD_dataset_10/run-20260715T120921Z-1368429-738
exec bash scripts/multivariate_detection/detect_score/ASD_dataset_10_script/CATCH.sh
