#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/ASD_dataset_1/run-20260715T120733Z-1347582-29581
exec bash scripts/multivariate_detection/detect_score/ASD_dataset_1_script/CATCH.sh
