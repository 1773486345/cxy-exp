#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/ASD_dataset_8/run-20260715T153406Z-377465-29501
exec bash scripts/multivariate_detection/detect_score/ASD_dataset_8_script/CATCH.sh
