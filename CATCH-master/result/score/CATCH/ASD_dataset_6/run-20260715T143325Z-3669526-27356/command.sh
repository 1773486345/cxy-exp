#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/ASD_dataset_6/run-20260715T143325Z-3669526-27356
exec bash scripts/multivariate_detection/detect_score/ASD_dataset_6_script/CATCH.sh
