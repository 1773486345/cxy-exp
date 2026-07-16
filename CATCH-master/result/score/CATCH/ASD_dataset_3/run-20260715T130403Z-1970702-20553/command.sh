#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/ASD_dataset_3/run-20260715T130403Z-1970702-20553
exec bash scripts/multivariate_detection/detect_score/ASD_dataset_3_script/CATCH.sh
