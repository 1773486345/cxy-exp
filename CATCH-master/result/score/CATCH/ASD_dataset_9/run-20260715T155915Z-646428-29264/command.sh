#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/ASD_dataset_9/run-20260715T155915Z-646428-29264
exec bash scripts/multivariate_detection/detect_score/ASD_dataset_9_script/CATCH.sh
