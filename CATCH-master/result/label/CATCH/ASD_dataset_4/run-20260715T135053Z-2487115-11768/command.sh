#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/ASD_dataset_4/run-20260715T135053Z-2487115-11768
exec bash scripts/multivariate_detection/detect_label/ASD_dataset_4_script/CATCH.sh
