#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/ASD_dataset_2/run-20260715T124813Z-1811546-28448
exec bash scripts/multivariate_detection/detect_label/ASD_dataset_2_script/CATCH.sh
