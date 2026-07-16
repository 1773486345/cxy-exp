#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/ASD_dataset_9/run-20260715T161140Z-790948-22025
exec bash scripts/multivariate_detection/detect_label/ASD_dataset_9_script/CATCH.sh
