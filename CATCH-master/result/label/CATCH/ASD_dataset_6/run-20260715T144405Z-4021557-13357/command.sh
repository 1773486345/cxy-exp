#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/ASD_dataset_6/run-20260715T144405Z-4021557-13357
exec bash scripts/multivariate_detection/detect_label/ASD_dataset_6_script/CATCH.sh
