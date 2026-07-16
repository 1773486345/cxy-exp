#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/ASD_dataset_11/run-20260715T124049Z-1715104-21708
exec bash scripts/multivariate_detection/detect_label/ASD_dataset_11_script/CATCH.sh
