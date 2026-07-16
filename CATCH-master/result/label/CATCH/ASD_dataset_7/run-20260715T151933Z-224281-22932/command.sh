#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/ASD_dataset_7/run-20260715T151933Z-224281-22932
exec bash scripts/multivariate_detection/detect_label/ASD_dataset_7_script/CATCH.sh
