#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/ASD_dataset_5/run-20260715T142327Z-3331415-3799
exec bash scripts/multivariate_detection/detect_label/ASD_dataset_5_script/CATCH.sh
