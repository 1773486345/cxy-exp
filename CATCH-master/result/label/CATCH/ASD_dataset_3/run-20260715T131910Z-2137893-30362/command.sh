#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/ASD_dataset_3/run-20260715T131910Z-2137893-30362
exec bash scripts/multivariate_detection/detect_label/ASD_dataset_3_script/CATCH.sh
