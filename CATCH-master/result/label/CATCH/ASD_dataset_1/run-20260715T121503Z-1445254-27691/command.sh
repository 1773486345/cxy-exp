#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/ASD_dataset_1/run-20260715T121503Z-1445254-27691
exec bash scripts/multivariate_detection/detect_label/ASD_dataset_1_script/CATCH.sh
