#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/ASD_dataset_10/run-20260715T122027Z-1501546-7788
exec bash scripts/multivariate_detection/detect_label/ASD_dataset_10_script/CATCH.sh
