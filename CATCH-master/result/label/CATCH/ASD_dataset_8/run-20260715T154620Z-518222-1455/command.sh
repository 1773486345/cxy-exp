#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/ASD_dataset_8/run-20260715T154620Z-518222-1455
exec bash scripts/multivariate_detection/detect_label/ASD_dataset_8_script/CATCH.sh
