#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/ASD_dataset_12/run-20260715T130926Z-2024314-1748
exec bash scripts/multivariate_detection/detect_label/ASD_dataset_12_script/CATCH.sh
