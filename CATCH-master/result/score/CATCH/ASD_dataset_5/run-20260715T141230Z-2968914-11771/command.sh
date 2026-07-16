#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/ASD_dataset_5/run-20260715T141230Z-2968914-11771
exec bash scripts/multivariate_detection/detect_score/ASD_dataset_5_script/CATCH.sh
