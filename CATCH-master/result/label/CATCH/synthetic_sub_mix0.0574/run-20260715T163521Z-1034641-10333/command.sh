#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/synthetic_sub_mix0.0574/run-20260715T163521Z-1034641-10333
exec bash scripts/multivariate_detection/detect_label/synthetic_sub_mix0.0574_script/CATCH.sh
