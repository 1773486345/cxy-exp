#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/synthetic_sub_mix0.0574/run-20260715T145436Z-4135816-12285
exec bash scripts/multivariate_detection/detect_score/synthetic_sub_mix0.0574_script/CATCH.sh
