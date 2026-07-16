#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/synthetic_sha0.049/run-20260715T140518Z-2723903-7715
exec bash scripts/multivariate_detection/detect_score/synthetic_sha0.049_script/CATCH.sh
