#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/synthetic_sha0.0742/run-20260715T143933Z-3865058-23478
exec bash scripts/multivariate_detection/detect_label/synthetic_sha0.0742_script/CATCH.sh
