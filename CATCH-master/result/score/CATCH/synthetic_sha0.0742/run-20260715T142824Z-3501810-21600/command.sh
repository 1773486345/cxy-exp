#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/synthetic_sha0.0742/run-20260715T142824Z-3501810-21600
exec bash scripts/multivariate_detection/detect_score/synthetic_sha0.0742_script/CATCH.sh
