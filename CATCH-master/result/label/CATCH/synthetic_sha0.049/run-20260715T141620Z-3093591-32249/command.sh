#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/synthetic_sha0.049/run-20260715T141620Z-3093591-32249
exec bash scripts/multivariate_detection/detect_label/synthetic_sha0.049_script/CATCH.sh
