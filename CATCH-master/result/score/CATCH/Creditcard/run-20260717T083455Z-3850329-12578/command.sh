#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/Creditcard/run-20260717T083455Z-3850329-12578
exec bash scripts/multivariate_detection/detect_score/Creditcard_script/CATCH.sh
