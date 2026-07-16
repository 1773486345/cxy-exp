#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/Creditcard/run-20260716T202925Z-3426561-275
exec bash scripts/multivariate_detection/detect_score/Creditcard_script/CATCH.sh
