#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/SWAT/run-20260715T184743Z-2499166-18894
exec bash scripts/multivariate_detection/detect_score/SWAT_script/CATCH.sh
