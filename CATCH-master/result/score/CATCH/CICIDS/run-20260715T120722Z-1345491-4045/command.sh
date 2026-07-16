#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/CICIDS/run-20260715T120722Z-1345491-4045
exec bash scripts/multivariate_detection/detect_score/CICIDS_script/CATCH.sh
