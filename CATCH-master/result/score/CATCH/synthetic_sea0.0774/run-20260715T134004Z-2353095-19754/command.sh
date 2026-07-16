#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/synthetic_sea0.0774/run-20260715T134004Z-2353095-19754
exec bash scripts/multivariate_detection/detect_score/synthetic_sea0.0774_script/CATCH.sh
