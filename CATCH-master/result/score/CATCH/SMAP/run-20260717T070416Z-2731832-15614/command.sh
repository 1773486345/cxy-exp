#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/SMAP/run-20260717T070416Z-2731832-15614
exec bash scripts/multivariate_detection/detect_score/SMAP_script/CATCH.sh
