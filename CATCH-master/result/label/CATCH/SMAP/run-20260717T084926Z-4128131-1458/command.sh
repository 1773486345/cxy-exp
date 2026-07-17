#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=label/CATCH/SMAP/run-20260717T084926Z-4128131-1458
exec bash scripts/multivariate_detection/detect_label/SMAP_script/CATCH.sh
