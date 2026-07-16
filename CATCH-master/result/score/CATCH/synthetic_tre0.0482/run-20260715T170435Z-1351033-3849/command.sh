#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/synthetic_tre0.0482/run-20260715T170435Z-1351033-3849
exec bash scripts/multivariate_detection/detect_score/synthetic_tre0.0482_script/CATCH.sh
