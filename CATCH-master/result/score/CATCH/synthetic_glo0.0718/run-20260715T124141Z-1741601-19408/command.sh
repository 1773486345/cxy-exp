#!/usr/bin/env bash
set -euo pipefail
cd /media/h3c/users/wangyueyang1/cxy/CATCH-master
export CATCH_SAVE_PATH=score/CATCH/synthetic_glo0.0718/run-20260715T124141Z-1741601-19408
exec bash scripts/multivariate_detection/detect_score/synthetic_glo0.0718_script/CATCH.sh
