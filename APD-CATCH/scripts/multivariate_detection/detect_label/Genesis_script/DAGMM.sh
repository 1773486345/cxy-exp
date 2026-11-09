#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
exec python scripts/run_archived_baseline.py \
  --task Genesis --entry label --method-path DAGMM --model-name merlion.DAGMM \
  --model-hyper-params '{"gmm_k":3,"hidden_size":5,"sequence_len":1,"lambda_energy":0.1,"lambda_cov_diag":0.005,"lr":0.001,"batch_size":256,"num_epochs":10,"max_score":1000,"enable_calibrator":true,"enable_threshold":true}' \
  --config-path config/unfixed_detect_label_multi_config.json \
  --startup-script "${BASH_SOURCE[0]}" --upstream "salesforce-merlion==1.3.1; DAGMMConfig defaults explicit" \
  --point-artifact "${BASELINE_POINT_ARTIFACT:-compact}" \
  --min-free-gib "${BASELINE_MIN_FREE_GIB:-4}"
