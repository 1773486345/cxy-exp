#!/usr/bin/env bash
set -euo pipefail

# Prepared only: paired CATCH baseline for TypeFusion GECCO seq_len=192 fairness.
# Source CATCH official_CATCH.sh: /media/h3c/users/wangyueyang1/cxy/CATCH-master/result/score/CATCH/GECCO/run-20260716T213302Z-16971-16047/official_CATCH.sh
# Source CATCH archive commit: a3ba73c56101778ac3df060814ae29d811fb31fb
# Configuration audit date (UTC): 2026-07-27

ROOT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1
  pwd
)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAVE_PATH="${CATCH_SAVE_PATH:-score/CATCH/GECCO_fair_seq192/run-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
MODEL_HYPER_PARAMS='{"Mlr":0.0001,"anomaly_ratio":1,"batch_size":__BATCH_SIZE__,"cf_dim":64,"d_ff":256,"d_model":128,"dropout":0.2,"e_layers":2,"head_dim":64,"head_dropout":0.1,"itr":1,"lr":0.0001,"n_heads":2,"num_epochs":1,"patch_size":16,"patch_stride":8,"patience":10,"seq_len":192,"small_kernel_merged":"False","temperature":0.1,"use_multi_scale":"False"}'
MODEL_HYPER_PARAMS="${MODEL_HYPER_PARAMS/__BATCH_SIZE__/${BATCH_SIZE}}"

exec python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "GECCO.csv" --model-name "catch.CATCH" --model-hyper-params "$MODEL_HYPER_PARAMS" --seed 2021 --gpus "$GPU_ID" --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH"
