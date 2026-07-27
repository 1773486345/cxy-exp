#!/usr/bin/env bash
set -euo pipefail

# Source CATCH official_CATCH.sh: /media/h3c/users/wangyueyang1/cxy/CATCH-master/result/score/CATCH/MSL/run-20260716T213635Z-58595-28964/official_CATCH.sh
# Source CATCH test report: /media/h3c/users/wangyueyang1/cxy/CATCH-master/result/score/CATCH/MSL/run-20260716T213635Z-58595-28964/test_report.1784244404.h3c-R5500-G5.59126.csv
# Source CATCH archive commit: a3ba73c56101778ac3df060814ae29d811fb31fb
# Configuration audit date (UTC): 2026-07-27
# GECCO fairness override: false
# TypeFusion compatibility override: none

ROOT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1
  pwd
)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAVE_PATH="${TYPEFUSION_SAVE_PATH:-score/TypeFusion-CATCH/MSL/run-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
MODEL_HYPER_PARAMS='{"batch_size":__BATCH_SIZE__,"branch_dim":128,"catch_train_epochs":5,"cf_dim":64,"d_ff":256,"d_model":128,"dropout":0.2,"e_layers":3,"fit_mode":"three_stage","fusion_heads":4,"fusion_layers":2,"head_dim":64,"joint_finetune_lr_scale":0.1,"lambda_freq":0.1,"lambda_mask":0.1,"lr":0.0005,"memory_size":32,"memory_topk":4,"n_heads":2,"patch_size":16,"patch_stride":8,"patience":3,"pattern_mask_ratio":0.25,"relation_mask_groups":4,"seed":2021,"seq_len":192,"temporal_hidden_dim":128,"temporal_layers":3,"training_budget_mode":"equal_total_steps"}'
MODEL_HYPER_PARAMS="${MODEL_HYPER_PARAMS/__BATCH_SIZE__/${BATCH_SIZE}}"
RUN_CONFIG_PATH="$ROOT_DIR/result/$SAVE_PATH/typefusion_run_config.json"
mkdir -p "$(dirname "$RUN_CONFIG_PATH")"
printf '{"data_name":"MSL.csv","model_name":"typefusion_catch.TypeFusionCATCH","seed":2021,"model_hyper_params":%s}\n' "$MODEL_HYPER_PARAMS" > "$RUN_CONFIG_PATH"
export TYPEFUSION_RUN_CONFIG_PATH="$RUN_CONFIG_PATH"

exec python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "MSL.csv" --model-name "typefusion_catch.TypeFusionCATCH" --model-hyper-params "$MODEL_HYPER_PARAMS" --seed 2021 --gpus "$GPU_ID" --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH"
