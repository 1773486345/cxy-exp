#!/usr/bin/env bash
set -euo pipefail

# Source CATCH official_CATCH.sh: /media/h3c/users/wangyueyang1/cxy/CATCH-master/result/score/CATCH/ASD_dataset_4/run-20260715T134133Z-2385568-17192/official_CATCH.sh
# Source CATCH test report: /media/h3c/users/wangyueyang1/cxy/CATCH-master/result/score/CATCH/ASD_dataset_4/run-20260715T134133Z-2385568-17192/test_report.1784123438.h3c-R5500-G5.2387988.csv
# Source CATCH archive commit: aedcc2512baa7e518829ec5d1a90c3e9837202c1
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
SAVE_PATH="${TYPEFUSION_SAVE_PATH:-score/TypeFusion-CATCH/ASD_dataset_4/run-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
MODEL_HYPER_PARAMS='{"batch_size":__BATCH_SIZE__,"branch_dim":128,"catch_train_epochs":10,"cf_dim":128,"d_ff":128,"d_model":128,"dropout":0.2,"e_layers":2,"fit_mode":"three_stage","fusion_heads":4,"fusion_layers":2,"head_dim":32,"joint_finetune_lr_scale":0.1,"lambda_freq":0.1,"lambda_mask":0.1,"lr":0.0001,"memory_size":32,"memory_topk":4,"n_heads":8,"patch_size":16,"patch_stride":16,"patience":3,"pattern_mask_ratio":0.25,"relation_mask_groups":4,"seed":2021,"seq_len":192,"temporal_hidden_dim":128,"temporal_layers":3,"training_budget_mode":"equal_total_steps"}'
MODEL_HYPER_PARAMS="${MODEL_HYPER_PARAMS/__BATCH_SIZE__/${BATCH_SIZE}}"
RUN_CONFIG_PATH="$ROOT_DIR/result/$SAVE_PATH/typefusion_run_config.json"
mkdir -p "$(dirname "$RUN_CONFIG_PATH")"
printf '{"data_name":"ASD_dataset_4.csv","model_name":"typefusion_catch.TypeFusionCATCH","seed":2021,"model_hyper_params":%s}\n' "$MODEL_HYPER_PARAMS" > "$RUN_CONFIG_PATH"
export TYPEFUSION_RUN_CONFIG_PATH="$RUN_CONFIG_PATH"

exec python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "ASD_dataset_4.csv" --model-name "typefusion_catch.TypeFusionCATCH" --model-hyper-params "$MODEL_HYPER_PARAMS" --seed 2021 --gpus "$GPU_ID" --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH"
