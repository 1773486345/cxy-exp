#!/usr/bin/env bash
set -euo pipefail

# Single-stage v2 script; public values mirror the corresponding CATCH task.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAVE_PATH="${TYPEFUSION_V2_SAVE_PATH:-score/TypeFusion-CATCH-V2/SMAP/run-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
MODEL_HYPER_PARAMS='{"batch_size":__BATCH_SIZE__,"branch_dim":128,"cf_dim":16,"d_ff":32,"d_model":64,"dropout":0.4,"e_layers":3,"joint_dim":128,"joint_heads":4,"joint_layers":2,"lambda_clean_score":0.1,"lambda_evidence":0.5,"lambda_pattern_frequency":0.1,"lambda_responsibility":0.5,"lambda_score":1.0,"lambda_score_rank":0.25,"lambda_state_usage":0.01,"lambda_synergy":0.25,"lr":0.005,"max_relation_attention_rows":2048,"n_heads":4,"num_epochs":10,"patch_size":16,"patch_stride":8,"patience":3,"prototype_temperature":1.0,"relation_correction_cap":2.0,"relation_mask_groups":4,"relation_temporal_kernel":5,"responsibility_margin":0.2,"score_margin":0.5,"seed":2021,"seq_len":192,"state_memory_size":32,"state_topk":4,"sufficient_temperature":1.0,"temporal_layers":3}'
MODEL_HYPER_PARAMS="${MODEL_HYPER_PARAMS/__BATCH_SIZE__/${BATCH_SIZE}}"
RUN_CONFIG_PATH="$ROOT_DIR/result/$SAVE_PATH/typefusion_v2_run_config.json"
mkdir -p "$(dirname "$RUN_CONFIG_PATH")"
printf '{"data_name":"SMAP.csv","model_name":"typefusion_catch_v2.TypeFusionCATCHV2","model_hyper_params":%s}\n' "$MODEL_HYPER_PARAMS" > "$RUN_CONFIG_PATH"
export TYPEFUSION_V2_RUN_CONFIG_PATH="$RUN_CONFIG_PATH"
exec python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "SMAP.csv" --model-name "typefusion_catch_v2.TypeFusionCATCHV2" --model-hyper-params "$MODEL_HYPER_PARAMS" --seed 2021 --gpus "$GPU_ID" --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH"
