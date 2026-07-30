#!/usr/bin/env bash
set -euo pipefail

# Public v2 script; configuration values mirror the corresponding CATCH script.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAVE_PATH="${TYPEFUSION_V2_SAVE_PATH:-score/TypeFusion-CATCH-V2/ASD_dataset_12/run-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
MODEL_HYPER_PARAMS='{"Mlr":0.00001,"anomaly_ratio":null,"auxi_lambda":0.005,"batch_size":__BATCH_SIZE__,"catch_train_epochs":10,"cf_dim":128,"d_ff":128,"d_model":128,"dc_lambda":0.005,"dropout":0.2,"e_layers":2,"head_dim":32,"inference_patch_size":32,"inference_patch_stride":1,"lr":0.0001,"n_heads":8,"patch_size":16,"patch_stride":16,"patience":3,"score_lambda":0.05,"seed":2021,"seq_len":192,"type_train_epochs":10,"joint_dim":128,"joint_layers":2,"joint_heads":4,"relation_mask_groups":4,"state_memory_size":32,"state_topk":4,"sufficient_temperature":1.0,"relation_correction_cap":2.0,"responsibility_margin":0.2,"score_margin":0.5,"synergy_margin":0.2}'
MODEL_HYPER_PARAMS="${MODEL_HYPER_PARAMS/__BATCH_SIZE__/${BATCH_SIZE}}"
RUN_CONFIG_PATH="$ROOT_DIR/result/$SAVE_PATH/typefusion_v2_run_config.json"
mkdir -p "$(dirname "$RUN_CONFIG_PATH")"
printf '{"data_name":"ASD_dataset_12.csv","model_name":"typefusion_catch_v2.TypeFusionCATCHV2","model_hyper_params":%s}\n' "$MODEL_HYPER_PARAMS" > "$RUN_CONFIG_PATH"
export TYPEFUSION_V2_RUN_CONFIG_PATH="$RUN_CONFIG_PATH"
exec python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "ASD_dataset_12.csv" --model-name "typefusion_catch_v2.TypeFusionCATCHV2" --model-hyper-params "$MODEL_HYPER_PARAMS" --seed 2021 --gpus "$GPU_ID" --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH"
