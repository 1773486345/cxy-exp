#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 GPU_ID [DATASET ...]" >&2
  echo "Example: $0 0 SMD_machine-1-1.csv.gz Weather.csv Energy.csv" >&2
  exit 2
fi

GPU_ID=$1
shift
DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=(SMD_machine-1-1.csv.gz Weather.csv Energy.csv)
fi

RELATION_MODE=${RELATION_MODE:-full}
SEED=${SEED:-2021}
case "$RELATION_MODE" in
  full|single_scale|no_graph) ;;
  *)
    echo "RELATION_MODE must be full, single_scale, or no_graph." >&2
    exit 2
    ;;
esac

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python}
RUN_NAME=${RUN_NAME:-mgad_real_v1_${RELATION_MODE}_seed${SEED}}
RESULT_DIR="$ROOT/result/$RUN_NAME"

if [[ -e "$RESULT_DIR" ]]; then
  echo "Refusing to overwrite an existing result directory: $RESULT_DIR" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is unavailable: $PYTHON_BIN" >&2
  exit 1
fi

MODEL_PARAMS=$(printf '%s' '{"device":"cuda:0","seq_len":64,"d_model":64,"graph_dim":24,"d_ff":128,"n_heads":4,"e_layers":2,"dropout":0.1,"temporal_kernels":[1,5,11],"relation_mode":"'"$RELATION_MODE"'","graph_target_chunk_size":16,"batch_size":32,"score_conditioning_batch_size":256,"num_epochs":20,"patience":4,"learning_rate":0.001,"weight_decay":0.0001,"point_mask_ratio":0.12,"variable_block_mask_ratio":0.2,"max_mask_block_length":8,"branch_loss_weight":0.25,"relation_consistency_weight":0.05,"score_top_k":3}')

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/run_benchmark.py \
  --config-path config/mechanism_graph_ad/real_benchmark_v1.json \
  --data-name-list "${DATASETS[@]}" \
  --model-name MechanismGraphAD.MechanismGraphAD \
  --model-hyper-params "$MODEL_PARAMS" \
  --seed "$SEED" \
  --eval-backend sequential \
  --num-workers 1 \
  --num-cpus 3 \
  --timeout 21600 \
  --save-path "$RUN_NAME"
