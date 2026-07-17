#!/usr/bin/env bash
set -euo pipefail

# Uses the original CATCH score settings and the common benchmark split/seed.
python_bin="${PYTHON_BIN:-/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python}"
save_path="${MSD_CATCH_SAVE_PATH:-score/MSD-CATCH}"
seed="${MSD_CATCH_SEED:-2021}"

run() {
  local dataset="$1"
  local params="$2"
  "$python_bin" scripts/run_benchmark.py \
    --config-path unfixed_detect_score_multi_config.json \
    --data-name-list "$dataset" \
    --model-name msd_catch.MSDCATCH \
    --model-hyper-params "$params" \
    --gpus 0 --num-workers 1 --timeout 60000 --seed "$seed" \
    --save-path "$save_path/$dataset"
}

run PSM.csv '{"Mlr": 0.001, "auxi_lambda": 0.01, "batch_size": 128, "cf_dim": 16, "d_ff": 32, "d_model": 16, "dc_lambda": 0.05, "dropout": 0.3, "e_layers": 1, "head_dim": 32, "inference_patch_size": 96, "lr": 0.005, "n_heads": 4, "num_epochs": 3, "patch_size": 16, "patch_stride": 8, "score_lambda": 0.5, "seq_len": 192}'
run Genesis.csv '{"Mlr": 0.0001, "batch_size": 4, "cf_dim": 64, "d_ff": 128, "d_model": 128, "e_layers": 3, "head_dim": 64, "inference_patch_size": 32, "inference_patch_stride": 1, "lr": 0.0001, "n_heads": 16, "num_epochs": 10, "patch_size": 8, "patch_stride": 8, "score_lambda": 1, "seq_len": 192, "temperature": 0.07}'
run CalIt2.csv '{"Mlr": 1e-05, "auxi_lambda": 0.1, "batch_size": 128, "cf_dim": 128, "d_ff": 128, "d_model": 128, "dc_lambda": 0.5, "e_layers": 2, "head_dim": 32, "inference_patch_size": 16, "lr": 0.0001, "n_heads": 8, "num_epochs": 10, "patch_size": 16, "patch_stride": 16, "score_lambda": 0.5, "seq_len": 192, "temperature": 0.07}'
