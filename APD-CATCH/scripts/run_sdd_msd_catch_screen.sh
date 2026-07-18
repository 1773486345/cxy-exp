#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?usage: run_sdd_msd_catch_screen.sh DATASET}"
python_bin="${PYTHON_BIN:-/media/h3c/users/wangyueyang1/.conda/envs/catch_env/bin/python}"
output_dir="${SDD_MSD_CATCH_OUTPUT_DIR:-result/sdd_msd_catch_screen}"
seed="${SDD_MSD_CATCH_SEED:-2021}"

case "$dataset" in
  PSM)
    params='{"Mlr": 0.001, "auxi_lambda": 0.01, "batch_size": 128, "cf_dim": 16, "d_ff": 32, "d_model": 16, "dc_lambda": 0.05, "dropout": 0.3, "e_layers": 1, "head_dim": 32, "inference_patch_size": 96, "lr": 0.005, "n_heads": 4, "num_epochs": 3, "patch_size": 16, "patch_stride": 8, "score_lambda": 0.5, "seq_len": 192}'
    ;;
  Genesis)
    params='{"Mlr": 0.0001, "batch_size": 4, "cf_dim": 64, "d_ff": 128, "d_model": 128, "e_layers": 3, "head_dim": 64, "inference_patch_size": 32, "inference_patch_stride": 1, "lr": 0.0001, "n_heads": 16, "num_epochs": 10, "patch_size": 8, "patch_stride": 8, "score_lambda": 1, "seq_len": 192, "temperature": 0.07}'
    ;;
  GECCO)
    params='{"Mlr": 0.00001, "anomaly_ratio": 2, "auxi_lambda": 0.3, "batch_size": 128, "cf_dim": 32, "d_ff": 128, "d_model": 128, "dc_lambda": 0.1, "dropout": 0.2, "e_layers": 1, "head_dim": 32, "head_dropout": 0.1, "lr": 0.00001, "n_heads": 16, "num_epochs": 3, "patch_size": 16, "patch_stride": 8, "score_lambda": 0.05, "seq_len": 192, "temperature": 0.07}'
    ;;
  CalIt2)
    params='{"Mlr": 0.00001, "auxi_lambda": 0.1, "batch_size": 128, "cf_dim": 128, "d_ff": 128, "d_model": 128, "dc_lambda": 0.5, "e_layers": 2, "head_dim": 32, "inference_patch_size": 16, "lr": 0.0001, "n_heads": 8, "num_epochs": 10, "patch_size": 16, "patch_stride": 16, "score_lambda": 0.5, "seq_len": 192, "temperature": 0.07}'
    ;;
  NYC)
    params='{"Mlr": 0.00001, "batch_size": 128, "cf_dim": 64, "d_ff": 256, "d_model": 256, "e_layers": 3, "head_dim": 64, "inference_patch_size": 32, "inference_patch_stride": 1, "lr": 0.0001, "n_heads": 2, "num_epochs": 2, "patch_size": 16, "patch_stride": 8, "seq_len": 192, "temperature": 0.07}'
    ;;
  MSL)
    params='{"Mlr": 0.00005, "batch_size": 128, "cf_dim": 64, "d_ff": 256, "d_model": 128, "e_layers": 3, "head_dim": 64, "lr": 0.0005, "n_heads": 2, "num_epochs": 5, "patch_size": 16, "patch_stride": 8, "seq_len": 192}'
    ;;
  *)
    echo "Unsupported dataset: $dataset" >&2
    exit 2
    ;;
esac

mkdir -p "$output_dir"
PYTHONUNBUFFERED=1 "$python_bin" -c "import json; from ts_benchmark.baselines.sdd_msd_catch.SDDMSDCATCH import run_sdd_msd_catch_screen; run_sdd_msd_catch_screen('$dataset', json.loads('''$params'''), '$output_dir', int('$seed'))"
