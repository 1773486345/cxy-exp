# TypeFusion-CATCH Formal Single-Task Commands

The first planned validation task is `PSM.csv`.  The repository already has a
complete detect-score CATCH script for this task at
`scripts/multivariate_detection/detect_score/PSM_script/CATCH.sh`; both commands
below use its `unfixed_detect_score` configuration and the default `large_detect`
data suite.  They are preparation only and have not been executed.

Both commands use the same data split, preprocessing and label handling from
the same configuration, `seed=2021`, `seq_len=192`, `patch_size=16`,
`patch_stride=8`, `batch_size=128`, `lr=0.005`, three CATCH reference epochs,
the same evaluator/protocol, one worker and GPU device 0.  The TypeFusion model
has its four fixed branches in code; its total three-stage optimizer updates are
constrained to the CATCH reference update count.

## CATCH

```bash
python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "PSM.csv" --model-name "catch.CATCH" --model-hyper-params '{"Mlr":0.001,"auxi_lambda":0.01,"batch_size":128,"cf_dim":16,"d_ff":32,"d_model":16,"dc_lambda":0.05,"dropout":0.3,"e_layers":1,"head_dim":32,"inference_patch_size":96,"lr":0.005,"n_heads":4,"num_epochs":3,"patch_size":16,"patch_stride":8,"score_lambda":0.5,"seq_len":192}' --seed 2021 --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/fair/PSM/CATCH_seed2021"
```

## TypeFusion-CATCH

```bash
python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "PSM.csv" --model-name "typefusion_catch.TypeFusionCATCH" --model-hyper-params '{"batch_size":128,"cf_dim":16,"d_ff":32,"d_model":16,"dropout":0.3,"e_layers":1,"head_dim":32,"lr":0.005,"n_heads":4,"seq_len":192,"patch_size":16,"patch_stride":8,"seed":2021,"fit_mode":"three_stage","training_budget_mode":"equal_total_steps","catch_train_epochs":3,"joint_finetune_lr_scale":0.1,"lambda_freq":0.1,"lambda_mask":0.1}' --seed 2021 --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/fair/PSM/TypeFusion-CATCH_seed2021"
```

The TypeFusion command keeps the fixed `lambda_freq=0.1`, `lambda_mask=0.1`
and `joint_finetune_lr_scale=0.1`.  It contains no score-level branch fusion,
threshold calibration or label-driven hyperparameters.
