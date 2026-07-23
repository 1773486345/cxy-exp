#!/usr/bin/env sh
python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-set-name "external_detect" --data-name-list "MetroPT3.csv" --model-name "tods.hbosski" --model-hyper-params '{"alpha":1e-06,"n_bins":3,"tol":0.9}' --seed 2021 --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/external_validation/MetroPT3/HBOS"
