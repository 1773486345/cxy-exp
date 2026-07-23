#!/usr/bin/env sh
python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-set-name "external_detect" --data-name-list "MTSB_SWAN_SF.csv" --model-name "tods.pcaodetectorski" --model-hyper-params '{"n_components":2,"window_size":1}' --seed 2021 --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/external_validation/MTSB_SWAN_SF/PCA"
