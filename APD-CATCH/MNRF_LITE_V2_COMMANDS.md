# MNRF-Lite V2 task commands

Each command runs exactly one task inside the registered `catch_env` Conda
environment. Execute them manually and sequentially; there is no run-all or
background runner. A preflight reads only the selected task, does not train or
create feature caches, and must be checked per task because gaps/boundaries
change usable window counts.

## Internal (`large_detect`)

```bash
bash ./scripts/multivariate_detection/detect_score/ASD_dataset_1_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/ASD_dataset_2_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/ASD_dataset_3_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/ASD_dataset_4_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/ASD_dataset_5_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/ASD_dataset_6_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/ASD_dataset_7_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/ASD_dataset_8_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/ASD_dataset_9_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/ASD_dataset_10_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/ASD_dataset_11_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/ASD_dataset_12_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/CICIDS_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/CalIt2_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/Creditcard_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/GECCO_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/Genesis_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MSL_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/NYC_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/PSM_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/SMAP_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/SMD_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/SWAT_script/MNRFLiteV2.sh
```

## External (`external_detect`)

```bash
bash ./scripts/multivariate_detection/detect_score/BATADAL_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/HAI20_07_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/MNRFLiteV2.sh
bash ./scripts/multivariate_detection/detect_score/MetroPT3_script/MNRFLiteV2.sh
```

The listed wrappers are candidates only. A task is available only if its
preflight confirms nonempty continuous-segment windows for each internal split
and test. The preflight command for the suggested first task is:

```bash
conda run --no-capture-output -n catch_env python scripts/mnrf_lite_v2_preflight.py --task PSM --data-source large_detect
```

First single-task command, prepared but not launched here:

```bash
GPU_ID=0 conda run --no-capture-output -n catch_env bash ./scripts/multivariate_detection/detect_score/PSM_script/MNRFLiteV2.sh
```

On the shared GPU, manually set `BATCH_SIZE` and optionally a smaller
`ENCODING_BATCH_SIZE`; encoding never exceeds training batch. `CACHE_CHUNK_SIZE`
controls disk I/O grouping only. Use `GPU_ID=cpu` only when explicitly choosing
CPU. GPU mode preserves caller visibility and selects its `cuda:0`.

Resume an existing epoch-boundary run explicitly:

```bash
GPU_ID=0 MNRF_RESUME_FROM=score/MNRF-Lite-V2/PSM/run-<UTC>-<PID> conda run --no-capture-output -n catch_env bash ./scripts/multivariate_detection/detect_score/PSM_script/MNRFLiteV2.sh
```

Resume validates task/configuration, variables, raw values, and time/index
fingerprints before restoring optimizer and RNG/sampler state.
