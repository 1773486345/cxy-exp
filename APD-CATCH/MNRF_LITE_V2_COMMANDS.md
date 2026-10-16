# MNRF-Lite V2 task commands

Each command runs exactly one task. Execute them manually and sequentially;
there is no run-all or background runner. A preflight does not train or create
feature caches.

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

Not available with the frozen L=128/purge protocol: MTSB Occupancy 01 and 02,
because their official training segments are too short to supply three
independent segments with at least one complete window each. The preflight
command for the suggested first task is:

```bash
python scripts/mnrf_lite_v2_preflight.py --task PSM --data-source large_detect
```
