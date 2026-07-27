# TypeFusion-CATCH Real-Task Manual Commands

These 23 single-task commands are prepared only and have not been run.
Run PSM first; after it is valid, run subsequent tasks manually one at a time.
Do not start all tasks in the background. Set `GPU_ID` explicitly when needed.
After a real CUDA OOM, set `BATCH_SIZE` manually to an allowed halved value,
inspect the completed result before proceeding, and prepare the paired CATCH rerun
before any comparison. The PSM command below can be manually prefixed with
`BATCH_SIZE=64` only after an actual OOM; no script retries or changes it automatically.

## ASD

GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_1_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_2_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_3_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_4_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_5_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_6_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_7_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_8_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_9_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_10_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_11_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_12_script/TypeFusionCATCH.sh

## Industrial / Cyber-Physical

GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/CICIDS_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/SWAT_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/GECCO_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/Genesis_script/TypeFusionCATCH.sh

## General Real Datasets

GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/CalIt2_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/Creditcard_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/MSL_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/NYC_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/PSM_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/SMAP_script/TypeFusionCATCH.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/SMD_script/TypeFusionCATCH.sh
