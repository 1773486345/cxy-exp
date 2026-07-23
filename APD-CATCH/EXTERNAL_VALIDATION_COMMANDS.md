# External Validation Commands

Run each command from the `APD-CATCH` directory after the external preparation and descriptor-freeze steps have completed. Every command is an independent `detect_score` task; none launches another task, retries a failure, or starts a background process.

```text
planned commands = 40
executable commands = 40
MetroPT3 status = valid; audited first complete calendar month = 2020-03
```

## Fixed Configuration

Default batch size: 128

Pre-run compatibility exception: `MTSB_OCCUPANCY_01` and `MTSB_OCCUPANCY_02` use batch size 64 for both CATCH and MSD-CATCH.

Reason: With batch size 128, the original frozen CATCH implementation produces fewer than 10 training batches and computes its mask update interval as zero, causing a deterministic `ZeroDivisionError`. The reduction was fixed before obtaining any valid result and was not selected using labels or performance.

## HAI 20.07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/MSDCATCH.sh
```

## BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/MSDCATCH.sh
```

## MetroPT-3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/MSDCATCH.sh
```

## mTSBench OPPORTUNITY

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/MSDCATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/MSDCATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/MSDCATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/MSDCATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/MSDCATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/MSDCATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/MSDCATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/MSDCATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/MSDCATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/MSDCATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/MSDCATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/MSDCATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/MSDCATCH.sh
```

## mTSBench Occupancy

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/MSDCATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/MSDCATCH.sh
```

## mTSBench Metro

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/MSDCATCH.sh
```

## mTSBench SWAN-SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/CATCH.sh
```

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/MSDCATCH.sh
```
