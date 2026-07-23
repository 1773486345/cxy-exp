# External Baseline Commands

- Baseline models: 15
- Tasks: 20
- Planned commands: 300
- Executable commands: 300
- Every command is foreground-only and operates on one model and one task.
- Activate `catch_env` for all models except OCSVM, PCA, and HBOS; activate `tods_legacy` for those three TODS models.
- Shell scripts intentionally use `python`, matching the existing CATCH/MSD external scripts.
- Each Baseline uses its own frozen PSM detect_score template on every task; any future CUDA OOM override must be recorded per model and task.

## ModernTCN

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/ModernTCN.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/ModernTCN.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/ModernTCN.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/ModernTCN.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/ModernTCN.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/ModernTCN.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/ModernTCN.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/ModernTCN.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/ModernTCN.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/ModernTCN.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/ModernTCN.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/ModernTCN.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/ModernTCN.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/ModernTCN.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/ModernTCN.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/ModernTCN.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/ModernTCN.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/ModernTCN.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/ModernTCN.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/ModernTCN.sh
```

## iTransformer

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/iTransformer.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/iTransformer.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/iTransformer.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/iTransformer.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/iTransformer.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/iTransformer.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/iTransformer.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/iTransformer.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/iTransformer.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/iTransformer.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/iTransformer.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/iTransformer.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/iTransformer.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/iTransformer.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/iTransformer.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/iTransformer.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/iTransformer.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/iTransformer.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/iTransformer.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/iTransformer.sh
```

## DualTF

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/DualTF.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/DualTF.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/DualTF.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/DualTF.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/DualTF.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/DualTF.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/DualTF.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/DualTF.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/DualTF.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/DualTF.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/DualTF.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/DualTF.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/DualTF.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/DualTF.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/DualTF.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/DualTF.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/DualTF.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/DualTF.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/DualTF.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/DualTF.sh
```

## AnomalyTransformer

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/AnomalyTransformer.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/AnomalyTransformer.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/AnomalyTransformer.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/AnomalyTransformer.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/AnomalyTransformer.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/AnomalyTransformer.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/AnomalyTransformer.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/AnomalyTransformer.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/AnomalyTransformer.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/AnomalyTransformer.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/AnomalyTransformer.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/AnomalyTransformer.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/AnomalyTransformer.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/AnomalyTransformer.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/AnomalyTransformer.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/AnomalyTransformer.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/AnomalyTransformer.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/AnomalyTransformer.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/AnomalyTransformer.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/AnomalyTransformer.sh
```

## DCdetector

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/DCdetector.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/DCdetector.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/DCdetector.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/DCdetector.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/DCdetector.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/DCdetector.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/DCdetector.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/DCdetector.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/DCdetector.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/DCdetector.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/DCdetector.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/DCdetector.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/DCdetector.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/DCdetector.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/DCdetector.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/DCdetector.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/DCdetector.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/DCdetector.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/DCdetector.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/DCdetector.sh
```

## TimesNet

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/TimesNet.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/TimesNet.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/TimesNet.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/TimesNet.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/TimesNet.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/TimesNet.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/TimesNet.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/TimesNet.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/TimesNet.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/TimesNet.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/TimesNet.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/TimesNet.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/TimesNet.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/TimesNet.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/TimesNet.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/TimesNet.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/TimesNet.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/TimesNet.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/TimesNet.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/TimesNet.sh
```

## PatchTST

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/PatchTST.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/PatchTST.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/PatchTST.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/PatchTST.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/PatchTST.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/PatchTST.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/PatchTST.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/PatchTST.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/PatchTST.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/PatchTST.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/PatchTST.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/PatchTST.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/PatchTST.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/PatchTST.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/PatchTST.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/PatchTST.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/PatchTST.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/PatchTST.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/PatchTST.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/PatchTST.sh
```

## DLinear

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/DLinear.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/DLinear.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/DLinear.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/DLinear.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/DLinear.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/DLinear.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/DLinear.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/DLinear.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/DLinear.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/DLinear.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/DLinear.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/DLinear.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/DLinear.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/DLinear.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/DLinear.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/DLinear.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/DLinear.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/DLinear.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/DLinear.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/DLinear.sh
```

## NLinear

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/NLinear.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/NLinear.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/NLinear.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/NLinear.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/NLinear.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/NLinear.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/NLinear.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/NLinear.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/NLinear.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/NLinear.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/NLinear.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/NLinear.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/NLinear.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/NLinear.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/NLinear.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/NLinear.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/NLinear.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/NLinear.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/NLinear.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/NLinear.sh
```

## TFAD

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/TFAD.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/TFAD.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/TFAD.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/TFAD.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/TFAD.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/TFAD.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/TFAD.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/TFAD.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/TFAD.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/TFAD.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/TFAD.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/TFAD.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/TFAD.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/TFAD.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/TFAD.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/TFAD.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/TFAD.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/TFAD.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/TFAD.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/TFAD.sh
```

## AutoEncoder

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/AutoEncoder.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/AutoEncoder.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/AutoEncoder.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/AutoEncoder.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/AutoEncoder.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/AutoEncoder.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/AutoEncoder.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/AutoEncoder.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/AutoEncoder.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/AutoEncoder.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/AutoEncoder.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/AutoEncoder.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/AutoEncoder.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/AutoEncoder.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/AutoEncoder.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/AutoEncoder.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/AutoEncoder.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/AutoEncoder.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/AutoEncoder.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/AutoEncoder.sh
```

## OCSVM

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/ocsvmski.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/ocsvmski.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/ocsvmski.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/ocsvmski.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/ocsvmski.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/ocsvmski.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/ocsvmski.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/ocsvmski.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/ocsvmski.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/ocsvmski.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/ocsvmski.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/ocsvmski.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/ocsvmski.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/ocsvmski.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/ocsvmski.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/ocsvmski.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/ocsvmski.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/ocsvmski.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/ocsvmski.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/ocsvmski.sh
```

## IsolationForest

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/IsolationForest.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/IsolationForest.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/IsolationForest.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/IsolationForest.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/IsolationForest.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/IsolationForest.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/IsolationForest.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/IsolationForest.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/IsolationForest.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/IsolationForest.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/IsolationForest.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/IsolationForest.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/IsolationForest.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/IsolationForest.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/IsolationForest.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/IsolationForest.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/IsolationForest.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/IsolationForest.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/IsolationForest.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/IsolationForest.sh
```

## PCA

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/pcaodetectorski.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/pcaodetectorski.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/pcaodetectorski.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/pcaodetectorski.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/pcaodetectorski.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/pcaodetectorski.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/pcaodetectorski.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/pcaodetectorski.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/pcaodetectorski.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/pcaodetectorski.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/pcaodetectorski.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/pcaodetectorski.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/pcaodetectorski.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/pcaodetectorski.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/pcaodetectorski.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/pcaodetectorski.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/pcaodetectorski.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/pcaodetectorski.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/pcaodetectorski.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/pcaodetectorski.sh
```

## HBOS

### HAI20_07

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/hbosski.sh
```

### BATADAL

```bash
sh ./scripts/multivariate_detection/detect_score/BATADAL_script/hbosski.sh
```

### MetroPT3

```bash
sh ./scripts/multivariate_detection/detect_score/MetroPT3_script/hbosski.sh
```

### MTSB_OPPORTUNITY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_01_script/hbosski.sh
```

### MTSB_OPPORTUNITY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_02_script/hbosski.sh
```

### MTSB_OPPORTUNITY_03

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_03_script/hbosski.sh
```

### MTSB_OPPORTUNITY_04

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_04_script/hbosski.sh
```

### MTSB_OPPORTUNITY_05

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_05_script/hbosski.sh
```

### MTSB_OPPORTUNITY_06

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_06_script/hbosski.sh
```

### MTSB_OPPORTUNITY_07

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_07_script/hbosski.sh
```

### MTSB_OPPORTUNITY_08

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_08_script/hbosski.sh
```

### MTSB_OPPORTUNITY_09

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_09_script/hbosski.sh
```

### MTSB_OPPORTUNITY_10

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_10_script/hbosski.sh
```

### MTSB_OPPORTUNITY_11

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_11_script/hbosski.sh
```

### MTSB_OPPORTUNITY_12

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_12_script/hbosski.sh
```

### MTSB_OPPORTUNITY_13

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OPPORTUNITY_13_script/hbosski.sh
```

### MTSB_OCCUPANCY_01

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_01_script/hbosski.sh
```

### MTSB_OCCUPANCY_02

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_OCCUPANCY_02_script/hbosski.sh
```

### MTSB_METRO

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_METRO_script/hbosski.sh
```

### MTSB_SWAN_SF

```bash
sh ./scripts/multivariate_detection/detect_score/MTSB_SWAN_SF_script/hbosski.sh
```
