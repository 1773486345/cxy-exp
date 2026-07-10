# PatternAD

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)

## Introduction

PatternAD is a multivariate time-series anomaly detection model built around context-conditioned reconstruction. Each time step is encoded as a full multivariate state, and training masks both individual variable points and whole variable traces inside a window.

The current default model no longer uses temporal context as a post-hoc score multiplier. Local scale, trend, high-frequency activity, and mask structure are encoded inside the reconstruction backbone through FiLM-style conditioning. During inference, PatternAD performs conditional reconstruction by hiding a deterministic subset of variables and scoring only the raw reconstruction residual on those hidden positions. The goal is to make reconstruction itself adapt to the operating state, rather than manually reweighting the anomaly score after reconstruction.

## Installation

```bash
pip install -r requirements.txt
```

The project was developed under Python 3.8.

## Data

Preprocessed datasets are stored under:

```text
dataset/anomaly_detect/data
```

Dataset metadata is stored in:

```text
dataset/anomaly_detect/DETECT_META.csv
```

The code supports both the original long format with `date,data,cols` and the added wide format with `date,<variables>,label`, including `.csv.gz` files.

## Core Model

Model entry:

```text
ts_benchmark/baselines/PatternAD/PatternAD.py
```

Scoring implementation:

```text
ts_benchmark/baselines/PatternAD/utils/pattern_scoring.py
```

Current default scoring is raw residual on conditionally hidden positions. Legacy aggregate and reliability-weighted scorers remain available only as explicit ablation paths.

## Standard Experiments

Run one of the existing multivariate datasets:

```bash
sh ./scripts/multivariate_detection/detect_label/Weather_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/Weather_script/PatternAD_raw.sh
```

The `_raw` script disables context-conditioned reconstruction and conditional scoring, then uses raw residual scoring. It is the no-context raw-control baseline for the current model.

## Added Dataset Scripts

These datasets have been adapted locally but may not yet exist on the server unless the new data files are synced.

MetroPT-3 is the lightweight real-industrial smoke-test dataset:

```bash
sh ./scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD_raw.sh
```

HAI21 and SMD have dev/full split scripts. The default scripts are lightweight dev runs:

```bash
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD_raw.sh

sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD_raw.sh
```

Full HAI21/SMD runs are explicit:

```bash
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD_full.sh
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD_raw_full.sh

sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD_full.sh
sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD_raw_full.sh
```

## Notes

Current local-added datasets:

```text
MetroPT3: single 15-variable industrial dataset, lightweight smoke test.
HAI21: 3 industrial process parts, stronger motivation dataset but heavier.
SMD: 28 machine entities, standard benchmark supplement; stored as .csv.gz.
```

See `PATTERN_AD_MODIFICATION_LOG.md` for implementation notes, result interpretation, and dataset adaptation history.
