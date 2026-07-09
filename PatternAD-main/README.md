# PatternAD

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)

## Introduction
This branch implements **PatternAD**, a multivariate anomaly detection model built around context-aware reconstruction scoring. The reconstruction backbone is a lightweight joint-variable masked reconstructor: each time step is encoded as a full multivariate state, and training masks both individual variable points and whole variable traces inside a window. The default anomaly score keeps raw reconstruction residual as the primary evidence and uses local scale, trend motion, high-frequency activity, and residual concentration only as reliability context. The goal is to make the same residual magnitude carry different anomaly meaning under different temporal states without letting auxiliary handcrafted components dominate the raw reconstruction signal.

## Quickstart

### Installation
Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

   ```bash
   pip install -r requirements.txt
   ```


## Data preparation
Prepare Data. You can obtain the pre-processed datasets from the ./dataset folder. If you need to add new datasets, you can also place them in this folder.

## Train and evaluate model
- To see the model structure, [click here](./ts_benchmark/baselines/PatternAD/PatternAD.py).

- Run a multivariate experiment:

```bash
sh ./scripts/multivariate_detection/detect_label/Weather_script/PatternAD.sh
```

## Modification Record
See [PATTERN_AD_MODIFICATION_LOG.md](./PATTERN_AD_MODIFICATION_LOG.md) for the current design and implementation notes.
