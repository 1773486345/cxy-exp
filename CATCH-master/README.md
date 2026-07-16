# <img src="docs/catch.png" alt="Image description" style="width:50px;height:50px;"> CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching

**This code is the official PyTorch implementation of our ICLR'25 paper: [CATCH](https://arxiv.org/pdf/2410.12261): Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching.**

[![ICLR](https://img.shields.io/badge/ICLR'25-CATCH-orange)](https://arxiv.org/pdf/2410.12261)  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-blue)](https://pytorch.org/)  ![Stars](https://img.shields.io/github/stars/decisionintelligence/CATCH) 

If you find this project helpful, please don't forget to give it a ⭐ Star to show your support. Thank you!

🚩 News (2025.6) The evaluation framework [TAB](https://arxiv.org/pdf/2506.18046) used in this study has been accepted by PVLDB 2025. Both the dataset and source code are available [here](https://github.com/decisionintelligence/TAB).

🚩 News (2025.1) CATCH has been accepted by ICLR 2025.

## Introduction

**CATCH**, a framework based on frequency patching, flexibly utilizing the channel correlations to reconstruct all the frequency spectrums in a fine-grained way to achieve remarkable detection accuracy. Technically,  we propose a **Channel Fusion Module** (CFM), which features a patch-wise **mask generator** and the masked-attention mechanism. Driven by a bi-level multi-objective optimization algorithm, the CFM is encouraged to iteratively discover appropriate patch-wise channel correlations and **cluster similar channels in the hidden spaces while isolate the adverse effects from irrelevant channels**, which provides both the **capacity and robustness** of the attention mechanism.

<div style="text-align: center;">
    <img src="docs/overview.png" alt="CATCH" style="zoom:80%;" />
</div>


## Quickstart

### Installation

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```
pip install -r requirements.txt
```

On this machine, run the original CATCH scripts from `catch_env`:

```shell
conda activate catch_env
python --version  # Python 3.8.x
```

### Data preparation

Prepare Data. You can obtained the well pre-processed datasets from [OneDrive](https://1drv.ms/u/c/801ce36c4ff3f93b/EVTDLHyvegpEn_Oxa6ZiuFIBjTsKk6m9JldUqWDqvrVCnQ?e=P2T3Vc) or [BaiduCloud](https://pan.baidu.com/s/1W7UoAWKZjoukSZ74FTipYA?pwd=2255). (This may take some time, please wait patiently.) Then place the downloaded data under the folder `./dataset`. 

### Train and evaluate model

- To see the model structure of CATCH,  [click here](./ts_benchmark/baselines/catch/CATCH.py).
- We provide the experiment scripts for CATCH and other baselines under the folder `./scripts/multivariate_detection`. For example you can reproduce a experiment result as the following:

```shell
sh ./scripts/multivariate_detection/detect_label/MSL_script/CATCH.sh

sh ./scripts/multivariate_detection/detect_score/MSL_script/CATCH.sh
```



## Official Reproduction Matrix And Traceable Results

The paper-level count of 18 is **not** 18 real datasets. It is 12 real-data
benchmark groups plus six synthetic anomaly families:

- Real groups: `CICIDS`, `CalIt2`, `SWAT`, `Creditcard`, `GECCO`, `Genesis`,
  `MSL`, `NYC`, `PSM`, `SMD`, `SMAP`, and `ASD`.
- Synthetic families: `con`, `glo`, `sea`, `sha`, `sub_mix`, and `tre`.

`ASD` consists of 12 physical files. The other 11 real groups each have one
file, so the official scripts reference 23 real files. Each synthetic family
has two supplied files, so they contribute 12 more. The repository therefore
contains 35 official physical-task entries per protocol and 70 official
protocol-task combinations: 23 real files plus 12 synthetic files, each under
`score` and `label`. The upstream Results paragraph's "10 real-world" wording
describes the displayed results rather than an executable matrix; it is not
used to invent or omit scripts.

The checked source of truth is
[`scripts/official_catch_manifest.tsv`](./scripts/official_catch_manifest.tsv).
It records every physical file, its paper-level group, type, and official
script directory in execution order. All 35 referenced data files and all 70
`CATCH.sh` scripts are present in this checkout. There are no missing entries
in this matrix. No configuration is created for a dataset that lacks an
official script or data file.

Only the `--save-path` argument in the official `CATCH.sh` files has been
changed. It now defaults to a protocol/task/unique-run location such as
`result/score/CATCH/Genesis/run-.../`; all model arguments, data names,
configuration files, strategy, seed, timeout, and hyperparameters remain the
official values. The two wrapper scripts are the supported entry points:

```shell
# Non-training preflight: parse one task, check its data, and print its result path.
bash scripts/run_official_catch.sh Genesis score

# Run exactly one task. --run is required before training can start.
bash scripts/run_official_catch.sh Genesis score --run

# Non-training preflight in manifest order. Default: all 70 combinations.
bash scripts/run_official_catch_matrix.sh

# Sequentially run one protocol or the complete 70-task matrix.
bash scripts/run_official_catch_matrix.sh score --run
bash scripts/run_official_catch_matrix.sh all --run
```

Every `--run` invocation creates a new timestamp/PID/random subdirectory.
That directory keeps the framework-generated CSV (and its original auxiliary
record), `command.sh`, `official_CATCH.sh` and its SHA-256, `metadata.txt`
(commit, data path and hashes, config, and save path), `environment.txt`, and
`console.log`. Re-running a task therefore cannot replace an older run.

`score` and `label` are separate original CATCH protocols. `score` uses the
continuous-score strategy and score metrics; `label` uses the original
`unfixed_detect_label` strategy and each script's unmodified `anomaly_ratio`
threshold protocol. Neither result is directly comparable to APD-CATCH's
past-to-next-point evaluation using a validation-set 1% FPR threshold and
Aff-F. Use this traceable original-CATCH matrix for any same-protocol baseline
comparison.

## Results

Extensive experiments on 10 real-world datasets and 12 synthetic datasets demonstrate that CATCH achieves state-of-the-art performance. We show the main results of all the 10 real-world datasets, and report the mean results of the 6 types of synthetic datasets:

<div align="center">
<img alt="exp" src="docs/experiment.png" width="100%"/>
</div>


## Setup for Running Baseline Models
If you want to test all baseline models, please refer to the Time Series Anomaly Detection Benchmark [TAB](https://github.com/decisionintelligence/TAB):


## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{wu2024catch,
  title     = {{CATCH}: Channel-Aware multivariate Time Series Anomaly Detection via Frequency Patching},
  author    = {Wu, Xingjian and Qiu, Xiangfei and Li, Zhengyu and Wang, Yihang and Hu, Jilin and Guo, Chenjuan and Xiong, Hui and Yang, Bin},
  booktitle = {ICLR},
  year      = {2025}
}

@inproceedings{qiu2025tab,
title      = {{TAB}: Unified Benchmarking of Time Series Anomaly Detection Methods},
author     = {Xiangfei Qiu and Zhe Li and Wanghui Qiu and Shiyan Hu and Lekui Zhou and Xingjian Wu and Zhengyu Li and Chenjuan Guo and Aoying Zhou and Zhenli Sheng and Jilin Hu and Christian S. Jensen and Bin Yang},
booktitle  = {Proc. {VLDB} Endow.},
year       = {2025}
}
```


## Contact

If you have any questions or suggestions, feel free to contact:
- [Xingjian Wu](https://ccloud0525.github.io/)  (xjwu@stu.ecnu.edu.cn)
- [Xiangfei Qiu](https://qiu69.github.io/) (xfqiu@stu.ecnu.edu.cn)

Or describe it in Issues.
