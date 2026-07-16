> [!IMPORTANT]
> **APD-CATCH v1/v2 已冻结。** `ts_benchmark/baselines/apd_catch/` 是冻结的 legacy exploratory line。本 README 中现存的 Causal-State-CATCH v2 命令仅用于复现旧探索，不是当前推荐研究流程；不应继续运行 15-task v2 screen，也不得将其结果自动并入新的分解研究主表。
>
> 当前活动研究边界由 [`docs/RESEARCH_RESET.md`](./docs/RESEARCH_RESET.md)、[`docs/REPOSITORY_AUDIT.md`](./docs/REPOSITORY_AUDIT.md) 与 [`docs/legacy/APD_CATCH_LEGACY_STATUS.md`](./docs/legacy/APD_CATCH_LEGACY_STATUS.md) 定义。新研究以原版 [`ts_benchmark/baselines/catch/`](./ts_benchmark/baselines/catch/) 的完整窗口重构协议为基座；在新协议通过人工确认前，不存在可运行的新分解模型。
>
> 以下内容是 legacy documentation，必须保留以说明和复现旧线。

# Causal-State-CATCH v2.0

本目录以 CATCH 官方仓库提交 `3647c69be5eb56649b072596cf89098e689e20c3` 为工程基座，用于验证方向 A 的异常保持条件分布。上游 `catch` 包保持不变；新增实现位于 `ts_benchmark/baselines/apd_catch`。v2 使用 CATCH 编码历史创新，而不是另建模型主干。

当前模型把 CATCH 的频率 patch 与通道融合主干改为严格的 `past-to-next-point` 协议：训练期标准化、因果状态、FFT 和通道路由只读取历史窗口，模型输出下一点的高斯条件分布。三个同参数预算版本为：

- `causal_catch`：目标盲 CATCH 主干，直接编码标准化历史；
- `state`：无参数因果 EMA 慢状态 + CATCH 创新编码；
- `state_scale`：`state` + 近期创新强度调制的条件尺度。

`fixed/adaptive` 是已否定的 v1 历史候选，不参与 v2。CATCH 保留其频率 patch、通道掩码和 masked cross-channel Transformer；状态由时间域因果 EMA 解释，CATCH 只解释去状态后的创新结构。

## 1. 环境与测试

本项目的唯一实验环境是全局 Conda 环境 `catch_env`：

```text
/media/h3c/users/wangyueyang1/.conda/envs/catch_env
```

`dyr` 不是 CATCH/APD-CATCH 的实验环境。运行测试、数据检查和训练前执行：

```bash
conda activate catch_env
python -m unittest tests.test_apd_catch_core tests.test_paper_runner
```

测试覆盖模型不变量、三版本同预算、官方长表数据读取、原 CATCH 参数映射和 12 个论文数据集展开。

## 2. 下载 CATCH/TAB 官方预处理数据

官方 TAB 数据压缩包约 1.9 GB，解压后放在本仓库的 `dataset/`。以下脚本支持断点续传，不会自动开始训练：

```bash
bash scripts/download_tab_datasets.sh
python scripts/run_apd_catch_paper.py --check-data --datasets all
```

也可以从上游 README 的 OneDrive/BaiduCloud 手动下载，然后把 `dataset` 文件夹放到本目录。

## 3. 并行运行 v2 三变体

从仓库根目录打开多个终端。首轮必须在同一 `seed=2021` 和同一协议下比较
`causal_catch`、`state`、`state_scale`；每行只运行一个“数据集 x 变体”，并默认保存诊断。
把第一个参数替换为该终端使用的 GPU 编号。每个任务写到独立 worker 目录，可安全并行。

```bash
# Genesis
bash scripts/run_causal_state_catch_variant.sh 0 Genesis causal_catch
bash scripts/run_causal_state_catch_variant.sh 1 Genesis state
bash scripts/run_causal_state_catch_variant.sh 2 Genesis state_scale

# CalIt2
bash scripts/run_causal_state_catch_variant.sh 0 CalIt2 causal_catch
bash scripts/run_causal_state_catch_variant.sh 1 CalIt2 state
bash scripts/run_causal_state_catch_variant.sh 2 CalIt2 state_scale

# GECCO
bash scripts/run_causal_state_catch_variant.sh 0 GECCO causal_catch
bash scripts/run_causal_state_catch_variant.sh 1 GECCO state
bash scripts/run_causal_state_catch_variant.sh 2 GECCO state_scale

# NYC
bash scripts/run_causal_state_catch_variant.sh 0 NYC causal_catch
bash scripts/run_causal_state_catch_variant.sh 1 NYC state
bash scripts/run_causal_state_catch_variant.sh 2 NYC state_scale

# PSM
bash scripts/run_causal_state_catch_variant.sh 0 PSM causal_catch
bash scripts/run_causal_state_catch_variant.sh 1 PSM state
bash scripts/run_causal_state_catch_variant.sh 2 PSM state_scale
```

全部 15 个窗口完成后汇总：

```bash
bash scripts/summarize_causal_state_catch.sh result/causal_state_catch_v2_screen
```

PSM 的旧 v1 运行表明每变体约需 2 小时训练和 17--22 分钟推理，因此纳入首轮。SMAP 虽与
PSM 同为 25 变量，但其官方配置为 10 epoch、3 层，测试段约为 PSM 的五倍；数据维度不能
单独代表成本。Creditcard、CICIDS、MSL、SMD、SMAP 的单锚点成本测量属于首轮 v2 三变体
比较之后的独立决策；SWAT 使用 `seq_len=2048`、`patch_size=256`、`batch_size=32`，当前明确
不运行。首轮按数据集独立比较：任一 v2 变体相对原 CATCH 持平/提升，或超过已有 baseline，
即可保留该结果；多种子只在需要进一步确认时追加。ASD 的 12 个子序列作为第二阶段独立扩展，
避免首轮将同一数据族重复计数。

原版 CATCH 的本地归档可直接汇总：

```bash
python scripts/analysis/summarize_original_catch_results.py \
  --original-root ../CATCH-master \
  --output result/causal_state_catch_v2/original_catch_local_reference.csv
```

当前归档已覆盖 CalIt2、CICIDS、SWAT 和 ASD 的 15 个实际文件。其余 8 个论文真实文件可由下列命令补齐；该命令逐字调用 `CATCH-master` 中既有的 `CATCH.sh`，不改原始种子、超参数或协议，已有报告会跳过：

```bash
bash scripts/run_missing_original_catch.sh \
  ../CATCH-master all "Creditcard GECCO Genesis MSL NYC PSM SMD SMAP"
```

每次训练同时输出 AUC-ROC、AUC-PR、R-AUC、VUS、验证集 1% FPR 校准阈值下的 Aff-F 和点级指标，不需要为 score/label 指标重复训练。默认沿用官方 `train_lens` 以及每个数据集原 CATCH 脚本中的窗口、模型容量、batch size、epoch 和学习率；训练标签不传给模型，只在结果中记录训练段污染率。原版特有但不适用于 Causal-State-CATCH 的重构分数权重不会传入。

v2 只映射原 CATCH 脚本中与其共享的窗口、容量、batch size、epoch 和学习率；Causal-State-CATCH 的状态比例、创新尺度和优化默认值是独立配置。每个结果 JSON 同时记录映射参数 `apd_params` 与完整有效配置 `effective_apd_params`，用于复现。

## 4. 结果位置

```text
result/causal_state_catch_v2_screen/
├── summary_runs.csv                 每个实际数据文件的结果
├── summary_paper_comparison.csv     ASD 聚合后与论文 CATCH 数字对照
└── workers/
    └── <dataset>_<variant>/<dataset>/<variant>/
        ├── seed_<seed>.json         参数、协议、耗时和指标
        └── seed_<seed>.npz          连续分数、标签和校准预测
```

首轮直接在同一数据集、同名指标上比较 v2、原 CATCH 和已有 baseline：v2 持平/提升原 CATCH，或超过 baseline，即保留该结果。原版 CATCH 重构包含待评分点的完整窗口，APD-CATCH 只用过去预测下一点；论文 Aff-F 的阈值方式也不同。这些协议差异在结果中注明即可，只有需要声明严格的协议改进时才将两者置于完全相同的可见性、训练和阈值协议下重跑。

与原版 CATCH 的复现清单已核对：12 个真实数据类别完全一致，即
`CICIDS`、`CalIt2`、`SWAT`、`Creditcard`、`GECCO`、`Genesis`、`MSL`、`NYC`、
`PSM`、`SMD`、`SMAP` 和 `ASD`；其中 ASD 展开后，双方都是 23 个实际数据文件。
因此没有改动 APD-CATCH v1.0 的模型、超参数或结果目录结构。原版 CATCH 的
`score` 连续分数指标和 `label` 的 `anomaly_ratio` 阈值协议，都不能与这里的验证集
1% FPR 校准 Aff-F 直接横比。

## 5. 合成机制门控

```bash
python scripts/analysis/evaluate_apd_catch_mechanism.py
```

三个种子的门控中，`adaptive` 相对 `causal_catch` 的四类异常平均 AP 从 `0.5165` 变为 `0.5191`。这只是机制筛选，不代替上述真实数据实验。完整修改和运行协议见 [`CODE_MODIFICATION_LOG.md`](./CODE_MODIFICATION_LOG.md)。

下文原样保留上游 CATCH README；其中 `CATCH.sh` 运行的是原版 `catch.CATCH`，不是 APD-CATCH。

---

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

### Data preparation

Prepare Data. You can obtained the well pre-processed datasets from [OneDrive](https://1drv.ms/u/c/801ce36c4ff3f93b/EVTDLHyvegpEn_Oxa6ZiuFIBjTsKk6m9JldUqWDqvrVCnQ?e=P2T3Vc) or [BaiduCloud](https://pan.baidu.com/s/1W7UoAWKZjoukSZ74FTipYA?pwd=2255). (This may take some time, please wait patiently.) Then place the downloaded data under the folder `./dataset`. 

### Train and evaluate model

- To see the model structure of CATCH,  [click here](./ts_benchmark/baselines/catch/CATCH.py).
- We provide the experiment scripts for CATCH and other baselines under the folder `./scripts/multivariate_detection`. For example you can reproduce a experiment result as the following:

```shell
sh ./scripts/multivariate_detection/detect_label/MSL_script/CATCH.sh

sh ./scripts/multivariate_detection/detect_score/MSL_script/CATCH.sh
```



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
