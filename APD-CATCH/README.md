# APD-CATCH v1.0

本目录以 CATCH 官方仓库提交 `3647c69be5eb56649b072596cf89098e689e20c3` 为工程基座，用于验证方向 A 的异常保持自适应分解。上游 `catch` 包保持不变；新增实现位于 `ts_benchmark/baselines/apd_catch`。当前是可以在 CATCH 论文数据上直接训练、评估和汇总的第一版模型。

当前模型把 CATCH 的频率 patch 与通道融合主干改为严格的 `past-to-next-point` 协议：归一化、FFT、分频和通道路由只读取历史窗口，模型输出下一点的高斯条件分布。三个同参数预算版本为：

- `causal_catch`：目标盲 CATCH 主干，不分频；
- `fixed`：固定截止频率的双频带分解；
- `adaptive`：仅由历史状态生成逐变量截止频率。

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

## 3. 运行第一版模型

先在一个论文数据集上运行三个同预算版本：

```bash
python scripts/run_apd_catch_paper.py \
  --datasets Genesis \
  --variants causal_catch fixed adaptive \
  --gpu 0
```

运行指定的多个数据集：

```bash
python scripts/run_apd_catch_paper.py \
  --datasets Genesis PSM SWAT \
  --variants causal_catch fixed adaptive \
  --gpu 0
```

运行论文全部 12 类真实数据集：

```bash
python scripts/run_apd_catch_paper.py \
  --datasets all \
  --variants causal_catch fixed adaptive \
  --gpu 0
```

`all` 会展开为 23 个实际文件：11 个单文件数据集和 ASD 的 12 个子序列，共 69 个训练任务。脚本每完成一个任务就立即写结果；再次执行相同命令会跳过已有结果，因此可以中断后续跑。添加 `--force` 才会覆盖已有结果。

每次训练同时输出 AUC-ROC、AUC-PR、R-AUC、VUS、验证集 1% FPR 校准阈值下的 Aff-F 和点级指标，不需要为 score/label 指标重复训练。默认沿用官方 `train_lens` 以及每个数据集原 CATCH 脚本中的窗口、模型容量、batch size、epoch 和学习率；训练标签不传给模型，只在结果中记录训练段污染率。原版特有但不适用于 APD-CATCH 的重构分数权重不会传入。

## 4. 结果位置

```text
result/paper_real_v1/
├── summary_runs.csv                 每个实际数据文件的结果
├── summary_paper_comparison.csv     ASD 聚合后与论文 CATCH 数字对照
└── <dataset>/<variant>/
    ├── seed_20261.json              参数、协议、耗时和指标
    └── seed_20261.npz               连续分数、标签和校准预测
```

论文数字与 APD-CATCH 不是完全同协议：原版 CATCH 重构包含待评分点的完整窗口，APD-CATCH 只用过去预测下一点；论文 Aff-F 的阈值方式也不同。因此 `summary_paper_comparison.csv` 首先用于检查模型是否发生灾难性退化，严格改进结论仍需同环境重跑原版 CATCH。

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
