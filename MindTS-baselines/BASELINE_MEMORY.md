# RMindTS / Baseline 实验 Memory

工作目录：`/media/h3c/users/wangyueyang1`

项目现在拆成两个目录：

- RMindTS 主模型：`cxy/MindTS-main`
- Baseline 全线结果：`cxy/MindTS-baselines`

## 严格阈值协议迁移（2026-07-13）

PatternAD 的正式评估切换到 `train_calibration`：模型只在官方正常训练段的前段
拟合，训练尾段 20% 仅用于分数校准；默认保留模型上下文长度减一的时间间隔。阈值
固定为校准分数的 1% 上尾有限样本分位数，测试段既不参与阈值计算，也不用于选择
阈值比例。

此前七数据集的历史 baseline，以及已删除的 `HAI21/SMD` 相关 baseline，均为 legacy
口径：阈值可依赖测试分数，旧 leaderboard 还会在多个比例中按测试指标取最大值。它们
保留用于复现，但不能与严格 PatternAD 结果直接比较。

严格 baseline 只改动 `MindTS-baselines` 和 `TAB`，不修改 `PatternAD-main`。新增
strategy 为 `ts_benchmark/evaluation/strategy/train_calibrated_anomaly_detect.py`，严格
配置为 `config/unfixed_detect_label_train_calibration_config.json`，固定：
`evaluation_protocol=train_calibration`、`anomaly_ratios=[1.0]`、
`calibration_fraction=0.2`、`seed=2021`。

- 七数据集重跑：`bash scripts/baselines/run_train_calibrated_baselines.sh`。
  结果写入 `result/label/strict_baselines_*`，汇总写入
  `_baseline_logs/strict_baselines_{summary,three_metrics}.csv`。
- 当前 PatternAD 数据集重跑：
  `bash scripts/baselines/run_train_calibrated_patternad_baselines.sh`。
  结果写入 `result/label/strict_patternad_baselines_*`。
- 七数据集 strict 入口默认包含 `AnomalyTransformer`；当前 PatternAD strict 入口默认
  跳过 `AnomalyTransformer`、`OmniAnomaly` 和 `InterFusion`。前者需要独占 GPU，后两者
  由 CPU-only TF1 环境运行。旧七数据集可单独执行：
  `BENCHMARK_CONFIG=unfixed_detect_label_train_calibration_config.json RESULT_NAMESPACE=strict_baselines MODEL_FILTER=AnomalyTransformer bash scripts/baselines/run_self_impl_deep_baselines.sh Genesis.csv`。

2026-07-14 的 MetroPT3-AnomalyTransformer GPU 训练在约 25 分钟内完成，但旧的
非重叠 100 点推理窗口遗漏了测试尾部 98 点，严格策略正确拒绝了长度
`1302000 != 1302098` 的得分；
`strict_patternad_baselines_MetroPT3_AnomalyTransformer/test_report.1783977769.*.csv`
因此是无效产物，三项指标为空，不能引用。`detect_score()` 已改为用一个覆盖结尾的
重叠窗口显式计算尾部得分，重跑时必须跳过该无效报告的自动复用：
`SKIP_EXISTING=0 MODEL_FILTER=AnomalyTransformer GPU=0 bash scripts/baselines/run_train_calibrated_patternad_baselines.sh MetroPT3`。

`Genesis, Weather, Energy, SKAB, MSDS, Daphnet, GECCO` 的官方训练段均无异常，
20% 校准点数范围为 260--5858。严格 strategy 的 3 项 baseline 单测、PatternAD
原有 10 项协议单测、Energy 的四个经典模型与 TAB-DAGMM 冒烟均已完成。DAGMM 在
固定 1% 阈值下可能产生零告警，此时 `Aff-F=NaN` 是有效结果，不应改写为 0。

## 当前 PatternAD 数据集适配（2026-07-14）

主模型数据集已经变为：

`Genesis, Weather, Energy, SKAB, MSDS, Daphnet, GECCO, MetroPT3, PSM, BATADAL`。

`HAI21` 与 `SMD` 数据、主模型脚本和 baseline 映射均已移除，不能再作为默认 baseline
目标。`scripts/baselines/run_patternad_dataset_baselines.sh` 是唯一对齐当前主模型数据的
多模型 runner；它支持 `DRY_RUN=1`，可在不启动训练时验证映射和命令。严格入口为：

```bash
cd /media/h3c/users/wangyueyang1/cxy/MindTS-baselines
bash scripts/baselines/run_train_calibrated_patternad_baselines.sh
```

`BATADAL` 保留主模型的双测试定义：同一任务传入 `BATADAL_dataset04.csv` 与
`BATADAL_test.csv`，两个序列各自用官方正常训练段拟合/校准，并在 `BATADAL` 结果目录
聚合。PSM 使用 `PSM.csv`。当前脚本的 TSLib 特征数也已同步为 PSM=25、BATADAL=43。

运行前预检（不生成结果）：

```bash
DRY_RUN=1 MODEL_FILTER=PCA,IsolationForest \
  bash scripts/baselines/run_train_calibrated_patternad_baselines.sh PSM BATADAL
```

## 已废弃的 PatternAD 补充数据集状态（2026-07-13）

该历史轮次的数据集是 `MetroPT3, HAI21, SMD`，与原先 9 个 RMindTS 数据集
分开记录；该段的结果和命令均不用于当前 PatternAD 10 数据集。历史统一结果文件为：

```text
cxy/MindTS-baselines/result/label/patternad_baseline_three_metrics.csv
cxy/MindTS-baselines/result/label/patternad_baseline_summary.csv
```

当时使用的主模型口径与脚本如下：

| Dataset | 正式脚本 | 数据规模 | PatternAD 配置 |
| --- | --- | --- | --- |
| MetroPT3 | `MetroPT3_script/PatternAD.sh` | 1 series, 15 features | batch 512, 30 epochs, patience 5 |
| HAI21 | `HAI21_script/PatternAD_full.sh` | 3 parts, 79 features | batch 256, 30 epochs, patience 5 |
| SMD | `SMD_script/PatternAD_full.sh` | 28 machines, 38 features | batch 512, 40 epochs, patience 7 |

HAI21/SMD 的非-`full` 脚本仅为 `10 / 3` 开发探针；`PatternAD_raw*.sh`
是 `relation_mode=no_graph` 对照，不能与 full 结果混用。

当时已归档 13/60 个 baseline-dataset 单元：四个 classic baseline 已覆盖
全部三个数据集，另有 MetroPT3-TranAD。HAI21-OCSVM 的 `Aff-F=nan` 是已执行
但无效的指标，不应替换为 0。其余 47 个单元尚未运行。

| Model | MetroPT3 Aff-F / V-PR / V-ROC | HAI21 Aff-F / V-PR / V-ROC | SMD Aff-F / V-PR / V-ROC |
| --- | --- | --- | --- |
| PCA | 0.8413 / 0.3212 / 0.9171 | 0.8511 / 0.2883 / 0.8086 | 0.8729 / 0.3338 / 0.7744 |
| IsolationForest | 0.7842 / 0.3331 / 0.9413 | 0.7001 / 0.1214 / 0.7350 | 0.7628 / 0.2205 / 0.7801 |
| LOF | 0.7586 / 0.0499 / 0.7265 | 0.8029 / 0.1878 / 0.7278 | 0.8030 / 0.2456 / 0.7105 |
| OCSVM | 0.8921 / 0.5212 / 0.9611 | NaN / 0.3162 / 0.7299 | 0.8782 / 0.3794 / 0.7808 |
| TranAD | 0.8183 / 0.3254 / 0.9519 | pending | pending |

PatternAD 本体暂时没有可报告结果。HAI21 的单-part 10-epoch 开发探针曾在
GPU 0 仅剩 228 MiB 时申请额外 316 MiB，发生 CUDA OOM；该空指标产物已清理。

本地统一环境和入口：

```text
/media/h3c/users/wangyueyang1/.env/envs/patternad_env
```

PatternAD 主模型使用全局 `patternad_env`，不能在 `cxy/.env/envs` 创建同名
副本；运行主模型前需要手动激活该环境。baseline 使用独立的
`cxy/.env/envs/baseline_env`，由
`scripts/baselines/run_baseline_python.sh` 启动；该环境从主模型环境克隆后补充了
`torch-geometric 2.6.1` 与 TAB-DAGMM 所需的 `salesforce-merlion 2.0.4`。主模型
环境本身没有被修改。两个环境均通过包装器
启动，以便 `conda run` 设置 MKL 动态库路径。

执行顺序：先在 GPU 释放至少约 32 GiB 后运行 MetroPT3 PatternAD full，随后
运行 HAI21 full 和 SMD full。不要串行直接启动完整 20-model sweep：MetroPT3 的
AnomalyTransformer 在共享 GPU 上曾显示约 28--32 小时 ETA，应单独在独占 GPU
中运行。其他未完成 baseline 通过 `MODEL_FILTER` 分组续跑；默认跳过已有结果。


## RMindTS 历史数据集

以下 9 个异常检测数据集只服务 RMindTS 历史实验：

`Genesis, Weather, Energy, SKAB, MSDS, Daphnet, GECCO, ExathlonSmall, Metro`


## 指标

统一比较三项：

- `Aff-F = affiliation_f`
- `V-PR = VUS_PR`
- `V-ROC = VUS_ROC`

表格单元格格式固定为：

`Aff-F / V-PR / V-ROC`

三项均越高越好。

## 结果来源

Baseline 全线三指标长表：

`cxy/MindTS-baselines/result/label/_baseline_logs/requested_baseline_three_metrics.csv`

Baseline 模型共 20 个：

`PCA, IsolationForest, LOF, OCSVM, USAD, OmniAnomaly, DAGMM, TranAD, AnomalyTransformer, GDN, MTAD-GAT, InterFusion, DADA, UniTS, Timer, LMixer, DLinear, PatchTST, iTransformer, TimesNet`

RMindTS 结果不在 baseline CSV 中，需要从：

`cxy/MindTS-main/result/label/{Dataset}_RMindTS/test_report.*.csv`

读取最新 report。

当前最新 RMindTS report：

- Genesis: `test_report.1780423957.h3c-R5500-G5.1633367.csv`
- Weather: `test_report.1778972620.h3c-R5500-G5.1244024.csv`
- Energy: `test_report.1778974176.h3c-R5500-G5.775693.csv`
- SKAB: `test_report.1780128396.h3c-R5500-G5.3728397.csv`
- MSDS: `test_report.1780350362.h3c-R5500-G5.327713.csv`
- Daphnet: `test_report.1781101799.h3c-R5500-G5.3484400.csv`
- GECCO: `test_report.1781142194.h3c-R5500-G5.3595153.csv`
- ExathlonSmall: `test_report.1781701051.h3c-R5500-G5.3288057.csv`
- Metro: `test_report.1781744678.h3c-R5500-G5.1924813.csv`

注意：Metro 的 RMindTS 有旧 report `1781457461`，最新应使用 `1781744678`。

## 实验结果分析

当前全线结果包含：

- 9 个数据集
- RMindTS + 20 个 Baseline，共 21 个模型
- 3 个指标：`Aff-F / V-PR / V-ROC`

按 9 个数据集 × 3 个指标的平均排名，RMindTS 当前整体第 1：

| Model | Avg Rank | Mean Aff-F / V-PR / V-ROC |
|---|---:|---|
| RMindTS | **5.85** | **0.8051 / 0.4753 / 0.7321** |
| TimesNet | 7.41 | 0.7896 / 0.4541 / 0.7270 |
| TranAD | 8.19 | 0.6708 / 0.4118 / 0.7049 |
| DLinear | 8.37 | 0.7892 / 0.4480 / 0.7062 |
| PatchTST | 8.44 | 0.7869 / 0.4520 / 0.7096 |
| iTransformer | 8.59 | 0.7851 / 0.4317 / 0.6982 |
| PCA | 8.85 | 0.6841 / 0.4601 / 0.7001 |
| IsolationForest | 9.52 | 0.7060 / 0.4034 / 0.7246 |

核心结论：

- RMindTS 是当前跨 9 个数据集整体平均排名最好的模型，说明鲁棒性最好。
- RMindTS 不是所有数据集都最强，优势集中在 `GECCO`、`Weather`、`SKAB`。
- `Metro` 是当前最明显负例，RMindTS 不占优。
- `MSDS` 上 RMindTS 的 `Aff-F` 第 1，但 `V-PR / V-ROC` 排名靠后，说明事件级检测较强，但 anomaly score 排序/校准能力不足。
- `ExathlonSmall` 接近天花板，很多模型接近 1，不能过度解释微小差异。

RMindTS 各数据集最新结果与排名：

| Dataset | RMindTS Aff-F / V-PR / V-ROC | Metric Ranks | Avg Rank |
|---|---|---|---:|
| Genesis | 0.8585 / 0.0232 / 0.8220 | 9 / 10 / 7 | 8.67 |
| Weather | 0.8064 / 0.5616 / 0.8335 | 2 / 1 / 1 | 1.33 |
| Energy | 0.7118 / 0.3614 / 0.6141 | 3 / 6 / 6 | 5.00 |
| SKAB | 0.7884 / 0.7521 / 0.8026 | 2 / 3 / 1 | 2.00 |
| MSDS | 0.7212 / 0.7293 / 0.4112 | 1 / 11 / 12 | 8.00 |
| Daphnet | 0.7448 / 0.2673 / 0.7652 | 9 / 4 / 3 | 5.33 |
| GECCO | 0.9506 / 0.6409 / 0.9929 | 1 / 2 / 1 | 1.33 |
| ExathlonSmall | 0.9986 / 0.9406 / 0.9798 | 11 / 8 / 10 | 9.67 |
| Metro | 0.6651 / 0.0015 / 0.3674 | 9 / 12 / 13 | 11.33 |

RMindTS 按指标排名统计：

| Metric | Top-1 | Top-3 | Top-5 | Avg Rank |
|---|---:|---:|---:|---:|
| Aff-F | 2/9 | 5/9 | 5/9 | 5.22 |
| V-PR | 1/9 | 3/9 | 4/9 | 6.33 |
| V-ROC | 3/9 | 4/9 | 4/9 | 6.00 |

适合写入论文/报告的结论：

> RMindTS achieves the best overall average rank across 9 datasets and 3 metrics, showing strong robustness across heterogeneous anomaly detection scenarios. The gains are most evident on GECCO, Weather, and SKAB. However, performance is dataset-dependent: RMindTS is less competitive on Metro, and on MSDS it achieves strong event-level Aff-F but weaker VUS-based ranking metrics, suggesting room for improving anomaly score calibration.

中文表述：

> RMindTS 在 9 个数据集和 3 个指标上的整体平均排名最好，说明其跨数据集鲁棒性较强。优势主要集中在 GECCO、Weather、SKAB；但在 Metro 上不占优，在 MSDS 上 Aff-F 高而 V-ROC 低，说明异常分数校准和排序能力仍有改进空间。

## 消融实验建议

主消融固定每个数据集当前 RMindTS 超参，只改模块开关，不重新调参。

推荐主消融变体：

1. Full RMindTS
2. Base MindTS: `use_relation_graph=false`
3. w/o Dynamic Graph: `relation_dynamic_graph=false`
4. w/o Numeric Relation Tokens: `relation_use_numeric_tokens=false`
5. w/o Conditional Recon: `relation_conditional_recon=false`
6. w/o Relation Aux Loss: `relation_aux_lma_weight=0, relation_rec_corr_weight=0, relation_corrupt_loss_weight=0`
7. w/o Corrupt Branch: `relation_corrupt_loss_weight=0, relation_score_corrupt_weight=0`
8. w/o Relation Score Fusion: `relation_score_lambda=1.0`
9. Relation Score Only: `relation_score_lambda=0.0`

最低实验规模：`9 datasets × 9 variants × 1 seed = 81 runs`。  
更稳妥：3 seeds，即 `243 runs`。

分析数据集：

- `GECCO, SKAB, Weather`: RMindTS 表现较强，适合看正贡献。
- `MSDS`: Aff-F 强但 V-ROC 弱，适合看分数融合问题。
- `Metro`: RMindTS 不占优，适合看关系模块是否负贡献。
- `ExathlonSmall`: 天花板数据集，需单独讨论。

结果解释必须看 `Δ = Ablation - Full`，不要只看平均值。
