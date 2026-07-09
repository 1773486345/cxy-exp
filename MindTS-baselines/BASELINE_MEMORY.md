# RMindTS / Baseline 实验 Memory

工作目录：`/media/h3c/users/wangyueyang1`

项目现在拆成两个目录：

- RMindTS 主模型：`cxy/MindTS-main`
- Baseline 全线结果：`cxy/MindTS-baselines`


## 数据集

当前共 9 个异常检测数据集：

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
