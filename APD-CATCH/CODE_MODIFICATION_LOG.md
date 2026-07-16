# APD-CATCH 第一版修改与运行记录

## v1.1（2026-07-16，Genesis 单种子负结果）

### 动机与范围

v1.0 的 Genesis 首轮结果出现异常的正常点 NLL 尾部：部分正常点分数最高达到
`7.8e15`，而标注异常的平均分数更低。排查到 v1.0 的预测尺度为局部历史标准差与
`[0.05, 3.0]` 的归一化头相乘；局部近常数窗口会把正常状态切换放大为极端 NLL。

v1.1 只修改条件尺度，不改变 CATCH 主干、频带划分、训练标签协议、参数预算或三组
对照关系。它检验的假设是：异常保持的条件分布必须给低局部波动状态提供由训练历史
支持的尺度下界，避免局部归一化统计主导异常排序。

### 模型修改

1. 每个变量从训练子段中均匀抽取至多 2048 个历史窗口，计算窗口 MAD 的 5% 分位数；
   同时以全训练段 MAD 的 1% 作为退化窗口的回退下界。
2. 评分时只由历史窗口计算 `local_MAD[t,d]`，并使用
   `s_eff[t,d] = sqrt((1.4826 * local_MAD[t,d])^2 + scale_floor[d]^2)`。
3. 高斯尺度改为 `s_eff[t,d] * scale_head(...)`。`scale_floor[d]` 是训练期固定 buffer，
   不可训练，且不读取测试目标、测试标签或未来点。
4. 添加 `--save-diagnostics`。启用后结果 NPZ 额外保存逐变量 `channel_nll`、
   `prediction_mean`、`prediction_scale`、`cutoff` 及训练期 `scale_floor`，用于审计
   高分正常点；不重复执行推理。

### 预注册运行与判定

使用独立输出目录，避免覆盖 v1.0：

```bash
python scripts/run_apd_catch_paper.py \
  --datasets Genesis PSM \
  --variants causal_catch fixed adaptive \
  --output-dir result/paper_real_v1_1_robust_scale \
  --save-diagnostics --gpu 0
```

先检查 Genesis 的最高分正常点是否仍由极小 `prediction_scale` 造成；随后只用连续
排名指标比较三个版本。v1.1 不因 Aff-F 上升而通过：`adaptive` 必须在 Genesis 与 PSM
上都不低于 `causal_catch` 和 `fixed` 的 AUC-PR/VUS-PR，且至少三个种子的方向一致。
若尺度修复后 `adaptive` 仍无独立优势，则停止把自适应截止频率作为方向 A 的默认贡献。

### 实际 Genesis 结果与结论

仅完成 Genesis、`seed=20261` 的三个变体。表面排名指标从 v1.0 的约
`AUC-ROC=0.40`、`AUC-PR=0.0035` 提升至 `0.7100`、`0.2256`，但这不是可接受的
性能结论：三个变体的分数和指标几乎完全相同，且最大点分数从 v1.0 的 `7.8e15`
恶化到 `3.8e22`。

诊断显示 18 个变量中多数在训练子段的窗口 MAD 和全局 MAD 都为零，导致其
`scale_floor` 退化为 float32 epsilon，测试期一旦出现变化就仍产生接近零的
`prediction_scale`（中位及 1% 分位约 `3.6e-7`）。这说明 v1.1 没有解决“训练期
恒定、测试期进入新工况”的识别问题；在纯过去可见设定下，训练中没有变化的变量也
没有可由数据支持的本变量正常尺度。

结论：停止 v1.1 的 PSM/SWAT 和多种子扩展，也不把 Genesis 排名上升记录为收益。
下一轮必须先定义可审计的冷启动变量协议，例如只评估训练期具有非零动态范围的变量，
或引入评分时可见的控制状态/同伴变量来条件化该变量的首次状态切换；二者都必须作为
新的研究设定和对照实验，不能把跨变量全局尺度或人为 epsilon 当作正常机制。

### 已完成验证

- `python -m unittest tests.test_apd_catch_core tests.test_paper_runner`：9 项通过。
- Genesis/PSM、三个变体及 `--save-diagnostics` 的 dry-run 通过。
- Genesis、三个变体、`seed=20261` 完成并保存诊断；结果如上，为数值门控失败。

## v1.0（2026-07-15）

### 基线来源

- 上游仓库：`decisionintelligence/CATCH`
- 上游提交：`3647c69be5eb56649b072596cf89098e689e20c3`
- 原版实现保留在 `ts_benchmark/baselines/catch`。
- 第一版 APD-CATCH 位于 `ts_benchmark/baselines/apd_catch`。

v1.0 是方向 A 修改后的第一版可运行模型，不是原版 CATCH 的直接复现。

### 模型修改

1. 将包含待评分点的全窗重构改成 `past-to-next-point` 条件预测；输入只包含 `x[t-L:t-1]`，目标 `x[t]` 不进入前向。
2. 将全窗 RevIN 改为 past-only 的逐样本、逐变量均值和标准差归一化。
3. 保留 CATCH 的频率 patch、通道掩码生成器和 masked cross-channel Transformer，移除全窗时域/频域重构分数。
4. 增加满足 `M_low + M_high = 1` 的低频/高频完备分解；两个分量共享同一个 CATCH-style 编码器。
5. 增加由历史频谱生成逐变量截止频率的 adaptive router，并提供三个同参数预算版本：
   - `causal_catch`：不分频；
   - `fixed`：固定截止频率；
   - `adaptive`：历史状态自适应截止频率。
6. 输出逐变量高斯条件分布，以 Gaussian NLL 形成一个连续异常分数，不再手工融合多个异常分数。
7. 官方训练切分按时间顺序再划分训练段和验证段；早停与 1% FPR 阈值校准均不读取标签。运行结果记录官方训练段异常率，但标签不传给模型，也不用于过滤样本。
8. 通道掩码训练时使用 Gumbel hard sampling，评估时使用确定性阈值；修复单批次输出的无条件 `squeeze` 风险。

### 论文数据运行入口

`scripts/run_apd_catch_paper.py` 直接读取 CATCH/TAB 官方预处理数据，并完成一次训练对应的全部 score/label 评估：

- 支持论文 12 类真实数据集：CICIDS、CalIt2、SWAT、Creditcard、GECCO、Genesis、MSL、NYC、PSM、SMD、SMAP、ASD。
- ASD 自动展开为 12 个子序列；全部数据共 23 个实际文件。
- 从每个数据集的原版 `detect_score/<dataset>_script/CATCH.sh` 自动读取并映射窗口长度、patch、模型容量、epoch、batch size 和学习率。
- 原版的 `score_lambda`、`auxi_lambda`、`anomaly_ratio` 和 inference patch 等重构/测试专属参数不传入 APD-CATCH。
- 第一个长度为 `seq_len` 的测试前缀没有足够历史，不填造分数，直接从所有指标中排除。
- 严格沿用官方 `train_lens`，训练标签只用于事后记录污染率，不传入模型或用于挑选样本。
- 每个任务立即保存 JSON 与 NPZ；已有结果默认跳过，可以断点续跑。
- 一次训练计算 AUC-ROC、AUC-PR、R-AUC-ROC、R-AUC-PR、VUS-ROC、VUS-PR、Aff-F 和点级指标。
- `summary_runs.csv` 保留实际文件结果，`summary_paper_comparison.csv` 对 ASD 求子序列均值后按论文 12 类数据汇总。

`scripts/download_tab_datasets.sh` 从 TAB 官方 Google Drive 下载约 1.9 GB 的预处理数据，支持断点续传，只下载和解压，不启动实验。

### 论文主表 CATCH 参考值

| 数据集 | AUC-ROC | Aff-F |
| --- | ---: | ---: |
| CICIDS | 0.795 | 0.787 |
| CalIt2 | 0.838 | 0.835 |
| SWAT | 0.345 | 0.755 |
| Creditcard | 0.958 | 0.750 |
| GECCO | 0.970 | 0.908 |
| Genesis | 0.974 | 0.896 |
| MSL | 0.664 | 0.740 |
| NYC | 0.816 | 0.994 |
| PSM | 0.652 | 0.859 |
| SMD | 0.811 | 0.847 |
| SMAP | 0.504 | 0.699 |
| ASD | 0.824 | 0.804 |

这些论文值来自原版目标可见重构和论文阈值协议，只作为第一轮外部健康检查。`../CATCH-master/result/` 已保留同机原版 CATCH 的本地运行结果：CalIt2、CICIDS、SWAT、12 个 ASD 子序列以及 12 组合成数据均有 score 记录，除 SWAT 外这些真实数据还保留了 label 记录。原版 score/label 使用分开训练和不同超参数，label 还依赖 `anomaly_ratio`，所以不能直接与 APD-CATCH 的 1% FPR Aff-F 横比；要声明严格改进，仍必须在相同目标可见性和阈值协议下比较。

### 已完成验证

- `python -m unittest tests.test_apd_catch_core tests.test_paper_runner`：8 项通过。
- 覆盖历史/目标隔离、三版本同参数预算、正尺度、频带完备性、确定性评估、23 个论文文件展开、全部 69 个任务的配置构造、官方长表数据读取、原 CATCH 参数映射和 ASD 论文级聚合。
- 统一入口的 Genesis/PSM/SWAT 三版本 dry-run 通过，未自动启动真实数据训练。

### 已完成的合成筛选

三个种子的合成机制门控使用正常均值/尺度/频率变化和 spike、level、periodic、relation 四类异常：

| `causal_catch` | `fixed` | `adaptive` |
| ---: | ---: | ---: |
| 0.5165 | 0.5125 | 0.5191 |

adaptive 相对 causal_catch 的平均 AP 绝对增益只有 0.0026，并且在 level 和 periodic 上更差。该结果不能作为真实有效性结论，也不能用于反向调整论文数据实验参数。
