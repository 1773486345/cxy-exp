# 方向 A：模式感知的残差语义建模

记录日期：2026-06-30  
更新日期：2026-07-10

方向定位：

```text
Pattern-Aware Residual Semantics for Context-Conditioned Reconstruction
模式感知的残差语义建模与条件重构
```

该方向的核心不是异常类型分类，也不是多证据修复。它关注的是：

```text
同样大小的预测/重构偏离，在不同动态状态、波动尺度和结构背景下，异常含义并不相同。
```

因此，方向 A 的代理任务可以概括为：

```text
给定多变量时序窗口和需要恢复的目标位置，
识别当前窗口所处的动态背景与变量状态，
使重构结果符合该背景下的合理取值，
再用条件重构残差判断异常。
```

---

## 1. 研究动机

真实多变量时序系统中的偏离并不具有固定含义。均值漂移、方差变化、周期强度改变、负载波动和变量间协同变化都会改变观测偏离的背景尺度，使得相同幅度的偏离在不同状态下具有不同风险含义。在稳定运行阶段，一个较小偏离可能已经具有异常指示性；而在剧烈波动阶段，相同偏离可能仍属于正常变化范围。

因此，多变量时序异常检测的关键并不只是度量偏离幅度，还在于理解偏离发生的结构背景。一个偏离可能出现在缓慢趋势变化中，也可能出现在短时突发扰动、局部高波动、周期结构改变或多变量同步变化中；这些背景模式会直接影响偏离是否应被视为异常。若将所有偏离统一压缩为单一分数并使用固定阈值判别，检测结果容易受到正常动态变化干扰，也难以准确保留真正具有异常意义的变化。

由此，如何对观测偏离进行模式感知的解释、校准与评分，使异常分数不仅反映偏离大小，也反映偏离所处的时序结构模式及其判别意义，是多变量时序异常检测中需要解决的关键问题。

---

## 2. 研究现状

多变量时序异常检测通常以观测序列与正常行为估计之间的差异作为异常依据，典型形式包括预测误差、重构误差、关联差异和表示距离。MSCRED（AAAI 2019）通过多尺度 signature matrices 表征系统状态，并利用残差矩阵进行异常检测与诊断；GDN（AAAI 2021）通过图结构学习变量间依赖关系，以预测期望行为并识别偏离依赖结构的异常；Anomaly Transformer（ICLR 2022）提出 association discrepancy，将异常判别扩展到时间点关联模式差异；DCdetector（KDD 2023）进一步通过多尺度 dual attention 和对比学习构造判别式异常表征。这些研究表明，异常检测中的有效信息并不局限于单一时域残差，而可以来自多尺度结构、变量关系和关联模式。

另一方面，时间序列分解、频域分析和非平稳建模研究说明，观测偏离的背景结构对时序建模具有重要影响。RobustSTL（AAAI 2019）利用鲁棒季节-趋势分解处理趋势变化、季节漂移和异常干扰；SR-CNN（KDD 2019）将频谱残差用于大规模在线服务监控，表明异常信号可能体现为频域显著性变化。Autoformer（NeurIPS 2021）、FEDformer（ICML 2022）、Non-stationary Transformer（NeurIPS 2022）和 TimesNet（ICLR 2023）分别从序列分解、频域表示、非平稳建模和多周期变化等角度证明了复杂时间结构的重要性。这些工作主要面向预测或通用时序分析，但也说明趋势、周期、频率、尺度和非平稳变化会影响观测偏离的语义。

现有研究已经证明，多尺度结构、频域信息、变量依赖和关联模式能够增强异常检测能力。然而，多数方法仍将这些信息隐式融合为某种统一异常判据，较少系统讨论偏离信号在不同结构背景下应如何解释和校准。换言之，已有工作更多关注如何获得更强的正常行为估计或更具判别性的表示，而对于“同一幅度偏离在不同动态模式下为何具有不同异常含义”这一问题仍缺少专门建模。因此，模式感知的异常证据评分仍是多变量时序异常检测中值得独立研究的方向。

---

## 3. 与方向 B 的边界

方向 A 研究的是偏离语义：

```text
给定一个 residual / deviation，
它处在什么时序结构模式下？
在该模式下，这个偏离是否具有异常意义？
```

方向 B 研究的是修复一致性：

```text
对同一个被遮蔽目标，
不同受限证据源是否能给出一致估计？
```

一句话区分：

```text
方向 A：解释和校准已有偏离信号的异常含义。
方向 B：构造多条修复路径并比较其一致性。
```

因此，方向 A 不应写成“异常类型识别”，也不应写成“多个证据源一致修复”。方向 A 的核心是 residual semantics / pattern-aware scoring。

---

## 4. 核心问题

### 4.1 结构背景如何刻画

方向 A 需要识别偏离发生时的背景模式，而不是把偏离幅度直接当成异常程度。可考虑的背景包括：

```text
stability context：
稳定阶段、剧烈波动阶段、局部方差变化。

trend context：
缓慢趋势漂移、突然水平跳变、持续偏移。

periodic / frequency context：
周期强度变化、频带能量变化、相位或周期结构改变。

multi-scale context：
短尺度异常、长尺度异常、短长尺度不一致。

cross-variable context：
单变量偏离、多变量同步偏离、变量间残差结构变化。
```

这些背景不是异常类型标签，而是解释 residual 含义所需的结构条件。

### 4.2 偏离证据如何构造

基础输入可以来自任意预测或重构模型：

```text
x: 原始窗口
x_hat: 预测或重构窗口
e = |x - x_hat|
```

在此基础上构造模式感知证据：

```text
local residual evidence：
点级或短窗口残差。

scale-normalized evidence：
残差相对于局部波动尺度的异常程度。

trend-aware evidence：
残差是否来自缓慢趋势变化或突然水平跳变。

frequency-aware evidence：
频带能量、谱残差或周期结构变化下的偏离。

multi-scale evidence：
短、中、长窗口残差的一致性与差异。

cross-variable residual evidence：
变量间残差同步性或局部相关残差变化。
```

关键不是把这些量拼接为普通特征，而是用它们解释：

```text
当前 residual 在其结构背景下是否异常。
```

### 4.3 分数如何校准

不同背景下 residual 的正常范围不同。需要使用训练集正常样本拟合每类证据的正常分布，例如：

```text
median / MAD
robust z-score
quantile score
tail probability
extreme value threshold
```

校准目标是让分数表达：

```text
在相似结构背景下，该偏离有多罕见？
```

而不是简单把所有 residual 归一化到相同范围。

### 4.4 聚合如何避免误报和漏报

不同偏离模式的判别逻辑不同。

```text
稳定阶段的小 residual：
可能需要升高异常意义。

高波动阶段的同等 residual：
可能需要降低异常意义。

趋势缓慢漂移：
可能是正常演化，也可能是慢性退化，需要持续性证据。

高频突发扰动：
即使整体 residual 不高，也可能具有异常意义。
```

因此，聚合不宜只是：

```text
S = w1 * z1 + w2 * z2 + ...
```

更合理的候选方式应首先保证 raw residual 的主体地位，再由结构背景调节该 residual 的异常含义，例如：

```text
context-conditioned calibration
context-conditioned reconstruction
tail-probability calibration
slow-drift vs abrupt-shift separation
multi-scale inconsistency scoring
```

前两轮实验也表明，简单把多个结构证据转化为异常分数再做 top-k / mean / max 聚合，或在重构之后手工调权，都不够稳健，容易让辅助证据噪声扰乱 raw residual 的排序。因此，后续实现应避免把背景描述量直接当成独立异常证据，而应优先让背景信息进入重构过程。

---

## 5. 主要挑战

### 5.1 不能把异常解释成正常变化

如果过度强调模式校准，真实异常可能被当作正常非平稳变化消除。例如突然水平跳变不能简单等同于缓慢均值漂移。

### 5.2 短窗口下结构估计不稳定

异常检测窗口通常较短，趋势、周期和频域估计可能不稳定。因此第一版应优先使用简单、稳健、可复现的结构证据，例如 moving average、局部 MAD、多窗口残差和频带能量。

### 5.3 需要证明不是普通后处理

该方向容易被质疑为阈值调整或特征后处理。需要通过实验说明：

```text
同样 residual 在不同结构背景下确实需要不同解释；
模式感知校准优于全局阈值或全局 z-score；
各类结构证据对不同异常场景有可消融贡献。
```

### 5.4 指标不能只依赖 point-adjusted F1

必须使用能够反映真实排序和事件检测质量的指标，例如：

```text
AUC-PR
AUC-ROC
VUS-PR
VUS-ROC
event-level F1
delay-aware metrics
```

---

## 6. 动机示例

### 6.1 同样 residual，不同波动背景

```text
稳定阶段：
prediction error = 0.5，可能异常。

剧烈波动阶段：
prediction error = 0.5，可能正常。
```

方向 A 要解决的是：残差大小相同，但背景模式不同，异常含义应不同。

### 6.2 趋势漂移与突然跳变

```text
缓慢均值变化：
可能是正常趋势演化。

突然均值跳变：
更可能具有异常意义。
```

方向 A 不能简单去除所有 mean shift，而应区分慢漂移和突变。

### 6.3 高频异常被整体残差掩盖

某变量整体趋势正常，但出现短时高频振荡。

```text
整体 residual 不高；
frequency-aware evidence 明显升高。
```

方向 A 应保留这种结构化异常证据。

---

## 7. 修订后的最小可行方案

前两轮实验后，方向 A 的最小可行方案需要从“残差后处理”进一步修订为“上下文条件重构”。核心判断是：raw reconstruction residual 本身仍然是最稳定的异常证据，但结构背景不应主要在评分阶段手工放大或压低 residual，而应进入重构过程，使模型先恢复出“在当前动态状态下合理的值”。

因此，落地流程应调整为：

```text
x_context = describe_local_dynamics(x)
x_masked = mask_target_variables(x)
x_hat = ContextConditionedReconstructor(x_masked, x_context, mask)
score = MSE(x_hat, x) on masked target positions
```

其中，`context` 用于帮助重构，而不是替代 residual，也不是额外生成一组并列异常分数：

```text
local scale context:
描述局部波动尺度，使模型在高波动阶段给出更符合当前状态的重构。

trend / shift context:
描述局部趋势运动和水平变化，使模型区分正常趋势演化与难以解释的偏离。

frequency context:
描述局部高频扰动，使模型在强振荡状态下避免把正常波动直接误报为异常。

mask context:
告诉模型哪些变量或时间点需要被条件恢复，防止评分目标直接泄漏给重构器。
```

修订后的目标不是：

```text
S = aggregate(raw_score, scale_score, trend_score, freq_score, sync_score)
S = raw_residual * handcrafted_context_weight
```

而是：

```text
先让 context 改善 x_hat，再用被遮蔽位置的 raw residual 做异常分数。
```

这样更符合方向 A 的动机：如果某个观测值虽然偏离自身历史趋势，但可以被当前局部波动、趋势阶段或其他变量状态合理解释，模型应尽量把它重构正确，从而降低误报；如果在给定这些上下文后仍无法恢复，残差才更能说明异常。

关键实验比较也应相应调整为：

```text
PatternAD_raw: no context conditioning + no conditional scoring + raw residual
PatternAD: context-conditioned reconstruction + conditional masked raw residual
legacy aggregate / reliability-weighted scorer: only as ablation, not main path
```

消融重点不再是堆叠更多评分组件，而是验证上下文进入重构是否有效：

```text
no context conditioning
local scale context only
scale + trend/frequency context
with / without variable-level conditional mask
with / without whole-variable training mask
```

---
## 8. 当前落地模型：PatternAD

当前代码中的落地版本仍命名为 `PatternAD`，实现位置为：

```text
PatternAD-main/ts_benchmark/baselines/PatternAD/PatternAD.py
PatternAD-main/ts_benchmark/baselines/PatternAD/utils/pattern_scoring.py
```

最新版本已经不再把贡献点放在“重构之后如何给 residual 加权”。前两轮实验说明，简单把 scale、trend、freq、sync 等结构量当作并列异常分数做聚合不稳定；进一步的 reliability-weighted residual 虽然优于旧聚合器，但整体仍弱于 raw residual control。因此，当前实现改为“模式感知的条件重构”：让局部动态背景和变量上下文在重构阶段发生作用，最终评分保持尽可能干净的 raw residual。

### 8.1 设计原则

方向 A 的核心判断是：同样大小的残差在不同动态背景下含义不同。但这并不意味着一定要在评分阶段手工调权。更合理的落地方式是让模型先利用当前背景重构出“在该状态下应当出现的合理值”。如果一个观测点只是由高负载、局部波动、趋势运动或变量协同变化造成，那么条件重构应尽量把它恢复正确，从而降低误报；如果该点在给定这些背景后仍难以被恢复，残差才更有异常意义。

因此，当前模型的重点不是增加异常分数种类，而是改变残差的来源：残差来自一个已经感知局部动态模式的重构器。

### 8.2 模型结构

模型输入为多变量窗口：

```text
x: [B, T, D]
```

每个时间步以完整 `D` 维变量状态作为联合输入。模型在重构前构造局部上下文特征，包括：

```text
local mean: 局部均值背景
local std: 局部波动尺度
trend: 局部一阶变化趋势
high-frequency residual: 去局部均值后的短时扰动
mask indicator: 当前哪些变量被遮蔽
```

这些上下文特征经过轻量 MLP 编码后，通过 FiLM-style gamma/beta 调制注入 Transformer 输入表示。也就是说，趋势、波动和高频活动不是在最后改变分数，而是在重构过程中改变模型如何解释当前窗口。

训练阶段继续采用 denoising reconstruction：随机遮蔽部分变量点，并额外随机遮蔽整条变量轨迹，使模型不能只复制单变量自身历史，而要利用时间上下文和其他变量共同恢复被遮蔽值。

### 8.3 推理与异常分数

推理阶段启用确定性的条件遮蔽。对于每个窗口，模型隐藏一部分变量位置，用剩余变量和局部动态上下文恢复它们。异常分数只在被隐藏的位置上计算：

```text
S_t = mean over i in M_t of (x_ti - x_hat_ti)^2
```

其中 `M_t` 是时间点 `t` 被遮蔽的变量集合。这样做的含义是：模型不能直接看到被评分的值，必须判断该值是否能被当前窗口的上下文和其他变量合理解释。

当前默认不再使用 post-hoc reliability weight。`pattern_scoring.py` 中的 aggregate / reliability-weighted 路径只作为显式消融保留。默认主模型为：

```text
PatternAD = context-conditioned reconstruction + conditional masked raw residual
```

对应 raw control 为：

```text
PatternAD_raw = no context conditioning + no conditional scoring + raw residual
```

### 8.4 与旧版本的区别

旧版本的逻辑是：

```text
先重构 -> 得到 residual -> 用 scale/trend/freq/sync 等结构量重新加权或聚合 residual
```

当前版本的逻辑是：

```text
先识别局部动态背景 -> 用该背景约束重构 -> 只在被遮蔽目标上计算 raw residual
```

这一区别很重要。旧版本仍然属于残差后处理，容易把结构估计噪声放大到最终排序中；当前版本则让结构背景直接影响重构结果本身，符合“关系、文本或状态信息应当帮助重构更符合实际意义，从而减少误报并提升检测”的总思路。

### 8.5 当前实验重点

下一轮实验不应再比较“多个手工评分组件谁更好”，而应比较上下文是否真正提升了条件重构下的异常检测：

```text
PatternAD_raw: 无上下文、无条件遮蔽的 raw residual baseline
PatternAD: 上下文条件重构 + 条件遮蔽 residual
```

优先观察 AUC-PR、AUC-ROC、VUS-PR、VUS-ROC，并辅以 event-level F1 和 point-adjusted F1。若主模型优于 raw control，说明模式背景进入重构过程是有效的；若仍然失败，应继续改重构机制，而不是回到手工分数聚合。
## 9. 数据集审计与新增数据集选择

### 9.1 当前数据集是否充分支持方向 A

第一轮结果不佳不能简单归因于数据集，但当前数据集确实不完全匹配方向 A 的核心动机。方向 A 要验证的是：

```text
同样大小的 reconstruction residual，
在不同局部动态背景下具有不同异常含义。
```

如果某个数据集中的异常主要表现为明显幅值偏离，那么 raw residual 本身就会很强，任何额外的上下文调权都有可能扰乱排序。因此，数据集审计需要区分两类问题：

```text
幅值可分性：
异常是否仅凭相对训练正常分布的幅值偏离就容易识别。

动态背景复杂度：
正常样本中是否存在明显波动尺度、趋势强度、负载状态或频率结构变化。
```

本地 7 个多变量数据集的快速审计结果如下。`amp_auc_mean/max` 是基于训练集 median/MAD 后的幅值偏离分数对测试异常标签的 AUC-ROC 近似值；数值越高，说明不需要复杂上下文，仅靠幅值也较容易分异常。`train_vol_p90/p10` 反映训练正常阶段局部波动强度变化范围；数值越高，说明存在更明显的正常动态背景变化。

```text
dataset   T       D   train   test_anom%  seg_n  seg_med  amp_auc_mean  amp_auc_max  train_vol_p90/p10  audit judgment
Weather   12339   4   9871    17.10       127    2        0.819         0.845        2.54               幅值已较强，适合做压力测试，不是最能证明动机的数据集
Genesis   16220   18  3604    0.40        3      22       0.735         0.978        8.10               max 幅值几乎可分异常，不适合作为核心动机数据集
SKAB      46806   8   9405    34.94       34     399      0.639         0.755        1.17               工业背景有价值，但异常占比高且训练波动变化弱
MSDS      58572   10  29286   72.25       577    1        0.916         0.726        1.77               测试异常占比过高，不适合验证正常背景下 residual 语义
GECCO     139566  9   16000   1.40        51     22       0.846         0.615        2.20               幅值均值很强，可保留但不能单独支撑动机
Energy    1622    9   1297    17.23       28     2        0.612         0.617        10.90              序列短，但动态背景变化明显，可作为辅助验证
Daphnet   28800   9   14610   10.90       5      256      0.485         0.463        22.10              raw 幅值不强且正常动态变化大，最接近方向 A 动机
```

结论：当前 7 个数据集仍应保留作为 benchmark，但它们不足以单独证明方向 A。尤其 Weather、Genesis、GECCO、MSDS 更容易让 raw residual 占优；如果新方法只在这些数据集上评估，很容易低估“上下文解释 residual”的价值。Daphnet 和 Energy 更适合验证动态背景调权，但二者领域和规模有限，不能作为唯一证据。

### 9.2 新增数据集筛选标准

新增数据集不应按“是否常见”选择，而应按是否能检验方向 A 的核心假设选择。准入标准如下：

```text
1. 必须是多变量时序，D > 1；优先 D >= 10。
2. 训练阶段应主要为正常样本，便于拟合正常 residual/context 分布。
3. 正常阶段应存在明显动态背景变化，例如负载切换、过程阶段变化、趋势/波动尺度变化、周期强度变化。
4. 异常不应全部是单点幅值尖峰；应包含持续事件、上下文异常、协同关系异常或隐蔽攻击。
5. 标签应至少支持 point-level 或 interval-level anomaly evaluation。
6. 数据规模不能过小，否则 context 统计不稳定。
```

### 9.3 新增数据集实际接入状态

已完成接入：HAI 21.03。

HAI 21.03 已从 HAI 官方 GitHub 仓库下载并转换为当前 benchmark 可直接读取的多变量宽表压缩 CSV。选择 21.03 而不是 22.04/23.05 的原因是：21.03 已经包含更紧密耦合的 HIL 工业过程和多种运行状态，同时原始文件是 `.csv.gz`，不依赖 Git LFS，服务器同步和复现实验更稳定。

当前接入结果如下：

```text
数据位置：PatternAD-main/dataset/anomaly_detect/data/HAI21_part*.csv.gz
文本占位：PatternAD-main/dataset/anomaly_detect/data/HAI21_text.csv
元数据：PatternAD-main/dataset/anomaly_detect/DETECT_META.csv
运行脚本：PatternAD-main/scripts/multivariate_detection/detect_label/HAI21_script/PatternAD.sh
Raw control：PatternAD-main/scripts/multivariate_detection/detect_label/HAI21_script/PatternAD_raw.sh
来源：HAI official repository, https://github.com/icsdataset/hai
```

适配方式如下：

```text
HAI21_part1.csv.gz = train1 + test1
HAI21_part2.csv.gz = train2 + test2
HAI21_part3.csv.gz = train3 + test3 + test4 + test5
```

原始 HAI 21.03 每行包含 `time + 79 个过程变量 + attack/attack_P1/attack_P2/attack_P3`。当前适配保留 79 个过程变量，用总 `attack` 作为 point-level 异常标签，删除分过程标签列。输出格式为 `date, <79 variables>, label`，并采用 gzip 压缩以减少同步体积。

快速审计结果如下：

```text
file              T       D   train    anom%   seg_n  seg_med  amp_auc_mean  amp_auc_max  train_vol_p90/p10
HAI21_part1       259202  79  216001   1.46    5      98       0.452         0.416        7.95
HAI21_part2       345602  79  226801   2.90    20     151      0.538         0.529        8.11
HAI21_part3       718804  79  478801   2.03    25     203      0.633         0.571        7.77
```

判断：HAI 21.03 比 SMD 更适合作为方向 A 的核心新增数据集。它的简单幅值可分性不强，三个实体的 `amp_auc_mean` 只有 0.452、0.538、0.633；同时训练正常阶段的局部波动变化范围较大，`train_vol_p90/p10` 约为 7.8-8.1。这说明异常并不是简单幅值尖峰即可稳定识别，而更需要解释 residual 所处的运行背景。

考虑到 HAI21 全量数据较长且变量数为 79，当前脚本已改成分层实验：

```text
HAI21_script/PatternAD.sh：默认 dev 实验，只跑 HAI21_part1，num_epochs=10，d_model=64。
HAI21_script/PatternAD_raw.sh：对应 raw-control dev 实验。
HAI21_script/PatternAD_full.sh：全量实验，跑 HAI21_part1/2/3，num_epochs=30，d_model=128。
HAI21_script/PatternAD_raw_full.sh：对应 raw-control full 实验。
```

因此服务器端应先跑 dev，确认方向和运行时间后再跑 full。

已完成接入：MetroPT-3。

MetroPT-3 已从 UCI 官方数据集下载并转换为当前 benchmark 可直接读取的单文件多变量宽表压缩 CSV。它来自真实 air production unit / air compressor predictive maintenance 场景，变量数只有 15，故障区间由官方说明给出，训练集采用官方建议的第一个月正常数据。

当前接入结果如下：

```text
数据位置：PatternAD-main/dataset/anomaly_detect/data/MetroPT3.csv.gz
文本占位：PatternAD-main/dataset/anomaly_detect/data/MetroPT3_text.csv
元数据：PatternAD-main/dataset/anomaly_detect/DETECT_META.csv
运行脚本：PatternAD-main/scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD.sh
Raw control：PatternAD-main/scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD_raw.sh
来源：UCI MetroPT-3 Dataset, https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset
```

快速审计结果如下：

```text
file       T        D   train    anom%   seg_n  seg_med  amp_auc_mean  amp_auc_max  train_vol_p90/p10
MetroPT3   1516948  15  214850   2.30    4      5511     0.945         0.881        17.45
```

判断：MetroPT-3 是更好的轻量开发集，而不是更好的核心论证数据集。它的优势是变量数低、真实工业场景、单文件易运行，适合快速验证代码、训练速度和多模态轻量 context 设计；限制是简单幅值可分性很强，`amp_auc_mean=0.945`，说明 raw amplitude 已能很好地区分故障区间。因此它不适合作为证明“raw residual 不足，需要上下文解释”的主要证据。

已完成接入：SMD。

SMD 已从 OmniAnomaly 官方仓库下载并转换为当前 benchmark 可直接读取的多变量宽表格式。当前接入结果如下：

```text
数据位置：PatternAD-main/dataset/anomaly_detect/data/SMD_machine-*.csv.gz
文本占位：PatternAD-main/dataset/anomaly_detect/data/SMD_text.csv
元数据：PatternAD-main/dataset/anomaly_detect/DETECT_META.csv
运行脚本：PatternAD-main/scripts/multivariate_detection/detect_label/SMD_script/PatternAD.sh
Raw control：PatternAD-main/scripts/multivariate_detection/detect_label/SMD_script/PatternAD_raw.sh
来源：OmniAnomaly, https://github.com/NetManAIOps/OmniAnomaly
```

SMD 共有 28 台机器，每个实体 38 维指标，训练/测试按原始数据对半划分，测试集提供点级异常标签。转换后的单个 CSV 使用 `date, channel1, ..., channel38, label` 格式，并已统一压缩为 `.csv.gz`；为支持这种宽表格式，`ts_benchmark/data/utils.py` 已增加兼容逻辑，同时保留原有 `date,data,cols` 长表读取路径。

快速审计结果如下：

```text
实体数：28
总长度：1,416,825
变量维度：38
平均测试异常比例：4.21%
幅值可分性 amp_auc_mean：0.782
幅值可分性范围：0.539 - 0.989
训练正常阶段局部波动变化均值 train_vol_p90/p10：6.99
```

判断：SMD 适合作为标准新增 benchmark。它的优势是多实体、多变量、公开且易复现；限制是部分机器的异常仍可被简单幅值分数较好区分，因此它更适合作为补充验证，而不是方向 A 的唯一核心数据集。

为降低实验成本，SMD 也已改成分层实验：

```text
SMD_script/PatternAD.sh：默认 dev 实验，只跑 5 个代表机器，num_epochs=10，d_model=64。
SMD_script/PatternAD_raw.sh：对应 raw-control dev 实验。
SMD_script/PatternAD_full.sh：全量实验，跑 28 个机器，num_epochs=30，d_model=128。
SMD_script/PatternAD_raw_full.sh：对应 raw-control full 实验。
```

SMD dev 机器选择为：

```text
SMD_machine-3-2：低幅值可分性，amp_auc_mean=0.543。
SMD_machine-2-2：高正常波动和较高异常比例，amp_auc_mean=0.633，train_vol_p90/p10=20.48。
SMD_machine-3-9：中等幅值可分性且高波动，amp_auc_mean=0.720，train_vol_p90/p10=15.28。
SMD_machine-1-1：常用中等难度机器，amp_auc_mean=0.735。
SMD_machine-2-8：幅值可分性很强的 sanity check，amp_auc_mean=0.989。
```

这组 dev 数据不是为了替代 full，而是用于快速检查模型趋势、运行稳定性和 raw-control 差距。

待申请：SWaT / WADI / BATADAL。

这些数据集来自工业控制或水处理/水分配系统，具有多传感器、多执行器、过程阶段和攻击事件。iTrust 官方页面列出 Secure Water Treatment、Water Distribution、BATADAL 等数据集，并说明数据需要通过 request dataset 获取。它们最适合验证“正常动态过程下 residual 含义会变化”这一动机，但当前不能无权限直接脚本化下载。

```text
建议定位：核心真实工业数据集。
当前状态：未接入，等待数据访问权限。
适配难度：中等，需要处理申请下载、列名、标签区间和 train/test split。
预期价值：高。若能获得权限，应作为主实验数据补充。
来源：iTrust Datasets, https://www.sutd.edu.sg/itrust/itrust-labs/datasets/
```

待接入：Exathlon。

Exathlon 来自 Apache Spark 集群上重复执行的大规模 stream processing jobs。异常包括 misbehaving inputs、resource contention、process failures，并提供 root-cause interval 和 extended-effect interval 标签。它的优势是天然存在不同运行执行、资源状态和异常传播区间，适合评估 calibration stability 和动态背景下的 residual 解释；限制是需要按 execution/job 组织数据和标签，适配成本高于 HAI/SMD。

```text
建议定位：AIOps / cloud 动态系统数据集。
当前状态：未接入。
适配难度：中到高。
预期价值：高，但不应先于 HAI21/SMD 实验。
来源：Exathlon paper, https://arxiv.org/abs/2010.05073
```

暂未接入：SMAP / MSL / PSM。

SMAP/MSL 是 NASA spacecraft telemetry，PSM 是常用服务器指标 benchmark。Anomaly Transformer 仓库提供 SMD、MSL、SMAP、PSM 的预处理数据下载入口，Telemanom 仓库提供 SMAP/MSL 的标签说明和预处理代码，但 Telemanom 本身不包含原始 `train/test` 数据文件。此类数据适合作为标准比较补充，但从方向 A 的动机强度看，优先级低于 HAI21 和 iTrust 工业过程数据。

```text
建议定位：标准公开 TSAD benchmark 补充。
当前状态：未接入。
适配难度：低到中，取决于下载入口是否可直接访问。
预期价值：中。需要先审计 raw 幅值可分性，再决定是否纳入主表。
来源：Anomaly Transformer, https://github.com/thuml/Anomaly-Transformer
来源：Telemanom, https://github.com/khundman/telemanom
```

待定：Tennessee Eastman Process，TEP。

TEP 是经典化工过程故障检测 benchmark。它的优势是过程变量多、故障类型丰富、工业控制语义明确。它更适合作为半控制工业过程数据集，用于分析不同过程扰动和故障下 residual/context 的响应。限制是不同版本 split 和标签协议不完全统一，需要明确采用哪一版数据和评价协议。

```text
建议定位：半控制工业过程机制验证数据集。
当前状态：未接入。
适配难度：中等。
预期价值：中到高。适合补充工业过程证据，但不能替代 HAI/SWaT/WADI 这类真实或半真实运行数据。
来源示例：TEP anomaly/fault detection studies, e.g. https://arxiv.org/abs/2303.05904
```

暂不优先：NAB、Yahoo、UCR univariate archive。

这些数据集在异常检测研究中常见，但大多是单变量或弱多变量形式，不适合作为 PatternAD 的主实验数据。它们可以用于机制示例或可视化说明，但不应作为核心多变量实验依据。

### 9.4 具体执行建议

下一步不应盲目扩大数据集数量，而应按以下顺序推进：

```text
Step 1：先运行 MetroPT3 与 HAI21 dev，用于检查代码、速度和方向趋势。
Step 2：运行 SMD dev，用 5 个代表机器检查标准 benchmark 上的趋势。
Step 3：若 dev 结果相对 raw-control 有价值，再运行 HAI21 full。
Step 4：SMD full 只放在最终补充实验或需要标准 benchmark 全量结果时运行。
Step 5：申请 SWaT/WADI/BATADAL，作为最重要的真实工业动态系统补充。
```

新增数据集进入主表前必须先做审计。审计至少包括：

```text
D, T, train/test split
异常比例与异常段长度分布
训练正常阶段局部波动变化范围
训练正常阶段趋势/均值状态变化范围
简单幅值分数的 AUC-ROC / AUC-PR
raw residual control 的完整指标
```

只有当数据集确实存在正常动态背景变化，且 raw residual 不是 trivially sufficient 时，它才适合作为方向 A 的核心支撑数据集。
