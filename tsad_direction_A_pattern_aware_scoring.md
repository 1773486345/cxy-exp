# 方向 A：模式感知的异常证据评分

记录日期：2026-06-30  
更新日期：2026-07-10

方向定位：

```text
Pattern-Aware Anomaly Evidence Scoring
模式感知的异常证据评分
```

该方向的核心不是异常类型分类，也不是多证据修复。它关注的是：

```text
同样大小的预测/重构偏离，在不同动态状态、波动尺度和结构背景下，异常含义并不相同。
```

因此，方向 A 的代理任务可以概括为：

```text
给定观测偏离或残差信号，
识别该偏离所处的时序结构模式，
并据此判断该偏离是否具有异常意义。
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
reliability-weighted residual scoring
tail-probability calibration
slow-drift vs abrupt-shift separation
multi-scale inconsistency scoring
```

第一轮实验也表明，简单把多个结构证据都转化为异常分数再做 top-k / mean / max 聚合并不稳健，容易让辅助证据噪声支配最终排序。因此，后续实现应避免把背景描述量直接当成独立异常证据。

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

第一版实验后，方向 A 的最小可行方案需要从“多组件异常证据聚合”修订为“残差语义校准”。核心判断是：raw reconstruction residual 本身仍然是最稳定的异常证据，结构背景不应直接变成一组并列异常分数，而应解释该 residual 在当前状态下是否可靠、是否需要放大或减弱。

因此，落地流程应为：

```text
x_hat = JointMultivariateReconstructor(x)
e_raw = MSE(x, x_hat)
context = describe_local_dynamics(x)
score = e_raw * reliability_weight(context)
```

其中，`context` 用于描述 residual 所处背景，而不是替代 residual：

```text
local scale context：
局部方差、局部波动强度，用于判断同等 residual 是否发生在高波动阶段。

trend / shift context：
局部趋势运动和水平变化强度，用于判断 residual 是否可能来自正常趋势演化或突发结构变化。

frequency context：
去趋势后的局部高频能量，用于判断当前窗口是否本身处于强振荡状态。

cross-variable concentration context：
残差是否集中在少数变量或同步出现在多个变量上，用于提供弱风险提示。
```

修订后的评分目标不是：

```text
S = aggregate(raw_score, scale_score, trend_score, freq_score, sync_score)
```

而是：

```text
S = raw_residual * context_reliability_weight
```

这样更符合方向 A 的动机：同样大小的 residual，在稳定状态下可以更可疑，在高波动或强动态状态下可以更谨慎；但上下文不能完全抹掉 residual 本身，也不能让手工构造的辅助分量主导排序。

关键实验比较也应相应调整为：

```text
raw residual control
reliability-weighted residual scoring
legacy aggregate scorer, only as an ablation
```

消融重点不再是不断堆叠 `raw + scale + trend + freq + sync`，而是验证不同上下文对 raw residual 的调权是否有效：

```text
raw only
raw * scale-context weight
raw * scale/trend/frequency-context weight
raw * dynamic-context + weak sync-risk weight
legacy top-k aggregate scorer
```

---

## 8. 当前落地模型：PatternAD

当前代码中的落地版本命名为 `PatternAD`，实现位置为：

```text
cxy/PatternAD-main/ts_benchmark/baselines/PatternAD/PatternAD.py
cxy/PatternAD-main/ts_benchmark/baselines/PatternAD/utils/pattern_scoring.py
```

该版本是独立的多变量异常检测模型，不再沿用 LLM prompt 编码主干。考虑到原 LLM prompt 主干在多变量实验中需要动态生成 prompt、逐窗口 tokenizer 并调用 DeepSeek 编码，实验速度过慢，当前版本已将重构 backbone 替换为轻量联合多变量 masked reconstructor。异常判定从原始重构误差改为模式感知的残差评分。模型关注的不是“重构误差有多大”，而是“该误差出现在怎样的局部动态背景下，以及在该背景下是否具有异常意义”。

### 8.1 输入与训练流程

模型仅面向多变量时序异常检测。核心输入为多变量时序窗口：

```text
X ∈ R^{B×T×D}, D > 1
```

训练阶段使用联合多变量 Transformer reconstruction 作为基础重构任务。每个时间步以完整变量状态输入，模型先将 `D` 维变量向量投影为联合状态表示，再沿时间维建模。训练时同时加入随机变量点遮蔽和整变量遮蔽，使模型必须利用其他变量和时间上下文恢复被遮蔽值，而不是各变量独立复制自身历史。训练结束后，使用最佳验证 checkpoint 在训练集正常窗口上重新生成重构结果，并拟合模式感知评分器的正常分布统计量。

```text
x_masked = point_mask(x, train_mask_ratio) + variable_mask(x, train_variable_mask_ratio)
x_hat = JointMultivariateReconstructor(x_masked)
L_rec = MSE(x_hat_masked, x_masked_target) + η · MSE(x_hat, x)
```

### 8.2 异常评分流程

推理阶段仍先得到原始窗口 `x` 和重构窗口 `x_hat`，但当前版本不再把所有结构证据都作为独立异常分数进行聚合。默认评分公式修订为：

```text
raw = MSE(x, x_hat)
weight = context_reliability_weight(x)
score = raw * weight
```

其中，`raw` 是主要异常证据；局部结构信息只用于估计当前 residual 的解释背景：

```text
scale_context：
局部波动尺度。高波动状态下，同等 raw residual 的异常意义应适当降低。

trend_context：
局部趋势运动或水平变化强度。动态变化强的阶段，对 residual 的判定应更谨慎。

freq_context：
去趋势后的高频能量。局部振荡较强时，残差可能更容易受正常动态影响。

sync_context：
残差在变量维度上的集中程度。该项只作为弱风险提示，而不是独立主导分数。
```

实现上，训练集正常窗口用于拟合这些上下文量的 median / MAD 统计量。测试阶段根据当前上下文相对训练正常分布的偏离程度生成权重。动态上下文主要提供 `relief`，用于降低高波动、高趋势运动或高频状态下的误报；残差集中上下文提供较弱的 `risk`，用于保留局部变量不一致的风险信号。权重被限制在固定范围内，避免上下文完全抹除 raw residual：

```text
relief = context_strength * tanh(dynamic_context_z)
risk = risk_strength * tanh(sync_context_z)
weight = clip(1 - relief + risk, min_weight, max_weight)
score = raw * weight
```

旧版的多组件聚合逻辑仍保留为显式消融路径：

```text
pattern_score_mode = "aggregate"
```

但它不再是默认方法。默认方法为：

```text
pattern_score_mode = "reliability_weighted"
```

### 8.3 第一轮实验反馈

第一轮实验比较了旧版默认 PatternAD scorer 和 raw residual control。结果表明，旧版“多组件校准 + top-k 聚合”没有获得实验支持，平均指标反而低于 raw control：

```text
              F1      Adjust-F1  Aff-F    AUC-ROC  AUC-PR  R-AUC-ROC  R-AUC-PR  VUS-ROC  VUS-PR
old PatternAD 0.3634  0.5664     0.7339   0.6571   0.3303  0.6280     0.3342    0.6241   0.3352
raw control   0.3921  0.6313     0.7518   0.7275   0.3897  0.6798     0.3763    0.6794   0.3775
```

这说明旧版实现的问题不在于“是否需要模式背景”，而在于“如何使用模式背景”。将 scale、trend、freq、sync 都当作独立异常分数并 top-k 聚合，会把辅助证据的估计噪声放大，破坏 raw residual 原本较稳定的排序能力。由此，当前版本改为 reliability-weighted residual scoring：保留 raw residual 的主体地位，只让上下文参与解释和调权。

### 8.4 当前实现边界

当前版本已经删除 `use_pattern_aware_scoring` 兼容开关，Pattern-Aware scoring 是 `PatternAD` 的默认评分机制。模型入口为：

```text
PatternAD.PatternAD
```

多变量实验脚本为：

```text
cxy/PatternAD-main/scripts/multivariate_detection/detect_label/*_script/PatternAD.sh
```

raw control 脚本仍应保留，用于固定比较基线：

```text
cxy/PatternAD-main/scripts/multivariate_detection/detect_label/*_script/PatternAD_raw.sh
```

单变量相关脚本已删除，`PatternAD` 在数据变量数 `D <= 1` 时直接拒绝运行。这是因为该方向当前主做多变量异常检测，变量维度上的 residual concentration 等上下文依赖多变量结构，继续保留单变量入口会增加无关复杂度。

当前版本没有引入关系图、文本拓扑同构、固定关系原型或 LLM zero-shot 异常打分。为了处理原 LLM prompt 主干多变量实验速度过慢的问题，文本暂不进入模型，multivariate loader 仅返回占位文本张量以保持 benchmark 接口一致。最终异常分数来自 raw residual 与局部动态上下文的可靠性调权。

### 8.5 当前版本的作用定位

该版本是方向 A 的第二版可运行实现，核心验证问题是：

```text
在同一重构 backbone 下，
reliability-weighted residual scoring 是否优于 raw residual control，
并且是否优于旧版 aggregate scorer？
```

因此，下一轮实验应比较：

```text
old aggregate PatternAD result
raw residual control
new reliability-weighted PatternAD result
```

评价不能只看 point-adjusted F1，应同时关注：

```text
AUC-PR
AUC-ROC
VUS-PR
VUS-ROC
event-level F1
adjust-F1
affiliation-F
```
