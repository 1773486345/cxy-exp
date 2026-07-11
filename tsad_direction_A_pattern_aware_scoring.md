# 方向 A：模式感知的残差语义建模

记录日期：2026-06-30  
更新日期：2026-07-11

方向定位：

```text
Pattern-Aware Residual Semantics via Conditional Residual Distributions
基于条件残差分布的模式感知残差语义建模
```

本文档负责研究动机、方法演进与论文主张边界。可执行因子、数据切分、seed、停止/推进判据和运行命令单独维护在 `PatternAD-main/EXPERIMENT_PLAN_DRAFT.md`，并在正式多 seed 实验前冻结为带版本号的 protocol，避免根据结果反向修改判据。

该方向的核心不是异常类型分类，也不是多证据修复。它关注的是：

```text
同样大小的预测/重构偏离，在不同动态状态、波动尺度和结构背景下，异常含义并不相同。
```

因此，方向 A 的代理任务可以概括为：

```text
给定多变量时序窗口和需要恢复的目标位置，
识别当前窗口所处的动态背景与变量状态，
使模型同时估计该背景下的合理取值与正常不确定性，
再用条件残差分布中的罕见程度判断异常。
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

不同背景下 residual 的正常范围不同。仅让背景改变条件均值，仍不能直接表达“相同 residual 在不同波动状态下含义不同”。因此当前主线进一步升级为直接建模条件残差分布，例如：

```text
median / MAD
robust z-score
quantile score
tail probability
extreme value threshold
heteroscedastic Gaussian / Student-t NLL
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

前两轮实验也表明，简单把多个结构证据转化为异常分数再做 top-k / mean / max 聚合，或在重构之后手工调权，都不够稳健，容易让辅助证据噪声扰乱 raw residual 的排序。因此，后续实现应避免把背景描述量直接当成独立异常证据，而应让背景信息进入条件均值与条件尺度的估计过程。

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

前两轮实验后，方向 A 的最小可行方案先从“残差后处理”修订为“上下文条件重构”；当前又进一步升级为“条件残差分布”。原因是：context-conditioned mean 可以改变 `x_hat`，但若最终仍只使用 raw MSE，则同样大小的 residual 仍会得到同样分数，尚未完整回答本方向的核心命题。更闭环的对象应是：

```text
q_theta(x_M | x_visible, context_visible)
score = -log q_theta(x_M | x_visible, context_visible)
```

其中，MSE 路径继续保留，作为兼容默认值和确定性条件均值基线；Gaussian / Student-t 路径用于显式估计条件尺度或重尾分布。

因此，落地流程应调整为：

```text
mask_groups = build_complementary_masks(x)

for mask in mask_groups:
    valid = not mask
    x_visible = replace_masked_targets(x, mask)
    context_visible = describe_local_dynamics(x_visible, valid)
    distribution = ContextConditionedReconstructor(
        x_visible, context_visible, mask
    )
    score[mask] = -log two_sided_tail(distribution, x)[mask]

score_t = mean_over_variables(score[t, :])
```

这里必须先确定 mask，再只使用 visible values 构造 context。旧伪码中的 `describe_local_dynamics(x)` 写在 mask 之前，容易被理解为上下文可读取待评分目标；这与实际代码不一致，现已纠正。实际实现通过显式 `valid=~mask` 和有效计数计算 rolling mean/std/trend，高频项在 masked 位置置零，不把真实 target 或 mask token 混入 context。

其中，`context` 用于帮助重构，而不是替代 residual，也不是额外生成一组并列异常分数：

```text
local scale context:
描述局部波动尺度，使模型在高波动阶段给出更符合当前状态的重构。

trend / shift context:
描述局部趋势运动和水平变化，使模型区分正常趋势演化与难以解释的偏离。

local high-frequency context:
描述相对局部均值的短时扰动，使模型在强振荡状态下避免把正常波动直接误报为异常。当前实现不是完整频谱模型。

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
先让 visible context 改善条件均值与条件尺度，再用被遮蔽目标的条件 NLL 评分。
MSE 版本作为只建模条件均值的兼容基线。
```

这样更符合方向 A 的动机：如果某个观测值虽然偏离条件均值，但当前状态本身具有较高正常不确定性，预测尺度可以表达其较低异常意义；反之，在稳定状态下，同样 residual 应获得更高 NLL。条件均值负责“当前应出现什么”，条件尺度/分布负责“允许偏离多少”。

训练目标也必须避免概率头在可见位置通过复制输入并压缩尺度获得虚假的低 NLL。当前实现为：

```text
MSE path:
L = masked_MSE(mu, x) + lambda * full_mean_MSE(mu, x)

Gaussian / Student-t path:
L = masked_NLL(mu, scale[, df], x) + lambda * full_mean_MSE(mu, x)

Current Gaussian development path:
L += lambda_transition * masked_transition_NLL(delta_mu, transition_scale, delta_x)
```

即概率模型只在隐藏目标上学习 NLL，辅助全窗口项始终约束条件均值，不使用 full-window NLL。

关键实验比较也应相应调整为同一 conditional-mask 协议下的最小公平消融：

```text
A00: context off + MSE + complementary conditional masks
A10: context on  + MSE + complementary conditional masks
A01: context off + Gaussian training NLL + conditional tail score + complementary masks
A11: context on  + Gaussian training NLL + conditional tail score + complementary masks
B00/B11: unmasked diagnostics, not the main conditional-density claim
```

消融重点不再是堆叠更多评分组件，而是验证上下文进入重构是否有效：

```text
context on/off under the same mask and distribution
MSE versus heteroscedastic Gaussian under the same context and mask
complementary conditional mask versus unmasked diagnostic
Gaussian confirmed first, then Student-t as heavy-tail robustness
point-only versus point + whole-variable training masks after the main mechanism is established
```

---
## 8. 当前落地模型：PatternAD

当前代码中的落地版本仍命名为 `PatternAD`，实现位置为：

```text
PatternAD-main/ts_benchmark/baselines/PatternAD/PatternAD.py
PatternAD-main/ts_benchmark/baselines/PatternAD/utils/pattern_scoring.py
```

最新版本已经不再把贡献点放在“重构之后如何给 residual 加权”。前两轮实验说明，简单把 scale、trend、freq、sync 等结构量当作并列异常分数做聚合不稳定；进一步的 reliability-weighted residual 虽然优于旧聚合器，但整体仍弱于 raw residual control。因此，当前实现改为“模式感知的条件残差分布”：让局部动态背景和变量上下文同时影响条件均值与条件尺度。默认 `reconstruction_distribution="mse"` 仍保留为兼容基线；`gaussian` 和 `student_t` 是显式 opt-in 的概率路径。

### 8.1 设计原则

方向 A 的核心判断是：同样大小的残差在不同动态背景下含义不同。但这并不意味着一定要在评分阶段手工调权。更合理的落地方式是让模型先利用当前背景重构出“在该状态下应当出现的合理值”。如果一个观测点只是由高负载、局部波动、趋势运动或变量协同变化造成，那么条件重构应尽量把它恢复正确，从而降低误报；如果该点在给定这些背景后仍难以被恢复，残差才更有异常意义。

因此，当前模型的重点不是增加手工异常分数种类，而是改变残差的参照系：残差来自一个感知局部动态模式、且可选择输出条件不确定性的重构器。

### 8.2 模型结构

模型输入为多变量窗口：

```text
x: [B, T, D]
```

每个时间步以完整 `D` 维变量状态作为联合输入。对每次遮蔽 pass，模型只用 visible values 构造局部上下文特征，包括：

```text
local mean: 局部均值背景
local std: 局部波动尺度
trend: 局部一阶变化趋势
high-frequency residual: 去局部均值后的短时扰动
mask indicator: 当前哪些变量被遮蔽
```

这些上下文特征经过轻量 MLP 编码后，通过 FiLM-style gamma/beta 调制注入 Transformer 输入表示。rolling mean/std 使用显式有效计数，trend 使用局部可见点线性回归，masked target 不参与 context。也就是说，趋势、波动和高频活动不是在最后改变分数，而是在重构过程中改变模型如何解释当前窗口。

当前开发版进一步把 visible local std 作为概率尺度的显式弱先验，而不是只期待通用 MLP 隐式学出该关系：`sigma = dataset_scale * local_visible_std^0.5 * bounded_learned_correction`。0.5 次幂用于保守地注入 regime scale，避免异常邻域直接把尺度完全抬高。该路径只影响 Gaussian/Student-t 的 `sigma`；MSE 路径不使用它，并可通过 `use_context_scale_prior=false` 回退到旧 learned-scale head。

trend 已从“仅相邻可见点差分”改为局部可见点线性回归斜率，从而在 target 位于跳变边缘时仍能跨过 masked gap 描述变化速度。曾尝试用该斜率直接压低 transition 附近的 `sigma`，但单 seed/单 epoch 诊断显示它虽改善一组 abrupt/gradual ordering，却破坏 same-residual ordering 与整体 AP，因此默认关闭；当前只把 slope 用作条件重构特征，不直接手工改变尺度。

训练阶段继续采用 denoising reconstruction：随机遮蔽部分变量点，并额外随机遮蔽整条变量轨迹，使模型不能只复制单变量自身历史，而要利用时间上下文和其他变量共同恢复被遮蔽值。

### 8.3 推理与异常分数

推理阶段启用确定性的互补条件遮蔽。`(time, variable)` 网格被划分为 K 个互斥 mask pass；每个位置恰好被隐藏并评分一次，所有 pass 的并集完整覆盖窗口，不再只给某一固定子集计分：

```text
MSE: S_t = mean_d (x_td - mu_td)^2
Tail: S_t = mean_d -log P(|X-mu_td| >= |x_td-mu_td| | visible context)
```

其中每个 `mu_td / q_theta` 都来自遮蔽 `(t,d)` 的对应 pass。这样做的含义是：模型不能直接看到被评分的值，但最终每个变量都有分数；实现还会检查 coverage 是否逐位置严格等于 1。

当前默认不再使用 post-hoc reliability weight。`pattern_scoring.py` 中的 aggregate / reliability-weighted 路径只作为显式消融保留。为兼容现有脚本，默认配置仍为：

```text
PatternAD default = context-conditioned mean reconstruction
                  + complementary conditional masks
                  + raw masked MSE
```

概率路径通过以下参数显式启用：

```json
{"reconstruction_distribution": "gaussian", "pattern_score_mode": "tail_probability"}
{"reconstruction_distribution": "student_t", "pattern_score_mode": "tail_probability"}
```

若选择非 MSE 分布且没有显式设置 `pattern_score_mode`，配置会自动切换为 `tail_probability`。训练目标仍为 masked NLL；`nll` 评分保留为显式消融。首轮开发只使用 Gaussian 识别 conditional-scale 机制；Student-t 已实现，但应在 Gaussian 获得支持后作为重尾鲁棒性实验。

### 8.4 与旧版本的区别

旧版本的逻辑是：

```text
先重构 -> 得到 residual -> 用 scale/trend/freq/sync 等结构量重新加权或聚合 residual
```

当前版本的逻辑是：

```text
先遮蔽目标 -> 只由 visible values 提取动态背景
-> 估计条件均值/尺度 -> 在所有目标各自被遮蔽时计算 MSE 或 NLL
```

这一区别很重要。旧版本仍然属于残差后处理，容易把结构估计噪声放大到最终排序中；当前版本则让多变量动态状态直接影响重构结果本身。当前 PatternAD 不消费文本，文本 CSV 仅是 benchmark 接口占位，不能把本轮结果解释为多模态贡献。

### 8.5 严格评估协议

当前 `unfixed_detect_label_multi_config.json` 已显式使用 `evaluation_protocol="train_calibration"`。严格路径先验证 official train 标签全部为 0；非零或非有限 train label 会使运行直接失败，当前代码不会静默过滤异常窗口。official train 的尾部被留作独立 temporal calibration，fit 与 calibration 之间默认保留 `seq_len - 1` 个点的 gap，避免重叠窗口跨段共享原始点。阈值只由 calibration scores 的有限样本 empirical/conformal-style quantile 决定；时间依赖与重叠窗口不满足普通 conformal 的交换性假设，因此不声称严格 coverage guarantee：

```text
alpha = anomaly_ratio / 100
k = ceil((n_cal + 1) * (1 - alpha))
threshold = kth order statistic of calibration scores
```

若 `k > n_cal`，阈值取 `+inf`，不静默截断。测试分数和测试标签都不参与阈值确定。严格 leaderboard 会把不同 ratio 保持为不同实验列；当前 factorial runner 预先固定 `anomaly_ratios=[1.0]`，summarizer 只读取阈值无关的 ranking/VUS 指标并验证重复 ratio 行完全一致，不再在测试集上取最大值。

旧行为只通过显式 `evaluation_protocol="legacy_test_contaminated"` 保留用于历史复现。该路径仍可能拼接 train/test scores 并在 leaderboard 中按测试指标取最大值，必须标为 test-label oracle，不能进入无偏主结果。

随机种子也已修复为在每个 model-series pair 的模型构造前读取当前 strategy seed，而不是无参数调用固定默认 seed。严格 runner 会把 dataset、variant、seed、配置 hash 和 attempt 目录绑定，便于配对重复实验与失败续跑。

### 8.6 版本兼容性边界

本轮为公平比较统一了所有 distribution variant 的输出头：末层均输出 `3 * D`，MSE 只使用其中的 mean，因此 MSE/Gaussian/Student-t 名义参数量一致。C0 不再删除条件路径，而是把可学习的 dataset-level 常量送入与 C1 相同的 `context_proj`、FiLM 和 `pre_encoder_norm`；C1 仅把该常量替换为 target-blind visible context。旧实现的 context-off 路径会同时跳过条件注入和该归一化。

因此，旧 checkpoint 通常会因输出头形状变化而无法直接加载；即使只比较旧结果，context-off 计算图也已经变化。此前生成的 PatternAD/PatternAD_raw 数值只能作为历史开发证据，不能与当前 A00/A10/A01/A11 结果直接拼表或据此声称增益。所有主消融 cell 必须在统一新代码、统一 seed 和统一严格协议下重跑。

### 8.7 当前实验重点

下一轮实验不应再使用同时关闭 context 和 conditional scoring 的 `PatternAD_raw` 来归因 context 效果，而应运行固定 complementary mask 的 2 x 2 主消融：

```text
A00 / A10: 在 MSE 下隔离 context 效应
A01 / A11: 在相同 Gaussian NLL 训练、conditional-tail 评分下隔离 context 效应
A00 / A01: context-off 时的 distribution 效应
A10 / A11: context-on 时的 distribution 效应
B00 / B11: 仅用于诊断 conditional mask 的作用
```

严格实验设计见 `PatternAD-main/EXPERIMENT_PLAN_DRAFT.md`，可执行配置与工具见 `PatternAD-main/config/patternad/` 和 `PatternAD-main/scripts/patternad/`。P1-v2 的 Gaussian 目标统一为 `masked Gaussian NLL + full mean-MSE`；已失败的 transition auxiliary 在所有正式 cell 中关闭。评分使用 model-fit prefix 内独立正常 score-reference segment 的 predicted-scale 分层经验尾概率，并向全局 ECDF 收缩；该段与优化和 early stopping 均按窗口长度隔离，外层 calibration 仍只负责固定阈值。

首个端到端检查建议从仓库根目录运行：

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/run_factorial_ablation.py \
  --group smoke --dataset Weather --variant A00 A10 A01 A11 \
  --seeds 2021 --gpus 0 --run-name p0_weather --dry-run
```

确认命令和数据后去掉 `--dry-run`，完成后使用 `scripts/patternad/summarize_factorial.py` 汇总。优先观察 AUC-PR、VUS-PR、AUC-ROC、VUS-ROC 及 paired seed deltas；固定阈值 F1 可作为协议诊断，但不能跨 ratio 选择测试最优值。

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

判断：HAI 21.03 是比 SMD 更契合方向 A 动机的候选验证集。它的简单幅值可分性不强，三个实体的 `amp_auc_mean` 只有 0.452、0.538、0.633；同时训练正常阶段的局部波动变化范围较大，`train_vol_p90/p10` 约为 7.8-8.1。这些统计说明 raw amplitude 未必充分、正常背景变化明显，但不能单独证明 context 方法必要或有效；必要性仍要由 A00/A10/A01/A11 与 synthetic matched-pair 实验检验。

考虑到 HAI21 全量数据较长且变量数为 79，当前脚本已改成分层实验：

```text
HAI21_script/PatternAD.sh：默认 dev 实验，只跑 HAI21_part1，num_epochs=10，d_model=64。
HAI21_script/PatternAD_raw.sh：对应 raw-control dev 实验。
HAI21_script/PatternAD_full.sh：全量实验，跑 HAI21_part1/2/3，num_epochs=30，d_model=128。
HAI21_script/PatternAD_raw_full.sh：对应 raw-control full 实验。
```

以上 shell 仅保留为历史便利入口；它们比较的是同时改变多个因子的 `PatternAD/PatternAD_raw`，不用于当前主消融或 locked confirmation。正式实验使用 factorial runner 的 motivation/confirmation 分组。

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

判断：MetroPT-3 是更好的轻量开发集，而不是更好的核心论证数据集。它的优势是变量数低、真实工业场景、单文件易运行，适合快速验证代码、训练速度和多变量动态 context 设计；限制是简单幅值可分性很强，`amp_auc_mean=0.945`，说明 raw amplitude 已能很好地区分故障区间。因此它不适合作为证明“raw residual 不足，需要上下文解释”的主要证据。

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

这组 dev 数据不是为了替代 full，而是用于快速检查模型趋势和运行稳定性。上述 shell 的 raw-control 比较仍有因子混杂；当前主结论只使用 factorial runner 中预声明的 A/B cells。

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

下一步不应盲目扩大数据集数量，而应按严格 factorial runner 的阶段推进：

```text
Step 0（旧模型工具链已完成）：Weather smoke 与 machine-readable scale 诊断通过，但结果早于 visible-context scale-prior 原型，只保留为 pipeline evidence。
Step 1（当前）：用单 seed、低 epoch 的 synthetic sanity check 迭代模型，优先验证 same-residual ordering、regime FPR 与 dependency break；此阶段不扩真实数据矩阵。
30-epoch generator-seed-3101 的 density-NLL 诊断中，A11 相对 A01 提升了 macro AP（0.0931 vs 0.0835）并略微缩小 regime FPR gap，但 matched ordering 从 1/5 降至 0/5。分解结果表明 `sigma` 方向正确，标准化残差已使 same-deviation 的 2/3 对排序正确，错误主要来自 density NLL 的 `log(sigma)` 项。改用 conditional two-sided tail surprisal 后，单 epoch A11 macro AP 达到 0.1224、same-deviation AP 达到 0.1573、ordering 提升到 2/5。当时据此先确认 30-epoch tail；其后结果与 transition 分支结论见下文。

30-epoch tail 结果保持 same-deviation 2/3，但 abrupt/gradual 仍为 0/2。随后两项开发修改均通过单 epoch sanity check：visible scale 同时归一化重构输入并反归一化条件均值，使 macro AP 从 0.1224 升至 0.1286；独立 masked transition NLL 利用 Gaussian 第三输出块学习 `transition_scale`，使 transition standardized residual 对 abrupt/gradual 达到 2/2。该阶段因此继续做同 seed 30-epoch A01/A11；完整训练结论见下一段。

完整训练确认 derived transition residual 可保持 2/2，但其 transition scale 近似常数；无条件联合 tail 破坏 same-deviation，基于 slope 或左右半窗的确定性 gate 也没有分离度。随后将 transition 分支升级为显式 head：直接预测 `delta_mu`，并用 visible difference scale prior 调制独立 `transition_scale`。该版本单 epoch 为 2/2，但 30-epoch 完整训练后 standardized transition ordering 退化到 1/2，第二个 margin 为 -0.0014；把 transition-loss weight 从 0.1 提高到 0.5 也未改善。P1-v1 的 120 个 cell 随后完整结束：A11-A01 matched-ordering 提升 `0.1867`，95% CI `[0.0667, 0.2933]`；A11-A00 macro AP 提升 `0.0637`，但 regime-FPR gap 只下降 `11.46%`，未达到预注册 `25%`，且 abrupt/gradual ordering 明显退化。方向 A 因而未彻底失败，但不能进入真实数据矩阵。P1-v2 停止 transition 路线，关闭其辅助损失，并只检验 normal-only、predicted-scale 分层的经验尾概率校准能否保留 same-deviation 收益同时修复跨 regime FPR。
Step 2：运行 motivation 组的三 seed 主消融，只依据预声明的 paired comparison 判断机制。
Step 3：用 `bootstrap_factorial.py` 对三个预声明的 A11 comparator 分别做 paired CI，再冻结候选并运行 robustness 组；MetroPT3 主要用于长序列时间/内存压力测试。
Step 4：保存代码、配置和数据 hash 后，只运行一次 locked confirmation 组。
Step 5：申请 SWaT/WADI/BATADAL 或接入 Exathlon/明确版本 TEP，补充真正未参与迭代的外部证据。
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
