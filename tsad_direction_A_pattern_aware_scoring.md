# 方向 A：动态结构感知的异常语义

记录日期：2026-06-30  
更新日期：2026-07-13

方向定位：

```text
Dynamic-Structure-Aware Anomaly Semantics
```

## 当前状态

方向 A 仍然开放。已关闭的是其第一条具体实现路线 **A-v1：条件残差语义**，
而不是“动态结构会改变异常含义”这个研究动机。

```text
A-v1 (closed):
  用一个条件 residual / innovation tail surprise 同时表达波动背景与 abrupt onset。

A2 (active definition, pre-model):
  给定事件前可观测状态，判断随后一段轨迹或状态转移是否与正常演化兼容。
```

A2 的可执行前置协议在
`PatternAD-main/research/direction_a/A2_EXPERIMENT_PLAN.md`。在该协议定义的
反事实合同完成前，不选择 Transformer、GRU、state-space model、图模型、对比学习、
外生模态或最终分数形式；这些都是待检验的实现选择，不是方向 A 的前提。

方向 B 的 B1 双证据修复保留为已确认的受控机制基线。B2a/B2c/B3 的连续关系漂移
transfer 假设已按各自冻结协议关闭，不能把 B 的未解决 transfer 问题误写成 A2 的
结论。

## 研究动机

多变量时序中的异常不只是一点偏离是否足够大。相同终点值、相同局部幅度，可能来自：

```text
正常的缓慢状态迁移
可解释的协同响应
不受支持的突然跳变
不相容的变化轨迹
```

它们的区别在于变化是如何从事件前状态发展而来，而不必能化约为同一个点级 residual
的大小或尾概率。因此，方向 A 研究的是：**正常动态过程允许哪些状态转移和局部轨迹，
观测到的变化是否与这些过程兼容。**

这一定义可以利用预测误差、重构误差、轨迹似然、转移能量、表示距离、变化点证据、
跨变量协同证据或真正事前可用的外生信息，但不预设其中任何一项必须是最终分数。

## A-v1：已关闭的条件残差路线

A-v1 将上述动机收窄为：模型从 target-blind visible context 估计条件均值/尺度，
再以单一条件 residual 或 innovation 的 tail surprise 评分。最终四格主消融为：

```text
A00: constant context + MSE
A10: visible context  + MSE
A01: constant context + Gaussian NLL + conditional tail
A11: visible context  + Gaussian NLL + conditional tail
```

它使用多变量 denoising Transformer、点/变量遮蔽、互补 conditional mask、局部
mean/std/trend/high-frequency context 与独立 normal-reference tail map；它不使用文本
或真正的外生多模态输入。完整实现和历史设计快照保留在：

```text
PatternAD-main/archive/direction_a/source_snapshot/
```

冻结的 P1-v2-holdout 完成了 `10 generator seeds x 3 model seeds x 4 cells`。虽然
`A11-A00` macro AP 提升 `+0.03494`，两个决定性 gate 均失败：

```text
A11-A01 matched ordering:                 +0.00667, 95% CI [-0.100, 0.100]
maximum regime-FPR gap relative reduction:  5.40%,   95% CI [-20.20%, 24.97%]
```

分解显示 A11/A01 提升 quiet/volatile same-deviation ordering `+0.4556`，却使
slow-drift versus abrupt-shift ordering 降低 `-0.6667`。随后严格 past-only level
innovation、delta innovation、predicted-scale tail 和 causal-state tail 也未能同时
通过两类机制：最终 state-tail 的 quiet/volatile 为 `1/3`、abrupt/gradual 为 `0/2`。

该结果关闭以下主张，而不是关闭方向 A：

```text
一个可校准的 scalar residual / innovation surprise
可以同时承担背景状态语义与突变 onset 语义。
```

不得重启 A-v1 的 tail bin、cluster、shrinkage、transition-loss、手工权重或
residual fusion 调参；也不得将它扩展到真实数据或 confirmation seed。可审计证据见
`PatternAD-main/archive/direction_a/README.md`。

## A2：事件与转移语义

对每个候选事件时刻 `t`，A2 将信息划分为：

```text
P_t = X[t-H : t)             # event-pre observable state
Y_t = X[t : t+L]             # trajectory to be judged
```

A2 的问题是：在正常过程下，`Y_t` 是否与 `P_t` 兼容。`Y_t` 可以是原始轨迹、
变化路径、状态转移表示或其组合；模型如何表示它必须由先冻结的反事实合同决定。

最低信息约束如下：

1. 用于描述或路由 event-pre state 的任何量只能读取 `P_t`，不能读 `Y_t`。
2. 训练、状态发现、normal reference、outer calibration 和 test 必须时间隔离；
   test labels 不能决定表征、阈值、模型选择或模态选择。
3. 若引入外生变量或其他模态，它必须在 `t` 前可获得，并需作为单独证据源记录其
   时间戳、缺失率和可用性，不能以未来结果或标签代理。
4. A2 必须包含与 terminal residual/endpoint magnitude 匹配的反事实对，证明其
   信号来自轨迹/转移兼容性，而不是重新包装点值偏离。

首个合成合同至少应同时包含：相同事件前状态和终点、但 gradual normal transition
与 abrupt incompatible transition 不同的配对；以及正常协同迁移、相同幅度但不受支持
的迁移、和无事件 normal control。通过该合同之前，不产生真实 benchmark 或部署优越性
主张。

## 与方向 B 的关系

方向 B 研究多个受限证据源能否一致修复同一目标；A2 研究事件前状态能否支持随后
一段演化。两者可共享数据审计、split/provenance 和某些基线，但不能互相替代：

```text
B: temporal/cross evidence 对终点目标是否一致。
A2: event-pre state 对后续 trajectory/transition 是否兼容。
```

B1 的 `R_T`、`R_C`、`D` 可以作为 A2 的外部基线或诊断，不能在没有独立协议的
情况下直接充当 A2 状态、标签或融合输入。
