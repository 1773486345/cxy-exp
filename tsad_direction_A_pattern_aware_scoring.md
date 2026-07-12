# 方向 A：模式感知残差语义建模（已关闭）

记录日期：2026-06-30  
关闭日期：2026-07-12

## 结论

方向 A 的核心主张是：一个可校准的条件 residual surprise 可以同时表达正常
波动背景与突变 onset 的异常语义。该主张已被本项目的冻结机制实验否定，因此
不再继续调参、融合分数、扩展 seed、进入真实数据或执行原 A factorial。

这不是训练未完成导致的暂停，而是主机制 gate 失败后的终止：

1. P1-v2-holdout 的 `10 generator seeds x 3 model seeds x 4 A cells` 中，
   `A11-A01` matched-ordering 仅 `+0.00667`，95% CI
   `[-0.100, 0.100]`；maximum regime-FPR gap 相对 A00 只下降 `5.40%`，
   未达到预注册的 `25%`。
2. past-only level innovation 的 same-deviation 为 `0/3`，
   abrupt/gradual 为 `1/2`。
3. past-only delta innovation 和 predicted-delta-scale tail 虽使
   abrupt/gradual 达到 `2/2`，但 same-deviation 仍是 `0/3`。
4. 最终 causal state-tail 诊断只读 `x_{<t}` 的 causal GRU state，在独立
   normal reference 上拟合 `state-cluster x variable` tail。它的 normal
   regime-FPR gap 为 `0.01873`，但 same-deviation 仅 `1/3`，
   abrupt/gradual 为 `0/2`，未通过推进 gate。

因此，当前证据不支持以单一 scalar residual surprise 同时承载这两类语义。
后续如重启相关问题，必须作为新的、与方向 A 原主张分离的研究项目。

## 已归档证据

为避免活跃工作区继续堆积无用 A 原始结果，保留了可审计的最小归档：

```text
PatternAD-main/archive/direction_a/p1_v2_holdout/
PatternAD-main/archive/direction_a/final_state_tail/
PatternAD-main/archive/direction_a/source_snapshot/
```

其中包括冻结输入、run plan、汇总统计、最终 score arrays 及未提交 A 代码快照。
原始 P1 cell 输出和可再生 synthetic artifact 已从活跃结果区删除；归档说明见
[`PatternAD-main/archive/direction_a/README.md`](PatternAD-main/archive/direction_a/README.md)。

## 当前替代路线

活跃路线已切换为方向 B：多证据一致性修复。它不是方向 A 的变体：B 不试图解释
同一个 residual 所处的模式，而是让受限 temporal/cross evidence 对同一 target
分别修复，并保留各自的 residual 与 prediction disagreement。

当前 B1 使用输入受限的 reliability calibration，五个 GPU seed 的 synthetic
机制确认全部通过。设计、结果与下一阶段边界见：

```text
tsad_direction_B_multi_evidence_repair.md
PatternAD-main/B1_EXPERIMENT_PLAN.md
```

方向 A 的完整历史理论与实验记录已经作为快照保存，不应再被当作当前执行方案。
