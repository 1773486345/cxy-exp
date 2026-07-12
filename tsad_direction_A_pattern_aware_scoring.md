# 方向 A：动态结构感知的异常语义

记录日期：2026-06-30  
更新日期：2026-07-13

方向定位：

```text
Dynamic-Structure-Aware Anomaly Semantics
```

## 当前状态

方向 A 仍然开放。已关闭的是 A-v1 以及 A2 下已经过冻结检验的四个具体模型路线，
而不是“动态结构会改变异常含义”这个研究动机。

```text
A-v1 (closed):
  用一个条件 residual / innovation tail surprise 同时表达波动背景与 abrupt onset。

A2 (archived synthetic research question; the v2 detector-development branch is paused):
  给定事件前可观测状态，判断随后一段轨迹或状态转移是否与正常演化兼容；
  M1 条件全轨迹混合模型、M2 对比兼容能量、M3 离散转移码、M4 显式地标/方向支持均已关闭。

A3 (active model design and implementation):
  将事件建模为 event-pre 可观测触发所允许的多通道响应图；模型分别预测
  每个通道是否响应、何时响应、向哪个方向响应，以定位延迟、缺失与错误传播。
```

A2 的协议、模型和开发状态在
`PatternAD-main/research/direction_a/A2_EXPERIMENT_PLAN.md`。A2 的 v2 反事实合同
已经通过多 seed 的构造审计；模型路线仍必须独立命名、先冻结再运行。模型的 GRU、
Transformer、state-space、图、对比学习、外生模态或最终分数形式都不是方向 A 的前提。

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

首个合成合同至少应同时包含：相同事件前状态、终点和实际观测轨迹的变化摘要、但 cue-compatible
scheduled transition 与 cue-incompatible timing transition 不同的配对。两类 onset 的边际
分布必须相同，避免仅按全局时刻或最大斜率通过；以及正常协同迁移、相同目标轨迹但不受
支持的迁移、和无事件 normal control。通过该合同之前，不产生真实 benchmark 或部署
优越性主张。

其中 timing cue 必须是 `P_t` 内可直接观测、可由预先声明的原始状态规则恢复的信号，
不能借助 generator latent state；但 cue 单独和 onset 单独又都不能预测正常/不兼容标签。

### 已完成的 A2 模型证据

A2 的 v2 合同已经完成构造审计；它要求 primary pair 的事件前状态、终点和增量摘要
相同，且 cue 单独、onset 单独均只能达到机会水平。下列路线都只在这一问题上被关闭，
不等于方向 A 的动机失效：

| 路线 | 已检验的主张 | 结果与停止结论 |
| --- | --- | --- |
| M1：条件全轨迹混合似然 | `P_t` 条件下整段 `Y_t` 的密度可形成稳定兼容性异常分数 | 冻结四 seed 确认 `0/4` complete passes，关闭 M1。 |
| M2：正常对比兼容能量 | `P_t` 与未来内部增量的连续 pair energy 可形成稳定校准分数 | v2 确认 `2/4` complete passes；关系排序稳定，但背景 FPR / normal control 不稳定，关闭 M2，不调温度、分桶或阈值。 |
| M3：离散转移码兼容性 | `P_t` 预测有限个正常未来转移码，候选码意外或远离正常码支持即异常 | 冻结开发种子 `5115/6320` 中，五码占用为 `[1021, 0, 0, 0, 0]`，码本塌缩；primary 仅 `8/16` 正方向，中位 tail margin `-0.01143`。虽二级协同 `16/16`、正常控制和全局背景 FPR 均通过，仍未通过主任务与码本覆盖门槛，关闭 M3-v1。 |

M3 的结果目录为
`PatternAD-main/result/a2/m3_v1_development_seed6320/`。由于开发 run 已失败，未运行其
unconditional control 或跨 seed confirmation，也不允许以改码数、权重、码本初始化或阈值
的方式重试该路线。

因此，当前 A2 不是“没有动机”，而是已有三类不同的可学习评分形式在同一受控合同上被
负证据约束：全轨迹密度、连续兼容能量和离散码支持都尚未给出可稳定确认的检测器。

M4 是对这一边界的最后一个冻结检验：它不学习整条轨迹的密度、pair energy 或有限隐码，
而是用 time-disjoint normal reference 检验两个显式可观测事实：事件前相近状态下，未来最强
内部变化是否发生在受支持的位置，以及该变化的跨变量方向是否受支持。M4 的 `P_t` 特征仅为
终点和最后七个内部增量，`Y_t` 表征仅为最大内部增量的位置和方向；不读 onset、cue mode、role
或生成器元数据。开发 `5120/6330` 全部通过（primary `14/16`、secondary `15/16`、背景 FPR
`4.95%`），matched unconditional control 的 primary 降至 `5/16`，因此 event-pre dependency
成立。然而冻结确认 `5121..5124/6331..6334` 仅 `1/4` complete passes：primary 分别为
`14/16、11/16、12/16、13/16`，而全部合同预检、背景 FPR 和 normal controls 均通过。失败在
原始地标/方向分数中同样存在，不能归咎于 tail 映射。M4 已关闭，不修改近邻数、特征长度、
地标规则、方向权重、阈值或 seed。

因此 A2-v2 的 detector-development 分支现在暂停：M1 全轨迹密度、M2 连续兼容能量、M3
离散码支持和 M4 显式地标/方向支持都未能通过各自的冻结确认。这不等于方向 A 的动机被否定，
也不等于合同没有可观测关系；它只意味着不能在这个合同上继续提出第五个分数形式。后续方向 A
工作必须先独立论证新的任务或数据合同，再选择模型。当前不产生真实 benchmark、与 B 的优劣
比较或多模态主张。

## A3：触发-响应图模型（进行中）

A2 暂停后，方向 A 不继续在同一个 `P_t -> Y_t` 合同上更换第五种兼容性分数，而是转向
一个不同的模型对象：事件前原始窗口中的可观测触发应当允许怎样的多通道响应结构。A3 的首个
模型 G1 以 past-only encoder 编码 `P_t`，并为每个原始通道预测
`(是否响应, 响应位置, 响应方向)`；`Y_t` 由固定的可观测 transition extractor 转为同样的
token，分别得到 activation、delay 和 direction 的机制证据。G1 已关闭；其有效结果仅作为
“独立节点预测不足以稳定校准完整响应图”的负证据保留。

其主任务不是 A2 的“轨迹整体是否兼容”，而是：**给定可观测触发，候选响应图是否为正常
机制允许的路由、延迟和传播。** A3-v1 将先在独立的两种正常 response mode 合同上检验：
misrouted response、partial propagation 和 untriggered response。primary 成对样本在两个
模式上平衡，使 trigger 单独和完整 future response mode 单独都不能区分正常/错误路由；
只有二者关系才决定标签。实现与预注册 gate 见
`PatternAD-main/research/direction_a/A3_EXPERIMENT_PLAN.md`。A3 与 A2 分置于
`config/a3`、`scripts/a3`、`result/a3` 和 `A3TriggerResponse`，不混入方向 B。

A3 的首个具体模型 G1（factorized GRU graph decoder）已完成唯一冻结开发 run：三种机制
配对关系均为 `15/16`，但 normal routed response 的 delay control 仅 `10/16`，ordinary
background 的 activation FPR 为 `10.23%`，超过 `10%` gate。因此 G1 已关闭，不做调参、
ablation 或 confirmation。该结果说明“按通道独立预测节点”不足以稳定地校准一张正常响应图，
不等于 A3 的触发-响应动机失败。

下一模型为 G2：`A3ObservableGraphGrammar`。它以固定原始规则从 `P_t` 提取末端线性、
有足够幅度的触发状态；将每个通道的固定 `(active, onset, direction)` 编码成可观测图 token；
再以自回归方式学习 `p(整张 response graph | 触发状态)`，保留节点依赖而不使用 learned
codebook。G2 的原始触发/图审计已在合同 seed `7101..7105` 通过，并完成唯一冻结 GPU 开发
run `7101/8201`：misrouted、partial propagation、untriggered 三个结构关系均为 `16/16`，
normal routed/no-trigger controls 分别为 `15/16`、`14/16`；但 ordinary background FPR 为
`11.60%`，超过冻结 `10%` gate。因此 G2 已关闭，不做 ablation、confirmation、阈值修改或
模型调参，也不进入真实数据。

G3：`A3CounterfactualEffectGraphGrammar` 已完成唯一冻结 GPU development run
`result/a3/a3_g3_development_seed8301_gpu/`（contract/model seed `7101/8301`）。它以 `P_t`
正常窗口拟合多通道 continuation，并按固定、可观测的 trigger state 加入 normal response
template；只将 `Y_t - baseline(P_t)` 的 activation/onset/direction 联合图送入 grammar，最终
证据是完整 effect graph 的 surprisal，而非 terminal residual 的尾分数。原始审计通过，
`Y_t` 替换不改变 trigger/baseline；misrouted、partial propagation、untriggered 三个配对 gate
均为 `16/16`，normal routed/no-trigger controls 为 `16/16`、`15/16`。但 ordinary background
FPR 为 `92/733 = 12.55%`，超过冻结 `10%` gate。说明反事实 effect representation 仍不能稳定
校准普通背景，G3 已关闭；不得执行 ablation、confirmation、retune、阈值/特征 probe 或真实数据实验。
方向 A 返回模型设计，下一路线必须在写代码前独立说明为何能改变 G1/G2/G3 共享的背景校准失败。

作为下一模型前的协议工作，A3-v2 已冻结 independent-background calibration contract：保留 `10%`
operating FPR，但不再以单条 768 点背景流的 733 个重叠窗口估计它，而是对每个正常噪声 regime
生成 1,024 个独立、固定 regime、独立 RNG 子流的 `H+L` 块（共 2,048 块）。未来模型必须同时
满足 pooled 与每个 regime 的 95% 单侧 Wilson 上界不超过 `10%`；pooled 2,048 块下最多允许
182 个 exceedances（`8.89%` point FPR）才有足够置信界余量。该协议只修复评估样本独立性与
不确定性，不能重新评分、追认或重启 G1/G2/G3，也没有授权任何 GPU detector run。详见
`PatternAD-main/research/direction_a/A3_BACKGROUND_CALIBRATION_V2.md`。

随后静态审计发现 A3-v1 的根本可辨识性问题：四个 response channels 上 normal latent loading
`[0.88, 0.61, -0.56, 0.43]` 与 injected route `[1.00, 0.78, -0.70, 0.42]` 的绝对余弦为
`0.99688`。因此正常 latent state transition 本来就会生成几乎同型的 response graph，G1/G2/G3
共同的 background failure 不能仅归因于模型容量或 calibration。新的 route-identifiable successor
contract 保留 A3 成对关系、normal process、splits、response amplitude 与双 mode 平衡，只改为与
背景 loading 正交的 route `[1,-1,1,0.6744186046511628]` 及其相反 mode；原始关系审计和 2,048
independent background audit 已通过。它是新任务合同，不能重跑 G1/G2/G3。N1
`Background-Nulling Route Graph` 仅冻结了 normal-only PCA preflight：先从 normal increments
估计背景一维子空间并投影去除，再构造 trigger-conditioned route graph；只有 preflight 通过后才会
实现 detector 或安排 GPU run。该 preflight 已按 all raw channels 重做并通过：normal optimization
increments 拟合的背景因子与 audit-only loading 对齐 `0.99974`，正交 route 在投影后保留
`0.99993`；完整 CPU fitting/reference/calibration smoke 也已通过。N1 的 detector 固定为
`A3BackgroundNullingRouteGraph`：只用 normal optimization 拟合 rank-1 全通道 PCA，投影 future
increments 后以 joint grammar 评分，outer calibration alpha 预注册为 `0.05`，并在 2,048
independent background blocks 上同时要求 pooled 与每 regime 的 95% 单侧 Wilson 上界不超过 `10%`。
其单次 GPU development (`seed 8401`) 已完整通过：三类配对 gate 均为 `16/16`，normal controls
均为 `15/16`，independent background pooled FPR 为 `84/2048 = 4.10%`、95% 上界 `4.89%`，两个
regime 的上界为 `1.13%`、`9.10%`，均低于冻结 `10%` gate。根据协议，唯一允许的下一步是将
`condition_on_event_pre` 固定替换为 false 的 past-free control；该控制已完成，且 primary 从
`16/16` 降至 `13/16`（median tail margin 从 `+4.2281` 降至 `+0.2644`），低于冻结 `14/16` 门槛。
控制分析无配置哈希违规，因此 N1 的效果依赖可用的 event-pre 状态，而不是未来信息替代。下一步且
仅允许执行 `background_nulling_n1_confirmation_v1.json` 预注册的四组 CUDA seed 对
`7202/8402`、`7203/8403`、`7204/8404`、`7205/8405`；模型、校准和协议不再改动。四组必须全部
complete pass，否则 N1 关闭；不得 replacement seed、sweep、retune、额外 ablation 或真实数据实验。

## 与方向 B 的关系

方向 B 研究多个受限证据源能否一致修复同一目标；A2 研究事件前状态能否支持随后
一段演化。两者可共享数据审计、split/provenance 和某些基线，但不能互相替代：

```text
B: temporal/cross evidence 对终点目标是否一致。
A2: event-pre state 对后续 trajectory/transition 是否兼容。
```

B1 的 `R_T`、`R_C`、`D` 可以作为 A2 的外部基线或诊断，不能在没有独立协议的
情况下直接充当 A2 状态、标签或融合输入。
