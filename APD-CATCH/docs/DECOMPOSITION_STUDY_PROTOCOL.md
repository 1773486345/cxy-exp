# 检测导向分解研究协议 v0

> 阶段二 A 只固定研究契约、结果规则和原版 CATCH 基座测试。本协议不定义 `decomp_catch`，不授权新增分解模型、真实数据训练或原版 CATCH 行为改动。

## 研究问题

在保持原版 CATCH 的训练和完整窗口重构不变时，对真实观测窗口和相应重构窗口应用同一个固定结构分解，并分别计算分量误差，是否能够提供区别于最终总残差的异常证据？

第一阶段的目标是验证分量证据是否存在，不是追求 SOTA，也不以任一数据集提升作为保留条件。

## 唯一研究变量

第一项实验只比较下列两种**评分读取方式**：

| 条件 | 允许的内容 |
| --- | --- |
| 原版 | 只使用 CATCH 现有连续异常分数 `original_score`。 |
| 固定分解对照 | 在完全相同的 checkpoint、相同输入窗口、相同原始重构输出上，额外计算固定分解后的 `slow_score`、`fast_score` 与 `fusion_score`。 |

除这个额外的后处理评分外，以下内容必须逐项保持不变：

- `CATCHModel` 及其参数；
- 模型参数、训练目标、optimizer、epoch 和 seed；
- 数据窗口、数据切分、RevIN、频率辅助损失和通道关系模块；
- 原始重构输出及原版连续评分输出。

不得重新训练、微调、替换 checkpoint，或为了分解结果修改任一原版 CATCH 超参数。

这里的“一次训练”指一个正式实验运行可严格按原版协议训练 CATCH 一次；`original_score`、`time_score`、`slow_score`、`fast_score` 与 `fusion_score` 必须共享这一次训练产生的同一个 checkpoint。不得为 slow、fast 或 fusion 分别训练或微调模型。已有合格 checkpoint 可以直接复用；没有 checkpoint 时，后续正式实验只能训练原版 CATCH 一次。本阶段不运行真实数据训练。

## 第一版固定分解

第一版只允许逐窗口、逐通道的固定移动平均：

```python
slow = moving_average(x)
fast = x - slow
```

对真实窗口 `x` 和原始 CATCH 重构窗口 `x_hat` 使用**同一个**算子、同一个预注册窗口和同一个边界规则：

```python
slow_x = moving_average(x)
fast_x = x - slow_x
slow_hat = moving_average(x_hat)
fast_hat = x_hat - slow_hat
```

移动平均采用中心窗口；窗口两端按端点复制后求等权平均。该边界规则对 `x` 与 `x_hat` 完全一致，不依赖数据集、异常、标签或分数。

## 当前阶段的数学解释边界

令：

```text
e = x - x_hat
M = fixed moving-average operator
```

由于同一个固定线性算子同时作用于观测与重构，当前分量误差满足：

```text
slow_x - slow_hat = M(e)
fast_x - fast_hat = (I - M)(e)
```

因此，本阶段是 **post-hoc reconstruction-error band decomposition**，也可准确称为“固定低频/高频重构误差证据门控”。它没有在模型输入、编码器或隐表示中分解序列。

本阶段能够回答：

1. CATCH 的低频重构误差和高频重构误差是否具有不同异常响应；
2. 分开读取这两类误差是否比总重构分数提供更多信息；
3. 等权融合是否值得进入下一阶段。

本阶段不能回答：

1. 不同时间序列成分是否被独立建模；
2. 分解是否改变了 CATCH 学习到的表示；
3. 趋势、周期和不规则分量是否形成专门化隐表示；
4. 检测导向分解是否整体成立。

第一版明确不允许：

- 可学习分解、adaptive cutoff、多窗口搜索或动态融合；
- EMA 在线状态、下一点预测、Gaussian NLL 或第三个分量；
- 由测试结果、测试标签或异常类型选择分解器、窗口或融合规则。

## 移动平均窗口预注册

对每一个原版 CATCH 配置，令 `P=patch_size`、`L=seq_len`。移动平均窗口唯一固定为：

```text
W = argmin_{w in {1, 3, 5, ..., <= L}} (abs(w - P), w)
```

其中二元比较先最小化与 `patch_size` 的距离，再在距离相同的合法奇数中选择较小值。因此 `W` 必为奇数、不会超过 `seq_len`，并且是与 CATCH patch size 最接近的合法奇数。相同数据集的所有方法、seed 和对照使用同一个由其原版 CATCH 配置导出的 `W`；第一轮不得搜索其他候选。

若该规则无法应用于某个现有配置或连续评分对齐方式，记录为阻塞并停止该配置的分解比较。不得替换为测试后最优窗口。

## 分数定义与无标签标准化

`original_score` 是原版 CATCH 连续评分接口直接返回的分数，保持其既有时域误差、频域误差及 `score_lambda` 语义，不重算或替代。

原版连续分数的组成还必须作为归因诊断保留：

```python
time_score = mean((x - x_hat) ** 2, channel_dim)
original_score = time_score + score_lambda * frequency_score
```

一次评分必须保存 `original_score`、`time_score`、`slow_score`、`fast_score` 与 `fusion_score`；建议同时保存 `frequency_score`。`original_score` 仍是主要原版基线，`time_score` 只用于诊断，不得用它替换原版结果或删除原版 frequency score。

对每个已对齐的时间位置，分量分数定义为：

```python
slow_score = mean((slow_x - slow_hat) ** 2, channel_dim)
fast_score = mean((fast_x - fast_hat) ** 2, channel_dim)
```

`fusion_score` 不学习权重。先对两个分数分别使用固定、非测试标签来源的统计量标准化：

```python
slow_z = (slow_score - slow_location) / (slow_scale + epsilon)
fast_z = (fast_score - fast_location) / (fast_scale + epsilon)
fusion_score = 0.5 * slow_z + 0.5 * fast_z
```

统计量来源固定如下：

1. 若原版运行已保留其既有的无标签 validation 段，则以最终选定的原版 checkpoint 在该 validation 段上产生 `slow_score` 与 `fast_score`，分别记录均值为 location、标准差为 scale。
2. 若现有原版协议没有独立 validation 段，则以其训练段重构分数的均值和标准差替代；不创建新切分。
3. `epsilon` 固定为 `1e-8`，只用于数值稳定性；不得按数据集、测试分数或标签调节。

用于估计这些统计量的 validation 或 train 段必须采用与测试连续评分相同的非重叠窗口协议：从原版 loader 的底层数据读取对应段，以与 `mode="thre"` 等价的窗口化方式重新构造只用于评分的 loader。不得使用 shuffled 或 step=1 的 train/validation loader 产生的重复窗口分数，也不创建新数据切分。每次参考评分必须记录原始数据段长度、实际评分长度、丢弃的尾部长度、评分窗口大小、窗口步长与时间索引对齐规则。

不得使用测试分数总体统计量、测试标签、异常比例或最终指标来确定 location、scale、窗口或融合权重。测试标签只可用于最终指标计算。

## 预注册判定

合成异常的类别、种子、数据、原版配置和评价指标必须在运行前写入 run manifest。以下判定在结果产生前固定：

1. 对每个含至少 10 个标注异常点的合成异常类别，计算异常点上的 Spearman 相关系数 `rho(slow_score, fast_score)`。若所有可评估类别的 `rho >= 0.98`，则判定分量证据未分化。
2. 每个合成异常类别分别计算 `slow_score` 与 `fast_score` 的 AUC-PR。只有当至少一个类别满足 `slow_auc_pr - fast_auc_pr >= 0.01`，且另一个类别满足 `fast_auc_pr - slow_auc_pr >= 0.01` 时，才判定至少两类异常呈现不同分支响应。
3. 对每个类别令 `K` 等于该测试段标注异常点数。若存在标注异常点不在 `original_score` 的 top-`K`，但位于 `slow_score` 或 `fast_score` 的 top-`K`，则记为一个“分量发现的原始低排名异常”。至少一个类别必须出现该事件。
4. 主比较为预注册合成类别与 seed 的平均 AUC-PR。以固定配对 bootstrap（10,000 次、随机种子 20260717）计算 `fusion_score - original_score` 的 95% 单侧下界；该下界不得低于 `-0.01`，才视为等权融合未显著低于原版。
5. 任一必要判定失败时，停止动态融合与可学习分解；不自动设计补救模块、路由器、额外分量或新主干。

真实数据只在上述合成证据已按预注册标准成立、且经过人工审核后才进入后续阶段；不得用“任一数据集提升即可保留”替代这些判定。

## 当前门控失败的适用范围

当前停止规则不能被解释为“本实验失败，因此检测导向分解整体失败”。若当前门控失败，只停止：

- 基于相同 post-hoc slow/fast 分数的动态融合；
- 基于相同残差频带划分的权重搜索；
- 对该固定后处理方案增加 adaptive cutoff 或 router。

它不直接否定下列独立且仍须重新预注册的研究方向：

- 在 CATCH 输入或隐表示阶段分别建模结构分量；
- 共享 CATCH 主干的分量级训练；
- 其他经过独立预注册的检测导向分解方案。

即使当前门控成功，也只能说明低频和高频重构误差存在互补信息，不能直接宣称已经得到检测导向的时序分解模型。

## 基座契约与单元测试边界

`tests/test_catch_reconstruction_contract.py` 只锁定原版 CATCH 的轻量接口：可初始化性、完整窗口重构形状、受控随机状态下的 evaluation 可复现性、连续评分长度与标签对齐，以及标签不进入连续评分接口。

原版 `CATCHModel` 的 channel mask 在 evaluation mode 仍调用 Gumbel-Softmax。因而不恢复 RNG 状态的两个连续前向并非严格确定；本阶段不修改这一既有行为。契约测试通过在两次前向前恢复同一 RNG 状态，验证受控条件下的可复现性，并把无 RNG 控制的严格确定性列为基座限制。

单元测试不能验证真实 checkpoint 的训练历史、真实数据指标、发布脚本的端到端行为、分解实现的正确性或分量证据的统计结论。这些都需要后续经人工确认的最小评分实现与独立运行记录。

## 阶段二 A 禁止事项

- 不创建 `ts_benchmark/baselines/decomp_catch/`；
- 不修改 `ts_benchmark/baselines/catch/`、`ts_benchmark/baselines/apd_catch/`、数据加载器或任何原版训练脚本；
- 不新增训练脚本，不运行真实数据训练或全部 benchmark；
- 不设计动态融合、可学习滤波器或其他新模型机制。
