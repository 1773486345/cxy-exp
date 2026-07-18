# 方向 A：基于结构分解的多变量时间序列异常检测——当前研究进展
## 一、研究起点与当前核心问题
本方向最初希望借鉴时间序列预测模型处理非平稳性的思路，将复杂多变量时间序列分解为具有不同动态特征的结构成分，再分别建模这些成分，以提高异常检测能力。

真实多变量时间序列通常同时包含缓慢变化趋势、周期与频率结构、快速局部波动以及不同时间尺度下的变量关系。原始序列中的这些结构相互叠加，可能使统一模型在同一表示空间中同时学习多种差异较大的动态规律，从而产生结构间干扰。

预测任务中的分解通常以降低预测误差为目标，但这种分解不能直接等同于适合异常检测的分解。对预测有利的分解可能把水平突变吸收到趋势中，把周期异常重新拟合为季节结构，或者将异常信息分散到多个成分。因此，本方向真正关心的不是单纯减小预测或重构误差，而是：
如何在现有多变量时间序列异常检测基座上引入结构分解，使不同动态成分获得有针对性的建模，并通过它们的联合重构或联合表示提高最终异常检测性能。

当前不再将研究问题限制为“残差校准”，也不预设异常只能存在于 residual 中。分解后的 trend、residual 等成分可以共同参与模型训练，但最终是否有效首先由真实数据集上的检测性能判断。

## 二、模型修改与实验的长期原则
当前已经明确的研究要求如下。
### 1. 修改必须契合研究动机
每项修改必须能够形成以下完整解释链：
> 具体难点 → 模型修改 → 作用机制 → 预期改善 → 真实数据结果

不能因为某个模块流行或容易实现就加入模型，也不能在结果不好后无目的堆叠模块。

### 2. 不限制修改幅度
大改、小改、参数增加或参数共享都不是优先顾虑。允许：
- 新增多分支；
- 使用独立主干；
- 改变训练损失；
- 加入分支交互；
- 使用可学习分解；
- 引入原始序列保留路径；
- 修改编码或解码结构。

是否继续某个方案，主要由两项条件决定：
① 修改能否清楚对应某个具体难点；
② 在真实数据集上能否相对原 CATCH 或其他可靠基线持平或更优。

### 3. 性能标准是“持平或更优即可继续”
不要求新模型在所有数据集全面超过原 CATCH。
可以保留的情况包括：
- 多数数据集优于原 CATCH；
- 部分数据集明显提升，其余基本持平；
- 平均结果持平，但改善了原 CATCH 较弱的数据集；
- 整体与原 CATCH 持平，但优于其他配置一致的基线；
- 性能基本持平，同时模型结构揭示了值得继续研究的有效信号。

### 4. 不再过度建设实验基础设施
此前曾围绕后处理分解实验建立协议、Gate、seed shard、finalizer、manifest、原子写入和中断恢复系统。这些工作严重偏离“直接改模型并跑真实数据”的目标。

当前原则是：
- 直接实现模型；
- 做必要 smoke test；
- 运行多个真实数据集；
- 与原 CATCH 和其他基线公平比较；
- 根据结果决定保留、调整或放弃。

不再为单一模型实验新增复杂 Gate、bootstrap、合成实验或运行基础设施。

## 三、历史探索及方向纠偏
### 1. APD-CATCH 旧探索
早期在 Codex 的推动下，研究问题逐渐被限制为预测残差和条件概率评分，并形成了 APD-CATCH 探索线，包括：
- 将原版 CATCH 的完整窗口重构改为 past-to-next-point 预测；
- Gaussian NLL；
- causal CATCH；
- EMA state；
- innovation encoding；
- recent innovation scale；
- adaptive frequency cutoff；
- target-blind 可见性约束。

这条线实际研究的是因果状态条件预测和异常分数校准，并没有直接回答最初的“检测导向结构分解”问题，因此已经冻结，不再作为方向 A 的活动主线。

### 2. 后处理分解 Gate
之后曾建立 decomp_catch，对原版 CATCH 的重构误差做移动平均低频/高频划分：
```
error = x - x_hat
slow_error = moving_average(error)
fast_error = error - slow_error
```
该实验实际上只检验低频和高频重构误差是否具有互补信息，并没有让结构分量进入模型训练。

围绕该实验又增加了：
- 合成异常生成器；
- 预注册 Gate；
- 三个固定 seed；
- seed shard；
- finalizer；
- manifest；
- 外部中断审计；
- CPU 运行包装；
- 原子写入工具。

正式实验未完成。后来确认，这个实验的科学价值不足以支撑如此高的工程成本，因此终止于 NOT_EVALUABLE，不用于判断检测导向分解是否有效。

曾经给出过删除这些废弃文件的清理指令，但当前对话中没有收到清理完成的确认。因此，相关文件是否仍存在，需要以实际仓库状态为准。它们即使仍存在，也不应继续开发或运行。

## 四、当前活动模型：MSD-CATCH
### 1. 模型定位
MSD-CATCH 的目标是直接在模型训练阶段引入结构分解，而不是仅在最终异常分数上做后处理。

当前模型主要包括：
① 三尺度自适应移动平均分解；
② 独立的 trend CATCH 分支；
③ 独立的 residual CATCH 分支；
④ 两个分支之间的双向 gated exchange；
⑤ trend 与 residual 重构相加形成最终重构；
⑥ 使用最终总重构异常分数作为主分数。

对应代码路径：
```
APD-CATCH/ts_benchmark/baselines/msd_catch/MSDCATCH.py
APD-CATCH/ts_benchmark/baselines/msd_catch/models/MSDCATCH_model.py
APD-CATCH/tests/test_msd_catch_smoke.py
APD-CATCH/scripts/run_msd_catch_screen.sh
```
原版 CATCH 路径（未修改）：
```
APD-CATCH/ts_benchmark/baselines/catch/
```

### 2. 三尺度自适应分解
模型对输入序列计算三个移动平均趋势候选。三个尺度与 CATCH 的 patch size 相关。
模型根据窗口和变量的时间统计特征，为三个尺度生成 softmax 权重，并得到自适应趋势：
```
trend = w1 × trend1 + w2 × trend2 + w3 × trend3
residual = x - trend
```
满足恒等式：`trend + residual = x`

该修改针对的难点是：
不同变量、窗口和数据集中的状态变化速度不同，单一固定趋势窗口难以同时适应短期状态变化和长期趋势。

### 3. 独立 trend / residual CATCH 分支
trend 与 residual 分别进入独立的 CATCH 分支。

该修改针对的难点是：
慢变化成分和快速变化成分具有不同的频率结构与变量关系，在统一主干中可能出现表示竞争，导致较弱的结构被主导成分覆盖。
两个独立分支允许慢变化结构和快速变化结构分别学习相应的正常模式。

### 4. 双向 gated exchange
两个分支并非完全独立。例如，正常快速波动的幅度可能受到慢变化运行状态控制；趋势状态变化也可能伴随局部快速变化。

因此，模型在分支频域通道 token 之间加入双向 gated exchange，恢复 trend 与 residual 之间的条件关系。

该修改针对的难点是：
直接分解可能切断原始序列中不同动态成分之间的联系。

### 5. 训练损失
两个分支分别保留原版 CATCH 的相关训练目标，包括：
- 时域重构；
- 频域相关损失；
- 通道关系损失。

此外还计算 trend 与 residual 重组后的总重构损失。
两个分支损失的权重固定为 0.5。

### 6. 当前唯一主异常分数
当前正式主结果只使用：`total_score`
计算逻辑：
```
x_hat = trend_hat + residual_hat
total_score = CATCH_score(x, x_hat)
```

模型同时输出过辅助分数：
- trend_score；
- residual_score；
- fixed fusion；
- anchored fusion。

但实验已经证明分支独立分数整体较弱，评分融合不能稳定改善 total，因此它们不再作为主结果。

## 五、评分融合实验及结论
### 1. 固定等权融合
最初将标准化后的 total、trend 和 residual 分数固定等权平均。
三数据集 PR 结果：

| Dataset | CATCH PR | MSD Total PR | Fixed Fusion PR |
| ---- | ---- | ---- | ---- |
| PSM | 0.4366 | 0.4304 | 0.4331 |
| Genesis | 0.2660 | 0.3108 | 0.2461 |
| GECCO | 0.4174 | 0.4065 | 0.3918 |

Genesis 中 total 明显优于原 CATCH，但 trend 和 residual 独立分数很弱，固定融合严重稀释了 total 的有效信号。

### 2. Anchored fusion
为避免弱分数无条件拉低 total，曾实现以 total 为锚点的增量融合。
模型根据训练正常段估计 trend/residual 相对于 total 的正常分歧阈值，只有当分支相对于 total 的额外偏离超过该阈值时，才增加 bonus。

三数据集完整对比：

| Dataset | CATCH | MSD Total | Fixed Fusion | Anchored Fusion |
| ---- | ---- | ---- | ---- | ---- |
| PSM | 0.4366 | 0.4304 | 0.4331 | 0.4314 |
| Genesis | 0.2660 | 0.3108 | 0.2461 | 0.3107 |
| GECCO | 0.4174 | 0.4065 | 0.3918 | 0.3957 |

Anchored fusion 基本保留了 Genesis total 的提升，但没有稳定超过 total，并且在 GECCO 上进一步下降。

**结论**：停止继续研究异常分数融合。MSD-CATCH 的唯一主分数设为 total_score。trend、residual 和 fusion 只保留为诊断输出。

## 六、六个真实数据集的冻结版对比
当前冻结版 MSD-CATCH 已经在六个真实数据集上与原版 CATCH 进行了公平对比。
固定实验条件：
- 相同数据切分；
- seed=2021；
- 相同 epoch；
- 相同 batch size；
- 相同 seq_len；
- 相同 patch 配置；
- 相同学习率；
- 相同 unfixed_detect_score 评价流程；
- MSD-CATCH 只使用 total_score 作为主异常分数。

### 全数据集指标总表
| Dataset | CATCH PR | MSD Total PR | Delta PR | CATCH ROC | MSD ROC | Delta ROC | CATCH/MSD Params (M) | CATCH/MSD Time (s) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| PSM | 0.43665 | 0.43044 | -0.00621 | 0.64755 | 0.64726 | -0.00029 | 3.77 / 7.55 | 604 / 3033 |
| Genesis | 0.26599 | 0.31077 | +0.04478 | 0.97354 | 0.98628 | +0.01274 | 230.11 / 460.61 | 2345 / 4844 |
| GECCO | 0.41742 | 0.40650 | -0.01093 | 0.96998 | 0.96438 | -0.00560 | 48.48 / 421.89 | 145 / 2199 |
| CalIt2 | 0.11331 | 0.12476 | +0.01145 | 0.83734 | 0.84874 | +0.01140 | 58.54 / 117.48 | 135 / 72 |
| NYC | 0.08012 | 0.06256 | -0.01756 | 0.81843 | 0.80806 | -0.01038 | 837.62 / 1676.81 | 218 / 435 |
| MSL | 0.16559 | 0.17403 | +0.00844 | 0.66196 | 0.66088 | -0.00109 | 210.88 / 422.16 | 6479 / 10115 |

### 汇总统计结果
- 平均 Delta AUC-PR：+0.00500
- 中位数 Delta AUC-PR：+0.00112
- 严格提升：3/6
- 明显提升：2/6
- 基本持平：2/6
- 明显下降：2/6
- 最大提升：Genesis，+0.04478
- 最大下降：NYC，-0.01756

定义：`基本持平 = |Delta AUC-PR| <= 0.01`

### 各分支独立 AUC-PR 对比
| Dataset | Trend | Residual | Total |
| ---- | ---- | ---- | ---- |
| PSM | 0.37701 | 0.42317 | 0.43044 |
| Genesis | 0.09845 | 0.09085 | 0.31077 |
| GECCO | 0.37777 | 0.27776 | 0.40650 |
| CalIt2 | 0.05700 | 0.11193 | 0.12476 |
| NYC | 0.06228 | 0.07620 | 0.06256 |
| MSL | 0.16715 | 0.16325 | 0.17403 |

### 实验结果文件路径
1. MSD-CATCH total 主实验输出：
```
APD-CATCH/result/msd_catch_total_screen/CalIt2.json
APD-CATCH/result/msd_catch_total_screen/NYC.json
APD-CATCH/result/msd_catch_total_screen/MSL.json
```
2. Anchored fusion 重训练产物：
```
APD-CATCH/result/msd_catch_anchored/
```

## 七、对当前结果的解释
### 1. 方向存在真实性能信号
MSD-CATCH 在 Genesis 上获得 +0.04478 AUC-PR，在 CalIt2 上获得 +0.01145，在 MSL 上获得 +0.00844。
因此，收益并不只存在于一个数据集。模型侧结构分解确实可能改善部分多变量异常检测任务。

### 2. 当前模型尚未稳定优于 CATCH
PSM 接近持平，GECCO 和 NYC 出现下降。
目前不能声称 MSD-CATCH 全面优于原 CATCH，只能得出：
当前结构分解方案整体平均略有正收益，并在部分数据集上表现明显改善，但跨数据集稳定性不足。

### 3. 分量单独评分不是当前贡献
trend_score 和 residual_score 在多数数据集上均弱于 total_score。尤其 Genesis 中，两个分支独立分数都很弱，但重组后的 total 明显改善。

这说明：
- 分支的价值主要体现在联合表示和联合重构；
- 分解不意味着各分量必须单独成为强异常检测器；
- 当前不能将“分量级异常分数互补”作为已经被结果支持的核心主张；
- 当前更合理的叙事是“分解后的专门化建模改善整体重构表示”。

### 4. 独立双主干成本较高
多数数据集的 MSD-CATCH 参数量约为原 CATCH 的两倍，符合双独立主干预期。
但 GECCO 存在异常参数膨胀：
- CATCH：48.48M
- MSD-CATCH：421.89M
参数增长达到约 8.7 倍。

推测与 gated exchange 或某个随通道数、token 数快速增长的映射有关，需要在下一轮修改前输出模块级参数统计。

### 5. 高维并不能直接解释下降
MSL 参数规模较大，但没有出现明显下降；NYC 明显下降，而 GECCO 的参数膨胀又远超一般双分支规律。

因此，当前不能简单归因于“通道越多，双分支越差”。更可能的结构问题包括：
- 硬分解丢失原始混合结构；
- 两个独立主干重复学习系统共性；
- 分支交互模块参数过大或行为不稳定；
- 某些数据需要原始序列中跨成分的整体关系。

## 八、当前明确保留与停止的决定
### 保留内容
- 三尺度自适应移动平均分解；
- 模型侧 trend/residual 分支建模；
- 分支之间恢复条件联系的总体思想；
- total_score 作为唯一主异常分数；
- 六数据集公平比较结果；
- 当前 MSD-CATCH 作为有效的中间基线。

### 停止方向
- fixed fusion；
- anchored fusion；
- 继续设计异常分数融合；
- 将 trend/residual 独立分数作为主证据；
- post-hoc reconstruction-error decomposition Gate；
- 因单一数据集下降而立即特调评分函数；
- 继续增加实验协议和运行基础设施。

### 尚未实施的下一版候选模型
候选模型名称：RSA-MSD-CATCH（Raw-Preserving Shared-Adapter MSD-CATCH）
设想架构：
- 共享 CATCH 主干；
- trend/residual 使用独立低秩 adapter；
- 低秩双向交互；
- 受分解置信度门控的 raw-preserving correction；
- 继续只使用 total_score。

截至当前对话，尚未收到 RSA-MSD-CATCH 已实现或已实验的报告，仅作为下一步候选，不能写入已完成工作。

## 九、下一步推荐方向
下一轮模型修改需要同时针对两个已经由六数据集结果暴露出的难点。
### 难点一：硬分解可能丢失原始混合结构
当前 MSD-CATCH 强制所有信息经过 trend/residual 分解。对 Genesis、CalIt2 有益，但 NYC、GECCO 的下降说明，某些数据中的正常模式可能依赖趋势、周期、快速波动和变量关系的联合表达。
可考虑增加受限的原始结构上下文，但不能直接添加 identity skip，也不建议再增加第三个完整 CATCH 主干。

### 难点二：两个独立完整主干重复学习系统共性
trend 和 residual 虽然动态不同，但来自同一个多变量系统，共享大量变量关系和时间规律。
优化方案：将两个独立 CATCH 主干改为「共享主干 + 分量专用低秩 adapter」，既保留分量专门化，也减少重复参数。

### 推荐候选完整结构流程
```
三尺度自适应分解
→ trend / residual
→ 共享 CATCH 主干
→ 两个分量专用低秩 adapter
→ 低秩双向交互
→ trend_hat + residual_hat
→ 受限 raw correction
→ final x_hat
→ total_score
```

### 前置准备与对比计划
1. 实现前前置任务：统计当前 MSD-CATCH 的模块参数量，定位 GECCO 参数膨胀来源
   - decomposition scale gate
   - trend CATCH branch
   - residual CATCH branch
   - gated exchange
   - 其他模块
   - total 总参数量
2. 下一轮统一实验配置，对比三组模型：
   - CATCH（原版基线）
   - MSD-CATCH（当前冻结版）
   - RSA-MSD-CATCH（新版候选）

## 十、当前方向 A 可以使用的阶段性结论
可以写入研究文档的稳健结论是：
在原版 CATCH 上引入多尺度自适应 trend–residual 分解、分支专门化建模和跨分支交互后，MSD-CATCH 在六个真实多变量时间序列异常检测数据集上获得了轻微但为正的平均 AUC-PR 增益，并在 Genesis 和 CalIt2 上取得明确提升。这表明模型侧结构分解具有实际潜力，但当前双独立主干结构尚不能稳定优于原 CATCH，在 NYC 和 GECCO 上仍存在性能下降和较高计算成本。

分支独立异常分数普遍弱于重组后的总重构分数，固定融合和基于正常分歧的 anchored fusion 均未稳定改善结果。因此，当前证据支持的是“分解后的专门化建模能够在部分数据中改善整体正常表示”，而不是“各分量能够独立形成可靠的异常证据”。

后续工作的重点应从异常分数融合转向模型内部结构：在保留分解优势的同时，通过共享表示、低秩专用适配和受限原始结构修正，降低双主干冗余并缓解硬分解造成的信息损失。

## 十一、当前仓库交接检查项
新对话或继续开发前，应核对以下事项。
### 已确认存在文件/目录
```
ts_benchmark/baselines/msd_catch/MSDCATCH.py
ts_benchmark/baselines/msd_catch/models/MSDCATCH_model.py
tests/test_msd_catch_smoke.py
scripts/run_msd_catch_screen.sh
result/msd_catch_total_screen/
result/msd_catch_anchored/
```

### 已确认未修改目录
```
ts_benchmark/baselines/catch/
```

### 需要实际仓库确认的废弃内容（建议清理）
此前曾提出清理以下废弃内容，但无清理完成报告；若存在则禁止运行、开发：
```
ts_benchmark/baselines/decomp_catch/
decomp_gate_v0 相关脚本
seed shard / finalizer / manifest 相关文件
docs 中的 decomposition Gate 文档
result/decomposition_study_v0/
```
清理注意事项：先根据当前 git status 和文件依赖核查，避免误删有用结果。

### 当前未确认完成任务
1. RSA-MSD-CATCH 或其他下一版模型实现与实验
2. GECCO 模块级参数统计
3. 六数据集上的下一版模型对比结果
