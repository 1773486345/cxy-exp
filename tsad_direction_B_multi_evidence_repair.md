# 方向 B：多证据一致性修复

记录日期：2026-06-30  
更新日期：2026-07-13

方向定位：

```text
Multi-Evidence Consistency Repair
多证据一致性修复
```

## 当前进度：B1 为已确认的受控机制证据；B2/B3 已关闭

B1 Evidence-Conditioned Reliability Calibration (ECRC) 是方向 B 唯一通过多
seed 的 controlled synthetic mechanism result。它保留两个严格隔离的 repair
evidence paths，不使用 score fusion：

```text
temporal branch: target[t-H:t] only
cross branch:    non-target[t-H+1:t+1] only

R_T = (y - mu_T)^2
R_C = (y - mu_C)^2
D   = (mu_T - mu_C)^2
```

两个 GRU/head 不共享参数。`R_T`、`R_C` 与 `D` 分开输出、分开校准；B1 没有
`lambda`、没有 learned fusion、没有 agreement loss，也不把 `D` 反传给两个
repair head。cross branch 观察同期其他变量是同步证据假设，不是因果方向主张。

已冻结的 B1 GPU seeds `3101..3105` 全部通过预注册 gates：跨变量修复相对
target-mean MAE 的提升为 `82.4%--87.6%`；背景 normal 的最大可观测
reliability-bin FPR 为 `5.18%--7.78%`，disagreement-bin FPR gap 为
`1.09%--4.40%`；两类 dependency-break 的 cross/disagreement median tail
margins 均超过 `4.0`，全部 `16/16` target spikes 同时触发两个 residual paths。
这只构成 controlled synthetic mechanism evidence，不构成真实 benchmark 优势主张。

后续连续关系漂移 transfer 中，B2a-GC 虽证明终点 dependency-break 信号，
却未通过 normal-control/FPR gates；B2c 的 calibration-only 修补与 B3a 的
relation-history-conditioned cross repair 也在预注册 GPU smoke 中失败。B3a 的
冻结 replay 排除了 temporal checkpoint 混杂，仍只通过 `8/72` gates。因此 B2/B3
是有效的负证据，不再调参、不再补跑 confirmation seed，也不再进入真实数据比较。

按当前清理边界，B1/B2/B3 的专用模型、配置、脚本与结果目录均不保留在主工作区；
历史证据由本文件和 Git 历史追溯，不能被误写为当前可执行资产。当前活跃工作是方向 A
中的 `PatternAD` 多模态条件异常检测；方向 B 保留为已完成的机制研究与未来独立立项的
理论基础。下文的三/四分支、频域/趋势分支及融合公式均是长期候选，不是当前实现或
当前可运行实验。

方向 B 关注的是修复任务本身：不要让一个统一模型通过一条混合信息通路给出一个重构值，而是让多个受限证据源分别估计同一个被遮蔽目标，并利用修复误差与证据冲突共同进行异常判别。

一句话概括：

```text
正常观测应当能够被多种受限证据源一致修复；
异常观测往往表现为某些证据源修复失败，或不同证据源之间出现显著冲突。
```

---

## 1. 与方向 A 的边界

方向 A 当前研究模式感知的条件异常语义：

```text
给定数值窗口与事前可用的文本/外生上下文，
当前观测是否与正常动态机制兼容？
```

方向 B 研究修复一致性：

```text
同一个被遮蔽目标能否被多个受限证据源一致估计？
不同证据源的估计是否冲突？
```

方向 B 不负责解释同一偏离在不同动态或外生上下文中的异常语义。它只关心不同信息
约束下的独立修复是否能够共同支持当前观测。二者可以共享严格的目标遮蔽、时间切分
和校准原则，但不能将 A 的上下文条件分数直接当作 B 的 disagreement，也不能将 B 的
双分支残差融合为 A 的多模态贡献。

---

## 2. 研究动机

多变量时序观测通常由自身时间演化、变量间协同关系、周期性结构和长期趋势等因素共同决定。异常并不总是表现为单个变量对自身历史模式的偏离，也可能表现为某一信息来源无法支持当前观测，或者多个信息来源对同一观测给出互相矛盾的判断。单一路径预测或重构将不同来源的信息提前融合为统一表示，容易掩盖这些潜在冲突。

因此，方向 B 关注多证据一致性修复问题。其基本出发点是：对同一个被遮蔽目标，不同受限证据源应分别给出自己的估计；若这些估计均能接近真实观测且彼此一致，则该观测更可能符合正常行为；若某些估计显著失败，或不同估计之间发生冲突，则这种修复失败与证据冲突本身具有异常指示意义。

---

## 3. 研究现状

现有多变量时序异常检测研究主要围绕正常行为建模展开。MSCRED（AAAI 2019）通过多尺度 signature matrices 表征变量间相关性与时间演化，并利用重构残差进行异常检测与诊断。MTAD-GAT（ICDM 2020）利用时间维和变量维的图注意力机制，同时优化预测与重构任务。GDN（AAAI 2021）显式学习传感器关系图，并基于图神经网络预测变量的期望行为。这类方法证明了变量间关系和多尺度结构对异常检测的重要性，但不同信息来源通常被融合为统一表示，并最终通过预测误差或重构误差进行判别。

Transformer 和注意力机制进一步拓展了异常判别方式。Anomaly Transformer（ICLR 2022）提出 association discrepancy，将异常点的关联模式差异作为判别信号；TranAD（PVLDB 2022）利用 Transformer 建模多变量时间依赖，并结合自条件机制提升检测性能。近年的 DCdetector（KDD 2023）通过多尺度 dual attention 和对比学习构造判别式异常表征，试图缓解重构式方法对异常样本的过度拟合问题。这些方法已经超越了简单点值误差，但其核心仍是学习统一表征或统一异常分数，而不是保留多个受限证据源对同一目标的独立估计。

总体来看，现有研究已经从时间依赖、变量关联、多尺度结构和表示学习等角度显著提升了多变量时序异常检测性能。然而，大多数方法仍倾向于将不同来源的信息提前融合，并基于单一预测误差、重构误差、关联差异或统一表征距离进行判别。对于由多种因素共同决定的多变量时序观测，这种统一判别方式可能难以揭示不同证据来源之间的潜在冲突。因此，如何在保持证据来源差异性的基础上，对同一观测进行多角度修复，并将证据间的一致性或冲突性纳入异常判别过程，是方向 B 的主要切入点。

---

## 4. 长期任务定义（非 B1）

最终任务仍然是多变量时序异常检测。方向 B 的代理任务是：

```text
给定一个被遮蔽的变量片段或时间片，
让多个受限证据源分别修复它，
再利用修复误差和多证据不一致性产生异常分数。
```

普通重构：

```text
x_masked -> model -> x_hat
score = |x - x_hat|
```

方向 B：

```text
x_masked -> temporal branch      -> x_hat_temporal
x_masked -> cross-variable branch -> x_hat_crossvar
x_masked -> frequency branch     -> x_hat_frequency
x_masked -> trend branch         -> x_hat_trend

score = residuals + disagreement
```

其中：

```text
residuals = 每个估计值与真实值之间的误差
disagreement = 多个估计值之间的不一致程度
```

---

## 5. 核心关键点

### 5.1 证据源必须受限

如果所有分支都能看到完整输入，它们会学习相同 shortcut，disagreement 将失去意义。因此，每个分支需要明确的信息边界：

```text
temporal branch：
只看目标变量自身历史和邻近时间，不看当前被遮蔽片段。

cross-variable branch：
看其他变量，不直接看目标变量当前片段。

frequency branch：
看频域或周期结构，弱化局部点值复制。

trend branch：
看低频趋势，弱化高频局部扰动。
```

核心不是分支数量多，而是不同分支回答问题时受到不同信息约束。

### 5.2 residual 和 disagreement 都应保留

最终分数至少包含两类量：

```text
R = mean_k |x_target - x_hat_k|
D = disagreement(x_hat_1, x_hat_2, ..., x_hat_K)
```

只使用 residual 难以发现不同证据源之间的冲突；只使用 disagreement 又可能漏掉所有证据源同时失败的异常。因此，两者应共同参与异常判别。

### 5.3 证据一致性不是普通 ensemble

普通 ensemble 通常让多个模型看到相同输入后进行投票。方向 B 的关键区别在于：

```text
多个受限证据源在不同信息约束下回答同一个目标应该是多少。
```

因此，方向 B 的 disagreement 不是模型随机性的副产品，而是由信息来源差异产生的判别信号。

### 5.4 跨变量关系只是证据源

方向 B 不做：

```text
构造关系图 -> 判断关系图是否异常
```

而做：

```text
其他变量能否为目标变量修复提供有效证据？
跨变量估计是否与时间、频域或趋势估计一致？
```

这可以避免把动态关系变化本身直接定义为异常。

---

## 6. 主要挑战

### 6.1 信息泄漏

这是方向 B 的最大风险。如果分支间共享过多信息，或 mask 不严格，模型会复制目标值，导致 residual 和 disagreement 人为降低。

需要检查：

```text
temporal branch 是否看到目标当前点？
cross-variable branch 是否通过 embedding 泄漏目标变量？
frequency branch 是否能完整逆变换恢复目标？
trend branch 是否包含目标局部扰动？
```

### 6.2 证据源并非统计独立

时间、趋势和频域本来就相关。方向 B 不能声称这些证据源彼此独立，更稳的表述是：

```text
它们是在不同信息约束下形成的估计视角。
```

### 6.3 disagreement 不一定总是异常

正常系统中也可能出现短时证据冲突。因此，disagreement 不能单独作为最终判据，需要与 residual、训练集正常统计和事件持续性共同使用。

### 6.4 所有证据可能同时失败

如果异常影响整个窗口，多证据可能一起给出错误估计，disagreement 不一定高。因此，方向 B 不能只依赖 `D`，必须保留修复残差 `R`。

### 6.5 复杂度和消融压力

多分支方法容易被质疑为模块堆叠。需要通过消融证明：

```text
每个 branch 都有具体贡献；
disagreement 不是 residual 的重复；
收益超过计算成本。
```

---

## 7. 动机示例

### 7.1 单变量看正常，但跨变量不支持

流量上升后，压力应随之上升，但压力保持平稳。

```text
temporal branch：
只看压力历史，认为压力平稳是正常。

cross-variable branch：
看到流量上升，认为压力也应上升。

结果：
cross-variable residual 高，disagreement 高。
```

### 7.2 数值突变，但有其他证据支持

业务负载突然上升，CPU 随之上升。

```text
temporal branch：
只看 CPU 历史，可能认为 CPU 突然上升异常。

cross-variable branch：
看到负载上升，认为 CPU 上升合理。
```

如果其他证据也支持该变化，则可以降低误报。

### 7.3 频域异常但点值范围正常

某变量值域正常，但出现异常高频振荡。

```text
temporal branch 可能 residual 不高；
frequency branch 会产生较高修复误差；
branch 之间可能出现冲突。
```

---

## 8. 后续扩展候选（非 B1）

第一版建议只做三个分支，避免过重：

```text
temporal branch
cross-variable branch
frequency / trend branch
```

### 8.1 输入与 mask

输入窗口：

```text
x in R^{B x T x D}
```

随机选择目标：

```text
target variable i
target time span t1:t2
```

构造 mask：

```text
x_masked
```

### 8.2 Temporal branch

只使用目标变量自身上下文：

```text
x[:, :, i] with target span masked
```

输出：

```text
x_hat_temporal[:, t1:t2, i]
```

### 8.3 Cross-variable branch

使用其他变量：

```text
x[:, :, -i]
```

不直接使用目标变量当前片段。

输出：

```text
x_hat_crossvar[:, t1:t2, i]
```

### 8.4 Frequency / trend branch

使用低频趋势或频域表示：

```text
moving average
FFT band features
multi-scale pooling
```

输出：

```text
x_hat_freqtrend[:, t1:t2, i]
```

### 8.5 分数

每个分支输出：

```text
mu_k      = 该分支对目标片段的修复值
logvar_k = 该分支对自身修复不确定性的估计，可选
h_k      = 该分支的证据表征，可选
```

基础分数：

```text
r_k = |x_target - mu_k|
D = variance(mu_1, mu_2, ..., mu_K)
S = calibrated(mean_k r_k) + lambda * calibrated(D)
```

如果加入不确定性，则使用：

```text
normalized_residual_k = r_k / sigma_k
```

但需要避免模型通过人为增大 `sigma_k` 掩盖异常。

---

## 9. 后续实验验证

需要验证：

```text
1. 多分支修复是否优于单一重构。
2. disagreement 是否提供 residual 之外的增益。
3. 信息隔离是否必要。
4. temporal / cross-variable / frequency-trend 分支是否各自有贡献。
5. 无 point-adjust 指标下是否仍然稳定。
```

关键消融：

```text
single repair head
multi-head without information restriction
multi-head residual only
multi-head disagreement only
multi-head residual + disagreement
```
