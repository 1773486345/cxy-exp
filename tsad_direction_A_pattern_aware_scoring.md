# 方向 A：面向多变量时间序列异常检测的结构分解与联合建模

## 1. 研究目标

方向 A 研究如何将时间序列预测领域中用于处理非平稳性和混合动态结构的分解思想，引入多变量时间序列异常检测模型，并使分解后的不同成分真正服务于异常识别，而不是仅作为预处理步骤或用于提高一般重构精度。

当前以 CATCH 作为基础模型。研究不从零构建新的异常检测框架，而是在 CATCH 的频域建模、通道关系建模和重构检测机制上，引入多尺度结构分解、分量专用建模及跨分量联合建模。

核心问题可以概括为：

> 真实多变量时间序列通常同时包含慢变趋势、周期变化、快速局部波动及复杂变量关系。若全部动态结构在同一频域空间中统一建模，不同成分可能相互干扰；但预测导向的传统分解又不保证分量对异常检测有判别价值。因此，需要研究如何在异常检测模型内部对不同动态成分进行有针对性的建模，并在保留原始系统结构的同时形成有效的联合异常分数。

本方向重点关注以下技术难点：

1. 不同动态结构在同一表示空间中的相互干扰；
2. 分解可能吸收、削弱或分散异常信息；
3. trend 与 residual 虽具有不同动态性质，但仍来自同一多变量系统；
4. 分量独立检测分数不一定比整体重构分数更有效；
5. 独立双分支能够保留分量专门化，但可能造成巨大参数和计算冗余；
6. 完整共享编码能够降低参数，却可能损害部分数据集所需的分量专用表示；
7. 模型复杂度、训练时间与检测性能之间需要形成合理折中。

研究过程中坚持以下原则：

* 所有修改必须对应明确的异常检测难点；
* 修改规模和参数量不是首要限制；
* 优先通过真实数据集与原始 CATCH 的公平对比判断方向；
* 性能持平或提升即可认为具有继续研究价值；
* 不使用测试标签训练模型或选择异常分数；
* 不将主要精力投入复杂审计、Gate、合成实验、manifest、shard、finalizer 等实验基础设施；
* 不继续已经证明无效的评分融合和后处理路线。

---

## 2. 基础模型：CATCH

本方向使用原版 CATCH 作为基础模型。

CATCH 主要在频域中进行 patch/token 表示学习，并建模多变量序列之间的通道关系，通过重构误差得到异常分数。它在多变量关联和频率结构建模方面具有较强基础，但原始模型直接处理完整序列，未显式区分慢变与快速变化结构。

原版目录保持不变：

```text
ts_benchmark/baselines/catch/
```

方向 A 的所有新模型均以独立 baseline 形式实现，不直接修改原始 CATCH。

---

## 3. 第一阶段：MSD-CATCH

### 3.1 模型动机

原始序列中的趋势、周期和快速局部扰动在统一频域空间中共同参与特征学习，可能导致不同动态结构相互干扰。

为此，第一版正式模型 MSD-CATCH 引入多尺度结构分解，并使用独立分支分别学习不同动态成分。

### 3.2 模型结构

MSD-CATCH 的主要结构包括：

1. 三尺度自适应移动平均分解；
2. 将输入分解为 trend 和 residual；
3. trend 与 residual 分别进入独立的完整 CATCH 分支；
4. 两个分支之间进行双向 gated exchange；
5. 分别得到 `trend_hat` 和 `residual_hat`；
6. 最终重构为：

```python
x_hat = trend_hat + residual_hat
```

7. 主异常分数为：

```python
total_score = original_catch_score(x, x_hat)
```

trend、residual 及 fusion 分数仅作为诊断，不进入正式主结果。

模型位置：

```text
ts_benchmark/baselines/msd_catch/
```

### 3.3 评分融合实验

早期尝试过：

* trend/residual 固定等权融合；
* anchored fusion；
* 基于训练正常分数统计的分量增益。

实验结果显示，trend 和 residual 的独立异常分数普遍弱于最终 `total_score`。尤其 Genesis 中，trend 和 residual 分数很弱，而总重构分数明显优于 CATCH。

这说明分解的主要价值不是让每个分量独立完成异常检测，而是帮助模型获得更好的联合重构表示。

因此正式决定：

> 后续所有 MSD 系列模型仅使用 `total_score` 作为主异常分数，停止继续研究分量分数融合。

### 3.4 六数据集公平结果

当前六个真实数据集包括：

```text
PSM
Genesis
GECCO
CalIt2
NYC
MSL
```

其中 GECCO 早期 CATCH 使用了 `seq_len=96`，后来补跑了与 MSD 一致的 `seq_len=192` 公平基线。公平的 GECCO CATCH 结果为：

```text
AUC-PR = 0.40931
AUC-ROC = 0.96346
```

MSD-CATCH 六数据集结果如下：

| Dataset | CATCH PR |  MSD PR | MSD − CATCH |
| ------- | -------: | ------: | ----------: |
| PSM     |  0.43665 | 0.43044 |    -0.00621 |
| Genesis |  0.26599 | 0.31077 |    +0.04478 |
| GECCO   |  0.40931 | 0.40650 |    -0.00281 |
| CalIt2  |  0.11331 | 0.12476 |    +0.01145 |
| NYC     |  0.08012 | 0.06256 |    -0.01756 |
| MSL     |  0.16559 | 0.17403 |    +0.00844 |

按照：

```text
|Delta AUC-PR| <= 0.01
```

定义基本持平，MSD 的结果可概括为：

* 明显提升：Genesis、CalIt2；
* 小幅提升：MSL；
* 基本持平：PSM、GECCO；
* 明显下降：NYC。

公平 GECCO 基线下，六数据集平均 Delta AUC-PR 约为正值，收益不再只由 Genesis 单独支撑。

### 3.5 MSD-CATCH 的结论

MSD-CATCH 证明了：

1. 模型内部结构分解能够在真实数据上产生性能信号；
2. 分解收益不仅出现在 Genesis，也出现在 CalIt2 和 MSL；
3. 分量单独异常分数不是核心，联合重构更重要；
4. 独立分支能够保留较强的分量专门化能力；
5. 当前主要缺点是参数量和训练成本较高。

因此当前定位为：

> **MSD-CATCH 是方向 A 的主性能模型。**

它已经可以进入全数据集主实验，与 CATCH 及配置可核查的其他基线进行正式比较。

---

## 4. 共享编码与轻量化探索

MSD-CATCH 使用两个完整 CATCH 分支，参数量和训练开销较高。后续模型围绕以下问题展开：

> trend 和 residual 是否可以共享系统共性表示，同时保留足够的分量专门化能力？

这一阶段先后形成 RSA、SDD 和 BHD 三个版本。

---

## 5. RSA-MSD-CATCH

### 5.1 结构

RSA-MSD-CATCH 使用：

* 一套可重入的共享 CATCH backbone；
* trend/residual 独立低秩 adapter；
* 低秩双向 exchange；
* 轻量 raw-preserving correction path；
* 受限 `raw_gate`；
* 最终仍使用 `total_score`。

raw path 不直接进行 identity skip，而是通过 depthwise temporal convolution 和低秩通道映射产生受限修正：

```python
x_hat = decomp_hat + raw_gate * raw_correction
```

其中：

```text
0 <= raw_gate <= 0.5
```

### 5.2 六数据集结果

| Dataset | CATCH PR |  MSD PR |  RSA PR |
| ------- | -------: | ------: | ------: |
| PSM     |  0.43665 | 0.43044 | 0.50551 |
| Genesis |  0.26599 | 0.31077 | 0.15125 |
| GECCO   |  0.40931 | 0.40650 | 0.43046 |
| CalIt2  |  0.11331 | 0.12476 | 0.09078 |
| NYC     |  0.08012 | 0.06256 | 0.06704 |
| MSL     |  0.16559 | 0.17403 | 0.16429 |

RSA 显著改善 PSM、GECCO，但 Genesis、CalIt2 明显下降。

### 5.3 raw path 归因

只读分析比较了：

```text
RSA Decomp Score
RSA Final Score
Raw Correction Score
```

关键结果：

| Dataset | RSA Decomp PR | RSA Final PR | Final − Decomp |
| ------- | ------------: | -----------: | -------------: |
| PSM     |       0.42265 |      0.50551 |       +0.08286 |
| Genesis |       0.07035 |      0.15125 |       +0.08091 |
| GECCO   |       0.43056 |      0.43046 |       -0.00010 |
| CalIt2  |       0.09042 |      0.09078 |       +0.00036 |
| NYC     |       0.06683 |      0.06704 |       +0.00020 |
| MSL     |       0.15873 |      0.16429 |       +0.00556 |

这表明：

* PSM 的 raw path 显著有效；
* Genesis 中 raw path 也提供正贡献，并不是性能下降原因；
* CalIt2 在 raw path 几乎关闭时已经下降；
* 主要问题发生在共享 decomp 表示阶段。

因此确认：

> RSA 的主要失败原因不是 raw gate，而是完整共享主干削弱了 trend/residual 的分量专门化。

RSA 不作为正式保留版本，但 raw-preserving path 的设计被后续轻量模型继承。

---

## 6. SDD-MSD-CATCH

### 6.1 结构

SDD-MSD-CATCH 在 RSA 基础上：

* 保留共享 encoder；
* trend/residual 使用独立 decoder；
* 将巨大 Flatten Head 替换为独立的全局 rank-64 factorized decoder；
* 保留 raw path、exchange 和训练损失。

目标是恢复分量专用解码，同时大幅降低参数量。

### 6.2 六数据集结果

| Dataset | CATCH PR |  MSD PR |  RSA PR |  SDD PR |
| ------- | -------: | ------: | ------: | ------: |
| PSM     |  0.43665 | 0.43044 | 0.50551 | 0.51329 |
| Genesis |  0.26599 | 0.31077 | 0.15125 | 0.02414 |
| GECCO   |  0.40931 | 0.40650 | 0.43046 | 0.44024 |
| CalIt2  |  0.11331 | 0.12476 | 0.09078 | 0.09729 |
| NYC     |  0.08012 | 0.06256 | 0.06704 | 0.03941 |
| MSL     |  0.16559 | 0.17403 | 0.16429 | 0.16174 |

SDD 在 PSM、GECCO 上继续提升，但 Genesis 和 NYC 严重下降。

参数量相对 MSD 下降约 80.5% 至 98.6%，说明巨型 Flatten Head 可以被大幅压缩；但统一的全局 rank-64 映射形成了过强信息瓶颈。

### 6.3 结论

SDD 的失败说明：

> 分量专用 decoder 本身并不足够；若将全部频率、patch 和通道信息统一压缩进全局 rank-64 潜空间，会严重损害局部重构能力。

因此放弃全局 rank-64 decoder，但保留“共享编码器加独立解码器”的探索方向。

---

## 7. BHD-MSD-CATCH

### 7.1 模型动机

SDD 的问题是全局低秩 decoder 需要同时保存所有 patch 和频率区域的信息。

BHD-MSD-CATCH 将其改为分支独立的 blockwise decoder，使每个 patch 在局部范围内完成复数频率重构，避免全局 token flatten 和统一低秩瓶颈。

### 7.2 模型结构

BHD decoder 的实际数据流为：

```text
[B * P, C, H]
→ [B, C, P, H]
→ [B, C, P, 2S]
```

其中：

* `P` 为 patch 数；
* `H = 2 × d_model`；
* `S = patch_size`；
* `2S` 对应 patch 级实部和虚部输出。

随后：

1. 每个 branch 使用独立局部 MLP；
2. 按原 patch stride 执行 overlap-add；
3. 重叠区域使用覆盖次数归一化；
4. 独立逐时间点复数到时域投影；
5. trend 和 residual 分别重构；
6. raw path 保持 RSA 版本不变。

模型位置：

```text
ts_benchmark/baselines/bhd_msd_catch/
```

### 7.3 六数据集结果

| Dataset | CATCH PR |  MSD PR |  BHD PR | BHD − CATCH | BHD − MSD |
| ------- | -------: | ------: | ------: | ----------: | --------: |
| PSM     |  0.43665 | 0.43044 | 0.55208 |    +0.11543 |  +0.12164 |
| Genesis |  0.26599 | 0.31077 | 0.02365 |    -0.24233 |  -0.28712 |
| GECCO   |  0.40931 | 0.40650 | 0.47225 |    +0.06294 |  +0.06575 |
| CalIt2  |  0.11331 | 0.12476 | 0.09417 |    -0.01914 |  -0.03059 |
| NYC     |  0.08012 | 0.06256 | 0.07882 |    -0.00130 |  +0.01626 |
| MSL     |  0.16559 | 0.17403 | 0.16213 |    -0.00346 |  -0.01190 |

按基本持平阈值计算：

* 明显提升：PSM、GECCO；
* 基本持平：NYC、MSL；
* 明显下降：Genesis、CalIt2。

相对 CATCH 为 2 个提升、2 个持平、2 个下降，即 `4/6` 数据集持平或提升。

### 7.4 BHD 的诊断结果

| Dataset | Decomp PR | Final PR | Raw Path Contribution |
| ------- | --------: | -------: | --------------------: |
| PSM     |   0.43540 |  0.55208 |              +0.11668 |
| Genesis |   0.02597 |  0.02365 |              -0.00232 |
| GECCO   |   0.47254 |  0.47225 |              -0.00029 |
| CalIt2  |   0.09420 |  0.09417 |              -0.00002 |
| NYC     |   0.07805 |  0.07882 |              +0.00077 |
| MSL     |   0.14979 |  0.16213 |              +0.01234 |

结论：

* PSM 的提升主要来自 raw-preserving path；
* GECCO 的提升主要来自分解与共享编码后的 decomp 重构；
* Genesis 在进入最终 raw correction 前，decomp 已经完全失效；
* Genesis 从 SDD 到 BHD 几乎没有恢复，说明问题不主要在 decoder 的局部 patch 重构能力；
* 完整共享 encoder 很可能在 decoder 之前已经丢失 Genesis 所需的分量专用信息。

### 7.5 参数量与效率

BHD 参数量约为：

```text
0.027M–1.130M
```

显著低于 MSD 的：

```text
7.55M–1676.81M
```

每个分支 block decoder 参数量仅为数千至十几万。

BHD 证明：

1. 原 CATCH 中巨型 Flatten Head 并非所有数据集都必需；
2. blockwise decoder 能在较低参数量下保持甚至提高部分数据集性能；
3. 参数量降低不等于训练时间同比下降；
4. BHD 的价值是形成轻量性能折中，而非全面替代 MSD。

当前定位为：

> **BHD-MSD-CATCH 是方向 A 的轻量共享编码版本。**

---

## 8. 当前正式保留与放弃的模型

### 8.1 正式保留

#### MSD-CATCH

定位：

```text
当前主性能模型
```

优点：

* 跨数据集平均性能具有正向信号；
* Genesis、CalIt2、MSL 有收益；
* 独立分支保留较强的分量专门化；
* 与研究动机最完整对应。

缺点：

* 参数量较大；
* 训练成本较高；
* NYC 仍有下降。

#### BHD-MSD-CATCH

定位：

```text
当前轻量共享编码模型
```

优点：

* 参数量极低；
* PSM、GECCO 明显提升；
* NYC、MSL 基本持平；
* 证明共享编码与 blockwise reconstruction 在部分数据集有效。

缺点：

* Genesis 严重失效；
* CalIt2 未恢复；
* 不能作为共享 encoder 路线的全面解决方案。

### 8.2 停止或放弃

以下版本不继续扩展主实验：

```text
RSA-MSD-CATCH
SDD-MSD-CATCH
fixed fusion
anchored fusion
post-hoc decomposition gate
旧 APD-CATCH 因果预测路线
```

这些版本可保留为模型演化和消融记录，但不作为正式候选。

---

## 9. 当前主实验线

当前主实验线准备在仓库中全部可运行真实数据集上比较：

```text
CATCH
MSD-CATCH
BHD-MSD-CATCH
```

其中：

* CATCH 为原始基线；
* MSD-CATCH 为主性能版本；
* BHD-MSD-CATCH 为轻量版本。

已有六数据集结果直接复用，不重新训练：

```text
PSM
Genesis
GECCO
CalIt2
NYC
MSL
```

其余数据集需先检查：

1. 数据是否完整；
2. CATCH、MSD、BHD 是否均可直接运行；
3. 配置是否可以严格对齐；
4. 是否已有可以核查的历史结果；
5. 预计训练时间和显存开销。

对于新增数据集：

* 先进行少量 batch 的轻量测速；
* 按实际预计耗时安排运行顺序；
* 不单独突出 SWaT 或任何特定数据集；
* 不因耗时缩减 epoch、seq_len、patch 或模型规模；
* 同一 GPU 上不同时运行多个完整训练任务；
* 短任务优先完成，长任务最后单独运行。

主实验必须保证三个模型在同一数据集上使用一致的：

```text
数据切分
seed=2021
epoch
batch size
seq_len
patch size
patch stride
学习率
优化器
评价代码
标签处理
point adjustment 设置
```

MSD 和 BHD 均只使用：

```text
total_score
```

进入正式主表。

最终主表至少包括：

| Dataset | CATCH PR | MSD PR | BHD PR | MSD − CATCH | BHD − CATCH | CATCH ROC | MSD ROC | BHD ROC |
| ------- | -------: | -----: | -----: | ----------: | ----------: | --------: | ------: | ------: |

效率表包括：

| Dataset | CATCH Params | MSD Params | BHD Params | CATCH Time | MSD Time | BHD Time | Peak Memory |
| ------- | -----------: | ---------: | ---------: | ---------: | -------: | -------: | ----------: |

主实验完成后需要分别判断：

### MSD-CATCH

* 是否在多数数据集持平或提升；
* 平均和中位 Delta 是否不明显为负；
* 收益是否跨越多个数据集；
* 性能收益是否能够说明额外计算成本。

### BHD-MSD-CATCH

* 是否在多数数据集接近或超过 CATCH；
* 是否形成稳定的轻量性能折中；
* 参数优势是否伴随可接受的性能；
* 实际训练时间是否也有优势。

---

## 10. 下一版模型候选

此前提出的下一版候选是：

```text
PSE-BHD-MSD-CATCH
Partially Shared Encoder BHD-MSD-CATCH
```

核心思想：

* FFT、patch 构造和 encoder 前段共享；
* trend/residual encoder 后段独立；
* 保留 BHD 的 blockwise decoder；
* 保留 exchange 和 raw-preserving path；
* 继续使用 `total_score`。

该模型针对的具体问题是：

> BHD 的完整共享 encoder 在 PSM、GECCO 等数据集上有效，但在 Genesis 中可能在进入 decoder 前已经丢失了分量专用表示。

部分共享编码器试图在以下两点之间折中：

1. 共享同一系统的公共频率结构和通道关系；
2. 为 trend 和 residual 保留后期分量专门化能力。

但当前决定是：

> 在主实验线完成前，暂不继续实现下一版模型。

原因是需要先通过更多数据集判断：

* MSD 的独立双分支是否已经足够稳定；
* BHD 的完整共享 encoder 失败是否主要集中于 Genesis；
* 共享编码的收益和失败是否具有明确的数据集模式；
* 是否值得继续部分共享路线，还是应回到 MSD 的独立分支并仅优化重构头成本。

---

## 11. 当前阶段结论

方向 A 已经从最初的分解概念验证，发展为具有两条明确模型线的研究工作。

### 主性能线

```text
CATCH → MSD-CATCH
```

该路线证明：

* 分解后的独立建模能够提高部分数据集的异常检测性能；
* trend/residual 分量需要联合重构，而非独立分数融合；
* 分量专门化对 Genesis 等数据集非常重要。

### 轻量化线

```text
MSD-CATCH → RSA → SDD → BHD
```

该路线证明：

* 两个完整 CATCH 分支存在明显参数冗余；
* 完整共享 encoder 在部分数据集上可行；
* 全局低秩 decoder 会造成过度压缩；
* blockwise decoder 能以极低参数保留局部重构能力；
* 但共享编码可能在部分数据集上提前损失分量专门化。

当前最终保留：

```text
MSD-CATCH：主性能模型
BHD-MSD-CATCH：轻量共享编码模型
```

当前尚未得到的结论：

1. MSD 是否能够在更大规模全数据集实验中保持总体优势；
2. BHD 是否具有稳定的性能—效率折中；
3. Genesis 的共享编码失败是否具有普遍性；
4. 是否需要部分共享 encoder；
5. 是否有必要进一步压缩 MSD 的独立重构头。

---

## 12. 下一阶段工作

下一阶段首先完成主实验线，而不是继续快速迭代新模型。

具体任务：

1. 扩展 CATCH、MSD、BHD 到仓库中全部可运行真实数据集；
2. 复用已有六数据集结果；
3. 对新增任务进行简单运行时间和显存估计；
4. 完成公平的 AUC-PR、AUC-ROC 对比；
5. 汇总参数量、训练时间和峰值显存；
6. 分别判断 MSD 的总体性能稳定性和 BHD 的轻量价值；
7. 分析完整共享 encoder 的失败是否只集中于 Genesis；
8. 主实验完成后，再决定是否实现 PSE-BHD。

后续模型开发只允许围绕完整数据集结果暴露出的一个主要问题展开。

可能的判断路径：

### 路径一：MSD 整体稳定优于或持平 CATCH

保留 MSD 作为最终主模型，下一步重点优化其参数和训练成本，但不能破坏独立分量建模能力。

### 路径二：BHD 在多数数据集形成良好折中

保留 BHD 作为轻量版本，并根据共享编码失败数据集判断是否需要部分共享 encoder。

### 路径三：完整共享 encoder 在多个数据集严重失败

停止强制共享表示的路线，回到 MSD 独立分支，只研究如何压缩或共享巨大重构头。

### 路径四：MSD 和 BHD 均缺乏全数据集稳定性

重新审视当前分解和联合建模机制，而不是继续调整 raw gate、评分融合或局部 decoder。

---

## 13. 当前工作状态摘要

截至目前：

```text
已完成：
原始 CATCH 公平基线核查
MSD-CATCH 六数据集实验
分量评分与 fusion 实验
RSA-MSD-CATCH 六数据集实验
raw path 只读归因
SDD-MSD-CATCH 六数据集实验
BHD-MSD-CATCH 六数据集实验

正式保留：
MSD-CATCH
BHD-MSD-CATCH

停止或放弃：
RSA-MSD-CATCH
SDD-MSD-CATCH
fixed fusion
anchored fusion
旧因果预测与 post-hoc gate 路线

当前正在进行：
CATCH、MSD、BHD 全数据集主实验比较

暂缓：
PSE-BHD-MSD-CATCH
其他新模型结构
```

方向 A 当前已经具备明确的研究主线、模型演化证据和正式候选版本。接下来的关键不再是继续增加模型模块，而是通过更完整的真实数据集比较，确认 MSD 和 BHD 各自的最终定位。

