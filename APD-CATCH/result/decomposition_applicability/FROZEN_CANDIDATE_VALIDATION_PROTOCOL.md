# 冻结候选验证协议与描述符公式

## 冻结范围

- 分析基准提交：`1c9cfb2151f4fca0477ad514638bb4a7b5bb08da`。
- 发现数据、正式来源映射、训练正常前缀、正式 `seq_len`、描述符公式、目标指标、分组和通过标准均冻结。
- 本协议不授权训练、推理、重评分、benchmark、模型修改、增加描述符、搜索阈值或提出新模型。

## 冻结的 Delta AUC-ROC 候选

令 `Delta AUC-ROC = AUC-ROC(MSD-CATCH total_score) - AUC-ROC(CATCH detect_score)`。以下四个候选仅针对该目标冻结：

| 描述符 | 预期方向 | 当前 task rho | 当前 paper rho | 去除 ASD 组后的 task rho |
| --- | --- | ---: | ---: | ---: |
| `mean_drift` | 正 | 0.589921 | 0.706294 | 0.627273 |
| `low_frequency_energy_ratio_mean` | 正 | 0.406126 | 0.419580 | 0.309091 |
| `periodicity_top3_ratio` | 正 | 0.468379 | 0.531469 | 0.454545 |
| `correlation_drift` | 负 | -0.591897 | -0.594406 | -0.554545 |

`low_frequency_energy_ratio_mean` 是较弱候选，但未来验证不得因为强度较低而事后删除。当前没有描述符通过 `Delta AUC-PR` 筛选。

## 公共数据口径

对每个任务，以正式 loader 重组后的训练正常前缀 $X \in \mathbb{R}^{T \times C}$ 计算全局通道均值和标准差：

$$
\mu_c = \operatorname{mean}_t X_{t,c}, \qquad
s_c = \max(\operatorname{std}_t X_{t,c}, \epsilon), \qquad
Z_{t,c} = \frac{X_{t,c} - \mu_c}{s_c}.
$$

其中 $\epsilon = 10^{-8}$。训练前缀按该任务正式 `seq_len` 切成连续、非重叠窗口 $w=1,\ldots,W$；不改变窗口长度、不搜索窗口长度。FFT 描述符使用 $Z$，漂移描述符的分子使用原始训练值 $X$，分母使用全局训练标准差 $s_c$。

## 冻结公式

设第 $w$ 个窗口内通道 $c$ 的原始均值和标准差分别为 $\bar X_{w,c}$、$q_{w,c}$。

### `mean_drift`

$$
\operatorname{mean\_drift} =
\frac{1}{W-1}\sum_{w=2}^{W}
\frac{1}{C}\sum_{c=1}^{C}
\frac{|\bar X_{w,c} - \bar X_{w-1,c}|}{s_c + \epsilon}.
$$

### `variance_drift`

$$
\operatorname{variance\_drift} =
\frac{1}{W-1}\sum_{w=2}^{W}
\frac{1}{C}\sum_{c=1}^{C}
\frac{|q_{w,c} - q_{w-1,c}|}{s_c + \epsilon}.
$$

`variance_drift` 保留在描述符输出中用于审计，但不是本协议的 ROC 候选。

### `low_frequency_energy_ratio_mean`

对标准化窗口 $Z_w$ 的每个通道，令 $P_{w,f,c}=|\operatorname{rFFT}(Z_{w,:,c})_f|^2$，去除 DC 项后得到频率索引集合 $F$。令 $K=\max(1,\lceil0.1|F|\rceil)$，则：

$$
r^{\mathrm{low}}_{w,c} =
\frac{\sum_{f \in \text{前 }K\text{ 个非零频率}}P_{w,f,c}}
{\sum_{f \in F}P_{w,f,c} + \epsilon}, \qquad
\operatorname{low\_frequency\_energy\_ratio\_mean} = \operatorname{mean}_{w,c} r^{\mathrm{low}}_{w,c}.
$$

### `periodicity_top3_ratio`

在同一非零频率功率谱上：

$$
r^{\mathrm{top3}}_{w,c} =
\frac{\sum_{f \in \text{功率最大的 }\min(3,|F|)\text{ 个频率}}P_{w,f,c}}
{\sum_{f \in F}P_{w,f,c} + \epsilon}, \qquad
\operatorname{periodicity\_top3\_ratio} = \operatorname{mean}_{w,c} r^{\mathrm{top3}}_{w,c}.
$$

### `correlation_drift`

令 $R^{\mathrm{global}}=\operatorname{corr}(Z)$，$R_w=\operatorname{corr}(Z_w)$；数值未定义的相关项按当前实现置零。对多通道任务：

$$
\operatorname{correlation\_drift} =
\frac{1}{W}\sum_{w=1}^{W}\left\|R_w-R^{\mathrm{global}}\right\|_F.
$$

## Delta AUC-ROC 通过标准

每个候选均独立按其预期符号审计，但不得把四者解释为四个独立机制。候选必须同时满足：

1. task-level 与 paper-level Spearman rho 同号，且两层均满足 `abs(rho) >= 0.35`。
2. task-level 与 paper-level 的所有逐行 leave-one-out rho 均不发生符号翻转。
3. task-level grouped leave-out 中，删除 ASD 全组和逐一删除每个非 ASD 论文数据集时均不发生符号翻转。
4. 删除 ASD 全组后，正关联候选 rho 必须大于零，负关联候选 rho 必须小于零。
5. ROC gain 至少 3 项，ROC loss/neutral 至少 3 项。
6. gain 与 loss/neutral 的描述符中位数差异必须符合预期方向：正关联为 `median(gain) > median(loss/neutral)`，负关联相反。

task-level 与 paper-level 是同批正式实验的两种聚合口径，不能视为独立复现。ASD 在 paper-level 中固定为其 12 个执行任务的等权 macro。

## 解释与外部验证边界

- 四个候选均为描述性关联，不是因果关系，不是直接选模规则。
- 四个候选可能相互相关；不得将它们解释为四个独立机制。
- 未来验证必须保持本协议的精确公式、全局标准化、正式 `seq_len` 连续非重叠窗口、Delta AUC-ROC 目标和上述通过标准。
- 只有未参与当前发现的新数据集，才可称为独立数据集外部验证；同一 23 个任务的重聚合、删一分析或重复读取不构成外部验证。

## 冻结记录

本协议只记录当前发现，不改变 [DECOMPOSITION_APPLICABILITY_REPORT.md](DECOMPOSITION_APPLICABILITY_REPORT.md) 中的正式来源、描述符和筛选结果。
