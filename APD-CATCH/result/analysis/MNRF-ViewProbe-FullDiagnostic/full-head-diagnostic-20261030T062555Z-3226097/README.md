# Full covariance / aggregation diagnosis

Only the eight pinned Full representations are compared. No encoder training, encoder forward, Flow, CATCH run, or MNRF branch ablation was performed. All GMM heads use the fixed source fit endpoints, full-layout fixed noise and frozen feature caches; only covariance type differs for new diagonal heads.

`full_four_configurations.csv` has full-covariance/max, full-covariance/mean, diagonal/max, diagonal/mean. Mean is over standardized per-variable NLL; max/mean thresholds are each calibrated from the same calibration endpoints at q=0.995. Predictions are strict score > threshold. Aff-F is the existing valid-continuous-segment macro diagnostic.

## Four configurations

| Dataset | Covariance | Aggregation | ROC-AUC | AP | P/R/F1 | FPR | Aff-F | Segments hit/scored/total |
|---|---|---|---:|---:|---|---:|---:|---:|
| PSM | full | max | 0.744926 | 0.437281 | 0.4592/0.6028/0.5213 | 0.273502 | 0.736402 | 48/71/72 |
| PSM | full | mean | 0.732468 | 0.473792 | 0.3414/0.3649/0.3528 | 0.271210 | 0.721166 | 45/71/72 |
| PSM | diag | max | 0.749547 | 0.455696 | 0.3425/0.3684/0.3550 | 0.272553 | 0.778198 | 50/71/72 |
| PSM | diag | mean | 0.736548 | 0.533102 | 0.4937/0.3234/0.3908 | 0.127813 | 0.711708 | 41/71/72 |
| MSL | full | max | 0.602445 | 0.159006 | 0.2903/0.0775/0.1224 | 0.022380 | 0.541380 | 13/36/36 |
| MSL | full | mean | 0.612376 | 0.162124 | 0.2903/0.0752/0.1195 | 0.021711 | 0.582247 | 14/36/36 |
| MSL | diag | max | n/a | n/a | n/a | n/a | n/a | failed: fixed K=4 diagonal GMM component covariance collapse |
| MSL | diag | mean | n/a | n/a | n/a | n/a | n/a | failed: fixed K=4 diagonal GMM component covariance collapse |
| GECCO | full | max | 0.442452 | 0.223303 | 0.7266/0.2548/0.3773 | 0.001023 | 0.288205 | 3/22/22 |
| GECCO | full | mean | 0.578429 | 0.245241 | 0.7348/0.2658/0.3903 | 0.001023 | 0.352493 | 4/22/22 |
| GECCO | diag | max | 0.847167 | 0.245850 | 0.3033/0.2767/0.2894 | 0.006780 | 0.299716 | 4/22/22 |
| GECCO | diag | mean | 0.844685 | 0.255553 | 0.2819/0.2726/0.2772 | 0.007408 | 0.314117 | 4/22/22 |
| Genesis | full | max | 0.899568 | 0.046805 | 0.0578/0.4600/0.1027 | 0.030303 | 0.854378 | 3/3/3 |
| Genesis | full | mean | 0.975780 | 0.100484 | 0.0695/0.7600/0.1273 | 0.041131 | 0.850635 | 3/3/3 |
| Genesis | diag | max | 0.929467 | 0.038311 | 0.0909/0.1600/0.1159 | 0.006465 | 0.855136 | 2/3/3 |
| Genesis | diag | mean | 0.856309 | 0.026525 | 0.0210/0.2200/0.0383 | 0.041535 | 0.776304 | 3/3/3 |
| NYC | full | max | 0.691339 | 0.044089 | 0.0000/0.0000/0.0000 | 0.001697 | 0.670732 | 0/1/1 |
| NYC | full | mean | 0.655087 | 0.039383 | 0.0000/0.0000/0.0000 | 0.004363 | 0.617377 | 0/1/1 |
| NYC | diag | max | 0.587783 | 0.034716 | 0.0870/0.0202/0.0328 | 0.005090 | 0.634825 | 1/1/1 |
| NYC | diag | mean | 0.618656 | 0.034252 | 0.0476/0.0101/0.0167 | 0.004847 | 0.656817 | 1/1/1 |
| CalIt2 | full | max | 0.713040 | 0.053971 | 0.0306/0.0429/0.0357 | 0.042054 | 0.613976 | 2/12/13 |
| CalIt2 | full | mean | 0.743300 | 0.064000 | 0.0521/0.0714/0.0602 | 0.040283 | 0.622530 | 3/12/13 |
| CalIt2 | diag | max | 0.624530 | 0.049229 | 0.0533/0.0571/0.0552 | 0.031430 | 0.706846 | 3/12/13 |
| CalIt2 | diag | mean | 0.655878 | 0.051175 | 0.0612/0.0857/0.0714 | 0.040726 | 0.696358 | 4/12/13 |
| Creditcard | full | max | 0.920188 | 0.051007 | 0.0352/0.2443/0.0615 | 0.010423 | 0.563857 | 39/202/204 |
| Creditcard | full | mean | 0.966333 | 0.106464 | 0.0606/0.8326/0.1129 | 0.020100 | 0.740304 | 165/202/204 |
| Creditcard | diag | max | 0.927364 | 0.094121 | 0.0550/0.3213/0.0940 | 0.008585 | 0.592303 | 54/202/204 |
| Creditcard | diag | mean | 0.957640 | 0.129505 | 0.0842/0.7873/0.1521 | 0.013332 | 0.747090 | 155/202/204 |
| CICIDS | full | max | 0.538825 | 0.000917 | 0.0000/0.0000/0.0000 | 0.016629 | 0.125632 | 0/65/70 |
| CICIDS | full | mean | 0.621907 | 0.001085 | 0.0000/0.0000/0.0000 | 0.010937 | 0.151893 | 0/65/70 |
| CICIDS | diag | max | 0.605432 | 0.001052 | 0.0000/0.0000/0.0000 | 0.020082 | 0.238366 | 0/65/70 |
| CICIDS | diag | mean | 0.698873 | 0.001372 | 0.0000/0.0000/0.0000 | 0.010359 | 0.213727 | 0/65/70 |

## Covariance 与 aggregation 效应

差值定义：diag−full 固定聚合；mean−max 固定协方差；交互=(diag mean−diag max)−(full mean−full max)。正 AP/AUC 差代表该改动提升排序指标。

| 数据集 | 协方差效应 AP max/mean | 协方差效应 ROC max/mean | 聚合效应 AP full/diag | 聚合效应 ROC full/diag | 交互 AP/ROC |
|---|---:|---:|---:|---:|---:|
| PSM | +0.01841/+0.05931 | +0.00462/+0.00408 | +0.03651/+0.07741 | -0.01246/-0.01300 | +0.04089/-0.00054 |
| MSL | n/a/n/a | n/a/n/a | +0.00312/n/a | +0.00993/n/a | n/a/n/a |
| GECCO | +0.02255/+0.01031 | +0.40472/+0.26626 | +0.02194/+0.00970 | +0.13598/-0.00248 | -0.01224/-0.13846 |
| Genesis | -0.00849/-0.07396 | +0.02990/-0.11947 | +0.05368/-0.01179 | +0.07621/-0.07316 | -0.06546/-0.14937 |
| NYC | -0.00937/-0.00513 | -0.10356/-0.03643 | -0.00471/-0.00046 | -0.03625/+0.03087 | +0.00424/+0.06713 |
| CalIt2 | -0.00474/-0.01283 | -0.08851/-0.08742 | +0.01003/+0.00195 | +0.03026/+0.03135 | -0.00808/+0.00109 |
| Creditcard | +0.04311/+0.02304 | +0.00718/-0.00869 | +0.05546/+0.03538 | +0.04614/+0.03028 | -0.02007/-0.01587 |
| CICIDS | +0.00013/+0.00029 | +0.06661/+0.07697 | +0.00017/+0.00032 | +0.08308/+0.09344 | +0.00015/+0.01036 |

## 共同协议排名参照

这些基线来自已验证的共同协议批次；PCA 的 NYC/CalIt2 行采用本次统一残差维修正规则，其旧全维重建结果保留在旧批次。

| 数据集 | Full full/max AP/ROC | Full diag/max AP/ROC | CATCH AP/ROC | IF AP/ROC | PCA residual AP/ROC |
|---|---:|---:|---:|---:|---:|
| PSM | 0.4373/0.7449 | 0.4557/0.7495 | 0.4241/0.6348 | 0.4397/0.6839 | 0.5348/0.7926 |
| MSL | 0.1590/0.6024 | n/a | 0.1696/0.6553 | 0.1423/0.6040 | 0.1440/0.5053 |
| GECCO | 0.2233/0.4425 | 0.2459/0.8472 | 0.4672/0.9714 | 0.4398/0.9561 | 0.3618/0.6796 |
| Genesis | 0.0468/0.8996 | 0.0383/0.9295 | 0.0144/0.7585 | 0.0055/0.6501 | 0.5155/0.9876 |
| NYC | 0.0441/0.6913 | 0.0347/0.5878 | 0.0500/0.6849 | 0.0378/0.6657 | 0.0238/0.5007 |
| CalIt2 | 0.0540/0.7130 | 0.0492/0.6245 | 0.1070/0.8358 | 0.1011/0.8002 | 0.0858/0.7637 |
| Creditcard | 0.0510/0.9202 | 0.0941/0.9274 | 0.0912/0.9593 | 0.0965/0.9551 | 0.0453/0.8007 |
| CICIDS | 0.0009/0.5388 | 0.0011/0.6054 | 0.0014/0.6938 | 0.0013/0.6747 | 0.0011/0.5739 |

## Common-protocol references

CATCH-common, IF and PCA values are read from the separately validated common-protocol batch. The PCA residual rule is `min(components_for_95_percent_variance, D-1)` on training rows; NYC and CalIt2 are recomputed because the old PCA retained all dimensions. Old PCA results remain in their original batch and are not overwritten.

## Effect decomposition

For each dataset, covariance effect is diagonal minus full at fixed aggregation; aggregation effect is mean minus max at fixed covariance. The interaction is (diag-mean minus diag-max) minus (full-mean minus full-max). These are post-hoc fixed-protocol contrasts, not a per-dataset selection of a new method.

## Scale and maximum-score contribution

Input scales and bounded sampled training-feature standard deviations are in `scale_diagnostics.csv`; TP/FP variable tie-split attribution is in `max_variable_contributions.csv`. They are localization evidence only, not a causal decomposition. Large MSL/CICIDS scale/NLL contributions are not clipped or removed.

## Interpretation boundary

An improvement from diagonal covariance supports covariance complexity/fitting as a possible source of loss; improvement from mean supports sensitivity to cross-variable max aggregation. If neither improves, this comparison does not explain the loss and is not proof that a representation branch is useless. Calibration pools for Genesis and especially CalIt2 are small; q=0.995 remains fixed.

Failures, if any, are preserved in `failures.json`; no failed configuration is treated as complete.

The diagonal MSL head failed under the fixed K=4, `reg_covar=0.001`, `n_init=1`, `max_iter=100` protocol because sklearn reported a collapsed/ill-defined component covariance. No retry, K reduction, regularization change, or parameter search was performed. Both original full-covariance MSL rows remain included and verified.

## 排序与报警损失定位

本节是只读事后诊断：没有重训编码器、重新编码、重拟合 GMM、重新校准或改变阈值。四种 GMM 配置和 CATCH-common、IF、PCA 均从已保存分数/NLL 计算；正式报警仍使用各自既有 calibration q=0.995 阈值及严格 `score > threshold`。`loss_decomposition.csv` 保留每种配置的 Recall@FPR、分布分位数及 Affiliation 指标；`calibration_tail_context.csv` 记录校准上尾点的原始标签、冻结窗口上下文及时间四分位；`variable_error_attribution.csv` 对 TP、FP、FN 和测试正常点分数最高 1% 做并列最大值分摊。

| 数据集 | 已确认的损失环节 | 支持证据 | 尚未解释 | 优先修改方向（待独立检验） |
|---|---|---|---|---|
| PSM | 报警误报及变量最大值集中 | Full/max AUC/AP=0.745/0.437；q995 FPR=27.35%、Recall=60.28%。q995 校准尾 66 点中无标签异常或异常上下文；正常误报最高 1% 的主导分数 100% 来自 feature_3，FN 最大项 feature_1 占 32.2%。 | max 排序 AP 低于 PCA 0.535；这些通道归因不是因果证据。 | 先评估有界、训练/校准冻结的跨变量稳健聚合；mean 提高 AP 0.0365 但 AUC 降 0.0125，不能直接替换正式结果。 |
| MSL | 排序弱，且对角头在协方差初始化阶段数值失败 | Full/max AUC/AP=0.602/0.159、Recall=7.75%；校准尾 29 点全在最后时间四分位，标签及窗口上下文均正常。col_51/col_37 有输入尺度触底标记且多维特征低于 feature floor；但 MSL 初始化失败变量 col_17 的输入有限、范围 [-179.95,214.84]。float32 同一初始化在单样本分量上得到最小经验协方差 -1.09e-4，float64 为 0.001 并可初始化。Full/max 测试分数校准中位数/99th/99.5th 为 77.5/537/2,202，测试正常点 99th/最大值为 65,128/1.02e18，异常点中位数/95th/99th 为 84.8/44,991/5.06e17；最大分数点在全局端点 104405、变量 col_37，标签为异常。该通道训练 scale 被 floor 到 1e-6，标准化缓存里稀有值达到 1e6，冻结前向特征窗口达到约 [-599531, 1000000]，明显超出训练该变量特征范围 [-64.8,72.8]。 | 对角初始化失败的直接证据是 float32 经验方差消减；Full/max 超大值与低基数通道的 scale-floor 放大和测试特征外推相符，但缓存不足以证明底层原始传感器值/数据拼接是否合理，也不能排除 GMM 对极端特征的数值敏感。异常分数在校准分布的中位百分位仅 55.9%，95th 已到 99.98%，显示排序并非所有异常都整体偏高。 | 优先为协方差统计实现稳定的高精度累积，并在保留 K/正则/seed 的受控独立运行验证；另独立审查 col_37 的低基数输入与 floor 语义。不得将本次 float64 初始化探针记作成功模型。 |
| GECCO | 排序与校准阈值均受异常污染风险影响 | Full/max AUC/AP=0.442/0.223，但 CATCH-common 为 0.971/0.467。校准 q995 的 35/35 个尾分数对应标签异常，且 35/35 窗口上下文包含异常。对角头 AUC 升至 0.847，但 AP 仅 0.246。 | 校准污染是阈值解释的重要证据，不能单独解释 AUC 排序分歧；源数据/标注语义仍需核实。 | 先审计数据的异常标注、purge 和校准假设；保留当前阈值作原协议结果，不用测试集重选。 |
| Genesis | 高 AUC 与低 AP 并存，报警强度受小校准集影响 | Full/max AUC/AP=0.900/0.0468；mean 为 0.976/0.1005，q995 Recall 从 46% 到 76%，FPR 从 3.03% 到 4.11%。校准仅 132 点，q995 尾只有 1 点。FN 最大变量为 IsAcceleration（70.4%）；最高分正常点主要由二值 Motor_Pos2reached 主导（81.5%）。 | PCA 的 AUC/AP 为 0.988/0.516，差距不由聚合单独解释；单个尾点使阈值尾部估计敏感。 | 先核对离散状态通道及事件/窗口语义，再评估对离散变量稳定的密度头；增加校准数据只可作为后续协议研究，不在此调分位数。 |
| NYC | 报警端未检出真阳性，max/mean 都未修复 | Full/max AUC/AP=0.691/0.0441，但 q995 TP=0、FN=99；mean 仍 TP=0。FN 的 63.6% 和高分正常点的 78.6% 都由 channnel3 主导，该通道有 7 个采样取值。 | 单变量稀疏/离散现象与记录的原始传感器含义是否一致尚未证实；排序和报警落差原因未定。 | 优先检查 channnel3 的测量、离散化和时间聚合语义；不能仅凭归因删除变量或分支。 |
| CalIt2 | 报警召回低且校准尾样本极少 | Full/max AUC/AP=0.713/0.0540，TP=3、FN=67、FPR=4.21%；校准仅 24 点、q995 尾 1 点（该点标签/上下文正常）。两输入变量样本取值数均为 10；FN 主要由变量 9（62.7%）与 7（37.3%）主导。 | 校准分位数方差、低基数输入和表示能力的相对影响无法由现有单次运行分开。 | 优先核对输入量化和极小校准切分对阈值的影响；不缩窗、不合并 split、不针对测试调阈值。 |
| Creditcard | 排序总体较好但极不平衡；报警阈值与召回权衡显著 | Full/max AUC/AP=0.920/0.0510，Recall=24.4%、FPR=1.04%；mean AP=0.1065、Recall=83.3%，但 FPR=2.01%。校准尾 71 点含 3 个标签异常、9 个异常上下文。FN 主导变量 V11 占 53.3%；正常高分主要由 V20/V25/V12 分担。 | 高分标签及上下文污染对阈值偏移的定量因果影响未知；对照头的 AP 受极低异常率影响。 | 先研究稳健校准/跨变量聚合对误报-召回曲线的影响；mean 结果是敏感性证据，不按测试结果替换 max。 |
| CICIDS | 排序接近随机且 q995 无真阳性，测试正常误报仍多 | Full/max AUC/AP=0.539/0.000917、TP=0、FN=74、FPR=1.66%；q995 校准尾 42 点标签全正常，但每个点的窗口上下文均覆盖标注异常。mean AUC/AP 升至 0.622/0.001085，仍无 TP。FN 主导为 Fwd IAT Min（28.4%）；正常高分主导为 Flow IAT Min（52.2%）。 | 上下文含异常不等于校准端点自身异常；标注边界、网络流量行序连续性和特征漂移仍需上游核实。 | 优先核查流量会话/时间拼接和异常窗口污染定义；本轮没有裁剪尺度或更改数据。 |

变量归因同时与尺度底板、低基数和特征尺度底板并列保存。它们提供可检验线索：例如 MSL 有若干低尺度/低方差通道，Genesis、NYC、CalIt2 的主要归因通道呈低基数；Creditcard、CICIDS 的错误变量并未普遍触发输入 scale floor 或 score MAD floor。因此“尺度极大/触底”不能作为八组共同根因。诊断使用的归因集合包括漏报和高分正常点，并对并列最大值平均分摊，不是删除/裁剪特征的依据。

Recall@FPR=0.1/0.5/1/5% 的四列值均逐个保存在 `loss_decomposition.csv`。Full/max 的对应召回依次为：PSM 0.07/0.76/1.38/12.27%，MSL 1.16/2.33/3.88/10.22%，GECCO 25.48/27.26/27.40/27.67%，Genesis 0/6/22/56%，NYC 0/0/3.03/10.10%，CalIt2 0/0/1.43/4.29%，Creditcard 10.41/20.81/24.43/68.78%，CICIDS 0/0/0/2.70%。GECCO 在低 FPR 区间的召回主要反映测试异常占比及排序点集，仍需结合其校准污染解释；不能将 FPR 曲线用于正式阈值选择。

Full/max 校准量分别为：PSM 13,020（q995 尾 66、无异常标签/窗口）；MSL 5,603（尾 29、集中在末四分位且均正常）；GECCO 6,812（尾 35、35 个端点及其上下文均含异常）；Genesis 132（尾 1、正常）；NYC 1,082（尾 6、正常）；CalIt2 24（尾 1、正常）；Creditcard 14,012（尾 71，其中 3 个尾端点标签为异常、9 个窗口上下文含异常）；CICIDS 8,283（尾 42 标签均正常，但 42 个上下文均覆盖异常标签）。上下文含异常不等于该 calibration 端点自身异常。

### 下一步优先项

优先修正 MSL 对角协方差初始化的数值稳定性：在相同 float32 输入值、分量分配、K=4、`reg_covar=0.001` 和 seed 下，仅让协方差矩统计使用 float64 或稳定的两遍累积；预期是消除负经验方差导致的非法初始化，而不是保证改善 AUC/AP。必须作为新独立诊断头验证，不改写当前失败状态。其次针对 GECCO/CICIDS 先核实异常标签与校准窗口语义，再做冻结协议下的校准稳健性实验。PSM/Creditcard 的 mean 对照显示跨变量 max 会影响排序/报警折衷；它不能解释 NYC、CalIt2 的低召回，也不能把所有差距归因为编码器表示。
