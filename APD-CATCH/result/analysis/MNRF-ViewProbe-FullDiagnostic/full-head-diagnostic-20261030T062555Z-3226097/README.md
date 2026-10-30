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
