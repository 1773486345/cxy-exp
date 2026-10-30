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
| GECCO | full | max | 0.442452 | 0.223303 | 0.7266/0.2548/0.3773 | 0.001023 | 0.288205 | 3/22/22 |
| GECCO | full | mean | 0.578429 | 0.245241 | 0.7348/0.2658/0.3903 | 0.001023 | 0.352493 | 4/22/22 |
| GECCO | diag | max | 0.847167 | 0.245850 | 0.3033/0.2767/0.2894 | 0.006780 | 0.299716 | 4/22/22 |
| GECCO | diag | mean | 0.844685 | 0.255553 | 0.2819/0.2726/0.2772 | 0.007408 | 0.314117 | 4/22/22 |
| Creditcard | full | max | 0.920188 | 0.051007 | 0.0352/0.2443/0.0615 | 0.010423 | 0.563857 | 39/202/204 |
| Creditcard | full | mean | 0.966333 | 0.106464 | 0.0606/0.8326/0.1129 | 0.020100 | 0.740304 | 165/202/204 |
| Creditcard | diag | max | 0.927364 | 0.094121 | 0.0550/0.3213/0.0940 | 0.008585 | 0.592303 | 54/202/204 |
| Creditcard | diag | mean | 0.957640 | 0.129505 | 0.0842/0.7873/0.1521 | 0.013332 | 0.747090 | 155/202/204 |

## Covariance 与 aggregation 效应

差值定义：diag−full 固定聚合；mean−max 固定协方差；交互=(diag mean−diag max)−(full mean−full max)。正 AP/AUC 差代表该改动提升排序指标。

| 数据集 | 协方差效应 AP max/mean | 协方差效应 ROC max/mean | 聚合效应 AP full/diag | 聚合效应 ROC full/diag | 交互 AP/ROC |
|---|---:|---:|---:|---:|---:|
| PSM | +0.01841/+0.05931 | +0.00462/+0.00408 | +0.03651/+0.07741 | -0.01246/-0.01300 | +0.04089/-0.00054 |
| MSL | failed/blocked | | | | |
| GECCO | +0.02255/+0.01031 | +0.40472/+0.26626 | +0.02194/+0.00970 | +0.13598/-0.00248 | -0.01224/-0.13846 |
| Genesis | failed/blocked | | | | |
| NYC | failed/blocked | | | | |
| CalIt2 | failed/blocked | | | | |
| Creditcard | +0.04311/+0.02304 | +0.00718/-0.00869 | +0.05546/+0.03538 | +0.04614/+0.03028 | -0.02007/-0.01587 |
| CICIDS | failed/blocked | | | | |

## 共同协议排名参照

这些基线来自已验证的共同协议批次；PCA 的 NYC/CalIt2 行采用本次统一残差维修正规则，其旧全维重建结果保留在旧批次。

| 数据集 | Full full/max AP/ROC | Full diag/max AP/ROC | CATCH AP/ROC | IF AP/ROC | PCA residual AP/ROC |
|---|---:|---:|---:|---:|---:|
| PSM | 0.4373/0.7449 | 0.4557/0.7495 | 0.4241/0.6348 | 0.4397/0.6839 | 0.5348/0.7926 |
| MSL | n/a | n/a | 0.1696/0.6553 | 0.1423/0.6040 | 0.1440/0.5053 |
| GECCO | 0.2233/0.4425 | 0.2459/0.8472 | 0.4672/0.9714 | 0.4398/0.9561 | 0.3618/0.6796 |
| Genesis | n/a | n/a | 0.0144/0.7585 | 0.0055/0.6501 | 0.5155/0.9876 |
| NYC | n/a | n/a | 0.0500/0.6849 | 0.0378/0.6657 | 0.0238/0.5007 |
| CalIt2 | n/a | n/a | 0.1070/0.8358 | 0.1011/0.8002 | 0.0858/0.7637 |
| Creditcard | 0.0510/0.9202 | 0.0941/0.9274 | 0.0912/0.9593 | 0.0965/0.9551 | 0.0453/0.8007 |
| CICIDS | n/a | n/a | 0.0014/0.6938 | 0.0013/0.6747 | 0.0011/0.5739 |

## Common-protocol references

CATCH-common, IF and PCA values are read from the separately validated common-protocol batch. The PCA residual rule is `min(components_for_95_percent_variance, D-1)` on training rows; NYC and CalIt2 are recomputed because the old PCA retained all dimensions. Old PCA results remain in their original batch and are not overwritten.

## Effect decomposition

For each dataset, covariance effect is diagonal minus full at fixed aggregation; aggregation effect is mean minus max at fixed covariance. The interaction is (diag-mean minus diag-max) minus (full-mean minus full-max). These are post-hoc fixed-protocol contrasts, not a per-dataset selection of a new method.

## Scale and maximum-score contribution

Input scales and bounded sampled training-feature standard deviations are in `scale_diagnostics.csv`; TP/FP variable tie-split attribution is in `max_variable_contributions.csv`. They are localization evidence only, not a causal decomposition. Large MSL/CICIDS scale/NLL contributions are not clipped or removed.

## Interpretation boundary

An improvement from diagonal covariance supports covariance complexity/fitting as a possible source of loss; improvement from mean supports sensitivity to cross-variable max aggregation. If neither improves, this comparison does not explain the loss and is not proof that a representation branch is useless. Calibration pools for Genesis and especially CalIt2 are small; q=0.995 remains fixed.

Failures, if any, are preserved in `failures.json`; no failed configuration is treated as complete.
