# MNRF Frozen-View Ablation Summary

Batch: `branch-ablation-20261024T210034Z-2771438`

Batch status: `failed`. Coverage: 7/8 complete datasets, 71/72 validated cells. Missing combinations are not filled with zero. `long_table.csv` is independently recalculated from each saved NLL and matched source labels. `paired_changes.csv` contains same-task, same-seed point and anomaly-segment additions/losses; bounded examples are in `change_cases.json`. `paired_point_scores.npz` contains only cells validated in this summary invocation.

For E/P/R, addition is `metric(S+branch)-metric(S)` and deletion is `metric(FULL)-metric(FULL-minus-branch)`. Positive deletion means removal hurt. S has deletion only. AP is average precision, not Aff-F. Segment hits mean at least one predicted point and are diagnostic only.

The head seed is fixed to 2021 for this matrix; it is a GMM-head seed on frozen features, not an encoder seed. These results do not prove causal representation contribution, paper-protocol equivalence, or superiority to CATCH. Full caches, encoders and GMM files stay in their server source runs.

## 变量评分聚合对照

本节由 `--aggregation-comparison` 只读生成；未改写原 max 分数、校准、manifest 或模型。
通过对照格：71/72；失败/未验证格：1。mean 只使用各格保存的 `calibration_nll.npy`、原 center/scale 和原 quantile。

max 复现要求保存 test NLL 聚合、原阈值、预测和指定 Full 的有效坐标轴全部一致。变量贡献按原 max 预测的 FP/TP 点统计，并列最大值按份额平分；该统计只描述变量评分聚合，不证明编码器分支因果贡献。

NYC 实际保存的逐变量 NLL 宽度为 3；因此本批次不是单变量特例，max 与 mean 不要求相等。若变量数为 1，CSV 的 `single_variable_check` 要求分数、阈值和预测逐项一致。

- PSM：Full mean-max AP +0.0365，ROC-AUC -0.0125；S-only AP +0.0905，Full mean AP +0.4738.
- MSL：Full mean-max AP +0.0031，ROC-AUC +0.0099；S-only AP +0.0007，Full mean AP +0.1621.
- GECCO：Full mean-max AP +0.0219，ROC-AUC +0.1360；S-only AP -0.0034，Full mean AP +0.2452.
- Genesis：Full mean-max AP +0.0537，ROC-AUC +0.0762；S-only AP -0.0425，Full mean AP +0.1005.
- NYC：Full mean-max AP -0.0047，ROC-AUC -0.0363；S-only AP -0.0066，Full mean AP +0.0394.
- CalIt2：Full mean-max AP +0.0100，ROC-AUC +0.0303；S-only AP -0.0038，Full mean AP +0.0640.
- Creditcard：Full mean-max AP +0.0555，ROC-AUC +0.0461；S-only AP +0.0737，Full mean AP +0.1065.
- CICIDS：Full mean-max AP +0.0002，ROC-AUC +0.0831；S-only AP -0.0008，Full mean AP +0.0011.
