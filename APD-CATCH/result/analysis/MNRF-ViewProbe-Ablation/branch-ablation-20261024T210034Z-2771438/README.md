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

- PSM（参与 9/9 格）：Full AP max/mean +0.4373/+0.4738，S-only AP max/mean +0.4142/+0.5047；Full mean-max AP +0.0365、ROC-AUC -0.0125。
  添加 E AP max/mean -0.0050/-0.0849、AUC max/mean -0.0110/+0.0266；添加 P AP max/mean +0.0010/-0.0475、AUC max/mean +0.0025/+0.0405；删除 E AP max/mean +0.0149/+0.0362、删除 P +0.0125/+0.0036、删除 R +0.0276/+0.0568。
- MSL（参与 9/9 格）：Full AP max/mean +0.1590/+0.1621，S-only AP max/mean +0.1464/+0.1472；Full mean-max AP +0.0031、ROC-AUC +0.0099。
  添加 E AP max/mean -0.0157/-0.0146、AUC max/mean -0.0511/-0.0590；添加 P AP max/mean -0.0083/-0.0034、AUC max/mean -0.0088/+0.0079；删除 E AP max/mean -0.0072/+0.0007、删除 P -0.0053/-0.0054、删除 R +0.0291/+0.0314。
- GECCO（参与 9/9 格）：Full AP max/mean +0.2233/+0.2452，S-only AP max/mean +0.1380/+0.1346；Full mean-max AP +0.0219、ROC-AUC +0.1360。
  添加 E AP max/mean -0.1013/-0.0834、AUC max/mean -0.3595/-0.3092；添加 P AP max/mean -0.0478/-0.0379、AUC max/mean -0.3749/-0.3250；删除 E AP max/mean -0.0161/+0.0310、删除 P -0.0246/+0.0087、删除 R +0.1379/+0.1573。
- Genesis（参与 9/9 格）：Full AP max/mean +0.0468/+0.1005，S-only AP max/mean +0.2851/+0.2426；Full mean-max AP +0.0537、ROC-AUC +0.0762。
  添加 E AP max/mean -0.2702/-0.2341、AUC max/mean -0.0278/+0.1163；添加 P AP max/mean -0.2432/-0.2040、AUC max/mean +0.2476/+0.3835；删除 E AP max/mean -0.0454/-0.0875、删除 P -0.0053/+0.0176、删除 R +0.0367/+0.0942。
- NYC（参与 9/9 格）：Full AP max/mean +0.0441/+0.0394，S-only AP max/mean +0.0477/+0.0411；Full mean-max AP -0.0047、ROC-AUC -0.0363。
  添加 E AP max/mean -0.0191/-0.0090、AUC max/mean -0.0738/+0.0976；添加 P AP max/mean -0.0141/-0.0082、AUC max/mean -0.0796/+0.0678；删除 E AP max/mean +0.0051/+0.0050、删除 P +0.0131/+0.0098、删除 R +0.0156/+0.0057。
- CalIt2（参与 9/9 格）：Full AP max/mean +0.0540/+0.0640，S-only AP max/mean +0.0915/+0.0877；Full mean-max AP +0.0100、ROC-AUC +0.0303。
  添加 E AP max/mean -0.0329/-0.0040、AUC max/mean -0.0904/-0.0163；添加 P AP max/mean -0.0357/-0.0157、AUC max/mean -0.0730/-0.0271；删除 E AP max/mean +0.0037/+0.0023、删除 P -0.0126/-0.0029、删除 R -0.0094/-0.0188。
- Creditcard（参与 9/9 格）：Full AP max/mean +0.0510/+0.1065，S-only AP max/mean +0.1395/+0.2132；Full mean-max AP +0.0555、ROC-AUC +0.0461。
  添加 E AP max/mean -0.1082/-0.1667、AUC max/mean -0.1188/-0.0789；添加 P AP max/mean -0.0964/-0.1506、AUC max/mean -0.1543/-0.0494；删除 E AP max/mean -0.0020/-0.0156、删除 P -0.0342/-0.0289、删除 R +0.0203/+0.0538。
- CICIDS（参与 8/9 格）：Full AP max/mean +0.0009/+0.0011，S-only AP max/mean +0.0025/+0.0017；Full mean-max AP +0.0002、ROC-AUC +0.0831。
  添加 E AP max/mean -0.0017/-0.0009、AUC max/mean -0.2466/-0.2428；添加 P AP max/mean -0.0011/-0.0004、AUC max/mean -0.1370/-0.1396；删除 E AP max/mean NA/NA、删除 P -0.0001/-0.0001、删除 R -0.0000/+0.0002。
