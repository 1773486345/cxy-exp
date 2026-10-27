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

## 正式 max 的 Aff-F 补算

本节使用原正式 max 分数、保存阈值与预测；逐格调用原生 `_evaluation()` 验证源/头、分数、标签、坐标及有效点绑定。没有重训、拟合、重新校准或修改预测。历史审查 ZIP 不包含本节或 `affiliation_results.csv`。

Affiliation Precision/Recall/F 使用仓库原始 event-affiliation 核心算法。评价仅覆盖有效预测点；连续有效区间依据有效掩码、全局坐标连续性及冻结段 ID 切分。单一完整连续区间报告原始 `affiliation_precision`、`affiliation_recall`、`affiliation_f` 并与原实现交叉核对；多个/部分区间报告各指标的 `*_valid_segment_macro`（仅对该指标有定义的区间取算术宏平均）。没有真实异常事件时相应指标未定义；没有预测事件时 Precision/F 可未定义而 Recall 可合法为 0。未定义值留空并保留原因，不替换为零。

阈值取每个组合原 `calibration.npz` 的 max 评分校准阈值，预测保持严格 `score > threshold`。无效点不补值、不压缩跨缺口，不作 point adjustment。

公平比较需要对齐标签、阈值选择协议、评分区间、边界处理和汇总方式；不同模型的数值阈值不必相同。本批次是冻结端点有效区间协议；原始 CATCH 文献/历史结果的窗口端点与区间汇总口径未由本次补算证明一致，因此仅并列作为协议参照，不据此宣称胜出。

### 八个 Full 的 Affiliation 指标

| Dataset | Precision | Recall | F | 状态 |
|---|---:|---:|---:|---|
| PSM | 0.733794 | 0.739027 | 0.736402 | passed |
| MSL | 0.568525 | 0.516708 | 0.541380 | passed |
| GECCO | 0.754876 | 0.178101 | 0.288205 | passed |
| Genesis | 0.757125 | 0.980298 | 0.854378 | passed |
| NYC | 0.511682 | 0.973254 | 0.670732 | passed |
| CalIt2 | 0.524786 | 0.739690 | 0.613976 | passed |
| Creditcard | 0.540385 | 0.589459 | 0.563857 | passed |
| CICIDS | 0.405221 | 0.074340 | 0.125632 | passed |

### 八数据集九组合 Affiliation F

| Dataset | S | SE | SP | SR | FULL | WITHOUT_S | WITHOUT_E | WITHOUT_P | WITHOUT_R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PSM | 0.742068 | 0.605262 | 0.595221 | 0.719942 | 0.736402 | 0.761485 | 0.766654 | 0.759949 | 0.502910 |
| MSL | 0.618293 | 0.639142 | 0.594304 | 0.695033 | 0.541380 | 0.632565 | 0.642318 | 0.691088 | 0.611414 |
| GECCO | 0.636898 | 0.164186 | 0.648101 | 0.347615 | 0.288205 | 0.288598 | 0.339607 | 0.246398 | 0.403135 |
| Genesis | 0.807630 | 0.734529 | 0.831355 | 0.843459 | 0.854378 | 0.853066 | 0.859403 | 0.845299 | 0.654298 |
| NYC | 0.970293 | 0.686231 | 0.683014 | 0.566432 | 0.670732 | 0.574838 | 0.646331 | 0.690584 | 0.971716 |
| CalIt2 | 0.762760 | 0.638886 | 0.708545 | 0.531445 | 0.613976 | 0.688930 | 0.478304 | 0.664010 | 0.664823 |
| Creditcard | 0.737431 | 0.574318 | 0.532108 | 0.674403 | 0.563857 | 0.579538 | 0.645223 | 0.587867 | 0.585182 |
| CICIDS | 0.247640 | 0.091810 | 0.110389 | 0.197971 | 0.125632 | 0.143713 |  | 0.153428 | 0.075103 |

### CATCH 论文 Aff-F 参照（非同协议复现）

| Dataset | CATCH Table 2 Aff-F | 本批 FULL Aff-F |
|---|---:|---:|
| PSM | 0.859 | 0.736402 |
| MSL | 0.740 | 0.541380 |
| GECCO | 0.908 | 0.288205 |
| Genesis | 0.896 | 0.854378 |
| NYC | 0.994 | 0.670732 |
| CalIt2 | 0.835 | 0.613976 |
| Creditcard | 0.750 | 0.563857 |
| CICIDS | 0.787 | 0.125632 |

CATCH 数值引自 Wu et al., *CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching*, ICLR 2025, Table 2。论文值按其原始评估协议报告；本批次使用本地 Full 源冻结的测试端点、有效掩码和分段边界。窗口端点、无效区间与分段汇总未被证明相同，因此仅作并列参照，不据此宣称统一口径下胜出。
