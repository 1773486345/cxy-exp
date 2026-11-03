# Dataset Diagnosis

This is a read-only synthesis of the existing Full GMM and frozen E/R CSVs. No model, encoder, decoder, GMM or calibration was run. All values below retain the source dataset version and valid-score protocol.

## Cross-dataset score snapshot

| Dataset | Full AUC | Full AP | Full P/R/F1 | Full FPR | R residual AP/AUC | E residual AP/AUC |
|---|---:|---:|---|---:|---:|---:|
| PSM | 0.744926 | 0.437281 | 0.459202/0.602757/0.521276 | 0.273502 | 0.529406/0.747066 | 0.386468/0.649196 |
| MSL | 0.602445 | 0.159006 | 0.290260/0.077517/0.122358 | 0.022380 | 0.130013/0.555096 | 0.124893/0.556382 |
| GECCO | 0.442452 | 0.223303 | 0.726562/0.254795/0.377282 | 0.001023 | 0.303731/0.870483 | 0.274894/0.838270 |
| Genesis | 0.899568 | 0.046805 | 0.057789/0.460000/0.102679 | 0.030303 | 0.090831/0.850694 | 0.018537/0.582681 |
| NYC | 0.691339 | 0.044089 | 0.000000/0.000000/0.000000 | 0.001697 | 0.025301/0.533637 | 0.021497/0.436728 |
| CalIt2 | 0.713040 | 0.053971 | 0.030612/0.042857/0.035714 | 0.042054 | 0.076635/0.781933 | 0.088731/0.811528 |
| Creditcard | 0.920188 | 0.051007 | 0.035202/0.244344/0.061538 | 0.010423 | 0.356110/0.954260 | 0.022876/0.940370 |
| CICIDS | 0.538825 | 0.000917 | 0.000000/0.000000/0.000000 | 0.016629 | 0.001262/0.679941 | 0.001233/0.674024 |

## Evidence and adjustment by dataset

### PSM

[已完成] 历史 Full 有效点评价为 AUC=0.744926、AP=0.437281，保存阈值报警为 P/R/F1=0.459202/0.602757/0.521276、FPR=27.35%（TP/FP/FN/TN=14694/17305/9684/45967；segments hit/scored=48/71）。FullDiagnostic 的协方差/max-mean 对照、正常高分变量归因和校准上尾审计均已保存。新增同点低误报诊断表明，当前分数在严格低 FPR 区域的排序尾部明显弱于可用基线；提高阈值只能沿当前分数排序曲线移动，不能补回这些预算下的召回差距。E/R 是否解释 Full 增益仍无因果证据。

Evidence record: `reference_metrics.csv` Full row with source `full_gmm-full-20261024T115119Z-3976714`; supporting files are `variable_metrics.csv`, `max_attribution.csv`, and `scale_diagnostics.csv` in the frozen analysis directory.

Evidence classification: the metric values and saved attribution/floor counts are directly recorded; causal explanations below are labelled as supported concerns or hypotheses and are not treated as proven.

Scale evidence: E/squared_residual: 3 of 25 scales at floor, E/nll: 0 of 25 scales at floor, R/squared_residual: 0 of 25 scales at floor, R/nll: 0 of 25 scales at floor.

R max attribution evidence (largest shares): anomaly:feature_1 0.409, normal:feature_1 0.314, anomaly:feature_3 0.311, normal:feature_3 0.280.

Existing diagnostic evidence:
- FullDiagnostic fixed-head contrasts (AP/AUC): full/max 0.4373/0.7449, full/mean 0.4738/0.7325, diag/max 0.4557/0.7495, diag/mean 0.5331/0.7365; these are fixed post-hoc contrasts, not selected replacement scores.
- FullDiagnostic calibration evidence: q=.995 tail has 66 points, 0 label anomalies and 0 contextual anomalies; this affects alarm interpretation, not ranking metrics.
- ConditionalScore post-hoc row: joint versus p(S,P|E,R) AP 0.4373->0.3633, AUC 0.7449->0.6934; it retains the source cache and is not a refit or proof of field causality.

#### Low-FPR comparison on the same points

This supplement uses the same 87,650 common test coordinates for all four methods (24,378 anomaly points, 63,272 normal points). Each cell reports `recall / actual FPR`; ties are kept whole and thresholds are selected post hoc using test labels solely to diagnose ranking, not as deployable thresholds.

| FPR ceiling | MNRF-Full-GMM | CATCH-common | IsolationForest | PCA |
|---:|---:|---:|---:|---:|
| 0.1% | 0.07% / 0.0000% | 0.38% / 0.0980% | 2.22% / 0.0996% | 0.32% / 0.0996% |
| 0.5% | 0.76% / 0.4963% | 2.14% / 0.4994% | 2.83% / 0.4994% | 1.49% / 0.4963% |
| 1% | 1.38% / 0.9973% | 3.96% / 0.9973% | 3.80% / 0.9910% | 2.51% / 0.9910% |
| 2% | 3.86% / 1.9977% | 6.74% / 1.9993% | 6.38% / 1.9946% | 14.38% / 1.9993% |

At the 1% ceiling MNRF and CATCH-common both produce 631 false positives, but MNRF detects 337 anomalies versus CATCH-common's 966. The saved MNRF `q=0.995` alarm point is separate: on these same 87,650 points it has recall 60.28% and FPR 27.35% (14,694 TP, 17,305 FP). The threshold and curve details are in the [operating-point CSV](../MNRF-PSM-LowFPR/low-fpr-20261103T121447Z/psm_low_fpr_operating_points.csv) and [low-FPR recall curve](../MNRF-PSM-LowFPR/low-fpr-20261103T121447Z/psm_low_fpr_recall_curve.png). This is evidence about the current score ranking at a restricted error budget; it does not justify replacing its saved calibration threshold or selecting a new model from test labels.

### MSL

[已完成] 历史拼接 Full 的排序和固定阈值指标为 AUC=0.602445、AP=0.159006、P/R/F1=0.290260/0.077517/0.122358，FPR=0.022380（TP/FP/FN/TN=602/1472/7164/64300；segments hit/scored=13/36）。序列边界约束下的修正版训练及 `col_0` 评分也已完成，不能再列为待运行。

Evidence record: `reference_metrics.csv` Full row with source `full_gmm-full-20261021T112147Z-2470335`; supporting files are `variable_metrics.csv`, `max_attribution.csv`, and `scale_diagnostics.csv` in the frozen analysis directory.

Evidence classification: the metric values and saved attribution/floor counts are directly recorded; causal explanations below are labelled as supported concerns or hypotheses and are not treated as proven.

Scale evidence: E/squared_residual: 53 of 55 scales at floor, E/nll: 49 of 55 scales at floor, R/squared_residual: 22 of 55 scales at floor, R/nll: 0 of 55 scales at floor.

R max attribution evidence (largest shares): normal:col_2 0.522, anomaly:col_2 0.483, anomaly:col_0 0.210, normal:col_0 0.185.

Existing diagnostic evidence:
- FullDiagnostic fixed-head contrasts (AP/AUC): full/max 0.1590/0.6024, full/mean 0.1621/0.6124, diag/max failed/ unavailable, diag/mean failed/ unavailable; these are fixed post-hoc contrasts, not selected replacement scores.
- FullDiagnostic calibration evidence: q=.995 tail has 29 points, 0 label anomalies and 0 contextual anomalies; this affects alarm interpretation, not ranking metrics.
- ConditionalScore post-hoc row: joint versus p(S,P|E,R) AP 0.1590->0.1455, AUC 0.6024->0.5656; it retains the source cache and is not a refit or proof of field causality.
- DataRepair MSL target-only post-hoc aggregation for col_0: AP 0.1590->0.2792, AUC 0.6024->0.7680; source Full threshold was 2201.67. This is post-hoc target aggregation, not a segmented retrain.
- `full_gmm-sequence-bounded-20261102T105427Z-2415811` keeps all 55 input fields and trains/windows within the 27 mapped test-sequence boundaries; final scoring uses `col_0`. On the 68,572 exact common test points (7,766 positives), it obtains AUC=0.778967, AP=0.307332, precision=0.437238, recall=0.107649 and FPR=1.7696% at its saved `q=0.995` threshold. It exceeds CATCH-common, IsolationForest and PCA-residual in both AUC and AP; its AP is +0.106561 versus CATCH-common. These method-specific thresholds are not an equal-FPR comparison.
- The same-point table and source/binding details are in [`MSL same-point comparison`](../MNRF-ViewProbe-MSL-Boundary/same-point-baselines-20261103T121447Z/README.md) and [`CSV`](../MNRF-ViewProbe-MSL-Boundary/same-point-baselines-20261103T121447Z/msl_same_point_comparison.csv). CATCH-common still uses concatenated-source training; each method retains its own saved threshold. Thus the comparison supports this sequence-bounded MNRF configuration over these common-protocol artifacts on this point set, but does not equalize training protocol or alarm threshold.
- The four-cell comparison separates two effects: for the sequence-bounded model, changing final score from all-55 max to `col_0` changes AP 0.1322→0.3073 (+0.1751); for all-55 max, changing from concatenated to sequence-bounded training changes AP 0.1715→0.1322 (-0.0392). Under `col_0` scoring, the boundary change is 0.3053→0.3073 (+0.0020). On these fixed coordinates, the larger observed AP change is associated with the scoring target, while the isolated boundary effect is small and differs by score target. This is a controlled diagnostic contrast, not a claim that boundary handling is unimportant for valid sequence semantics.

### GECCO

[已完成] Full GMM 排序较弱而 E/R residual 排序较高（Full AUC=0.442452, AP=0.223303, P/R/F1=0.726562/0.254795/0.377282, FPR=0.001023, TP/FP/FN/TN=186/70/544/68366, segments hit/scored=3/22; R residual AUC=0.870483, AP=0.303731, P/R/F1=0.068384/0.767123/0.125575, FPR=0.111476, TP/FP/FN/TN=560/7629/170/60807, segments hit/scored=20/22). [已完成] FullDiagnostic 已比较 full/diag 与 max/mean，并保存校准上尾和变量归因。 [支持但未证实] 校准污染可能影响报警，不等于已证明是唯一原因。 [真正新增验证] 仅需预注册校准标签审计或边界控制。

Evidence record: `reference_metrics.csv` Full row with source `full_gmm-full-20261020T204951Z-1642355`; supporting files are `variable_metrics.csv`, `max_attribution.csv`, and `scale_diagnostics.csv` in the frozen analysis directory.

Evidence classification: the metric values and saved attribution/floor counts are directly recorded; causal explanations below are labelled as supported concerns or hypotheses and are not treated as proven.

Scale evidence: E/squared_residual: 0 of 9 scales at floor, E/nll: 0 of 9 scales at floor, R/squared_residual: 0 of 9 scales at floor, R/nll: 0 of 9 scales at floor.

R max attribution evidence (largest shares): normal:Tp 0.662, anomaly:Leit 0.482, anomaly:Redox 0.284, normal:Leit 0.260.

Existing diagnostic evidence:
- FullDiagnostic fixed-head contrasts (AP/AUC): full/max 0.2233/0.4425, full/mean 0.2452/0.5784, diag/max 0.2459/0.8472, diag/mean 0.2556/0.8447; these are fixed post-hoc contrasts, not selected replacement scores.
- FullDiagnostic calibration evidence: q=.995 tail has 35 points, 35 label anomalies and 35 contextual anomalies; this affects alarm interpretation, not ranking metrics.
- ConditionalScore post-hoc row: joint versus p(S,P|E,R) AP 0.2233->0.2278, AUC 0.4425->0.4551; it retains the source cache and is not a refit or proof of field causality.

### Genesis

[已完成] Full has strong AUC but weak AP and alarm precision (AUC=0.899568, AP=0.046805, P/R/F1=0.057789/0.460000/0.102679, FPR=0.030303, TP/FP/FN/TN=23/375/27/12000, segments hit/scored=3/3). FullDiagnostic 的正常高分变量归因、校准尾部和 R/persistence 对照已保存；仅三个 scored anomaly segments 限制外推。 [尚未证实] 工况变化是否导致正常高分；[真正新增验证] 仅复核预注册的正常尾部窗口，不以测试标签改阈值。

Evidence record: `reference_metrics.csv` Full row with source `full_gmm-full-20261020T205005Z-1645522`; supporting files are `variable_metrics.csv`, `max_attribution.csv`, and `scale_diagnostics.csv` in the frozen analysis directory.

Evidence classification: the metric values and saved attribution/floor counts are directly recorded; causal explanations below are labelled as supported concerns or hypotheses and are not treated as proven.

Scale evidence: E/squared_residual: 9 of 18 scales at floor, E/nll: 2 of 18 scales at floor, R/squared_residual: 2 of 18 scales at floor, R/nll: 0 of 18 scales at floor.

R max attribution evidence (largest shares): anomaly:MotorData.IsForce 0.680, normal:MotorData.ActSpeed 0.244, normal:MotorData.IsAcceleration 0.192, anomaly:MotorData.ActSpeed 0.160.

Existing diagnostic evidence:
- FullDiagnostic fixed-head contrasts (AP/AUC): full/max 0.0468/0.8996, full/mean 0.1005/0.9758, diag/max 0.0383/0.9295, diag/mean 0.0265/0.8563; these are fixed post-hoc contrasts, not selected replacement scores.
- FullDiagnostic calibration evidence: q=.995 tail has 1 points, 0 label anomalies and 0 contextual anomalies; this affects alarm interpretation, not ranking metrics.
- ConditionalScore post-hoc row: joint versus p(S,P|E,R) AP 0.0468->0.0204, AUC 0.8996->0.8740; it retains the source cache and is not a refit or proof of field causality.

### NYC

[已完成] Full ranking is above random but its fixed alarm has no true positives (AUC=0.691339, AP=0.044089, P/R/F1=0.000000/0.000000/0.000000, FPR=0.001697, TP/FP/FN/TN=0/7/99/4119, segments hit/scored=0/1). 单一有评分事件的 k=8 neighborhood summaries 已保存，并区分客流目标与日历条件。 [尚未证实] 预测误差降低是否包含异常信息；[真正新增验证] 只可做预注册 target-only/context-only 诊断。

Evidence record: `reference_metrics.csv` Full row with source `full_gmm-full-20261020T211341Z-1931786`; supporting files are `variable_metrics.csv`, `max_attribution.csv`, and `scale_diagnostics.csv` in the frozen analysis directory.

Evidence classification: the metric values and saved attribution/floor counts are directly recorded; causal explanations below are labelled as supported concerns or hypotheses and are not treated as proven.

Scale evidence: E/squared_residual: 2 of 3 scales at floor, E/nll: 0 of 3 scales at floor, R/squared_residual: 1 of 3 scales at floor, R/nll: 0 of 3 scales at floor.

R max attribution evidence (largest shares): anomaly:channnel3 0.485, normal:channnel2 0.393, normal:channnel3 0.304, normal:channnel1 0.303.

NYC fixed-neighborhood rows available: 20; this is one scored event and is not a general mechanism test.

Existing diagnostic evidence:
- FullDiagnostic fixed-head contrasts (AP/AUC): full/max 0.0441/0.6913, full/mean 0.0394/0.6551, diag/max 0.0347/0.5878, diag/mean 0.0343/0.6187; these are fixed post-hoc contrasts, not selected replacement scores.
- FullDiagnostic calibration evidence: q=.995 tail has 6 points, 0 label anomalies and 0 contextual anomalies; this affects alarm interpretation, not ranking metrics.
- ConditionalScore post-hoc row: joint versus p(S,P|E,R) AP 0.0441->0.0267, AUC 0.6913->0.5171; it retains the source cache and is not a refit or proof of field causality.

### CalIt2

[已完成] The historical modulo-10 source has low Full AP (AUC=0.713040, AP=0.053971, P/R/F1=0.030612/0.042857/0.035714, FPR=0.042054, TP/FP/FN/TN=3/95/67/2164, segments hit/scored=2/12). The completed RawCount report is a separate data version: MNRF AUC/AP changes to 0.853538/0.170287, while CATCH, IF and PCA are 0.881578/0.281496, 0.890166/0.253774 and 0.822361/0.146648. These four-method results must not be mixed into the old eight-task average; RawCount S-only is not a prerequisite for this conclusion. E/R residuals exceed persistence in some rankings but do not constitute a branch gain over the target-mean reference. [尚未证实] E/R 是否提供稳定 target-mean 以上增量；RawCount 与旧版本不得混合。

Evidence record: `reference_metrics.csv` Full row with source `full_gmm-full-20261021T112212Z-2474004`; supporting files are `variable_metrics.csv`, `max_attribution.csv`, and `scale_diagnostics.csv` in the frozen analysis directory.

Evidence classification: the metric values and saved attribution/floor counts are directly recorded; causal explanations below are labelled as supported concerns or hypotheses and are not treated as proven.

Scale evidence: E/squared_residual: 0 of 2 scales at floor, E/nll: 0 of 2 scales at floor, R/squared_residual: 0 of 2 scales at floor, R/nll: 0 of 2 scales at floor.

R max attribution evidence (largest shares): anomaly:7 0.686, normal:9 0.579, normal:7 0.421, anomaly:9 0.314.

Existing diagnostic evidence:
- FullDiagnostic fixed-head contrasts (AP/AUC): full/max 0.0540/0.7130, full/mean 0.0640/0.7433, diag/max 0.0492/0.6245, diag/mean 0.0512/0.6559; these are fixed post-hoc contrasts, not selected replacement scores.
- FullDiagnostic calibration evidence: q=.995 tail has 1 points, 0 label anomalies and 0 contextual anomalies; this affects alarm interpretation, not ranking metrics.
- ConditionalScore post-hoc row: joint versus p(S,P|E,R) AP 0.0540->0.0474, AUC 0.7130->0.6964; it retains the source cache and is not a refit or proof of field causality.
- DataRepair RawCount paired results (separate data version): MNRF Full RawCount AP 0.0540->0.1703, AUC 0.7130->0.8535; CATCH RawCount AP 0.1070->0.2815, AUC 0.8358->0.8816; IsolationForest RawCount AP 0.1011->0.2538, AUC 0.8002->0.8902; PCA RawCount AP 0.0801->0.1466, AUC 0.7328->0.8224.

### Creditcard

[已完成] Full ranking is high but AP and calibrated precision are low relative to its AUC (AUC=0.920188, AP=0.051007, P/R/F1=0.035202/0.244344/0.061538, FPR=0.010423, TP/FP/FN/TN=54/1480/167/140512, segments hit/scored=39/202). R squared-residual AP is 0.356110 while R auxiliary NLL AP is 0.001756; the ranking loss is already present at the single-variable NLL layer, and cross-variable scaling/max aggregation can worsen it. Pointwise mu/sigma are not saved, so high-variance masking is not established. [尚未证实] 没有逐点 mu/sigma，不能断言高 sigma 掩盖异常；[真正新增验证] 仅导出预注册 mu/sigma 样本。

Evidence record: `reference_metrics.csv` Full row with source `full_gmm-full-20261024T115131Z-3977864`; supporting files are `variable_metrics.csv`, `max_attribution.csv`, and `scale_diagnostics.csv` in the frozen analysis directory.

Evidence classification: the metric values and saved attribution/floor counts are directly recorded; causal explanations below are labelled as supported concerns or hypotheses and are not treated as proven.

Scale evidence: E/squared_residual: 0 of 29 scales at floor, E/nll: 0 of 29 scales at floor, R/squared_residual: 1 of 29 scales at floor, R/nll: 0 of 29 scales at floor.

R max attribution evidence (largest shares): anomaly:V14 0.471, anomaly:V17 0.371, normal:V24 0.191, normal:V4 0.180.

Existing diagnostic evidence:
- FullDiagnostic fixed-head contrasts (AP/AUC): full/max 0.0510/0.9202, full/mean 0.1065/0.9663, diag/max 0.0941/0.9274, diag/mean 0.1295/0.9576; these are fixed post-hoc contrasts, not selected replacement scores.
- FullDiagnostic calibration evidence: q=.995 tail has 71 points, 3 label anomalies and 9 contextual anomalies; this affects alarm interpretation, not ranking metrics.
- ConditionalScore post-hoc row: joint versus p(S,P|E,R) AP 0.0510->0.0432, AUC 0.9202->0.7497; it retains the source cache and is not a refit or proof of field causality.

### CICIDS

[已完成] Full ranking is near random and the fixed alarm has no true positives (AUC=0.538825, AP=0.000917, P/R/F1=0.000000/0.000000/0.000000, FPR=0.016629, TP/FP/FN/TN=0/1411/74/83440, segments hit/scored=0/65). E/R residuals are also weak at the alarm threshold, although their ranking can exceed Full. The source lacks verified entity/session continuity, so row adjacency and event interpretation remain unknown. [尚未证实] E/R 与 Full 的差异是否来自表示或边界；源没有可验证 entity/session 连续性。 [真正新增验证] 获取 flow/session 边界映射，不改变评分。

Evidence record: `reference_metrics.csv` Full row with source `full_gmm-full-20261024T115157Z-3980319`; supporting files are `variable_metrics.csv`, `max_attribution.csv`, and `scale_diagnostics.csv` in the frozen analysis directory.

Evidence classification: the metric values and saved attribution/floor counts are directly recorded; causal explanations below are labelled as supported concerns or hypotheses and are not treated as proven.

Scale evidence: E/squared_residual: 28 of 72 scales at floor, E/nll: 15 of 72 scales at floor, R/squared_residual: 48 of 72 scales at floor, R/nll: 3 of 72 scales at floor.

R max attribution evidence (largest shares): anomaly:Fwd IAT Std 0.568, anomaly:Bwd Packet Length Std 0.189, anomaly:URG Flag Count 0.135, normal:Flow Bytes/s 0.110.

Existing diagnostic evidence:
- FullDiagnostic fixed-head contrasts (AP/AUC): full/max 0.0009/0.5388, full/mean 0.0011/0.6219, diag/max 0.0011/0.6054, diag/mean 0.0014/0.6989; these are fixed post-hoc contrasts, not selected replacement scores.
- FullDiagnostic calibration evidence: q=.995 tail has 42 points, 0 label anomalies and 42 contextual anomalies; this affects alarm interpretation, not ranking metrics.
- ConditionalScore post-hoc row: joint versus p(S,P|E,R) AP 0.0009->0.0012, AUC 0.5388->0.5885; it retains the source cache and is not a refit or proof of field causality.

## Updated overall assessment

The historical eight-dataset snapshot above and the paper-baseline reference remain unchanged. The new MSL and PSM tables are separately scoped supplements, not replacements for those historical rows and not a new pooled eight-dataset average. MSL now has evidence supporting the sequence-bounded, 55-field-input/`col_0`-scoring task definition: on 68,572 common points its AUC/AP exceed all three available common-protocol comparators, and the four-cell contrast associates the larger AP change with the score target rather than the boundary-only change. PSM shows the opposite operational constraint: despite useful overall ranking and a high-recall saved `q=.995` alarm, its current low-FPR tail recalls fewer anomalies than every best baseline selected at the corresponding FPR ceiling; changing only the threshold cannot improve the ordering of that tail. CalIt2 remains split by data version: the historical modulo-10 result stays in the old eight-group snapshot, while the completed RawCount four-method paired comparison is reported separately (RawCount MNRF AUC/AP 0.853538/0.170287; CATCH 0.881578/0.281496; IF 0.890166/0.253774; PCA 0.822361/0.146648).

These results support preserving each dataset's explicit task semantics and reporting score-ranking and thresholded alarm behavior separately. They do not establish that one head or score rule should be selected per test dataset. E/R residual AP above persistence remains only a reference comparison, not proof of conditional information gain; target-mean and latent-mean references still matter. Full GMM, E/R decoder residual/NLL, conditional-score diagnostics, and any complete coupling-flow score are separate heads and must not be pooled.

## Remaining decision-changing validations

1. If changing the MSL training objective is under consideration, compare a predeclared target/condition-aware objective against this completed sequence-bounded 55-input, `col_0`-scored model. The current four-cell result establishes the score-target effect more directly than the boundary-only AP change, but does not test objective-level separation of telemetry and commands.
2. For GECCO alarm interpretation and CICIDS segment interpretation, preserve the existing scores and first resolve whether GECCO calibration contamination affects the chosen alarm rule and whether CICIDS entity/session boundaries can be recovered. This does not alter the already observed ranking metrics.
3. Creditcard's sampled residual-vs-NLL ranking reversal is established, but pointwise decoder `mu`/`sigma` are still needed to distinguish mechanisms; the existing sample export proposal is diagnostic only and must not choose variables or a new test-set scoring rule.
