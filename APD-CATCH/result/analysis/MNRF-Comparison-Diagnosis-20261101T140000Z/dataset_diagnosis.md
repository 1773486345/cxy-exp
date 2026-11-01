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

Full ranking is useful (AUC=0.744926, AP=0.437281, P/R/F1=0.459202/0.602757/0.521276, FPR=0.273502, TP/FP/FN/TN=14694/17305/9684/45967, segments hit/scored=48/71). R squared-residual is a stronger diagnostic ranking than E, but its calibrated alarm still has high FPR; E/R residuals do not explain the Full score by themselves. Evidence-supported adjustment: inspect calibration upper tail and max-variable attribution before changing the head. A fixed-threshold recalibration on the same scores is the smallest test; it cannot be used to claim an AUC/AP change.

Scale evidence: E/squared_residual: 3 of 25 scales at floor, E/nll: 0 of 25 scales at floor, R/squared_residual: 0 of 25 scales at floor, R/nll: 0 of 25 scales at floor.

R max attribution evidence (largest shares): anomaly:feature_1 0.409, normal:feature_1 0.314, anomaly:feature_3 0.311, normal:feature_3 0.280.

### MSL

Full is moderate in ranking and low in recall at its fixed threshold (AUC=0.602445, AP=0.159006, P/R/F1=0.290260/0.077517/0.122358, FPR=0.022380, TP/FP/FN/TN=602/1472/7164/64300, segments hit/scored=13/36). E/R residual diagnostics are close to simple references, while the prior audit records 27 telemetry sequences concatenated into 55 fields. The sequence-boundary effect and mixed target/condition roles are supported concerns, not proven causal causes. Minimum test: score the existing frozen outputs on non-cross-sequence coordinates, then separately report col_0; a boundary-aware retrain would be a later experiment.

Scale evidence: E/squared_residual: 53 of 55 scales at floor, E/nll: 49 of 55 scales at floor, R/squared_residual: 22 of 55 scales at floor, R/nll: 0 of 55 scales at floor.

R max attribution evidence (largest shares): normal:col_2 0.522, anomaly:col_2 0.483, anomaly:col_0 0.210, normal:col_0 0.185.

### GECCO

The Full GMM ranking is poor while E/R residual rankings are high (Full AUC=0.442452, AP=0.223303, P/R/F1=0.726562/0.254795/0.377282, FPR=0.001023, TP/FP/FN/TN=186/70/544/68366, segments hit/scored=3/22; R residual AUC=0.870483, AP=0.303731, P/R/F1=0.068384/0.767123/0.125575, FPR=0.111476, TP/FP/FN/TN=560/7629/170/60807, segments hit/scored=20/22). This separates a density/aggregation/calibration issue from a lack of information in the frozen readout; calibration contamination is possible because the source audit records labeled anomalies in fitting ranges, but it is not proved as the sole cause. Minimum test: replay the same cached NLL with a fixed non-max diagnostic only for analysis, and compare calibration-tail composition without selecting a test-optimal threshold.

Scale evidence: E/squared_residual: 0 of 9 scales at floor, E/nll: 0 of 9 scales at floor, R/squared_residual: 0 of 9 scales at floor, R/nll: 0 of 9 scales at floor.

R max attribution evidence (largest shares): normal:Tp 0.662, anomaly:Leit 0.482, anomaly:Redox 0.284, normal:Leit 0.260.

### Genesis

Full has strong AUC but weak AP and alarm precision (AUC=0.899568, AP=0.046805, P/R/F1=0.057789/0.460000/0.102679, FPR=0.030303, TP/FP/FN/TN=23/375/27/12000, segments hit/scored=3/3). R residual improves over persistence in ranking but its fixed threshold still produces false positives; only three scored anomaly segments limit generalization. Minimum test: inspect normal high-score variable attribution and calibration-tail windows; any threshold change must be a predeclared calibration protocol, not a test optimization.

Scale evidence: E/squared_residual: 9 of 18 scales at floor, E/nll: 2 of 18 scales at floor, R/squared_residual: 2 of 18 scales at floor, R/nll: 0 of 18 scales at floor.

R max attribution evidence (largest shares): anomaly:MotorData.IsForce 0.680, normal:MotorData.ActSpeed 0.244, normal:MotorData.IsAcceleration 0.192, anomaly:MotorData.ActSpeed 0.160.

### NYC

Full ranking is above random but its fixed alarm has no true positives (AUC=0.691339, AP=0.044089, P/R/F1=0.000000/0.000000/0.000000, FPR=0.001697, TP/FP/FN/TN=0/7/99/4119, segments hit/scored=0/1). The frozen E/R report has one scored event and k=8 neighborhood summaries; lower prediction error in calendar/target readout does not establish anomaly information. Passenger target and calendar context must be kept distinct. Minimum test: one predeclared target-only/context-only scoring comparison on the same event boundaries, explicitly marked as single-event evidence.

Scale evidence: E/squared_residual: 2 of 3 scales at floor, E/nll: 0 of 3 scales at floor, R/squared_residual: 1 of 3 scales at floor, R/nll: 0 of 3 scales at floor.

R max attribution evidence (largest shares): anomaly:channnel3 0.485, normal:channnel2 0.393, normal:channnel3 0.304, normal:channnel1 0.303.

NYC fixed-neighborhood rows available: 20; this is one scored event and is not a general mechanism test.

### CalIt2

The historical modulo-10 source has low Full AP (AUC=0.713040, AP=0.053971, P/R/F1=0.030612/0.042857/0.035714, FPR=0.042054, TP/FP/FN/TN=3/95/67/2164, segments hit/scored=2/12). The existing RawCount report is a separate data version and showed improved MNRF/CATCH/IF/PCA ranking, but no RawCount S-only result was verified; it must not be mixed into the old eight-task average. E/R residuals exceed persistence in some rankings but do not constitute a branch gain over the target-mean reference. Minimum test: compare old and RawCount Full under the same coordinate/calibration contract, then run the already prepared RawCount S-only head if needed.

Scale evidence: E/squared_residual: 0 of 2 scales at floor, E/nll: 0 of 2 scales at floor, R/squared_residual: 0 of 2 scales at floor, R/nll: 0 of 2 scales at floor.

R max attribution evidence (largest shares): anomaly:7 0.686, normal:9 0.579, normal:7 0.421, anomaly:9 0.314.

### Creditcard

Full ranking is high but AP and calibrated precision are low relative to its AUC (AUC=0.920188, AP=0.051007, P/R/F1=0.035202/0.244344/0.061538, FPR=0.010423, TP/FP/FN/TN=54/1480/167/140512, segments hit/scored=39/202). R squared-residual AP is 0.356110 while R auxiliary NLL AP is 0.001756; the ranking loss is already present at the single-variable NLL layer, and cross-variable scaling/max aggregation can worsen it. Pointwise mu/sigma are not saved, so high-variance masking is not established. Minimum test: export fixed pointwise mu/sigma for a predeclared sample and compare per-variable residual/NLL ranks without choosing variables from test AP.

Scale evidence: E/squared_residual: 0 of 29 scales at floor, E/nll: 0 of 29 scales at floor, R/squared_residual: 1 of 29 scales at floor, R/nll: 0 of 29 scales at floor.

R max attribution evidence (largest shares): anomaly:V14 0.471, anomaly:V17 0.371, normal:V24 0.191, normal:V4 0.180.

### CICIDS

Full ranking is near random and the fixed alarm has no true positives (AUC=0.538825, AP=0.000917, P/R/F1=0.000000/0.000000/0.000000, FPR=0.016629, TP/FP/FN/TN=0/1411/74/83440, segments hit/scored=0/65). E/R residuals are also weak at the alarm threshold, although their ranking can exceed Full. The source lacks verified entity/session continuity, so row adjacency and event interpretation remain unknown. Minimum test: obtain a source-level session/flow boundary mapping; until then, treat boundary-sensitive Affiliation and window interpretation as limited rather than changing the score.

Scale evidence: E/squared_residual: 28 of 72 scales at floor, E/nll: 15 of 72 scales at floor, R/squared_residual: 48 of 72 scales at floor, R/nll: 3 of 72 scales at floor.

R max attribution evidence (largest shares): anomaly:Fwd IAT Std 0.568, anomaly:Bwd Packet Length Std 0.189, anomaly:URG Flag Count 0.135, normal:Flow Bytes/s 0.110.

## Interpretation boundary

A residual AP above persistence is only a comparison to that reference; it is not proof of conditional information gain. The target-mean and latent-mean references are also required. Lower prediction error is not equivalent to better anomaly ranking. Full GMM, E/R decoder residual/NLL, and any complete coupling-flow score are separate heads and must not be pooled.

## Up to three next validations

1. Freeze the common coordinates and calibration contract, then run one predeclared score-head replay on GECCO/MSL to separate density/aggregation effects from representation information.
2. Run the corrected CalIt2 RawCount Full versus the existing modulo-10 Full, then the prepared RawCount S-only head only if the Full comparison warrants it.
3. Export fixed pointwise decoder mean/sigma for Creditcard and a boundary-aware MSL/CICIDS coordinate sample; this directly tests variance ranking and sequence/session assumptions without changing the production head.
