# PSM Same-FPR Recall Diagnostic

## Result

On the 87,650 coordinates shared by the indexed PSM comparison and all three saved baseline bundles, the saved MNRF `q=0.995` alarm point has recall **60.28%** and FPR **27.35%**. Under a 1% FPR ceiling, the best attainable MNRF threshold on these same scores detects **337 / 24,378 anomalies** (recall **1.38%**, actual FPR **1.00%**). That is **14,357 fewer detections** than the original MNRF alarm point. The best baseline at this ceiling is CATCH-common: it detects 966 anomalies (recall 3.96%), so MNRF detects **629 fewer anomalies**, a recall difference of **-2.58 percentage points**.

| FPR ceiling | MNRF detected / recall | Best baseline | Baseline detected / recall | MNRF TP difference | Recall difference |
|---:|---:|---|---:|---:|---:|
| 0.1% | 18 / 0.07% | IsolationForest | 542 / 2.22% | -524 | -2.15 pp |
| 0.5% | 185 / 0.76% | IsolationForest | 689 / 2.83% | -504 | -2.07 pp |
| 1% | 337 / 1.38% | CATCH-common | 966 / 3.96% | -629 | -2.58 pp |
| 2% | 940 / 3.86% | PCA | 3,505 / 14.38% | -2,565 | -10.52 pp |

Selected work points use whole tied-score groups and strict `score > threshold`; there is no interpolation. For each FPR ceiling, the method-specific threshold is chosen post hoc on PSM test labels to maximize recall without exceeding the ceiling, then minimize FP among equal-recall points. These are diagnostic operating points, not a deployable threshold-selection procedure. Full per-method points, including the original saved alarm thresholds, are in [`psm_low_fpr_operating_points.csv`](psm_low_fpr_operating_points.csv); the tie-aware low-FPR curve is [`psm_low_fpr_recall_curve.png`](psm_low_fpr_recall_curve.png).

## Sources And Alignment

- The four PSM rows in `result/analysis/MNRF-Comparison-Adapter-20261102/common_comparison.csv` fixed the source Full run and reported 87,650 common valid points. The MNRF source is `full_gmm-full-20261024T115119Z-3976714`.
- The original MNRF Full bundle contains 87,650 valid endpoints, global rows 132,672–220,321. A later frozen prefix adapter has scores across the 87,841-point official test axis, adding the first 191 points; those new points are excluded here because the three baseline bundles do not score them. Thus this is an effective-point comparison, not a claim that all methods cover the full official axis.
- The adapter score coordinates, times, labels, and threshold were checked against its bound source/full-axis inputs; the original Full bundle has the same 87,650 coordinates, labels, times, predictions, and threshold. Each baseline bundle was checked for sorted unique integer coordinates, exact coordinate equality, per-point label/time equality, prediction equality to its saved threshold, complete status, file digest, and calibration record.
- All methods' original alarm points use their own saved independent calibration threshold at `q=0.995` and strict `score > threshold`. This is not a shared threshold or equal-FPR evaluation. The MNRF threshold remains 15.7021261311 and was not changed.
- The original saved MNRF alarm point is reported on the same 87,650 common points as the low-FPR analysis. It is not an estimate over the 87,841-point full axis.

## Interpretation

The existing MNRF score ranks anomalies (the indexed table reports ROC-AUC 0.7449 and AP 0.4373), but its saved calibration alarm threshold operates at a very high test FPR on this common subset. At the fixed 1% diagnostic budget, MNRF retains only 337 true positives, fewer than CATCH-common, while the baseline with best recall changes across FPR ceilings. This shows that the original high-recall alarm point depends on tolerating many false alarms, and that the MNRF score's low-FPR tail is weaker than the best available baseline tail on this PSM point set. Because the thresholds for the budgeted points were selected using test labels, these results diagnose score ranking only; they do not justify replacing the saved calibration threshold or claim a calibrated deployment result.

No training, model forward pass, fitting, recalibration, or baseline rerun was performed.
