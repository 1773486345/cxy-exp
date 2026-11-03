# MSL Same-Point Comparison

## Result

The sequence-bounded MNRF Full GMM scored `col_0` on 68,572 common test coordinates. Its AP (`0.307332`) and ROC-AUC (`0.778967`) are higher than all three locally available pointwise comparators on those exact coordinates:

| Method | ROC-AUC | AP | MNRF minus method AP |
|---|---:|---:|---:|
| MNRF sequence-bounded Full GMM, `col_0` score | 0.778967 | 0.307332 | -- |
| CATCH-common | 0.668308 | 0.200771 | +0.106561 |
| IsolationForest | 0.607384 | 0.153582 | +0.153750 |
| PCA-residual | 0.509302 | 0.154128 | +0.153204 |

The same-point alert results are not identical: MNRF has precision `0.4372`, recall `0.1076`, and FPR `1.7696%`; CATCH-common has `0.3172`, `0.0650`, and `1.7877%`; IsolationForest has `0.5577`, `0.0075`, and `0.0757%`; PCA-residual has `0.1930`, `0.1353`, and `7.2263%`. Each method retains its own saved threshold. All thresholds are from independent calibration at `q=0.995`, use strict `score > threshold`, and are not a shared-FPR comparison.

## Scope And Binding

- MNRF source/run: `result/score/MNRF-ViewProbe-MSL-Boundary/MSL/full_gmm-sequence-bounded-20261102T105427Z-2415811`.
- The starting MNRF result is the `sequence-bounded × col_0_only` row in `result/analysis/MNRF-ViewProbe-MSL-Boundary/full_gmm-sequence-bounded-20261102T105427Z-2415811/four_cell_comparison.csv`: 68,572 common valid points and AP `0.3073322544639995`.
- Its `col_0` score is replayed from the saved 55-column `results/test_nll.npy` and saved calibration `center`/`scale`; the existing four-cell record's threshold `14.898866634368899` is reproduced from saved calibration NLL at `q=0.995`. The original calibration files and source run were not changed.
- The test axis has 73,729 official points. The 68,572 MNRF points omit the first 191 points of each of the 27 mapped test sequences (5,157 points total), where a complete length-192 window is unavailable. MNRF and each baseline were joined by sorted unique global row coordinate. Labels and timestamps match point-for-point; all four methods have 7,766 positives and 60,806 negatives on the intersection.
- CATCH-common and the IsolationForest/PCA bundles each originally contain 73,538 timestamped points. Their bundle coordinates, labels, times, saved predictions, source/data identity, and calibration threshold records were checked before taking the exact coordinate intersection. No truncation by array length was used.
- The CATCH-common inference coordinates in the intersection are at least 191 rows into each mapped test sequence, so its length-192 evaluation context does not cross those test sequence starts. Its training protocol remains the original concatenated-source protocol and is not sequence-bounded. IsolationForest and PCA are row-wise and use no temporal window.
- `../CATCH-master/result/score/CATCH/MSL/run-20260716T213635Z-58595-28964` was inspected. It contains an aggregate test report and a compressed aggregate CSV, but no coordinate-aligned point-score bundle; it is therefore not eligible for this same-point recomputation. Paper-level or aggregate-only values are not substituted for pointwise evidence.

## Interpretation Limits

On the common valid coordinates, the MNRF `col_0` score ranks anomalies better than these three existing common-protocol artifacts. This does not establish equivalence of training protocols: MNRF uses sequence-bounded training/windows, while CATCH-common training follows the concatenated source; the row-wise methods have no window context. The comparison is also not a common-threshold or common-FPR test. The result is specifically a comparison of the sequence-bounded MNRF Full GMM with all 55 fields retained as inputs and `col_0` used for final scoring; it is not a target-only trained model.

The source artifacts and metrics are listed in [`msl_same_point_comparison.csv`](msl_same_point_comparison.csv). No training, forward pass, fit, or recalibration was run for this comparison.
