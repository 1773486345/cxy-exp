# CalIt2 RawCount data-repair validation

This directory is independent from the historical `dataset/anomaly_detect/data/CalIt2.csv` and shared `DETECT_META.csv`. The old files were not modified (their SHA256 is recorded in `data_root/RAWCOUNT_PROTOCOL.json`). The UCI `CalIt2.data` flow counts were parsed as integers, preserving two-digit values (flow 9 max 62, flow 7 max 54). The historical benchmark columns equal the UCI counts modulo 10; the producer of that modulo projection is not present in this repository, so responsibility for the old conversion is recorded as unknown.

The RawCount protocol keeps the old 5040 half-hour axis, variable order `[9, 7]`, and train boundary 2520. The UCI `CalIt2.events` intervals, using inclusive half-hour endpoints, reproduce all 206 historical positive labels exactly. The read-only preflight produced 1519/22/24/2329 valid windows for train/validation/calibration/test. No labels were cleaned and no split was changed.

## Four-method paired result

`paired_comparison.json` contains old modulo-10 common-protocol rows and the actual RawCount CPU rerun. The RawCount MNRF run was trained from scratch with the frozen Full settings; CATCH, Isolation Forest and PCA used the same RawCount training rows, calibration endpoints and test coordinates. The old rows are references only and were not mixed into an eight-task average.

| method | old ROC-AUC | old AP | RawCount ROC-AUC | RawCount AP | RawCount P/R/F1 | RawCount FPR |
|---|---:|---:|---:|---:|---:|---:|
| MNRF Full RawCount | 0.713040 | 0.053971 | 0.853538 | 0.170287 | 0.109524/0.328571/0.164286 | 0.082780 |
| CATCH RawCount | 0.835812 | 0.106986 | 0.881578 | 0.281496 | 0.130548/0.714286/0.220751 | 0.147410 |
| IsolationForest RawCount | 0.800190 | 0.101147 | 0.890166 | 0.253774 | 0.179372/0.571429/0.273038 | 0.081009 |
| PCA RawCount | 0.732786 | 0.080117 | 0.822361 | 0.146648 | 0.215909/0.271429/0.240506 | 0.030544 |

RawCount improves MNRF ranking over the historical modulo representation (AUC 0.713040 -> 0.853538; AP 0.053971 -> 0.170287). CATCH and IF also improve their ranking/AP; PCA AUC increases while AP decreases slightly. The restored magnitude therefore explains a substantial part of the old CalIt2 loss, but it does not establish that MNRF is best: IF and CATCH remain higher on RawCount AUC/AP. The result is a single corrected-data task and must not be merged into the historical eight-task aggregate.
