# MNRF and Baseline Comparison Protocol

## Scope and evidence classes

This report accepts the confirmed baseline premise: the baseline settings and paper-table results are valid. Absence of baseline result files in this checkout is not treated as a failed experiment. The numeric paper reference is the ICLR 2025 conference-paper Table 2; `baseline_settings.csv` records the server entry points and effective settings for both `detect_score` and `detect_label`. CATCH-master archive rows, where available, are original-protocol records. APD-CATCH rows are common-protocol diagnostics. These three classes are never merged into one score.

## What produces the MNRF table score

The table's MNRF score is the saved Full GMM probe, not the complete MNRF-Lite V2 flow and not the frozen E/R decoder diagnostics. `views.py:44-83` constructs per-window causal state/evolution/pattern/relation views and concatenates the layout. The complete model's optional coupling head is `fusion.py:63-125` (`AlternatingAffineFlow`); its training NLL is implemented in `training.py:327-334` and stage-2 fitting in `training.py:358-460`. No Full table row in this report silently substitutes that flow.

For the Full GMM probe, `probe.py:1128-1354` fits one GMM per original variable on the selected 25 feature coordinates and the fixed fit-endpoint sample. `probe.py:1359-1393` scores each variable with the saved model and averages the four fixed-noise evaluations. `probe.py:1395-1425` computes validation median and MAD scale, then the calibration 0.995 quantile. `probe.py:1427-1455` aggregates the normalized variable scores by a maximum and predicts with strict `score > threshold`. Thus, for endpoint t and variable c:

`nll[t,c] = mean_j(-log p_gmm((feature[t,c,selected] - center_feature[c]) / scale_feature[c] + noise_j))`;

`z[t,c] = (nll[t,c] - median(validation_nll[c])) / max(1.4826*MAD(validation_nll[c]), score_mad_floor)`;

`score[t] = max_c z[t,c]`, and `threshold = quantile(max_c z_calibration[t,c], 0.995)`.

The E/R decoder rows in `MNRF-FrozenDecoder-Analysis-20261102-final/reference_metrics.csv` use the auxiliary decoder's squared residual or Gaussian NLL, with their own validation median/MAD and calibration threshold. They are diagnostics of conditional readout quality, not the table's Full detector. The coupling flow is a separate complete-model design path; the Full table does not prove it was used.

## Alignment decisions

The directly comparable MNRF row is the common-protocol Full GMM row when the baseline comparison uses the same source version, labels, valid test coordinates and metric definition. Its ranking metrics (ROC-AUC and average precision) can be compared after exact coordinate alignment; changing a threshold cannot change either ranking metric. Point P/R/F1/FPR and Affiliation metrics must remain separate when a baseline uses a different threshold source, point adjustment, flattened full-window coverage, or segment definition.

MNRF currently uses frozen source train/validation/calibration/test endpoints, no second split, preserved labels and strict endpoint `score > threshold`. Its standardization statistics are fit on the frozen feature-training cache; the alarm threshold is independent calibration q=.995. The baseline settings table shows that native CATCH and several transformer adapters use full non-overlapping windows flattened to positions, while label paths use their own train+test percentile/anomaly-ratio conventions. Those native methods retain their heads, losses, contexts and aggregation. Equal hidden widths, learning rates, epochs, or a forced GMM head are not fairness requirements.

For a fair next comparison, report three side-by-side blocks: (1) paper Table 2 values, (2) original-protocol archive values with their native threshold/post-processing, and (3) common-protocol values on the shared valid coordinates. Do not relabel a common calibrated MNRF alarm as the paper's Aff-F. If a baseline's historical point score uses point adjustment or test-derived anomaly ratio, retain that fact in its original-protocol column rather than silently converting it.

## Executable comparison settings

For MNRF, use the registered Full source for each task, preserve its input hashes, split/end-point hashes, variable order and valid mask, and report the saved Full GMM score as the primary common diagnostic. Use the source's native sequence length and variables, CPU/GPU only as an execution resource, and no test-label tuning. Report ranking metrics on the exact valid test points. Report alarms with the independent calibration threshold and strict `score > threshold`, while putting native baseline alarms in their own protocol column. Affiliation precision/recall/F must use the same valid contiguous intervals and boundaries; no point adjustment or gap concatenation is allowed.

The resulting comparison is fair at the data identity, evaluated coordinates, provenance and metric-definition layers while preserving each method's native detector. It is not a claim that all heads or threshold protocols are mathematically identical.

## Evidence limits

The paper/reference values and server settings are confirmed by the existing investigation artifacts. Local common MNRF rows are saved and validated. Historical multi-anomaly-ratio selection rules for some native label paths remain unknown; this does not block the independent common diagnostic. Creditcard and CICIDS entity/session continuity is not recoverable from the frozen input and remains a boundary limitation.
