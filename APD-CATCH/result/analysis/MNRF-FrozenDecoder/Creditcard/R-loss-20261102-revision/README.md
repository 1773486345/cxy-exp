# Creditcard R-loss statistics revision

This is a read-only correction of `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/analysis/MNRF-FrozenDecoder/Creditcard/R-loss-20261102`. It consumes only its existing sample CSV and summary; it does not load a checkpoint or source cache.

## Residual-bin correction

For each variable, bins reuse the previous pooled sampled squared-residual quantile rule (10 requested quantiles, duplicate edges collapsed). The table contains 290 variable/bin rows: 285 bins contain both classes and 5 preserve a missing class or empty-bin state. Each row reports both class counts, signed-residual and squared-residual ranges/medians, sigma medians, and an anomaly/normal sigma ratio only when both classes are present. These are coarse bins, not exact residual matches. No cross-bin pooled ratio is reported; the earlier 63.4x/76.2x wording is withdrawn as a residual-controlled claim.

For the prespecified variables, V14: 10 overlapping bins, 0 bins without anomalies, coarse-bin sigma ratios 1.06–7.05; V17: 9 overlapping bins, 1 bins without anomalies, coarse-bin sigma ratios 1.11–8.52. These within-bin ratios are still descriptive only: each coarse interval can contain materially different residuals between classes, so they do not isolate a sigma effect at fixed residual. The corrected evidence is therefore insufficient to quantify an independent residual-controlled sigma difference.

## Max tie shares

For each endpoint, NLL and residual standardized values are compared jointly across all 29 variables using the original `np.isclose(rtol=1e-06, atol=1e-07)`. If k variables tie at the maximum each now receives 1/k; other variables receive zero. Endpoint shares sum to one. The full-test `weighted_dominance` summary was not recalculated from this sample and remains byte-for-byte copied from the original summary.

## Preserved evidence

The original residual/NLL full-test AUC/AP, confusion matrices, thresholds, fixed-sigma counterfactual, and source bindings are copied unchanged into this revision summary. The revised per-point CSV retains the original sample selection: all 221 anomalies, 2,048 seed-2021 sampled normals, and 29 variables (65,801 rows).

Synthetic checks exercised the actual binning and tie-share functions: within-bin equal sigmas yielded ratios of one despite differing class proportions across bins; two- and three-way ties yielded 1/2 and 1/3 shares summing to one.
