# Creditcard R auxiliary-decoder NLL loss diagnosis

Source: `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/score/MNRF-ViewProbe/Creditcard/full_gmm-full-20261024T115131Z-3977864`  
Bound prior frozen-decoder diagnostic: `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/analysis/MNRF-FrozenDecoder/Creditcard/full_gmm-full-20261024T115131Z-3977864`  
New output: `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/analysis/MNRF-FrozenDecoder/Creditcard/R-loss-20261102`

## Verified full-test result

From the existing full-test saved scores, squared-residual AP=0.356109599376, ROC-AUC=0.954259611592; auxiliary Gaussian-NLL AP=0.001755611372, ROC-AUC=0.590763733041. Fresh frozen R decoder reproduction passed with max absolute residual error 0 and NLL error 0 (rtol=1e-06, atol=1e-07).

## Findings

The loss appears at two distinct stages.

1. **Within-variable Gaussian scoring already loses top-rank quality.** For V14, full-test squared-residual AP/AUC are `0.772511 / 0.953681`, while NLL AP/AUC are `0.141851 / 0.941809`. For V17 they are `0.707879 / 0.945541` versus `0.058146 / 0.945620`. Thus AUC changes little, but AP collapses: a small number of normal observations entering the highest ranks can dominate AP under Creditcard's rare-positive prevalence. Because each saved center/scale is a positive per-variable affine transform, that normalization cannot explain the within-variable AP loss; it can still affect which variable wins the later max.
2. **The error-weighting term strongly compresses large anomaly residuals.** In the fixed sample, among each variable's largest-residual anomaly decile (23 points), V14's median squared residual is `275.57`, but median sigma is `12.17` and the NLL weighted-error term is only `0.938`; its `log(sigma)` median is `2.499`, giving median NLL `4.388`. V17's corresponding values are `438.28`, `13.55`, `1.045`, `2.607`, and `4.622`. The constant `0.5*log(2*pi)` is identical for every point and cannot change rankings. The positive log-scale term partly offsets, but does not undo, the division of squared error by `2*sigma^2`.
3. **Scale association is visible at similar residual magnitudes, but is not a causal test.** In within-variable pooled residual decile bins containing both classes, sampled anomaly sigma medians are `63.4x` normal for V14 and `76.2x` for V17. The binning is coarse and observational. A fixed training-sigma counterfactual (per-variable medians from 2,048 seed-2021 training endpoints) is included in the CSV; it is not recalibrated and is not a formal score result.
4. **Max aggregation adds a separate failure mode.** Under the saved NLL calibration threshold, the aggregate has `0/221` TP and `222` FP; tie-split max attribution assigns `95.5%` of these FP maxima to V24. Among anomalies, NLL maxima are led by V17 (`54.3%`) and V14 (`33.5%`). By contrast, the squared-residual aggregate has `184/221` TP and `1,167` FP, with TP maxima led by V14 (`51.6%`) and V17 (`44.6%`). These are different outcomes from the same R decoder, not evidence about the Full GMM head.

| Variable | Residual AP (full test) | NLL AP (full test) | Residual AUC | NLL AUC | Sample normal sigma median | All-anomaly sigma median |
|---|---:|---:|---:|---:|---:|---:|
| V14 | 0.772511 | 0.141851 | 0.953681 | 0.941809 | 0.071845 | 4.555580 |
| V17 | 0.707879 | 0.058146 | 0.945541 | 0.945620 | 0.067791 | 5.238712 |

`sampled_test_point_variables.csv` contains every valid test anomaly plus 2,048 valid normal endpoints sampled without replacement with seed 2021, across all 29 variables. It includes y, mu, sigma, residual, squared residual, all NLL terms, saved NLL, the fixed training-sigma counterfactual, saved normalization parameters, and max-variable/tie contribution. `summary.json` contains full-test per-variable rankings, the sampling/source bindings, distributions, and max-variable attribution. The bounded sample is not used to estimate full-test AP.

## Interpretation boundary

The evidence confirms that heteroscedastic weighting changes the within-variable ranking of V14/V17 and that cross-variable max introduces strong V24-driven false alarms for aggregate NLL. It does **not** identify whether sigma is miscalibrated, whether the decoder learned a legitimate conditional uncertainty, or whether a different scoring rule would generalize. The one fixed-sigma counterfactual is a mechanism probe, not a proposal to replace the model's likelihood or tune it on test labels.

## Scope

This concerns only the frozen Stage-1 R auxiliary decoder. It does not establish the cause of the separate Full GMM score. No training, encoder forward, GMM fitting, recalibration, or threshold search was run.
