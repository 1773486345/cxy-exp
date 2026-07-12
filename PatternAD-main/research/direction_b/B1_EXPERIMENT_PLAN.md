# Direction B1 Experiment Plan

Status: B1 synthetic confirmation completed on 2026-07-12. B1 is the retained
controlled-mechanism baseline and executable repair implementation. A-v1 is
closed and archived under [`archive/direction_a/`](archive/direction_a/README.md);
Direction A2 is a separate active pre-model research definition in
[`../direction_a/A2_EXPERIMENT_PLAN.md`](../direction_a/A2_EXPERIMENT_PLAN.md).

## Claim Boundary

B1 is a controlled mechanism result, not yet a real-benchmark claim. It
establishes that independently constrained temporal and cross-variable repair
evidence can expose two complementary relation breaks while keeping normal
false alarms bounded under observable evidence-reliability variation.

It does not claim that a learned fused anomaly score is optimal. B1 exports
three components separately and does not tune a mixture weight on any test
set.

## Frozen B1 Contract

For a normalized terminal window `W in R^{H+1 x D}`, target channel `i=0`, and
terminal target `y=W[H,i]`:

```text
temporal input T = W[0:H, i]                # target history only
cross input    C = W[0:H+1, j != i]         # all non-target channels, synchronous
mu_T = TemporalGRU(T)
mu_C = CrossGRU(C)

R_T = (y - mu_T)^2
R_C = (y - mu_C)^2
D   = (mu_T - mu_C)^2
```

The GRUs and heads have disjoint parameter sets. There is no shared encoder,
agreement loss, distillation, score fusion, target-current access in the
temporal branch, or target-column access in the cross branch.

`C` may observe non-target values at the terminal time. This is a synchronous
evidence assumption, not a causal-direction claim. A future lagged-cross
variant must be evaluated as a separate protocol.

## Reliability Calibration

B0's global tail calibration failed under explicitly heteroscedastic normal
states. B1 retains the repair model and changes only normal-only calibration.
For standardized windows, its observable reliability features are:

```text
v_T = sqrt(mean(diff(W[0:H, i])^2) + 1e-6)
v_C = sqrt(mean(diff(W[0:H+1, j != i])^2) + 1e-6)
v_D = sqrt(v_T^2 + v_C^2)
```

`v_T` cannot inspect drivers or `y`; `v_C` cannot inspect any target value;
`v_D` only combines the two allowed reliability values. The optimization-normal
split fixes three equal-frequency boundaries. Reference-normal windows fit a
separate empirical upper tail for every `(R_T, R_C, D) x reliability-bin`.
Every reference bin must contain at least 50 samples; otherwise the run is
invalid rather than silently merged or retuned. The independent outer-normal
calibration split provides one finite-sample conformal upper threshold per
component after this stratified tail mapping. No hidden generator regime,
test score, or test label routes a bin or fits a threshold.

This is evidence-reliability calibration, not A-v1 residual semantics:
it conditions only on the permitted input quality of each evidence route, not
on target residual history or a latent pattern label.

## Synthetic Contract

The suite is a two-regime factor-VAR normal process. A coherent control is a
directly sampled normal factor-VAR terminal window, rather than an injected
smooth profile. For each pair, a non-overlapping normal donor with a different
latent realization is used only by the generator to construct relation breaks:

```text
coherent control:         target and drivers from the same normal window
unsupported-target break: target window from coherent; drivers from donor
target-omission break:    drivers from coherent; target window from donor
target spike:             coherent window plus a terminal target spike
```

The generator saves and verifies exact counterfactual ties. It does not expose
latent state or hidden regime to the model or reliability calibration.

## Split And Gates

All normal intervals are temporally disjoint with an `H=24` gap between
optimization, validation, reference, and outer-calibration segments. The
normalizer is fitted only on optimization normal data. B1 uses `3072` normal
training points and `1536` normal background test points so every fixed
reference reliability bin has adequate support.

| Gate | Required |
| --- | --- |
| Branch/routing isolation and disjoint parameters | exact up to `1e-7` |
| Counterfactual input ties | exact up to `1e-7` |
| Each dependency break, `R_C` and `D` | at least `14/16` positive paired deltas and median tail delta `>= 0.5` |
| Coherent control | at most `2/16` cross-or-disagreement threshold exceedances |
| Background normal | every component/reliability-bin FPR `<= 0.10`; `D` bin FPR gap `<= 0.05` |
| Cross normal skill | MAE at least 10% below target-mean baseline |
| Target spike | both residual components exceed in at least `14/16` pairs |

## Completed Confirmation

The frozen config is
[`config/multi_evidence/b1_reliability.json`](config/multi_evidence/b1_reliability.json).
Five GPU runs with seeds `3101..3105` all passed. Their source hashes and
per-seed gates are checked by:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/multi_evidence/summarize_b1.py \
  --input-dir result/multi_evidence/b1_ecrc_seed3101_gpu \
  --input-dir result/multi_evidence/b1_ecrc_seed3102_gpu \
  --input-dir result/multi_evidence/b1_ecrc_seed3103_gpu \
  --input-dir result/multi_evidence/b1_ecrc_seed3104_gpu \
  --input-dir result/multi_evidence/b1_ecrc_seed3105_gpu \
  --output-dir result/multi_evidence/b1_ecrc_summary_3101_3105
```

The summary reports cross-repair MAE improvement of `82.4%` to `87.6%` over a
target-mean predictor, maximum reliability-bin FPR of `5.18%` to `7.78%`, and
disagreement-bin FPR gap of `1.09%` to `4.40%`. Every target-spike pair was
detected by both residual paths. Full machine-readable evidence is in
[`result/multi_evidence/b1_ecrc_summary_3101_3105/`](result/multi_evidence/b1_ecrc_summary_3101_3105).

## Next Decision

B2 may proceed only as a new frozen protocol: hold out a structurally changed
synthetic generator before any implementation adjustment, then evaluate on
real labelled multivariate data with no point adjustment and without selecting
component weights on test labels. Frequency/trend evidence, learned
uncertainty, graph modules, and a fused deployment score are explicitly
deferred until B2 establishes that B1's three independently reported
components transfer beyond this mechanism suite.
