# Direction B2a-v1 Held-Out Transfer Record

Status: closed after the frozen GPU development smoke on 2026-07-12. B2a-v1
tested whether B1's dual evidence mechanism transfers to a structurally
distinct normal process and to every target channel. It remains a synthetic
mechanism stage, not a real-data performance claim.

## Recorded Outcome

The pre-registered `4101` smoke failed eight gates and therefore did not
authorize seeds `4102..4105` or B2b real-data work. The complete immutable
artifact is retained at:

```text
result/multi_evidence/b2a_drift_seed4101_gpu/b2a_evaluation.json
```

All six cross heads nonetheless improved reference MAE over the target-mean
baseline by `58.1%--74.0%`. The failure is thus not a global optimization
collapse. It contains two findings that must remain separate:

1. The frozen B1 transfer protocol failed its all-target gates, including
   target-specific paired ordering, coherent controls, and a disagreement-bin
   FPR-gap check.
2. The v1 synthetic donor contract was insufficient for a terminal-point
   score: it required only a nonzero replacement somewhere in the window,
   while several source/donor pairs had negligible target-specific terminal
   cross-compatibility gap. This prevents v1 from serving as the final
   transfer inference even though its failed result is preserved.

The next experiment must use a new name, config, seeds, and frozen protocol;
it must not edit or rerun B2a-v1 under the same label.

## Immutable Boundary

B2a does not change B1's repair objective, add a third branch, share a trunk,
learn uncertainty, or fuse scores. It wraps one fully independent B1-style
temporal/cross repair pair for each of the six target channels. Parameter IDs
are checked across all `6 x 2` branches; target-specific models, tails, bins,
and outer-calibration thresholds are never flattened or shared.

Each component remains a matrix:

```text
R_T[t, i], R_C[t, i], D[t, i]
```

No B2a scalar timestamp score is emitted. A later real-data protocol may
pre-register a component-wise max-over-target aggregation and calibrate it on
outer normal data, but that is explicitly outside B2a.

## Held-Out Generator

The B2a normal process is a six-channel, rank-two, continuously drifted VAR:

```text
r_t = sin(2*pi*t / 192)
f_t = [[0.78, 0.12], [-0.08, 0.66]] f_(t-1) + Q epsilon_t
x_t = (A0 + r_t A1) x_(t-1) + (L0 + r_t L1) f_t + eta_t
```

This differs from B1's single-factor, two-regime process. Relation phase and
latent factors exist only in generator metadata and post-hoc FPR diagnostics;
they are not visible to the model, normalizer, reliability features, tail fit,
or outer calibration.

For each target and phase quadrant, four pairs are generated:

```text
coherent control:         source normal window
unsupported-target break: source target column + donor non-target columns
target-omission break:    source non-target columns + donor target column
target spike:             source plus a 3-sigma terminal target perturbation
```

Every target has 16 pairs. A separate set of 16 unmodified high relation-drift
normal controls per target tests that normal structural drift alone is not
reported as an anomaly.

## Calibration And Split

The four normal segments are temporally disjoint with `H=24` gaps:

```text
optimization -> gap -> validation -> gap -> reference -> gap -> outer calibration
```

The channel normalizer is fitted only on optimization data. For every target
and component, B1's input-restricted ECRC uses three fixed reliability bins,
reference-only empirical tails, and an outer-normal conformal threshold. Every
reference bin requires at least 50 samples; the frozen `4101` precheck has a
minimum of 93.

## Frozen Gates

All six targets must pass individually. No mean or pooled target statistic can
rescue a target failure.

| Gate | Requirement per target |
| --- | --- |
| Branch/routing isolation and all parameter sets | exact up to `1e-7` |
| Counterfactual ties | temporal residual tie for coherent/unsupported; cross prediction tie for coherent/omission, up to `1e-7` |
| Both dependency breaks, cross and disagreement tails | `>=14/16` positive paired deltas; median `>=0.5` |
| Coherent control | cross-or-disagreement exceedances `<=2/16` |
| Normal relation-drift control | cross-or-disagreement exceedances `<=2/16` |
| Target spike | both residual paths exceed `>=14/16` |
| Cross normal skill | at least 10% MAE improvement over the target-mean baseline |
| Background FPR | each component/reliability bin and hidden drift-phase diagnostic `<=0.10`; disagreement bin/phase gap `<=0.05` |

## Historical Execution

Frozen config:

```text
config/multi_evidence/b2a_drift_rotation.json
```

Development integrity smoke, then unchanged confirmation seeds:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/multi_evidence/run_b2a_transfer.py \
  --config config/multi_evidence/b2a_drift_rotation.json \
  --output-dir result/multi_evidence/b2a_drift_seed4101_gpu \
  --seed 4101 --device cuda:0 --torch-threads 1 --strict
```

Only if `4101` had passed without code or config changes would the same command
have run for `4102..4105`. It did not pass. Do not run those confirmation seeds
and do not start B2b real-data evaluation from this protocol.

## B2b Handoff

If B2a passes all fixed seeds, B2b starts with HAI21 part 1 under a separate
frozen no-point-adjust protocol. HAI21 parts 2/3 remain untouched external
confirmation data. The B2b runner must separately calibrate a pre-registered
max-over-target score for each component; B2a intentionally does not perform
that aggregation.
