# Direction B2a-GC Counterfactual-Contract Transfer Plan

Status: closed after the frozen GPU development smoke `4201` on 2026-07-12.
B2a-GC was a new validity protocol, not a retry or relabeling of B2a-v1. It
changed only how synthetic counterfactual donors were certified. The B1 dual
repair heads, normal process, splits, ECRC, score components, gates, and
no-fusion rule remained unchanged.

## Recorded Outcome

The complete artifact is retained at:

```text
result/multi_evidence/b2a_gc_seed4201_gpu/b2a_gc_evaluation.json
```

The terminal contract passed exactly: its observed minimum structural cross
gap was `2.164` target standard deviations, its minimum terminal target gap
was `2.136`, and relation-value difference was at most `0.080`. All `24/24`
dependency-break paired ordering gates passed, with positive counts of at
least `14/16` and median tail deltas of at least `3.332`. Every target also
passed cross normal-skill (`60.4%--72.5%` MAE improvement).

The run nevertheless failed normal-control/FPR gates: coherent controls were
`5/16`, `4/16`, and `3/16` for targets 0, 1, and 3; target 3 relation-drift
controls were `4/16`; target 1 cross phase FPR reached `11.17%`; and its
disagreement reliability-bin gap was `5.12%`. No confirmation seeds run.

Thus B2a-GC validates the dependency-break mechanism, but rejects the B1
global-after-stratified outer cutoff on this continuous-drift transfer setting.
The separately named B2c-FW-ECRC protocol addresses that calibration failure;
it does not change this result or rerun B2a-GC.

## Motivation

B2a-v1 scored only the terminal point, but its donor contract required a
nonzero replacement somewhere in the `H+1` window. Its broad phase-bin match
also allowed appreciable relation-value differences. Consequently, some donor
pairs supplied almost no target-specific terminal incompatibility, making a
failed paired ordering ambiguous between a detector limitation and an invalid
counterfactual.

For target `i`, B2a-GC computes this generator-only structural cross support:

```text
g_i(t) = sum_{j != i} A_ij(r_t) x_j(t-1) + L_i(r_t) f_t
delta_C = |g_i(source) - g_i(donor)| / std_train(x_i)
delta_Y = |x_i(source) - x_i(donor)| / std_train(x_i)
```

`g_i`, factors, relation values, and the two deltas are recorded only in suite
metadata. They are never passed to a repair head, reliability feature, tail
map, threshold, or gate calculation other than validating the synthetic input
contract.

## Frozen Donor Contract

For every target and every pair, a donor must be non-overlapping, be in the
same hidden phase quadrant, and satisfy:

```text
|r_source - r_donor| <= 0.10
delta_C >= 1.50
delta_Y >= 1.50
```

Each phase is split into four chronological blocks. Sixteen evenly spaced
generator-only source candidates are examined per block, then one valid pair
is selected deterministically: largest minimum structural margin first, then
lowest relation-value difference, then stable index tie-break. Its donor uses
the local-relation-first ordering. The suite records the selection block, both
terminal structural quantities, and direct copy ties for `unsupported-target`
and `target-omission` roles.

This is not oracle-assisted detection. The oracle exists only to ensure that
the constructed test input actually instantiates the dependency break it
claims to represent.

## Unchanged Evaluation

Each of six targets owns an independent B1 temporal/cross repair pair and an
independent ECRC. The runner still exports only targetwise matrices of:

```text
temporal_residual, cross_residual, disagreement
```

There is no shared trunk, target pooling, score fusion, model capacity change,
threshold change, phase input, or test-data fitting. All B2a-v1 per-target
paired, spike, normal-control, cross-skill, reliability-bin FPR, and hidden
phase diagnostic gates remain unchanged.

## Historical Execution

The development integrity smoke is seed `4201`:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/multi_evidence/run_b2a_gc_transfer.py \
  --config config/multi_evidence/b2a_gc_drift_rotation.json \
  --output-dir result/multi_evidence/b2a_gc_seed4201_gpu \
  --seed 4201 --device cuda:0 --torch-threads 1 --strict
```

Only if `4201` had passed with no code or configuration change would unchanged
confirmation seeds `4202..4205` have run. The overall smoke did not pass, so
they must remain unrun. B2c is a separately motivated calibration protocol;
it must not be introduced into or retrospectively applied to B2a-GC.
