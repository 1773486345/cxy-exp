# Direction B2c-FW-ECRC Plan

Status: closed after the frozen GPU development smoke `4301` on 2026-07-12.
B2c was a calibration innovation motivated by the completed B2a-GC `4201`
result. It retained B2a-GC's valid terminal counterfactual generator and B1's
independent repair heads exactly.

## Recorded Outcome

The complete artifact is retained at:

```text
result/multi_evidence/b2c_fw_ecrc_seed4301_gpu/b2c_evaluation.json
```

The run completed on `cuda:0`, wrote all model/suite/score artifacts, and left
no B2c runner process. Its terminal contract, information/parameter isolation,
reliability-routing isolation, counterfactual ties, and all cross normal-skill
gates passed. Overall, `64/70` frozen gates passed; the strict status is
`failed_gates`.

The six failed gates were target-0 coherent control (`3/16`, allowed `2`),
target-0 target spike (`10/16`, required `14`), target-2 target spike
(`13/16`), target-3 unsupported-break disagreement ordering (`13/16`,
required `14`), target-4 coherent control (`3/16`), and target-5 disagreement
reliability-bin FPR gap (`0.05825`, limit `0.05`). No confirmation seeds ran.

A no-retraining CPU diagnostic restored the saved `4301` checkpoint and suite,
then fitted B1's global outer ECRC on the same normal splits. It also failed
(`58` pass / `8` fail performance gates), at:

```text
result/multi_evidence/b2c_fw_ecrc_seed4301_gpu/analysis/b2c_global_ecrc_same_model.json
```

The replay's saved-versus-recomputed raw-score maximum differences were at most
`0.00195` because this environment could not expose CUDA to PyTorch after the
original run. Treat it as supporting diagnostic evidence, not an exact
same-device causal comparison. The direct `4301` failure is sufficient to
close the frozen B2c protocol: do not tune it or run `4302..4305`. The next
separately motivated proposal is B3 in `B3_EXPERIMENT_PLAN.md`.

## Motivation

In B1 ECRC, each component has a reliability-stratified reference tail but all
strata return to one global outer-normal cutoff at alpha `0.05`. B2a-GC showed
that this is insufficient under continuous relation drift: all 24 dependency
break ordering gates passed under a valid terminal contract, but normal
coherent/drift controls and target-1 FPR stability failed.

The operational normal-control rule is `cross OR disagreement`. Calibrating
both components independently at `0.05` does not control that family. B2c
therefore introduces **Family-Wise Evidence-Conditioned Reliability
Calibration (FW-ECRC)**:

```text
temporal residual:     alpha_T = 0.050
cross residual:        alpha_C = 0.025
disagreement:          alpha_D = 0.025
threshold scope:       target x component x reliability stratum
threshold data:        outer normal only
```

`alpha_C + alpha_D = 0.05` is a pre-declared Bonferroni allocation for the
cross/disagreement alert family. It is component-wise stratum calibration with
a family-wise budget, not a claim of exact joint-stratum FWER. Each outer
stratum requires at least 50 normal samples; there is no global-threshold
fallback and no merging of targets or strata.

## Immutable Boundary

B2c does not add a branch, shared encoder, graph, lag module, score fusion,
target pooling, learned uncertainty, phase input, or label-driven choice. It
uses the same six independent B1 temporal/cross repair pairs and same three
exported matrices:

```text
R_T[t, i], R_C[t, i], D[t, i]
```

The B2a-GC terminal donor contract remains generator-only. Its phase, factor,
structural support, and source/donor diagnostics do not enter model input,
normalization, reliability features, tails, or thresholds.

## Frozen Gates

Every B2a-GC gate remains unchanged for every target: exact information and
counterfactual ties, terminal synthetic contract, both dependency-break tail
orderings, coherent and drift controls, target spikes, cross normal skill, and
per-bin/per-phase FPR checks. A missing reference or outer stratum is an
invalid run, not a pass. No target average may rescue a failed gate.

## Historical Execution

Development smoke, to be run once:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/multi_evidence/run_b2c_fw_ecrc.py \
  --config config/multi_evidence/b2c_fw_ecrc_drift_rotation.json \
  --output-dir result/multi_evidence/b2c_fw_ecrc_seed4301_gpu \
  --seed 4301 --device cuda:0 --torch-threads 1 --strict
```

The runner emitted `b2c_evaluation.json` when `--strict` returned exit code 2.
The overall smoke did not pass, so seeds `4302..4305` must remain unrun.
