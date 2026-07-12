# Direction B3: Observable Relation-State Conditioned Cross Repair

Status: closed after the frozen-temporal GPU smoke `4401`. The comparison is
valid but failed eight strict gates, so no B3 confirmation seed or retuning is
authorized. Two earlier B3 development candidates were discarded because their
nominally unchanged temporal path was not identical to the baseline; they are
not B3 results. This was a model-level successor to the closed B2c calibration
hypothesis, not a retuning of B2c thresholds.

## B3a-4401 Execution Record

The B2a-GC baseline completed at:

```text
result/multi_evidence/b3a_baseline_seed4401_gpu/b2a_gc_evaluation.json
```

It is a valid same-seed (`4401`) control: normal streams, episodes, terminal
contract, splits, and all target-blind B2a-GC settings are retained exactly.
It failed seven gates: coherent controls for every target (`3/16`, `4/16`,
`4/16`, `3/16`, `3/16`, `3/16`), plus target-2 temporal-tail background FPR.
This baseline result is retained and must not be rerun.

The initial B3a output was not used for a comparison. Adding relation modules
changed PyTorch constructor RNG consumption before the nominally unchanged
temporal head, so its temporal residuals differed from the baseline. Its raw
directory was removed after this cause and its non-result status were recorded;
it must not be reconstructed or reported.

The implementation then constructed and discarded the B2a-GC-width driver GRU
before creating B3a-only cross modules, preserving exact same-seed temporal
GRU/head initialization. This passed the initialization and fixed-epoch CPU
tests. The resulting GPU candidate was nevertheless also discarded: although
all branch parameters are disjoint, global validation loss chose a different
best epoch for targets 1 and 3, changing the selected temporal checkpoints.
Thus it still cannot isolate the relation-conditioned cross modification.

The raw development directories were deleted after their causes were recorded.
The frozen-temporal protocol below is the resulting correction. It changes
neither B2a-GC nor B2c thresholds, generator margins, dimensions, or the
temporal model: it freezes the selected B2a-GC temporal checkpoints and trains
only the newly introduced B3 cross path.

## Frozen-Temporal Result

The sole authorized GPU smoke completed at:

```text
result/multi_evidence/b3a_frozen_temporal_seed4401_gpu/b3a_evaluation.json
```

It is a valid comparison: all frozen B2a-GC model/suite hashes matched; every
temporal GRU/head checksum was unchanged before and after B3 cross training;
and same-device background replay had exactly zero difference for `target`,
`mu_temporal`, and `temporal_residual`. The terminal information and parameter
isolation, synthetic contract, counterfactual ties, capacity control, all 24
dependency-break paired ordering families, target-spike, and cross-normal-skill
gates passed. Its strict status is nevertheless `failed_gates` (`64/72`).

The eight failed gates are target coherent controls 0/1/2/4/5 at
`3/16`, `4/16`, `6/16`, `3/16`, and `3/16` exceedances respectively (limit
`2/16`); target-2 temporal-residual background reliability-bin FPR
`10.46%` (limit `10%`); and disagreement reliability-bin FPR gaps of `5.46%`
for target 1 and `5.05%` for target 3 (limit `5%`). Thus relation-history
conditioning preserves the certified dependency-break signal but does not
solve the normal-control/FPR limitation; it worsens the target-2 coherent
control relative to the valid target-blind `4401` control. B3 is closed.

## Motivation

B2a-GC made each terminal dependency break valid at the generator level. Its
`4201` result showed that the target-blind cross repair can detect a certified
relation break, but it did not stabilize normal coherent and continuous-drift
controls. B2c then changed only outer-normal calibration. Its frozen `4301`
smoke also failed: despite target/component/stratum thresholds and the
predeclared `0.025 + 0.025` cross/disagreement budget, it retained coherent
false alerts, a target-3 disagreement ordering miss, and a target-5
disagreement reliability-bin gap.

The joint evidence is therefore not consistent with a cutoff-only remedy. In
the B2a-GC process, the cross relation is a continuously varying, target-
specific conditional relation. The existing cross head sees only driver
channels. It cannot use the observed pre-terminal coevolution of the target
and drivers to identify the current relation state; a target-blind predictor
must treat distinct target-driver compatibility states as the same driver-only
input family.

B3 tests the following falsifiable claim:

```text
For a changing multivariate relation, cross repair should condition on an
observable target-driver relation history, while remaining blind to the target
terminal value being repaired.
```

## B3a Model

For target `i` at terminal time `t`, B3 retains the independent temporal
repair path unchanged:

```text
mu_T(t, i) = T_i(x_i[t-H : t-1])
```

Its cross path has two disjoint learned modules:

```text
q_i(t)    = Q_i([x_i[t-H : t-1], x_-i[t-H : t-1]])
h_i(t)    = C_i(x_-i[t-H : t])
mu_C(t,i) = Head_i([q_i(t), h_i(t)])
```

`Q_i` is a relation-history GRU and `C_i` is the existing driver GRU. The
history representation may use the target only before the repaired terminal.
The driver path may use all non-target channels through the terminal, as in
B1/B2. `Q_i`, `C_i`, and the temporal path do not share parameters; each
target owns an independent copy. The exported residuals remain exactly:

```text
R_T[t, i], R_C[t, i], D[t, i]
```

There is no score fusion, shared trunk, target pooling, graph input, hidden
phase/factor input, structural-support input, learned threshold, or test-data
fitting.

The B3a configuration retains the B2a-GC temporal width (`32`) and fixes its
two cross encoders/head to widths `22/22/20`. This gives each B3a target cross
branch `4,815` parameters, below the B2a-GC target-blind cross branch's
`4,833`. The same-seed baseline and B3a use the same normal generator, splits,
Adam optimizer, learning rate, early stopping, and deterministic Torch batch
ordering; B3a's relation-history input is the only model-side change.

## Information Contract

Because B3 deliberately permits target *history* in the cross path, B2a's
former `coherent -> target-omission` cross-prediction equality is no longer a
valid information tie. It is replaced by two stricter and relevant checks:

1. `coherent` and `unsupported-target` retain identical target histories, so
   the temporal residual must tie exactly.
2. `coherent` and `target-spike` differ only in the target terminal, so B3's
   cross prediction must tie exactly. The relation-history tensor itself is
   unit-tested to be unchanged under that terminal perturbation and to change
   when an allowed target-history point changes.

The target terminal is never supplied to `Q_i`, `C_i`, their heads, the
reliability feature, a reference tail, or an outer threshold. The B2a-GC
generator's hidden relation phase, latent factors, structural support, donor
indices, and labels remain audit-only metadata.

## Frozen B3a Protocol

- Generator: unchanged B2a-GC terminal donor contract.
- Calibration: unchanged B1 targetwise global-after-stratified ECRC at
  `alpha=0.05`, to isolate the model change. B2c's family-wise calibration is
  closed and is not combined with B3a.
- Gates: retain all B2a-GC paired ordering, coherent/drift, target-spike,
  cross-skill, and FPR thresholds. Replace only the invalid B2a cross-input
  tie with the B3 terminal-blind cross tie above.
- Comparison: the B3 runner loads the retained `4401` B2a-GC control rather
  than regenerating it. It verifies the recorded SHA-256 hashes for the model
  state, normal streams, episodes, and suite manifest before doing work.
- Checkpoint isolation: for every target it loads and freezes only
  `temporal_gru` and `temporal_head` from the retained B2a-GC state. Those
  tensors are excluded from the optimizer; early stopping selects the cross
  path using cross validation loss only. The output records per-target source
  and final temporal tensor hashes and must replay the control's temporal
  background outputs on the same CUDA device.

The baseline is not a post-hoc ablation: it is a predeclared same-seed control.
B3a `4401` failed, so no confirmation seed is authorized. Do not tune
dimensions, epochs, losses, thresholds, or donor margins on this result.
Direction B has exhausted B2a-GC's calibration-only and relation-history
model-level remedies under the frozen continuous-drift transfer protocol.

## Execution Status

The complete CPU multi-evidence regression suite passed `24/24` before the
GPU run. The retained B3 GPU artifact passed the checkpoint-isolation and
same-device replay gates, but failed its frozen performance gates. Do not run
the command again, do not run another seed, and do not alter this protocol
after the observed result.
