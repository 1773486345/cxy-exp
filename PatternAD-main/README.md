# PatternAD Research Workspace

## Active Direction

Direction B2c is closed after its frozen `4301` smoke failed six strict gates.
The active design is Direction B3a: **Observable Relation-History Conditioned
Cross Repair** on top of the independently audited B1 Multi-Evidence
Consistency Repair heads. It is implemented independently from the archived
PatternAD-A backbone so temporal and cross-variable evidence paths can be
audited for information leakage.

```text
temporal repair: target history only
cross repair:    terminal-blind target/driver history + all non-target channels
                 including synchronous terminal values

exported components: temporal residual, cross residual, disagreement
deployment fusion:   intentionally absent in B1
```

The B1 synthetic five-seed confirmation passed every frozen gate. B2a-GC
validated the terminal dependency-break signal (`24/24` paired gates), but
failed normal-control/FPR gates; B2c's calibration-only remedy also failed.
See [B2C_EXPERIMENT_PLAN.md](B2C_EXPERIMENT_PLAN.md) for retained negative
evidence and [B3_EXPERIMENT_PLAN.md](B3_EXPERIMENT_PLAN.md) for the frozen
next model-level test.

## B3a Status

The same-seed B2a-GC `4401` control is complete and retained. Both prior B3a
development candidates were discarded because their selected temporal
checkpoints did not match that baseline; they are not research results. The
frozen replacement loads those temporal checkpoints from the control, excludes
them from optimization, and hashes both inputs and final temporal tensors. The
single authorized `4401` GPU smoke is documented in
`B3_EXPERIMENT_PLAN.md`; do not run B2c confirmation seeds `4302..4305` or a
second B3 seed.

## Run B1

Use the existing project environment:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/multi_evidence/run_b1_reliability.py \
  --config config/multi_evidence/b1_reliability.json \
  --output-dir result/multi_evidence/b1_ecrc_new_seed \
  --seed 3106 \
  --device cuda:0 \
  --torch-threads 1 \
  --strict
```

The runner uses one foreground process and one GPU only. It refuses to
overwrite an existing output directory, writes the model state, generated
suite, normalizer, tails, thresholds, per-episode scores, and machine-readable
gate decisions. A failed gate still produces a complete result directory.

The five completed confirmation runs are:

```text
result/multi_evidence/b1_ecrc_seed3101_gpu/
result/multi_evidence/b1_ecrc_seed3102_gpu/
result/multi_evidence/b1_ecrc_seed3103_gpu/
result/multi_evidence/b1_ecrc_seed3104_gpu/
result/multi_evidence/b1_ecrc_seed3105_gpu/
result/multi_evidence/b1_ecrc_summary_3101_3105/
```

Run the focused tests with:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  -m unittest discover -s tests -p 'test_multi_evidence_b0.py' -v
```

## Active Files

```text
ts_benchmark/baselines/MultiEvidenceRepair/MultiEvidenceRepair.py
scripts/multi_evidence/generate_b0_synthetic.py
scripts/multi_evidence/reliability_calibration.py
scripts/multi_evidence/run_b1_reliability.py
scripts/multi_evidence/summarize_b1.py
config/multi_evidence/b1_reliability.json
```

`tsad_direction_B_multi_evidence_repair.md` contains the research motivation;
`scripts/multi_evidence/README.md` contains the runner-level usage note.

## Archived Direction A

Direction A, Pattern-Aware Residual Semantics, is closed. Its conditional
residual scalar could not satisfy the matched quiet/volatile and
abrupt/gradual mechanism gates simultaneously. The compact reproducibility
archive, including final negative evidence and source snapshot, is at
[archive/direction_a/README.md](archive/direction_a/README.md). The former
PatternAD-A model and old dataset shell entry points are historical artifacts,
not current experiment commands.

## Repository Data

Existing benchmark data remains under `dataset/anomaly_detect/data`; it has not
yet been used to make a B1 performance claim. Any B2 real-data study must use
a new frozen protocol and must not choose component weights, bins, or
thresholds on test labels.
