# PatternAD Research Workspace

## Active Direction

The active research path is Direction B1: **Multi-Evidence Consistency Repair
with Evidence-Conditioned Reliability Calibration (ECRC)**. It is implemented
independently from the archived PatternAD-A backbone so its temporal and
cross-variable evidence paths can be audited for information leakage.

```text
temporal repair: target history only
cross repair:    all non-target channels, including synchronous terminal values

exported components: temporal residual, cross residual, disagreement
deployment fusion:   intentionally absent in B1
```

The B1 synthetic five-seed confirmation passed every frozen gate. See
[B1_EXPERIMENT_PLAN.md](B1_EXPERIMENT_PLAN.md) for the exact contract,
calibration, gates, result locations, and the boundary before real-data work.

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
