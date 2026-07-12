# PatternAD Research Workspace

## Current Research State

The active research definition is **Direction A2: dynamic-structure-aware
event and transition semantics**. A2 asks whether a post-boundary trajectory
is compatible with an event-pre observable state; it does not preselect a
residual, model backbone, graph module, or modality. Its synthetic development
task and
falsification contract are in
[research/direction_a/A2_EXPERIMENT_PLAN.md](research/direction_a/A2_EXPERIMENT_PLAN.md).
The A2 generator, audit, and separate model families live under
[`config/a2/`](config/a2/) and [`scripts/a2/`](scripts/a2/). The first
conditional-mixture-density family (M1) failed its frozen four-seed
confirmation; the current development candidate is the separately implemented
contrastive compatibility-energy family (M2). M2's initial A2-v1 contract was
not seed-stable; A2-v2 fixed that defect, but M2-v2 passed only `2/4` complete
confirmation gates because normal-score calibration was unstable. M2 is closed
as an A2 detector route. Neither route has a real-data claim.

Direction B1 remains the active executable controlled-mechanism baseline.
B2c and B3a are closed after their respective frozen smokes failed strict
transfer gates. B3a, **Observable Relation-History Conditioned Cross Repair**,
was implemented independently from the historical PatternAD-A backbone so its
temporal and cross-variable evidence paths could be audited for leakage.

```text
temporal repair: target history only
cross repair:    terminal-blind target/driver history + all non-target channels
                 including synchronous terminal values

exported components: temporal residual, cross residual, disagreement
deployment fusion:   intentionally absent in B1
```

The B1 synthetic five-seed confirmation passed every frozen gate. B2a-GC
validated the terminal dependency-break signal (`24/24` paired gates), but
failed normal-control/FPR gates; B2c's calibration-only remedy and B3a's
relation-history model remedy also failed. See
[research/direction_b/closed/B3_EXPERIMENT_PLAN.md](research/direction_b/closed/B3_EXPERIMENT_PLAN.md) for the final comparable B3
negative evidence.

## B3a Status

The same-seed B2a-GC `4401` control and B3a frozen-temporal `4401` comparison
are retained. B3a froze all temporal checkpoints, verified every control hash,
and replayed temporal outputs with exact same-device equality, yet failed
`8/72` strict gates. The remaining failures are normal coherent-control/FPR
instability, not a checkpoint confound. Do not run B2c confirmation seeds
`4302..4305`, another B3 seed, or a B3 retune.

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
[`research/direction_b/B_RESULT_INDEX.md`](research/direction_b/B_RESULT_INDEX.md)
is the immutable index for retained B artifacts and their allowed interpretation.

## A-v1 Archive

A-v1, **Pattern-Aware Residual Semantics**, is closed. Its conditional
residual scalar could not satisfy the matched quiet/volatile and
abrupt/gradual mechanism gates simultaneously. This does not close Direction
A's broader dynamic-structure motivation. The compact negative-evidence and
source archive is at [archive/direction_a/README.md](archive/direction_a/README.md).
The former PatternAD-A model and old dataset shell entry points are historical
artifacts, not A2 experiment commands.

## Repository Data

Existing benchmark data remains under `dataset/anomaly_detect/data`; it has not
yet been used to make a B1 performance claim. Any B2 real-data study must use
a new frozen protocol and must not choose component weights, bins, or
thresholds on test labels.
