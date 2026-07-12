# Direction B runners

`run_b1_reliability.py` is the active Direction B entry point. It trains two
isolated GRU repair paths on a clean normal split, calibrates each exported
component with input-restricted normal reliability bins, and writes all inputs,
model state, scores, thresholds, and gate decisions below a new result
directory.

```bash
python scripts/multi_evidence/run_b1_reliability.py \
  --config config/multi_evidence/b1_reliability.json \
  --output-dir result/multi_evidence/b1_new_seed \
  --seed 3106 \
  --device cuda:0 \
  --torch-threads 1 \
  --strict
```

The B1 result has no fused anomaly score. Its only exported decision components
are `temporal_residual`, `cross_residual`, and `disagreement`; the fixed gates
in `b1_evaluation.json` decide whether a later B phase is justified.

`run_b0.py` and `b0_synthetic.json` remain only as the documented failed
global-tail diagnostic that motivated B1's reliability calibration. Do not use
them as the active experiment command.

`run_b2c_fw_ecrc.py` is a closed calibration-only transfer experiment. It reused B2a-GC's
generator-only terminal contract and B1's isolated heads, but fits a distinct
outer-normal conformal cutoff for every target/component/reliability stratum.
Cross and disagreement use the pre-declared `0.025 + 0.025` Bonferroni budget;
temporal remains at `0.05`. Its completed `4301` smoke failed; do not run
confirmation seeds. See `B2C_EXPERIMENT_PLAN.md` for the retained artifact and
`B3_EXPERIMENT_PLAN.md` for the active model-level direction.

`run_b3_relation_conditioned.py` is the frozen B3a checkpoint-isolation smoke.
It reads the retained B2a-GC `4401` control, verifies hashes of its model and
inputs, freezes each selected temporal GRU/head, and trains only the
relation-conditioned cross path. It writes per-target temporal hashes and a
same-device temporal-output replay check. The sole authorized command and
stop rule are in `B3_EXPERIMENT_PLAN.md`; do not change its seed, control, or
output identity.
