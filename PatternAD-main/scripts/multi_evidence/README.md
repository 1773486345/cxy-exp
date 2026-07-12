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
