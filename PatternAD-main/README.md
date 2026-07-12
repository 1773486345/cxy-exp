# PatternAD-main

Active model: `MechanismGraphAD`.

```text
ts_benchmark/baselines/MechanismGraphAD/  model implementation
config/mechanism_graph_ad/                benchmark configuration
scripts/mechanism_graph_ad/               single-GPU foreground runner
tests/test_mechanism_graph_ad.py          CPU contract tests
```

`ts_benchmark/baselines/PatternAD/` and
`ts_benchmark/baselines/time_series_library/` are retained only for fair
baseline comparison. Closed synthetic routes, their checkpoints, configs,
runners, tests, and research records have been removed.

Run the frozen development experiment on one GPU:

```bash
bash scripts/mechanism_graph_ad/run_real_benchmark_v1.sh 0
```

Use `RELATION_MODE=single_scale` or `RELATION_MODE=no_graph` for the two
predeclared structural ablations. The runner is foreground-only and refuses to
overwrite an existing result directory.
