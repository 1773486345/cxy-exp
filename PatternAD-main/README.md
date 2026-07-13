# PatternAD-main

Active model: `PatternAD`.

```text
ts_benchmark/baselines/PatternAD/  model implementation
config/patternad/                  benchmark configuration
scripts/patternad/                 single-GPU foreground runner
tests/test_patternad.py            CPU contract tests
```

`ts_benchmark/baselines/time_series_library/` is retained for fair baseline
comparison. Closed synthetic routes, their checkpoints, configs, runners,
tests, and research records have been removed.

Run the frozen development experiment on one GPU:

```bash
bash scripts/patternad/run_real_benchmark_v1.sh 0
```

Use `RELATION_MODE=single_scale` or `RELATION_MODE=no_graph` for the two
predeclared structural ablations. The runner is foreground-only and refuses to
overwrite an existing result directory.
