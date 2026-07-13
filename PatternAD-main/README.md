# PatternAD-main

Active model: `PatternAD`.

```text
ts_benchmark/baselines/PatternAD/  model implementation
config/unfixed_detect_label_multi_config.json  strict evaluation protocol
scripts/multivariate_detection/detect_label/   per-dataset runners
tests/test_patternad_core.py                    CPU contract tests
```

`ts_benchmark/baselines/time_series_library/` is retained for fair baseline
comparison. Closed synthetic routes, their checkpoints, configs, runners,
tests, and research records have been removed.

Run a dataset experiment on one GPU from the repository root:

```bash
bash scripts/multivariate_detection/detect_label/GECCO_script/PatternAD.sh
```

Every runner is foreground-only, sequential, and uses `cuda:0`. `PatternAD.sh`
uses the full multi-scale relation mechanism. The existing `PatternAD_raw.sh`
files are the matched `relation_mode=no_graph` controls. The benchmark text
argument is retained and validated, but is not treated as an auxiliary signal
until its provenance has been audited.

Training progress is printed to the terminal: every ten batches (and the final
batch of each epoch) it reports `iters`, `epoch`, seconds per iteration, and
estimated remaining time. Each epoch reports training/validation loss and their
changes, graph entropy, elapsed time, and early stopping state. Scoring progress
is printed in roughly 10% increments. No extra log file or background process is
created.

The runners use the following default `batch_size` / inference
`score_conditioning_batch_size` pairs for a roughly 30 GB single-GPU budget:

- HAI21: `256 / 512`
- Daphnet, Energy, GECCO, Genesis, MSDS, MetroPT3, SKAB, SMD, Weather: `512 / 1024`

The current implementation is a research candidate, not a claimed performance
improvement. Shell syntax checks and 18 CPU contract tests pass; no real-data
GPU result has been recorded yet.
