# PatternAD-main

Active model: `PatternAD`.

```text
ts_benchmark/baselines/PatternAD/  model implementation
config/unfixed_detect_label_multi_config.json  strict evaluation protocol
scripts/multivariate_detection/detect_label/   per-dataset runners
tests/test_patternad_core.py                    CPU contract tests
CODE_MODIFICATION_LOG.md                        implementation and run configuration record
```

`ts_benchmark/baselines/time_series_library/` is retained for fair baseline
comparison. Closed synthetic routes, their checkpoints, configs, runners,
tests, and research records have been removed.

Run a dataset experiment on one GPU from the repository root:

```bash
bash scripts/multivariate_detection/detect_label/GECCO_script/PatternAD.sh
```

Every runner is foreground-only, sequential, and uses `cuda:0`. `PatternAD.sh`
uses the full multi-scale relation mechanism and the fixed four-mode candidate:
`normal_mode_count=4`, `normal_mode_scope="system"`, and
`mode_transition=false`. The shared system mode is inferred from target-blind
states across variables, while the conditional Gaussian decoder remains
variable-specific. The existing `PatternAD_raw.sh` files are the matched
`relation_mode=no_graph` controls and do not use this fixed candidate
configuration. The benchmark text argument is retained and validated, but is
not treated as an auxiliary signal until its provenance has been audited.

Training progress is printed to the terminal: every ten batches (and the final
batch of each epoch) it reports `iters`, `epoch`, seconds per iteration, and
estimated remaining time. Each epoch reports training/validation loss and their
changes, graph entropy, elapsed time, and early stopping state. Scoring progress
is printed in roughly 10% increments. No extra log file or background process is
created.

The model reads the actual feature count from each training series and builds
its variable embedding and dynamic relation graph for that count. Capacity
presets are selected from official training-window scale before testing; the
main mechanism and evaluation protocol are unchanged across datasets.

The runners use the following default `batch_size` / inference
`score_conditioning_batch_size` pairs for a roughly 30 GB single-GPU budget:

- PSM: `1024 / 2048`
- BATADAL: `512 / 1024`
- Energy: `128 / 1024`
- Genesis: `256 / 1024`
- Daphnet, GECCO, MSDS, MetroPT3, SKAB, Weather: `512 / 1024`

The full-run `num_epochs / patience` pairs are: PSM and MetroPT3 `30 / 5`;
MSDS `40 / 7`; Daphnet and GECCO `60 / 8`; SKAB and Weather `80 / 10`;
Genesis `100 / 12`; BATADAL `120 / 15`; Energy `180 / 15`.

PSM and BATADAL are prepared by `scripts/prepare_psm_batadal.py`. PSM uses
its public normal-training/test-label split; BATADAL uses normal-only
`dataset03` for training and runs `dataset04` and `test_dataset` as two
independent evaluation traces. Two previously registered high-cost datasets are intentionally not
included because the current dense multi-scale graph exceeds the project
runtime and memory budget on them.

Capacity presets are documented in `CODE_MODIFICATION_LOG.md`. They are shared
between each `PatternAD` runner and its matched `PatternAD_raw` ablation;
normal-mode configuration differs as described above.

The current implementation is a research candidate, not a claimed performance
improvement. Shell syntax checks and 18 CPU contract tests pass; no real-data
GPU result has been recorded yet.
