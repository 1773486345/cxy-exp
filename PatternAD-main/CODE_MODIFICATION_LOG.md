# PatternAD Code Modification Log

This file records implementation, runtime, and reproducibility details. The
research motivation, model claim, and experimental argument belong in
`../tsad_direction_A_pattern_aware_scoring.md`.

## Current Implementation (2026-07-13)

### Model

- Active implementation: `ts_benchmark/baselines/PatternAD/PatternAD.py`.
- Mechanism: causal multi-scale temporal encoding, dynamic directed
  cross-variable relation states, target-blind conditional Gaussian decoding,
  and conditional negative log-likelihood scoring.
- The number of input variables is read from each training series. Variable
  embeddings and relation layers are constructed for that series rather than
  using a fixed channel count.
- `detect_multi_fit` and `detect_multi_score` retain and validate benchmark
  text inputs. Text is not used as a scoring feature until provenance and
  time-availability are audited.

### Capacity And Training Policy

The presets below use only official training-window scale and channel-memory
requirements. They were selected before test-result comparison. Each full
model and its `PatternAD_raw` control share the same preset.

| Dataset group | `d_model / d_ff / e_layers / graph_dim` | `batch_size` | `num_epochs / patience` |
|---|---|---:|---:|
| Energy | `32 / 64 / 1 / 16` | 128 | 180 / 15 |
| Genesis, Weather, SKAB, Daphnet, GECCO | `48 / 96 / 1 / 24` | 256 for Genesis; 512 otherwise | 100 / 12 for Genesis; see scripts for the remaining schedules |
| MSDS, SMD, MetroPT3, HAI21 | `64 / 128 / 2 / 24` | 256 for HAI21; 512 otherwise | see dataset scripts |

- Inference uses `score_conditioning_batch_size=512` for HAI21 and `1024`
  for the remaining datasets.
- HAI21 and SMD have short development scripts (`PatternAD.sh` and
  `PatternAD_raw.sh`) and full-coverage scripts (`*_full.sh`).
- Exact per-dataset epoch and patience settings live in the existing scripts,
  which remain the executable source of truth.

### Evaluation

- Config: `config/unfixed_detect_label_multi_config.json`.
- Protocol: `train_calibration`, with an anomaly-free official training split
  and a disjoint calibration segment for labels.
- Report metrics: AUC-PR, VUS-PR, AUC-ROC, VUS-ROC, F-score, and
  Affiliation-F.
- The accelerated VUS implementation is retained in
  `ts_benchmark/evaluation/metrics/vus_metrics.py`; VUS-ROC and VUS-PR share
  cached curve computation in `classification_metrics_label.py`.

### Runtime Behavior

- Runners execute foreground-only and sequentially. They use `cuda:0` inside
  the selected visible-GPU environment.
- Training prints an iteration progress line every ten batches and at each
  epoch end: iteration, epoch, seconds per iteration, estimated remaining
  time, training loss, validation loss, loss deltas, graph entropy, and early
  stopping status.
- Scoring prints progress in roughly ten-percent increments.

### Validation

- `tests.test_patternad_core` and `tests.test_anomaly_protocol`: 18 CPU
  contract tests pass.
- All restored `PatternAD*.sh` runners pass `bash -n`.
- No real-data GPU result is recorded in this workspace.

## Boundaries

- `PatternAD_raw` is the matched `relation_mode=no_graph` structural control,
  not a legacy handcrafted residual scorer.
- Do not claim a multimodal result from current scripts: text is validated but
  not scored.
- Do not compare strict train-calibrated Affiliation-F directly against legacy
  test-contaminated baseline reports.
