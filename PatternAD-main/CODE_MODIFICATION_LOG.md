# PatternAD Code Modification Log

This file records implementation, runtime, and reproducibility details. The
research motivation, model claim, and experimental argument belong in
`../tsad_direction_A_pattern_aware_scoring.md`.

## Current Implementation (2026-07-15)

### Model

- Active implementation: `ts_benchmark/baselines/PatternAD/PatternAD.py`.
- Mechanism: causal multi-scale temporal encoding, dynamic directed
  cross-variable relation states, target-blind conditional Gaussian decoding,
  optional normal-mode mixture decoding, and conditional negative
  log-likelihood scoring.
- The number of input variables is read from each training series. Variable
  embeddings and relation layers are constructed for that series rather than
  using a fixed channel count.
- `detect_multi_fit` and `detect_multi_score` retain and validate benchmark
  text inputs. Text is not used as a scoring feature until provenance and
  time-availability are audited.
- When `normal_mode_count > 1`, the decoder learns a normal-mode memory and
  predicts a Gaussian component for each mode. The predictive mean and scale
  are the mixture moments, not a post-hoc score ensemble.
- `normal_mode_scope="system"` infers one shared mode distribution from the
  mean target-blind decoder state across variables, then applies each
  variable's decoder to that shared distribution. `mode_transition` optionally
  applies a learned causal transition prior to the mode probabilities.

### Active Candidate Configuration

- Every existing `scripts/multivariate_detection/detect_label/*/PatternAD.sh`
  fixes `normal_mode_count=4`, `normal_mode_scope="system"`, and
  `mode_transition=false` while retaining its dataset-specific capacity and
  training schedule.
- The fixed candidate uses independent per-timestamp mode probabilities; it
  does not enable the optional learned transition prior.

### Capacity And Training Policy

The presets below use only official training-window scale and channel-memory
requirements. They were selected before test-result comparison. Each full
model and its `PatternAD_raw` control share the same preset.

| Dataset group | `d_model / d_ff / e_layers / graph_dim` | `batch_size` | `num_epochs / patience` |
|---|---|---:|---:|
| Energy | `32 / 64 / 1 / 16` | 128 | 180 / 15 |
| Genesis, Weather, SKAB, Daphnet, GECCO | `48 / 96 / 1 / 24` | 256 for Genesis; 512 otherwise | 100 / 12 for Genesis; see scripts for the remaining schedules |
| PSM, BATADAL | `48 / 96 / 1 / 24` | 1024 for PSM; 512 for BATADAL | 30 / 5 for PSM; 120 / 15 for BATADAL |
| MSDS, MetroPT3 | `64 / 128 / 2 / 24` | 512 | see dataset scripts |

- PSM uses `score_conditioning_batch_size=2048`; BATADAL and the remaining
  datasets use `1024`.
- The PSM runner uses a 25-variable configuration with batch size 1024. The
  BATADAL runner uses 43 water-system variables, batch size 512, and a longer
  early-stopped schedule because its normal training segment has 8,761 points.
- BATADAL now writes and evaluates `dataset04` and `test_dataset` separately,
  each prefixed only with the same official normal `dataset03` training trace.
  This prevents sliding windows, calibration, or reported scores from crossing
  an artificial boundary between the two source evaluation traces.
- Two high-cost datasets, their source data, metadata, runners, and
  development probes were removed. This is a resource-budget decision made
  before evaluating either dataset: the current dense multi-scale graph made
  complete runs too memory- and time-intensive for the project budget.
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
- `PatternAD_raw` does not carry the four-system-mode candidate configuration;
  it remains a no-graph ablation for the original relation comparison.
- Do not claim a multimodal result from current scripts: text is validated but
  not scored.
- Do not compare strict train-calibrated Affiliation-F directly against legacy
  test-contaminated baseline reports.
