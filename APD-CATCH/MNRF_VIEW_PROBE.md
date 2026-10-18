# MNRF View Probe

`MNRFViewProbe` is an independent frozen-feature ablation experiment. It does
not modify `MNRFLiteV2`, its encoder, losses, Flow, score aggregation, or the
existing launchers. The probe imports the existing `FourViewEncoder` and
stage-one `MNRFTrainer` implementation, then writes a separate
`result/score/MNRF-ViewProbe/...` run directory.

Artifacts use schema `mnrf-view-probe-v2`. Each run is bound to the task, raw
features, normalized time coordinates, entity boundaries, train split, purge
ranges, continuous segments, endpoint indices, canonical S/E/P/R layout,
encoder/input-statistics fingerprint, and dependency versions. A changed
source, split, variable order, layout, or time gap is rejected before an
existing artifact can be written. Stage-one, head-fitting, complete, and
failed states are recorded explicitly.

## Protocol

Stage one fits the existing E/P/R self-supervised tasks, saves the best encoder
and the initial random encoder state, and exports read-only S/E/P/R feature
caches with full-series endpoint indices. A GMM head fits one
`sklearn.mixture.GaussianMixture` per original variable. It uses negative
`score_samples` as NLL, bounded deterministic training-window sampling, feature
statistics fitted only on the training cache, validation median/MAD scaling,
and the independent calibration 0.995 quantile. The default head is
`K=4`, `covariance_type=full`, `reg_covar=0.001`, `n_init=1`, `max_iter=100`,
seed 2021, at most 20,000 fitting windows and one CPU thread.

The first round is explicitly **frozen feature deletion ablation**. Selecting
`S`, `S/E/P/R`, `E/P/R`, `S/P/R`, `S/E/R`, or `S/E/P` selects exact indices from
the saved canonical layout; it does not delete architecture modules or invent a
new layout with zero-width branches. Every selected view gets a new GMM,
feature statistics, calibration and result directory. Head parameter counts
and selected dimensions are recorded. A random comparison uses the saved
pre-training encoder initialization and an independent cache/fingerprint; it
never labels the trained encoder as random.

`flow` is an optional full-layout control: it reuses a validated source
encoder/cache and trains the existing Flow head in a separate probe directory.
It is not used for arbitrary deleted-view layouts.

GMM and Flow use the same four deterministic feature-noise vectors. GMM fits
one full-covariance head per original variable and averages negative
`score_samples` over those perturbations; Flow uses its existing NLL head with
the same wrapper. GMM variables are atomically written under
`heads/gmm_variables/` with `gmm_progress.json`, so explicit recovery can skip
verified variables. Stage-one and Flow checkpoints include optimizer, RNG,
sampler, and best-state information at epoch boundaries.

## Artifacts and safety

Each run contains `probe_manifest.json`, `input_statistics.npz`, encoder
checkpoints, read-only-source references, `cache/`, `heads/`, and
`results/metrics.json` plus `aligned_scores.npz`. A source directory is never
modified. Layout, variable order, selected views and source identity are
checked before reuse. A missing or incompatible source artifact fails rather
than defaulting to 1/8/8. The current model's strict FeatureLayout and time
window protocols remain in force.

This round does not make a performance claim. Paper CATCH values and any
CATCH-original result are references only; they are not substituted for probe
results. AUC-ROC/AUC-PR are measured only on the valid endpoint intersection
written in each run.

## Common comparison

`scripts/evaluate_mnrf_view_probe_common.py` is a read-only adapter for an
existing CATCH score artifact and a probe score artifact. It computes the
explicit endpoint intersection, coverage and excluded anomaly points, then
uses the repository affiliation implementation on the original coordinate
axis without joining gaps or entities. It does not train CATCH or silently
re-split data.

The evaluator calls a score pair an `index_intersection_reference` by default.
Predeclared split files alone do not prove that either model trained and
calibrated on those coordinates.  The strict `CATCH-common` name is granted
only when `scripts/register_mnrf_catch_common.py` has registered a completed
common-split training/score/calibration evidence manifest; pass that manifest
with `--common-manifest` together with all three split files.  The adapter
verifies sorted, disjoint coordinates and records their hashes instead of
creating an internal split.  Paper/original CATCH scores and hand-entered
thresholds cannot create this protocol identity.

Validate a completed run with:

```bash
conda run --no-capture-output -n catch_env python scripts/validate_mnrf_view_probe.py <RUN_DIR>
```

The validator checks complete stage/head state, hashes, finite statistics,
calibration, fixed noise, aligned score arrays, and reloads a Flow encoder and
head before scoring the validated source test cache.

### Recovery and time identity

ViewProbe writes `run_request.json` and its immutable source identity before
stage updates. Stage-one epoch checkpoints and per-variable GMM progress are
validated read-only before a resume can write status, caches, or progress
files. A damaged completed GMM variable is rejected before any target file is
changed. Flow accepts either an independent stage-one source or the same frozen
encoder/cache contained in a completed GMM artifact.

`IndexedViewProbeStrategy` preserves timestamp indices. Timestamp gaps and
entity/session boundaries are hard window boundaries; row-order data without a
real timestamp is explicitly recorded as the row-order protocol. It never
resets a real time index or pads scores across an invalid interval.

For an explicit CATCH-common run, use
`scripts/run_mnrf_catch_common.py`. It trains CATCH on supplied train
endpoints, validates on supplied validation endpoints, calibrates only on the
supplied calibration endpoints, and writes checkpoint/calibration evidence.
The adapter preserves CATCH's pointwise score contract: the last point of each
explicit window is scored as pointwise time-domain MSE plus
`score_lambda * frequency_criterion` (default `score_lambda=0.05`, inference
patch size 32 and stride 1). The scaler is fitted on the explicit
`--train-rows` raw training-row file, never on a window-endpoint union or on
validation, calibration, or test rows. Test windows are filtered per window
when a timestamp gap or any entity/session boundary is encountered; invalid
cold-start intervals remain invalid rather than being silently bridged. Split
windows must be mutually disjoint and cannot cross a timestamp or any declared
boundary column. The optimizer and mask-generator schedule follow the original
CATCH adapter: mask gradients use its delayed update interval and OneCycle
learning rates are adjusted at epoch boundaries. This explicit-split adapter is
named `CATCH-common`, not presented as the benchmark's internal split.
The registry refuses empty evidence, missing checkpoints, missing calibration
scores, cross-run evidence, or hand-edited thresholds. Supplying
`--common-manifest` to the evaluator is strict and exits nonzero when evidence
is invalid; use `--reference-mode` only when an explicit
`index_intersection_reference` result is intended.

Strict registration also reconstructs the recorded CATCH model and loads its
checkpoint with `strict=True`. The checkpoint, normalization statistics,
scaler-fit row file, calibration score array/endpoints, serialized threshold,
and score output are cross-bound. Probe and CATCH may have different run IDs,
but their raw feature values, variable order, canonical time values, boundary
columns, official train/test boundary, and frozen split identity must agree.
The scaler's raw fit rows are the exact Probe internal training range, not the
union of rows covered by training windows; this retains valid short-entity rows
that cannot form a window. CATCH windows are materialized lazily per DataLoader
batch for train, validation, calibration, and test. The I/O/cache chunk size
does not alter GPU forward batch size.

Calibration thresholds are generated and verified with the saved score array's
dtype and NumPy linear-quantile rule. The evaluator uses the external evidence
object only after checking its embedded copy is identical, and verifies the
calibration endpoint array against the frozen calibration split. A report is
strict CATCH-common only after these checks; it is not equivalent to a paper
reference or an index-intersection-only comparison.
