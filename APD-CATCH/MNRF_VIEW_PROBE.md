# MNRF View Probe

`MNRFViewProbe` is an independent frozen-feature ablation experiment. It does
not modify `MNRFLiteV2`, its encoder, losses, Flow, score aggregation, or the
existing launchers. The probe imports the existing `FourViewEncoder` and
stage-one `MNRFTrainer` implementation, then writes a separate
`result/score/MNRF-ViewProbe/...` run directory.

## Prepared task entries

The five manually runnable entries are
`PSM_script/MNRFViewProbe.sh`, `MSL_script/MNRFViewProbe.sh`,
`GECCO_script/MNRFViewProbe.sh`, `SMD_script/MNRFViewProbe.sh`, and
`Genesis_script/MNRFViewProbe.sh` under
`scripts/multivariate_detection/detect_score/`. They are self-contained,
resolve the repository root from their own location, run the read-only
preflight first, and then invoke the MNRF CLI with explicit parameters. They
expect the caller to run `conda activate catch_env` once; they do not activate
an environment or alter `CUDA_VISIBLE_DEVICES`. Their default device is the
inherited logical `cuda:0`, and each run gets a new timestamp/PID output
directory. See `MNRF_VIEW_PROBE_COMMANDS.md` for the five independent
commands. These entries are prepared, not executed, and do not claim a
benchmark result.

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

Each run writes a flushed `run.log` containing startup configuration, the
read-only preflight result, stage/cache/GMM/calibration/scoring progress and
any traceback. The full artifact policy (the generic API default) additionally
contains `training.jsonl`, `run_request.json`, `input_statistics.npz`, encoder
checkpoints, source references, `cache/`, `heads/`, and the result files. A
source directory is never modified. Layout, variable order, selected views
and source identity are checked before reuse. A missing or incompatible source
artifact fails rather than defaulting to 1/8/8. The current model's strict
FeatureLayout and time window protocols remain in force.

The five prepared task scripts intentionally use `artifact_policy=results`.
After scores and metrics have been validated, the run performs an atomic
retention transaction and keeps only `run.log`, optional `training.jsonl`,
`probe_manifest.json`, `run_status.json`, `results/metrics.json`, and
`results/aligned_scores.npz`. Its manifest declares that model loading,
training resume, source reuse and strict common recomputation are unavailable;
`validate_mnrf_view_probe.py` and `summarize_mnrf_view_probe.py` can still
validate/recompute the saved result metrics. Use the generic CLI or set
`ARTIFACT_POLICY=full` for a reusable full run. An interrupted cleanup can be
finished without training with:

```bash
python scripts/organize_mnrf_view_probe.py /absolute/path/to/run
```

The organizer validates the result hashes before removing recovery material.
`load_probe`, source reuse and resume reject a results-only bundle explicitly.

The live log is attached before the first post-preflight output chunk; that
chunk is copied once and every later stdout/stderr write is flushed to the
same `run.log`. The shell preflight records both the Python and `tee` exit
statuses, so a logging failure cannot start training. Flow and GMM result
metrics both contain `valid_points`, `excluded_points`, and `coverage`; the
validator checks these as finite, non-negative integer/range values without
truncating malformed decimals. Predictions must be exact binary values (with
`-1` only at invalid aligned positions).

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

Flow recovery has three explicit outcomes. An in-progress `stage2_latest.pt`
is restored from its next epoch; a checkpoint marked complete only performs the
remaining calibration/score finalization and takes no new optimizer steps; an
invalid checkpoint is rejected before any run file is created or changed. The
preflight strictly loads the Flow and best-state keys/shapes/dtypes, validates
AdamW parameter groups and moment shapes, and loads Python, NumPy, torch and
available CUDA RNG states into private objects. It preserves the caller's RNG
state while constructing the validation model, so a rejected check cannot
change the subsequent run.

Flow manifests distinguish the raw input scaler from the encoded-feature
scaler. `input_statistics.npz` is bound by `raw_input_statistics_sha256`,
while `heads/feature_statistics.npz` is bound by
`feature_statistics_sha256`. Older Flow artifacts from the release that wrote
the feature hash under `input_statistics_sha256` are accepted only when that
legacy value exactly matches the feature-statistics file and an independent
raw-input binding is also present and valid; the loader never re-authenticates
an altered file by writing a new digest. A completed Flow checkpoint that was
interrupted after scoring therefore resumes only calibration/result
finalization, with zero additional optimizer steps. A validated result bundle
with cleanup failure is handled by `organize_mnrf_view_probe.py` only.

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
Older common manifests without the production score receipt remain read-only
references; regenerate a new independent common run before claiming strict
CATCH-common identity.

Strict registration also reconstructs the recorded CATCH model and loads its
checkpoint with `strict=True`. The checkpoint, normalization statistics,
scaler-fit row file, calibration score array/endpoints, serialized threshold,
and score output are cross-bound. The CATCH adapter writes a production score
receipt only after the stable checkpoint, calibration and test NPZ exist;
registration and evaluation require that receipt and reject a replaced,
cross-run, or hand-completed score file. Probe and CATCH may have different run IDs,
but their raw feature values, variable order, canonical time values, boundary
columns, official train/test boundary, and frozen split identity must agree.
The scaler's raw fit rows are the exact Probe internal training range, not the
union of rows covered by training windows; this retains valid short-entity rows
that cannot form a window. CATCH windows are materialized lazily per DataLoader
batch for train, validation, calibration, and test. The I/O/cache chunk size
does not alter GPU forward batch size.

The common exporter keeps the Probe context and endpoint files unchanged.  It
can derive a separate CATCH endpoint set with `--catch-seq-len` (the formal
default is 192) from the same frozen raw ranges and continuous segments.  A
split that is too short for that context is rejected before training.  The
versioned v2 contract records the Probe manifest/source identity, canonical
time kind, every joint boundary column and value identity, frozen temporal
rule, raw split ranges, continuous segments, raw training rows, labels, and
the exact CATCH endpoint hashes.  The CATCH runner consumes
`--common-contract` and refuses a wrong length, a hand-selected/subsampled
endpoint file, or a self-consistent but incorrectly derived contract.
Different model window lengths are therefore a declared protocol property,
not an endpoint rewrite.  Timestamped CATCH runs consume the Probe-frozen
training cadence from this contract; cadence is never re-estimated from
overlapping windows or later test entities, so short entities can contribute
to raw training identity even when they cannot form a window.

Time identity is canonical absolute time for `timestamp_utc_ns`: a CSV round
trip may change a display/source timezone name (for example,
`Asia/Shanghai` to `UTC`) when the UTC nanosecond sequence is unchanged. The
runner accepts that representation change only when the time kind, length and
canonical hash still match. Naive timestamps and row-order coordinates retain
strict metadata semantics; a real shift, gap, reorder, or aware/naive change is
rejected.

Calibration thresholds are generated and verified with the saved score array's
dtype and NumPy linear-quantile rule. The evaluator uses the external evidence
object only after checking its embedded copy is identical, and verifies the
calibration endpoint array against the frozen calibration split. A report is
strict CATCH-common only after these checks; it is not equivalent to a paper
reference or an index-intersection-only comparison.

### Runtime logging and result retention

The five task scripts capture their read-only preflight into the same live
`run.log` as the Python run. Once the run directory has been created, the log
is flushed incrementally and includes stage/cache/GMM/calibration/scoring
progress, stderr, warnings, and tracebacks; a rejected resume keeps its
preflight log outside the target run without modifying that run. SIGINT and
SIGTERM return nonzero and preserve the recovery tree.

Stage-one and Flow stage-two now also emit intra-epoch events at real completed
optimizer-step boundaries. Each `*_batch` event records `epoch`,
`epoch_batch`, `total_batches`, cumulative `completed_steps`/
`optimizer_steps`, the already-computed train NLL, seconds per step, and an
epoch ETA. The first and last batch and either the step or elapsed-time
interval are logged without extra forward/backward work. Epoch-start events and
the checkpoint's next epoch remain authoritative after recovery; a failed
optimizer step does not produce a completion event.

`artifact_policy=results` describes the requested final retention policy. An
interrupted Stage 1, GMM, or Flow run remains resumable while its recovery
artifacts are present. Only a validated, committed `result_bundle` is terminal
for model loading or training resume. Result coordinates, time values, labels,
valid masks, counts, coverage, and AUCs are cross-checked against the frozen
source identity before cleanup. If deletion of disposable training artifacts
fails after that validation, the run is reported as
`results_valid_cleanup_failed`; the scores remain usable and
`organize_mnrf_view_probe.py` retries the same retention transaction without
retraining or rewriting metrics.

GMM progress uses a versioned numerical identity that excludes logging,
storage, retention, and load-device fields while retaining model, sample,
statistics, noise, cache, endpoint, and source bindings. Progress written by
the previous release is accepted only after its persisted configuration is
reconstructed and all immutable bindings match, then migrated atomically.
