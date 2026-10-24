# MNRF View Probe

`MNRFViewProbe` is an independent frozen-feature ablation experiment. It does
not modify `MNRFLiteV2`, its encoder, losses, Flow, score aggregation, or the
existing launchers. The probe imports the existing `FourViewEncoder` and
stage-one `MNRFTrainer` implementation, then writes a separate
`result/score/MNRF-ViewProbe/...` run directory.

## Prepared task entries

The eight manually runnable entries are
`PSM_script/MNRFViewProbe.sh`, `MSL_script/MNRFViewProbe.sh`,
`GECCO_script/MNRFViewProbe.sh`, `Genesis_script/MNRFViewProbe.sh`,
`NYC_script/MNRFViewProbe.sh`, `CalIt2_script/MNRFViewProbe.sh`,
`Creditcard_script/MNRFViewProbe.sh`, and `CICIDS_script/MNRFViewProbe.sh` under
`scripts/multivariate_detection/detect_score/`. They are self-contained,
resolve the repository root from their own location, run the read-only
preflight first, and then invoke the MNRF CLI with explicit parameters. They
expect the caller to run `conda activate catch_env` once; they do not activate
an environment or alter `CUDA_VISIBLE_DEVICES`. Their default device is the
inherited logical `cuda:0`, and each run gets a new timestamp/PID output
directory. See `MNRF_VIEW_PROBE_COMMANDS.md` for the eight independent
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

For a new source-backed Flow run, the probe deliberately preserves the
historical 411e91a constructor/RNG sequence: after the formal seed it creates
the discarded pre-source trainer and then the trainer used for stage2, before
restoring the validated source encoder and input statistics. This keeps Flow
initialization and subsequent training-noise draws comparable to that release;
resume runs instead restore the checkpoint's saved weights and RNG without an
extra construction.

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

The eight prepared task scripts intentionally use `artifact_policy=full`.
After validation, the source keeps checkpoints, complete split caches, GMM
heads, calibration and score evidence for later read-only reuse. A separate
bounded analysis directory can be generated without training with:

```bash
python scripts/export_mnrf_view_probe_analysis.py --run-dir /absolute/path/to/run --output-dir /absolute/path/to/analysis
```

The exporter performs the full source validation once internally; no boolean
or command-line flag can bypass it. It refuses source/target overlap and only
allows `--overwrite` for a recognized analysis package bound to that same
source. Replacement is staged so an export failure preserves the old package.
The source includes the complete raw test label/time/global-coordinate axis
and training-time code, script, data and environment provenance. Labels are
accepted only when one-dimensional, finite, and exactly 0/1; integer, unsigned,
float (including exact `0.0`/`1.0`), and Boolean arrays are explicitly accepted
and stored as `int8`. Fractional, near-binary, non-finite, object, complex, and
wrong-shape labels fail before model training. The CLI writes the immutable
training receipt and startup-script snapshot after read-only resume checks and
run-directory preparation, before the first optimizer step.

Complete-axis continuous-segment IDs are canonical. Test-axis IDs are sliced
from those full-axis IDs, so a gap before the official test boundary does not
renumber a later segment. The complete source validator also opens
`cache/normalized_values.npy` read-only as a memmap, checks its frozen shape,
`float32` dtype and finite values by chunks, and hashes the actual `.npy` file
bytes against the authoritative manifest binding. A missing or stale digest is
not repaired automatically; restore a trusted source copy before reuse.
The same check runs inside the Python source-loader used by `select_gmm`,
source-backed `s_only`, and `flow`, as well as their resume preflight. It runs
before a derived output directory or status is written; a damaged source
cannot produce a derived run marked complete. This check does not require the
source's optional analysis-input package.
After this read-only load, the validated encoder state and input statistics
are explicitly restored into the trainer used for the derived fit. Flow
`stage2_identity` is computed from that frozen source fingerprint; a checkpoint
whose manifest and payload agree with each other but bind a different/random
encoder is rejected before any target write.

If writing auxiliary inputs fails after training, the model remains complete.
With the original receipt present, supplement against the same parsed local
data and export without training. For an older completed full source whose
support write failed before any receipt was saved, use the explicit
`--recover-legacy` mode. Its training-time Git, script, versions, and original
raw-file byte identity remain `unknown`; recovery records only the currently
reparsed loader identity and the scored endpoint label agreement. It does not
claim those current facts were captured during training. The explicit
`--migrate-segment-ids` option only accepts the recognized old local-test-ID
renumbering signature, preserves old auxiliary evidence, and records a
migration. Core hashes must validate before either operation. A normal export
remains read-only and never supplements or migrates its source.

Auxiliary evidence writes use a private uniquely-created transaction directory.
Only files published by that transaction are rolled back. Existing similarly
named `.tmp` files, directories, and symlinks are never globbed, traversed, or
removed. If rollback itself fails, the error names both failures and retains
the transaction backup for manual recovery.

The old SMD entry remains a results-only historical artifact because its
prior budget was terminated; it is not restarted by this round. `load_probe`,
source reuse and resume require full artifacts for new runs.

Feature overview variance is measured over windows for each variable-feature
pair before aggregation by S/E/P/R. Cross-variable mean differences do not
count as temporal variation. Diagnostic sampling is bounded and deterministic;
test samples stratify available TP/FP/TN/FN and rare anomaly points. These
summaries do not establish causal branch contribution. The exporter reports
logical disk bytes for source and analysis; this is not peak memory or a
compressed transfer-size estimate. Preflight reports retained-disk components
and target-volume free space without imposing an arbitrary free-space gate.

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

The eight task scripts capture their read-only preflight into the same live
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

The summary treats `run_status.json` as a typed contract. Its `status` must be
a known probe state (`running`, `stage1_training`, `stage1_complete`,
`gmm_fitting`, `flow_training`, `organizing_results`, `complete`, `failed`, or
`cleanup_failed`). A malformed root, missing field, boolean/number/list value,
or unknown string produces one `invalid` row for that run with a file/field
reason; it cannot stop other runs from being summarized. Legitimate incomplete
states remain `not_applicable`, while a validated result with cleanup failure
remains `results_valid_cleanup_failed`. Stale metrics in a failed run are not
reported as a valid score. CSV and Markdown output errors remain fatal output
errors rather than being swallowed by the per-run isolation.

Resume preflight snapshots and restores Python, NumPy, PyTorch CPU, and the
available CUDA RNG state around the entire read-only validation call. This
includes temporary source encoders and optimizers constructed while checking
source/cache/checkpoint bindings. Both rejected and successful preflights leave
the caller's RNG unchanged; the subsequent real fit still follows its normal
seed and checkpoint protocol.

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

## Frozen-view branch ablations

Branch ablations select columns from the saved 25-dimensional S/E/P/R cache;
they do not remove network branches or retrain an encoder. The nine combinations
are S, SE, SP, SR, FULL, EPR (without S), SPR (without E), SER (without P), and
SEP (without R), with widths 1/9/9/9/25/24/17/17/17. Each combination fits an
independent CPU GMM and calibration on the source's fixed endpoints. Positive
deletion means that removing the branch hurt the metric; negative deletion is
a head/fusion diagnostic, not proof that the representation is useless.

The read-only summary validates the pinned Full source and every candidate again
with the native validator. It requires the exact Full scoring axes, masks,
labels, times, and segment identities, and reproduces each candidate score from
its saved test NLL and calibration center/scale. Missing, failed, blocked, or
invalid cells remain explicit and make the summary non-complete; no score or
point bundle is retained from a failed verification. The current ablation
matrix is fixed to head seed 2021 (72 positions across eight datasets); paired
reports separate added/lost TP, added/removed FP, and gained/lost anomaly
segments.

## Frozen-cache recovery

Missing split caches in the four pinned Full sources can first be searched in
scoped project result/source/backup/review roots, including ZIP and TAR
members. Archive contents are streamed rather than extracted; candidates must
match the source manifest byte SHA256 and expected float32 shape. Installation
uses an atomic no-replace link. A concurrently-created equal target is reused;
a different target or dangling symlink is preserved and reported as conflict.

The saved best encoder may be used for a bounded frozen forward only when an
exact backup is absent. It runs in evaluation/no-gradient mode and installs a
cache only if its complete `.npy` SHA256 matches the immutable source manifest.
It never invokes Stage 1, backward, an optimizer, GMM fitting, Flow, or
recalibration. Full success additionally requires all four split caches and
endpoints, all native full-GMM artifacts, successful `validate_mnrf_view_probe`
validation, and the same read-only source loader used by `select_gmm`. A
split-limited repair or missing validation/calibration/statistics/checkpoint/
GMM/calibration/result artifact stays partial/blocked and exits nonzero.

After a prior forward hash mismatch, a bounded diagnostic is available:
`python scripts/recover_mnrf_view_probe_cache.py --diagnose-forward --task PSM
--task Creditcard --task CICIDS --device cuda:0`. It replays the beginning,
middle, last full batch and true tail of cached validation/calibration splits,
reports per-view exact equality/error counts, runtime precision/thread state,
and `.npy` header details. Unknown historical runtime settings remain unknown;
the diagnostic does not invent a cause or compare unavailable historical
train/test bytes. Reports live under
`result/analysis/MNRF-ViewProbe-Ablation/cache-recovery-<UTC>-<PID>/`.
