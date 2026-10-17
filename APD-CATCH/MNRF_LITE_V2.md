# MNRF-Lite V2

Independent two-stage model at `ts_benchmark/baselines/mnrf_lite_v2`. The
implementation does not change original CATCH, TypeFusion v1/v2, their scripts,
or the fixed research design. No performance conclusion follows from smoke tests.

## Architecture

Input values use per-variable statistics fitted on the internal train subset.
A shared point embedding represents value, valid adjacent difference,
observation mask, and difference mask. The shared TCN has six residual blocks
with dilations `[1,2,4,8,16,32]`, width 32, two depthwise-separable kernel-3
layers per block, left-only padding, per-time LayerNorm, and GELU. Layers have
no residual; each block computes `x + layer2(layer1(x))` once. Its receptive
field is 253 points.

The four views are `[S,E,P,R]` with 25 features per variable: current state,
causal evolution, 32-point pattern, and cross-variable relation. E excludes the
target value. R constructs source tokens independently and uses identity-only
target queries; a batched MHA call blocks the diagonal, so no target's own
current/history source enters its relation representation. Features condition
four alternating attention-conditioned affine coupling blocks. Mechanisms A
and B are `(S,R)` and `(E,P)`, updated B/A/B/A. The only detector score is the
four-fixed-noise mean Gaussian-base joint NLL with log-determinant correction.

## Training

One task command performs the complete protocol. The dataset's official
train/test boundary is kept. The training side is internally purged by `L-1`
points at each split boundary, then divided 80/10/10 into train/validation/
calibration. Each split is intersected with continuous record segments.
Windows must fit wholly inside one segment and one split; the first `L-1`
points of every new segment are cold-start invalid.

Timestamp indexes are checked for parseability, order, duplicates, and `NaT`.
The normal cadence is fitted only from the internally selected training portion
of the official training split, then frozen for validation, calibration, test,
preflight, cache identity, and loaded-model scoring. Integer-multiple intervals
within 5% of that cadence split a sequence; unexplained intervals fail rather
than being reinterpreted from a test segment. Aware timestamps are normalized to
absolute UTC nanoseconds for comparisons and fingerprints; naive timestamps use
a timezone-independent naive-nanosecond protocol. Explicit
`entity_id`, `session_id`, `record_id`, or `scenario_id` changes also split the
sequence. The implementation never sorts, interpolates, or resamples. If only
row indices exist, metadata records the row-order protocol and does not invent
time information.

DatetimeIndex, numpy datetime64 arrays, and explicit timestamp columns are
canonicalized to nanoseconds before cadence inference, identity hashing,
segmentation, and saved score coordinates. Training freezes the exact set of
entity/session boundary columns; scoring rejects missing or additional
boundary definitions rather than silently changing the segmentation protocol.

Stage 1 samples E/P/R auxiliary prediction tasks and uses an auxiliary Gaussian
head only for training. P masks eight consecutive raw values and validity flags
before recomputing differences. Validation masking is deterministic from stable
window end indices. The best encoder and auxiliary state are saved; the encoder
is then frozen. Stage 2 trains only the four affine coupling blocks. Validation
NLL supplies per-variable median/MAD scales; an independent calibration split
sets the fixed 0.995 score quantile. Test labels do not select parameters or the
threshold. Default settings: seed 2021, `L=128`, pattern length 32, width 32,
view dimension 8, train batch 64, stage limits 50/30 epochs, patience 5,
AdamW lr 0.001, weight decay 0.0001, feature scale floor 0.05, and training
noise std 0.01.

## Memory, Progress, and Recovery

Input statistics and feature statistics are accumulated in chunks. The
standardized training sequence and frozen feature views are `.npy` memmaps.
Stage 2, validation, calibration, and test scoring standardize and transfer
only a current batch; features are never materialized as another full ndarray
or placed wholesale on the GPU. `CACHE_CHUNK_SIZE` controls disk I/O grouping
only. `ENCODING_BATCH_SIZE` controls encoder GPU batch and is capped by
`BATCH_SIZE`; by default it equals the training batch. Exact calibration
medians/MADs use per-variable reads from disk-backed NLL arrays.

Each epoch prints and flushes stage, epoch, optimizer steps, train NLL,
validation NLL, best value, patience, and elapsed seconds to stdout and
`logs/training.jsonl`. Cache extraction reports bounded progress. Atomic
checkpoints are under `checkpoints/`: `best_encoder.pt`, `best_stage1.pt`,
`stage1_latest.pt`, `best_fusion.pt`, and `stage2_latest.pt`. A normal launch
creates an independent `run-<UTC>-<PID>` directory. Recovery is explicit via
`MNRF_RESUME_FROM` and is epoch-boundary only. Checkpoints record whether a
stage is in progress or completed, its completion reason (`early_stopping` or
`epoch_limit`), best epoch/loss, next epoch, patience, optimizer, RNG, and
sampler state. A completed stage only reloads its selected best state; normal
resume never revives an early-stopped stage. Stage 2 is bound to the selected
frozen encoder, input/feature statistics, feature order, window-end indices,
data identity, and frozen temporal rule. Missing stage-2 cache files may be
rebuilt only with that same encoder/statistics and accepted only if their
content hashes match the saved binding. Incompatible or legacy checkpoints are
rejected without overwriting run artifacts; start a new run for a new protocol.
The time/cache/deployment schema is versioned as
`mnrf-lite-v2-tcn6b-2l-r1-segment-v3-timefreeze`. The deployment package
excludes the auxiliary head. Caches, statistics, manifests, aligned scores,
metrics and the benchmark report remain inside the run directory. The run
validator fails on missing, malformed, cross-task, or inconsistent artifacts.

## Scoring and Task Status

Each stride-one complete window contributes a score only at its endpoint.
Invalid cold-start points receive no score; there is no padding, repeated score,
or future-window backfill. Saved `time_indices` are full-series positions and
`local_time_indices` address the test-label vector; both masks and label
selection are checked against these indices. The aligned result stores the
full-series valid mask, test-local mask, labels, predictions, scores, threshold,
and timestamps. Optional per-variable test NLL diagnostics are off by default.
A one-class valid test interval keeps its undefined-metric reason and is not
reported as a complete evaluation.

When all three feature caches are present and match the stage-2 binding, resume
reuses them directly and restores their verified hashes into the final manifest
without re-encoding. A missing or damaged cache is rebuilt only when the same
frozen encoder/statistics reproduce the bound cache and index hashes; otherwise
resume fails before replacing the existing manifest. A legacy time coordinate
is never rescaled based only on its field name.

Every benchmark invocation gets a fresh identity in benchmark_context.json.
After the report is written, run_benchmark atomically records its run-local
path, SHA-256, model column, task/run identity, invocation ID, and AUC values in
report_reference.json. Historical test_report CSVs remain in place. Validation
checks the referenced report hash and metrics against aligned scores, not report
count or modification time. An unreferenced report from a crash between report
writing and reference update is not accepted; retrying creates a new
report/reference pair. A legacy run without an invocation ID is accepted only
when exactly one report is unambiguous.

The 23 internal and 18 external wrappers are candidate task entries, not an
assertion that all 41 remain supported after temporal segmentation. Run
preflight per task: it uses the same internal long-table or external wide-table
source parser as benchmark loading, reads only that task, and prints official boundary,
segments, cadence/gaps, post-purge train/validation/calibration window counts,
and test windows. External parsing removes timestamp/label using the benchmark
source semantics and never sends the wide table through the internal long-table
reader. Empty splits fail. A validation or calibration count of one
is called out as too small for stable scale/high-quantile estimation. No task
is silently kept supported by shortening `L` or dropping purge.

Read-only server preflight at the default `L=128` confirmed usable windows for
PSM (`D=25`) and BATADAL (`D=43`). Other candidate tasks remain unverified in
this round and need their own preflight before a run.

PSM's internal source currently converts its integer date ordinal (1, 2, ...)
to a naive datetime upstream, so preflight reports a 1 ns cadence. Window
availability is confirmed, but this is not a claim about PSM's physical
sampling period; cadence-aware gap interpretation for that source remains
limited by the upstream loader representation.

```bash
conda run --no-capture-output -n catch_env python scripts/mnrf_lite_v2_preflight.py --task PSM --data-source large_detect
```

The dedicated `mnrf_valid_detect` strategy owns exact score/label alignment.
For the first real single task, after reviewing preflight:

```bash
GPU_ID=0 conda run --no-capture-output -n catch_env bash ./scripts/multivariate_detection/detect_score/PSM_script/MNRFLiteV2.sh
```

The launcher defaults to the first GPU already visible as `cuda:0`; it preserves
an existing `CUDA_VISIBLE_DEVICES` restriction and prints the final device and
mask. Use `GPU_ID=cpu` for explicit CPU mode. Shared-GPU memory overrides are
manual, for example `BATCH_SIZE=16 ENCODING_BATCH_SIZE=4`; no automatic OOM
retry or batch adjustment exists. `MNRF_SAVE_PATH` sets a new relative path
under `result/`. To resume a verified epoch-boundary run:

```bash
GPU_ID=0 MNRF_RESUME_FROM=score/MNRF-Lite-V2/PSM/run-<UTC>-<PID> conda run --no-capture-output -n catch_env bash ./scripts/multivariate_detection/detect_score/PSM_script/MNRFLiteV2.sh
```

No formal task training or benchmark was launched during implementation.
Commands for candidate tasks are in [MNRF_LITE_V2_COMMANDS.md](MNRF_LITE_V2_COMMANDS.md).
