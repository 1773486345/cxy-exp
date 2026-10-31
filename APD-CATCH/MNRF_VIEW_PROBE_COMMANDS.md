# MNRF ViewProbe commands

These eight independent task entries are prepared only; they have not been
run as formal experiments. Activate the existing global environment once,
then launch each task manually from the repository root. The scripts do not
activate Conda, alter `CUDA_VISIBLE_DEVICES`, search for another Python, or
retry after an error.

```bash
cd /media/h3c/users/wangyueyang1/cxy/APD-CATCH
conda activate catch_env
```

The first command runs a read-only preflight and then, only if it succeeds,
starts a new `full_gmm` run with `artifact_policy=full`. Each output directory
contains a UTC timestamp and PID, so existing source runs are not overwritten.
The default device is
the inherited logical `cuda:0`; edit the literal `DEVICE="cuda:0"` near the
top of one script to `cpu` when a CPU run is intended.

## Frozen-view branch ablation

Use `python scripts/run_mnrf_branch_ablation.py --all --mode plan` for a
read-only 8x9 matrix (72 positions, fixed `head_seed=2021`). The nine
combinations are S, SE, SP, SR, FULL, WITHOUT_S, WITHOUT_E, WITHOUT_P and
WITHOUT_R (widths 1/9/9/9/25/24/17/17/17). There is no repeat/multi-seed
entry in the current experiment. Missing cells stay `planned/pending`, while
blocked and conflicting cells remain visible.

After reviewing the plan, start one dataset at a time with its independent
entry, for example:

```bash
bash scripts/multivariate_detection/detect_score/PSM_script/MNRFBranchAblation.sh
```

The other seven entries use the same filename under MSL, GECCO, Genesis, NYC,
CalIt2, Creditcard and CICIDS. Derived heads reference the pinned Full cache;
they do not copy an encoder or cache and add no GPU lock or scheduler. Summarize
a completed plan read-only with:

```bash
python scripts/summarize_mnrf_branch_ablation.py \
  --plan /absolute/path/to/result/analysis/MNRF-ViewProbe-Ablation/<batch>/plan.json
```

The summary independently rechecks scores, labels, confusion counts, full-axis
anomaly segments and paired contribution deltas. For E/P/R, addition is
`metric(S+branch)-metric(S)` and deletion is
`metric(FULL)-metric(FULL-minus-branch)`; a positive deletion means removal
hurt. S has deletion only. This evaluates frozen representations and heads; it
is not causal proof, paper-protocol equivalence, or a claim over CATCH.

Each summary invocation revalidates the pinned Full source and each candidate
with the native read-only validator, requires exact equality of candidate and
Full scoring axes/masks, and recomputes scores from saved per-variable NLL plus
that head's calibration statistics. Invalid cells are excluded from metrics
and point bundles. `summary.json` reports expected/present cells, fully
complete datasets, and complete/reused/pending/running/failed/blocked/conflict/
invalid counts; failed, blocked, conflicting, or invalid plans exit nonzero.
All summaries are fixed-seed summaries for head seed 2021. Paired changes distinguish added/lost TP, added/removed FP, and
gained/lost anomaly-segment hits, with bounded examples in `change_cases.json`.

The eight entries use `artifact_policy=full` literally. A successful run keeps
the encoder checkpoints, complete split caches, GMM heads, calibration,
per-variable NLL and score evidence for later read-only reuse. The log can be
followed with `tail -f <RUN_DIR>/run.log` while a task runs. A separate
read-only analysis export is written under
`result/analysis/MNRF-ViewProbe/<TASK>/<RUN_ID>`; an export failure leaves the
complete source intact and returns nonzero.
If the preflight pipeline reports separate Python and `tee` statuses, training
has not started; retain the printed temporary log path for diagnosis.
The old SMD entry is intentionally excluded from this list: its prior budget
was terminated and its historical results-only artifacts are not restarted or
modified. The eight new/full entries retain recovery artifacts directly.
Any earlier PSM/MSL/GECCO/Genesis `results`-policy run has already discarded
its reusable cache/checkpoints and cannot be upgraded in place; start one of
these new unique full runs instead.

Flow recovery keeps raw and encoded-feature statistics separate:
`input_statistics.npz`/`raw_input_statistics_sha256` describe the input
standardization, and `heads/feature_statistics.npz`/
`feature_statistics_sha256` describe the Flow feature standardization. A
legacy Flow manifest using `input_statistics_sha256` for the latter is accepted
only with an independently verified raw-input hash. A completed checkpoint
resumed after scoring performs no new optimizer steps; it only finishes
calibration, validation, and result organization.

Training logs are flushed at epoch starts and at completed optimizer-step
boundaries. Stage-one and Flow stage-two batch events include the epoch and
batch position, total and cumulative optimizer steps, NLL, speed, and ETA.
The configured step/time intervals control observation frequency only and do
not alter numerical training or recovery state.

The read-only summary validates `run_status.json` before touching any run
branch. Invalid status roots or values are retained as one `invalid` row with
their precise reason; valid runs in the same root still appear in the CSV and
Markdown. `failed` and other unfinished states are shown as not applicable,
not as valid scores, while `cleanup_failed` keeps the already validated result
metrics. Summary input trees are never rewritten. Resume preflight likewise
restores the caller's random states even when source or statistics validation
rejects a run, so no manual reseeding is required.

## Task commands

```bash
bash ./scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
bash ./scripts/multivariate_detection/detect_score/MSL_script/MNRFViewProbe.sh
bash ./scripts/multivariate_detection/detect_score/GECCO_script/MNRFViewProbe.sh
bash ./scripts/multivariate_detection/detect_score/Genesis_script/MNRFViewProbe.sh
bash ./scripts/multivariate_detection/detect_score/NYC_script/MNRFViewProbe.sh
bash ./scripts/multivariate_detection/detect_score/CalIt2_script/MNRFViewProbe.sh
bash ./scripts/multivariate_detection/detect_score/Creditcard_script/MNRFViewProbe.sh
bash ./scripts/multivariate_detection/detect_score/CICIDS_script/MNRFViewProbe.sh
```

Run these manually as independent commands. They are not a loop, runner,
scheduler, or background launcher. Tasks may run concurrently when shared
resources allow; the scripts impose no global serialization or GPU-availability
gate. SMD remains paused and is not included.
The task scripts keep the following CATCH reference window and batch values,
but the MNRF training/head settings are independent:

| task | CATCH reference seq_len | CATCH reference batch | MNRF stage1/stage2 | MNRF batch | MNRF views |
| --- | ---: | ---: | ---: | ---: | --- |
| PSM | 192 | 128 | 50/30 | 128 | S,E,P,R |
| MSL | 192 | 128 | 50/30 | 128 | S,E,P,R |
| GECCO | 96 | 128 | 50/30 | 128 | S,E,P,R |
| Genesis | 192 | 4 | 50/30 | 4 | S,E,P,R |
| NYC | 192 | 128 | 50/30 | 128 | S,E,P,R |
| CalIt2 | 192 | 128 | 50/30 | 128 | S,E,P,R |
| Creditcard | 192 | 128 | 50/30 | 128 | S,E,P,R |
| CICIDS | 192 | 128 | 50/30 | 128 | S,E,P,R |

All eight scripts explicitly fix seed `2021`, dimensions `1/8/8/8`, pattern
length `32`, hidden width `32`, Adam learning rate `0.001`, weight decay
`0.0001`, GMM `K=4/full/reg_covar=0.001/n_init=1/max_iter=100`, at most
`20000` fit windows, noise standard deviation `0.01` with seed `20261013`,
feature scale floor `0.05`, score MAD floor `0.001`, calibration quantile
`0.995`, CPU threads `1`, and cache I/O chunk `256`. Encoding batch equals
the task batch; the cache chunk is not a GPU forward batch.

To use a different batch or CPU, edit the corresponding literal at the top of
that task script and record the change as a new manual run. There is no
automatic OOM reduction or retry. The older
`scripts/run_mnrf_view_probe_task.sh` remains an advanced environment-driven
entry for explicit recovery/source workflows; it is not used by these eight
independent commands.

Read-only utilities after a manual run:

```bash
python scripts/mnrf_lite_v2_preflight.py --task PSM --data-source large_detect --batch-size 128 --seq-len 192 --encoding-batch-size 128 --view-dim 8 --evolution-dim 8 --pattern-dim 8 --relation-dim 8
python scripts/validate_mnrf_view_probe.py /absolute/path/to/run
python scripts/export_mnrf_view_probe_analysis.py --run-dir /absolute/path/to/run --output-dir /absolute/path/to/analysis
python scripts/summarize_mnrf_view_probe.py --root result/score/MNRF-ViewProbe
```

The exporter runs the full read-only validator exactly once as part of export;
there is no validation-bypass flag. It consumes only a completed, self-contained
full source; it never retrains, refits a GMM, recalibrates, or removes source
files. `--overwrite` only replaces a recognized completed analysis package
bound to the same source, and stages the replacement before commit. It rejects
source/analysis path overlap, symlink targets, and unknown directories. The local metadata has
one row and a CSV for each of the eight tasks; the read-only preflight remains
the authority for actual window counts and disk estimates. CalIt2's validation
and calibration counts are reported there and may be too small for stable MAD
and 0.995-quantile estimates. The approximate feature payload figures are
planning values, not measured RAM or VRAM usage.

The first two commands above are read-only checks. Formal task runs are not
part of this repository change and no benchmark result is implied by preparing
these entries.

Preflight reports retained-disk estimates for the float32 feature caches,
normalized values, endpoint indices, NLL, checkpoints/head/logs and bounded
analysis payload, together with free space on the target volume. Those values
are not peak RAM/VRAM. For the protocol-only synthetic layout of 5,040 rows,
train boundary 2,520 and window 192, the purged counts are 1,519/22/24/2,329
for train/validation/calibration/test. This is a formula check, not live-data
statistics; the small validation/calibration counts limit robust MAD and
0.995-quantile estimation.

If auxiliary full-axis/provenance writing fails after training, the completed
model source is retained. For a new run with its training receipt, supplement
the verified source and export without rerunning Stage 1 or GMM:

```bash
python scripts/supplement_mnrf_view_probe_analysis_inputs.py --run-dir /absolute/path/to/full-run --output-dir /absolute/path/to/analysis
```

For a source whose auxiliary inputs already exist, export alone:

```bash
python scripts/export_mnrf_view_probe_analysis.py --run-dir /absolute/path/to/full-run --output-dir /absolute/path/to/new-analysis
```

Use `--overwrite` only for a recognized analysis package from the same source.
An older completed `full` run that has no receipt because its historical CLI
failed during the post-training auxiliary write can be recovered explicitly
from the original local dataset. The recovery records current loader/data
identity separately and leaves training-time provenance fields unknown; it
does not claim the current checkout or file digest is training-time evidence:

```bash
python scripts/supplement_mnrf_view_probe_analysis_inputs.py \
  --run-dir /absolute/path/to/completed-full-run \
  --output-dir /absolute/path/to/analysis \
  --recover-legacy
```

To migrate only the previously released bug where test segments were numbered
from zero independently of the full axis, use the explicit migration mode:

```bash
python scripts/supplement_mnrf_view_probe_analysis_inputs.py \
  --run-dir /absolute/path/to/completed-full-run \
  --output-dir /absolute/path/to/analysis \
  --migrate-segment-ids
```

Both write modes first validate the completed core, frozen source/data identity,
and scored endpoint labels; neither trains, refits, re-encodes or recalibrates.
Normal export remains read-only. A changed `normalized_values.npy` is rejected
using its actual `.npy` byte digest and cannot be re-bound by these commands.
Old `results`-policy runs cannot regain discarded encoders or caches.
The `select_gmm`, source-backed `s_only`, and `flow` Python entry points apply
the same raw-cache, statistics, variable-order and digest checks before
creating their derived run. Resume performs this source check during its
read-only preflight. A rejected source or resume target remains unchanged.
For Flow, the preflight also binds the saved stage-2 checkpoint and manifest
to the validated source encoder fingerprint. The fit restores that exact
encoder/statistics state after the historical two-construction initialization
sequence for a new source-backed Flow run, so head initialization and the
following training RNG stream remain compatible with 411e91a. Resume runs use
the checkpoint's saved weights and RNG directly and do not repeat that
sequence.

Auxiliary writes stage under an exclusive per-operation directory and only
publish the known manifest/provenance/analysis files. Rollback never globs
`.tmp` names in the run tree, so pre-existing temporary evidence, ordinary
directories and symlinks are preserved. If restoration itself fails, the
error reports the recovery directory instead of claiming success.

## Frozen-cache recovery

The former PSM/Creditcard/CICIDS cache-recovery campaign is closed. Do not run
another recovery/forward audit for those old Run IDs; their mismatched missing
caches will not be used. The recovery utility remains available for other
explicitly authorized operations, but these three tasks now use new Full runs.

After each new Full run and analysis export succeed, the task script performs a
separate source-registration step. That step reruns native validation and the
actual read-only `select_gmm` source loader before recording the exact output
path, manifest digest, source identity, feature fingerprint, cache and endpoint
bindings under `result/analysis/MNRF-ViewProbe-Ablation/current_sources/`.
Planning, execution, and summary all consume this same explicit per-task
registration. No source is chosen by latest timestamp or score. Until a new
source is registered, PSM, Creditcard, and CICIDS remain blocked in ablation
plans. Only after registration is atomically written does the helper remove
that task's one fixed retired residual Run ID; it refuses symlink paths and
never removes the newly registered run.

```bash
bash scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
bash scripts/multivariate_detection/detect_score/Creditcard_script/MNRFViewProbe.sh
bash scripts/multivariate_detection/detect_score/CICIDS_script/MNRFViewProbe.sh
```

Create a fresh 72-cell plan after the desired Full runs have registered; do
not reuse the historical inventory batch for these three tasks. A newly
saved all-dataset plan removes older plan-only inventory folders and matching
review ZIPs only when their rows contain no executed `complete`, `failed`, or
`running` cells; executed batches are retained.

## Frozen E/R decoder diagnostic

Run one registered Full source at a time. This is a CPU, read-only diagnostic;
it restores `best_stage1.pt` and consumes raw cached E/R coordinates without
encoder forward, training, GMM fitting, or cache writes:

```bash
python scripts/run_mnrf_frozen_decoder_diagnostic.py --task PSM
python scripts/run_mnrf_frozen_decoder_diagnostic.py --task MSL
python scripts/run_mnrf_frozen_decoder_diagnostic.py --task GECCO
python scripts/run_mnrf_frozen_decoder_diagnostic.py --task Genesis
python scripts/run_mnrf_frozen_decoder_diagnostic.py --task NYC
python scripts/run_mnrf_frozen_decoder_diagnostic.py --task CalIt2
python scripts/run_mnrf_frozen_decoder_diagnostic.py --task Creditcard
python scripts/run_mnrf_frozen_decoder_diagnostic.py --task CICIDS
```

The diagnostic output is stored below
`result/analysis/MNRF-FrozenDecoder/<TASK>/<source-run>`. P masking is not
silently approximated from an existing cache; it is reserved for a later
masked-input/re-encoding experiment.

For the existing RawCount CalIt2 source, the S-only head can be launched by
the existing frozen-source CLI. Verify the explicitly registered RawCount run
before using the illustrative path:

```bash
python scripts/run_mnrf_view_probe.py --task CalIt2 --mode select_gmm --views S \
  --source-dir result/score/MNRF-ViewProbe-RawCount/CalIt2/rawcount-full-20261031T20261031T092320Z-1093827 \
  --output-dir result/score/MNRF-ViewProbe-RawCount/CalIt2/s-only-$(date -u +%Y%m%dT%H%M%SZ)-$$ \
  --seed 2021 --seq-len 192 --pattern-len 32 --hidden-dim 32 --batch-size 128 \
  --stage1-epochs 50 --stage2-epochs 30 --patience 5 --learning-rate 0.001 --weight-decay 0.0001 \
  --feature-std-floor 0.05 --device cpu --cache-chunk-size 256 --encoding-batch-size 128 \
  --view-dim 8 --evolution-dim 8 --pattern-dim 8 --relation-dim 8 \
  --gmm-components 4 --gmm-covariance-type full --gmm-reg-covar 0.001 \
  --gmm-n-init 1 --gmm-max-iter 100 --gmm-fit-max-windows 20000 \
  --cpu-threads 1 --train-noise-std 0.01 --noise-seed 20261013 \
  --calibration-quantile 0.995 --score-mad-floor 0.001 --artifact-policy full
```

This reuses the RawCount source cache and fits only its S head; it does not
retrain or overwrite the RawCount Full source.
