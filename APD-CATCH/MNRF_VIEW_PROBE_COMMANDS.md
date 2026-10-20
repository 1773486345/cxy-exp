# MNRF ViewProbe commands

These eight independent task entries are prepared only; they have not been
run as formal experiments. Activate the existing global environment once,
then run one command at a time from the repository root. The scripts do not
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
scheduler, or background launcher. Do not paste all eight into a background job.
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

The exporter refuses an existing analysis directory unless `--overwrite` is
explicit. It consumes only a completed full source; it never retrains,
refits a GMM, recalibrates, or removes source files. The local metadata has
one row and a CSV for each of the eight tasks; the read-only preflight remains
the authority for actual window counts and disk estimates. CalIt2's validation
and calibration counts are reported there and may be too small for stable MAD
and 0.995-quantile estimates. The approximate feature payload figures are
planning values, not measured RAM or VRAM usage.

The first two commands above are read-only checks. Formal task runs are not
part of this repository change and no benchmark result is implied by preparing
these entries.
