# MNRF ViewProbe commands

These five independent task entries are prepared only; they have not been
run as formal experiments. Activate the existing global environment once,
then run one command at a time from the repository root. The scripts do not
activate Conda, alter `CUDA_VISIBLE_DEVICES`, search for another Python, or
retry after an error.

```bash
cd /media/h3c/users/wangyueyang1/cxy/APD-CATCH
conda activate catch_env
```

The first command runs a read-only preflight and then, only if it succeeds,
starts a new `full_gmm` run. Each output directory contains a UTC timestamp
and PID, so existing source runs are not overwritten. The default device is
the inherited logical `cuda:0`; edit the literal `DEVICE="cuda:0"` near the
top of one script to `cpu` when a CPU run is intended.

The prepared scripts use `artifact_policy=results`: after successful scoring
they retain the validated result bundle and `run.log`, but discard model,
cache, checkpoint, head and request files. This is the default for these five
task entries so results do not accumulate large reusable artifacts. For a
reusable full artifact, edit the script literal `ARTIFACT_POLICY="results"` to
`full` before starting a new run; do not change an existing run in place.
The log can be followed with `tail -f <RUN_DIR>/run.log` while a task runs.
The requested `results` policy does not disable recovery: interrupted stage,
GMM, and Flow runs keep their model/cache/checkpoint tree until completion.
After scoring, validation and cleanup are separate; a cleanup error is
reported as `results_valid_cleanup_failed` and can be retried with
`python scripts/organize_mnrf_view_probe.py <RUN_DIR>`.

## Task commands

```bash
bash ./scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
bash ./scripts/multivariate_detection/detect_score/MSL_script/MNRFViewProbe.sh
bash ./scripts/multivariate_detection/detect_score/GECCO_script/MNRFViewProbe.sh
bash ./scripts/multivariate_detection/detect_score/SMD_script/MNRFViewProbe.sh
bash ./scripts/multivariate_detection/detect_score/Genesis_script/MNRFViewProbe.sh
```

Run these manually as independent commands. They are not a loop, runner,
scheduler, or background launcher. The task scripts keep the following CATCH reference
window and batch values, but the MNRF training/head settings are independent:

| task | CATCH reference seq_len | CATCH reference batch | MNRF stage1/stage2 | MNRF batch | MNRF views |
| --- | ---: | ---: | ---: | ---: | --- |
| PSM | 192 | 128 | 50/30 | 128 | S,E,P,R |
| MSL | 192 | 128 | 50/30 | 128 | S,E,P,R |
| GECCO | 96 | 128 | 50/30 | 128 | S,E,P,R |
| SMD | 192 | 64 | 50/30 | 64 | S,E,P,R |
| Genesis | 192 | 4 | 50/30 | 4 | S,E,P,R |

All five scripts explicitly fix seed `2021`, dimensions `1/8/8/8`, pattern
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
entry for explicit recovery/source workflows; it is not used by these five
independent commands.

Read-only utilities after a manual run:

```bash
python scripts/mnrf_lite_v2_preflight.py --task PSM --data-source large_detect --batch-size 128 --seq-len 192 --encoding-batch-size 128 --view-dim 8 --evolution-dim 8 --pattern-dim 8 --relation-dim 8
python scripts/validate_mnrf_view_probe.py /absolute/path/to/run
python scripts/summarize_mnrf_view_probe.py --root result/score/MNRF-ViewProbe
```

The first two commands above are read-only checks. Formal task runs are not
part of this repository change and no benchmark result is implied by preparing
these entries.
