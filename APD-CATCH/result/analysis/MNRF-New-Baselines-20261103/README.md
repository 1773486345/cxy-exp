# Added baseline preparation (static only)

This directory records the added-baseline preparation against commit
`5e7ffe9e3b4daeab80ab5a27d94a766d69abfb8f`, extended to all eight APD tasks.
The active environments are `catch_env` and `tods_legacy` (the latter has
TensorFlow 2.4.0); `catch_env` has no TensorFlow.
(Python 3.8.20, PyTorch 2.4.1, NumPy 1.24.4, pandas 1.5.3, scikit-learn 1.3.2,
Merlion 1.3.1). TensorFlow is not installed.

## What is actually connected

Only Merlion DAGMM has a verified in-repository registration and an executable
eight-task wrapper set. It uses the existing `merlion.DAGMM` adapter and the existing
`run_benchmark.py` pipeline. The explicit configuration is the Merlion 1.3.1
`DAGMMConfig`: `gmm_k=3`, `hidden_size=5`, `sequence_len=1`,
`lambda_energy=0.1`, `lambda_cov_diag=0.005`, `lr=0.001`, `batch_size=256`,
`num_epochs=10`. DAGMM trains its joint autoencoder/GMM on the training side;
the native sample-energy score is higher-is-more-anomalous. Its native label
threshold is owned by Merlion's postrule/calibrator and is not guessed by this
wrapper.

GANF, MTGFlow, tcNF, OmniAnomaly, ConFlow and AFNF are not represented by
placeholder adapters. Their per-task status, source, license/dependency gap and
score unit are in `candidate_status.csv` and
`config/added_baselines_registry.json`. MTGFlow is the only blocked candidate
with a verified MIT source and explicit PSM/MSL scope, but its legacy runtime
must be isolated before an adapter can be reviewed. The other unresolved
names are not safe to infer from a paper title or an unrelated repository.

## Archive contract

`run_archived_baseline.py` creates the unique `result/{score,label}/DAGMM/<task>/run-UTC-pid/`
directory before starting the benchmark. It atomically records the startup
script snapshot/hash, command, effective JSON configuration, git state, package
versions, data/metadata hashes, inherited `CUDA_VISIBLE_DEVICES`, stdout,
stderr, exit status and failure reason. The strategy's point-artifact capture
is opt-in and is enabled only by this wrapper. It stores sorted integer
`global_indices`, source timestamps, labels, valid mask, score direction and
score/prediction arrays under `artifacts/`; a length or label/coordinate
mismatch fails export instead of padding or truncating.

The raw native benchmark report remains in the same run directory. Native
`detect_label` outputs are kept separate from continuous score artifacts;
because the generic adapter does not expose a portable calibration series, a
label run is not silently presented as a q=.995 common-protocol result.

The launcher checks free space before starting `run_benchmark.py`. The default
requires a 4 GiB reserve plus the estimated compact artifact size. It records
the check and returns `blocked_disk_space` without starting the model if the
budget is not met. `BASELINE_POINT_ARTIFACT=compact` (default) keeps one
coordinate-bound NPZ for offline evaluation; `native-only` omits that duplicate
point payload and is suitable only when no common pointwise comparison is
needed. `BASELINE_MIN_FREE_GIB` changes the reserve explicitly.

By default archives and native reports remain below this project's `result/`.
An external result root is supported with `BASELINE_STORAGE_ROOT`, but the
selected directory must already exist and be writable. On this server
`/media/h3c/data1` and `/media/h3c/data3` have free capacity but their mount
roots are not writable by this account; ask the administrator to create and
grant a per-user directory before using either mount. The wrapper records the
effective configuration once and stores the source config path/hash rather
than copying a second full config file. stdout/stderr are captured once by the
wrapper and are not piped through a second `tee` in these launchers.

## Manual DAGMM commands

After manually activating `catch_env`, from `APD-CATCH` run one command at a
time. The same pattern is available for all eight datasets:

```bash
# Optional, only after an administrator creates and grants a writable data1/data3 directory:
# export BASELINE_STORAGE_ROOT=/media/h3c/data3/$USER/apd-catch-results
# export BASELINE_POINT_ARTIFACT=compact
# export BASELINE_MIN_FREE_GIB=4
bash scripts/multivariate_detection/detect_score/PSM_script/DAGMM.sh
bash scripts/multivariate_detection/detect_score/MSL_script/DAGMM.sh
bash scripts/multivariate_detection/detect_label/PSM_script/DAGMM.sh
bash scripts/multivariate_detection/detect_label/MSL_script/DAGMM.sh
bash scripts/multivariate_detection/detect_score/GECCO_script/DAGMM.sh
bash scripts/multivariate_detection/detect_score/Genesis_script/DAGMM.sh
bash scripts/multivariate_detection/detect_score/NYC_script/DAGMM.sh
bash scripts/multivariate_detection/detect_score/CalIt2_script/DAGMM.sh
bash scripts/multivariate_detection/detect_score/Creditcard_script/DAGMM.sh
bash scripts/multivariate_detection/detect_score/CICIDS_script/DAGMM.sh
bash scripts/multivariate_detection/detect_label/GECCO_script/DAGMM.sh
bash scripts/multivariate_detection/detect_label/Genesis_script/DAGMM.sh
bash scripts/multivariate_detection/detect_label/NYC_script/DAGMM.sh
bash scripts/multivariate_detection/detect_label/CalIt2_script/DAGMM.sh
bash scripts/multivariate_detection/detect_label/Creditcard_script/DAGMM.sh
bash scripts/multivariate_detection/detect_label/CICIDS_script/DAGMM.sh
```

Success requires `run_manifest.json` with status
`complete_native_score_artifact` (score) or
`complete_native_label_no_threshold_export` (label), a non-empty native report,
and at least one expected `artifacts/score_*.npz` for score. A failure or artifact mismatch
is retained in the same manifest and exits non-zero. These commands are not
run by this preparation.

There are deliberately no executable APD-CATCH commands for the other six
methods yet. Their 48 dataset-method rows are blocked preparations, not failed
experiments. MTGFlow is pinned and MIT-licensed and its upstream documents
PSM/MSL, but it needs a separate compatible runtime and a reviewed APD adapter.
`tods_legacy` supplies TensorFlow 2.4.0, but OmniAnomaly still needs its author
implementation adapted to this benchmark. GANF's official repository gives no
explicit reuse license and documents different datasets. The tcNF paper says
open-source code is available, but the candidate HistCondFlow repository's
identity and license are not yet verified. AFNF and ConFlow lack a verified
compatible source in this checkout. No dependencies were installed and no new
environment was created because the user quota filesystem is full.

## Offline evaluation after a score run

The read-only evaluator intersects sorted coordinates and never joins by array
length. Use the strict identity-bound command below; a loose invocation without
manifests is not a common-protocol admission.

`PSM_results_template.csv` and `MSL_results_template.csv` are the two empty
post-run table schemas. `affiliation_f` remains blank until the artifact also
binds the benchmark event boundaries.

## Static checks and limits

The sixteen DAGMM scripts are syntax-checked and the registry/config parse is
checked. These checks do not prove training, forward, score, convergence,
native threshold behavior, or fairness against MNRF. Those claims require the
manual runs and the source-specific adapters for the six blocked candidates.

## Source and protocol status

The source checks are recorded per method and task in `candidate_status.csv`.
MTGFlow is source-identified at commit `f8cb9ea225fd7e64cfb4556dae3525accac285f5`; its
official README lists PSM/MSL but requires Python 3.8.5, PyTorch 1.7.1, NumPy 1.19.2,
SciPy 1.6.1 and scikit-learn 0.24.1. GANF is pinned to
`bce38333b109f325a403dae9aff4987ae5bd6e1f` and its official experiment scripts cover
SWaT/Metr-LA/PMU, not PSM/MSL. OmniAnomaly is pinned to
`7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4`; TensorFlow 2.4.0 is present in
`tods_legacy` but absent from `catch_env`, and no APD adapter is present. The tcNF
paper (arXiv:2603.09490) names open-source code; `2er0/HistCondFlow` is a candidate
but its paper identity and license require confirmation. ConFlow (IEEE TIM 2025)
and AFNF (ISA Transactions 143, 2023) lack verified compatible adapters. These
are preparation blockers, not failed experiments.

For a strict common-protocol comparison, bind every artifact to its task, source
manifest or embedded source digest, sorted global coordinates, labels, time values,
and `higher_is_anomalous` direction. After a DAGMM score run, use the run manifest and
the fixed source manifest explicitly:

```bash
python scripts/evaluate_added_baselines.py --task PSM \
  --require-identity \
  --data-file dataset/anomaly_detect/data/PSM.csv \
  --source-manifest MNRF=result/score/MNRF-ViewProbe/PSM/full_gmm-full-20261024T115119Z-3976714/probe_manifest.json \
  --source-manifest DAGMM=result/score/DAGMM/PSM/<run-id>/run_manifest.json \
  --score-direction MNRF=higher_is_anomalous \
  --artifact DAGMM=result/score/DAGMM/PSM/<run-id>/artifacts/score_default_0.npz \
  --artifact MNRF=result/score/MNRF-ViewProbe/PSM/full_gmm-full-20261024T115119Z-3976714/results/aligned_scores.npz \
  --output result/analysis/MNRF-New-Baselines-20261103/PSM-same-point.csv
```

For MSL use the same command with `--task MSL`, the MSL dataset file, the MSL
sequence-bounded Full manifest
`result/score/MNRF-ViewProbe-MSL-Boundary/MSL/full_gmm-sequence-bounded-20261102T105427Z-2415811/probe_manifest.json`,
and its `results/aligned_scores.npz`. The evaluator reports AUC/AP only after the
coordinate and identity checks. Alert metrics require a verified threshold in the
artifact; DAGMM native labels remain a separate native-protocol result because no
portable independent calibration artifact is exported.

The PSM/MNRF duplicate-artifact check reproduced 87,650 common points, ROC-AUC
0.7449263161, AP 0.4372811313, saved-threshold recall 0.6027566 and FPR
0.2735017. This validates the evaluator's coordinate/mask handling against an
existing artifact; it is not a new DAGMM result.

During this preparation a shell `--help` check was mistakenly passed through the
score wrapper, which entered PSM benchmark scheduling. The process was stopped
before completion. Because it had entered scheduling, model initialization is
not claimed to have been avoided; its incomplete run directory was removed and
no score/checkpoint/result is claimed. This incident is disclosed here rather
than treated as a successful or completed experiment.
