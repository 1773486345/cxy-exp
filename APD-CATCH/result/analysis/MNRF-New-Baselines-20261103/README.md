# Added baseline preparation (static only)

This directory records the first PSM/MSL preparation against commit
`146b14bc4419792db4921a3b14e04d68513e3122`. No model was instantiated, fitted,
run, or scored in this preparation. The active environment was `catch_env`
(Python 3.8.20, PyTorch 2.4.1, NumPy 1.24.4, pandas 1.5.3, scikit-learn 1.3.2,
Merlion 1.3.1). TensorFlow is not installed.

## What is actually connected

Only Merlion DAGMM has a verified in-repository registration and an executable
PSM/MSL wrapper. It uses the existing `merlion.DAGMM` adapter and the existing
`run_benchmark.py` pipeline. The explicit configuration is the Merlion 1.3.1
`DAGMMConfig`: `gmm_k=3`, `hidden_size=5`, `sequence_len=1`,
`lambda_energy=0.1`, `lambda_cov_diag=0.005`, `lr=0.001`, `batch_size=256`,
`num_epochs=10`. DAGMM trains its joint autoencoder/GMM on the training side;
the native sample-energy score is higher-is-more-anomalous. Its native label
threshold is owned by Merlion's postrule/calibrator and is not guessed by this
wrapper.

GANF, MTGFlow, tcNF, OmniAnomaly, ConFlow and AFNF are not represented by
placeholder adapters. Their exact status, source, license/dependency gap and
PSM/MSL score unit are in `candidate_status.csv` and
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

## Manual DAGMM commands

After manually activating `catch_env`, from `APD-CATCH` run one command at a
time:

```bash
bash scripts/multivariate_detection/detect_score/PSM_script/DAGMM.sh
bash scripts/multivariate_detection/detect_score/MSL_script/DAGMM.sh
bash scripts/multivariate_detection/detect_label/PSM_script/DAGMM.sh
bash scripts/multivariate_detection/detect_label/MSL_script/DAGMM.sh
```

Success requires `run_manifest.json` with status
`complete_native_score_artifact` (score) or
`complete_native_label_no_threshold_export` (label), a non-empty native report,
and at least one expected `artifacts/score_*.npz` for score. A failure or artifact mismatch
is retained in the same manifest and exits non-zero. These commands are not
run by this preparation.

There are deliberately no executable APD-CATCH commands for the other six
names yet. Their rows are blocked preparations, not failed experiments. The
only safe next checks are source/environment checks in an isolated location,
for example `git ls-remote https://github.com/zqhang/MTGFLOW
f8cb9ea225fd7e64cfb4556dae3525accac285f5` for MTGFlow and
`python -c 'import tensorflow'` in a separately prepared OmniAnomaly
environment. Neither check is a substitute for an adapter. GANF, tcNF,
ConFlow and AFNF need a pinned implementation and reuse terms before a
launcher can be supplied; a method name or paper score must not enter the
PSM/MSL tables by itself.

## Offline evaluation after a score run

The read-only evaluator requires explicit coordinate-bearing NPZ files and
intersects sorted coordinates; it never joins by array length:

```bash
python scripts/evaluate_added_baselines.py --task PSM \
  --artifact DAGMM=result/score/DAGMM/PSM/<run-id>/artifacts/score_default_0.npz \
  --artifact MNRF=result/score/MNRF-ViewProbe/PSM/full_gmm-full-20261024T115119Z-3976714/results/aligned_scores.npz \
  --output result/analysis/MNRF-New-Baselines-20261103/PSM-same-point.csv
```

Replace `<run-id>` only after inspecting the manifest. The evaluator reports
ROC-AUC/AP when both labels are present and only reports alert metrics when a
verified threshold is stored in the artifact. A future common q=.995 report
must additionally bind an independent calibration score artifact; this
preparation does not use test labels or invent that threshold.

`PSM_results_template.csv` and `MSL_results_template.csv` are the two empty
post-run table schemas. `affiliation_f` remains blank until the artifact also
binds the benchmark event boundaries.

## Static checks and limits

The four DAGMM scripts are syntax-checked and the registry/config parse is
checked. These checks do not prove training, forward, score, convergence,
native threshold behavior, or fairness against MNRF. Those claims require the
manual runs and the source-specific adapters for the six blocked candidates.
