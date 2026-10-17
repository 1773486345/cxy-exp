# MNRF View Probe

`MNRFViewProbe` is an independent frozen-feature ablation experiment. It does
not modify `MNRFLiteV2`, its encoder, losses, Flow, score aggregation, or the
existing launchers. The probe imports the existing `FourViewEncoder` and
stage-one `MNRFTrainer` implementation, then writes a separate
`result/score/MNRF-ViewProbe/...` run directory.

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
