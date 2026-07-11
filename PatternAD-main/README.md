# PatternAD

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)

## Introduction

PatternAD is a multivariate time-series anomaly detection model for context-conditioned residual semantics. Each time step is encoded as a full multivariate state, and training masks both individual variable points and whole variable traces inside a window.

The current research path no longer uses temporal context as a post-hoc score multiplier. Local scale, trend, high-frequency activity, and mask structure are encoded inside the reconstruction backbone through FiLM-style conditioning. The model can estimate either a deterministic conditional mean or a conditional Gaussian/Student-t distribution. The repository keeps `reconstruction_distribution="mse"` as the backward-compatible default; probabilistic variants train with masked NLL and score conditional two-sided tail surprisal.

## Installation

```bash
pip install -r requirements.txt
```

The project was developed under Python 3.8. On the current server, the intended Conda environment is `patternad_env` and can be used without activating the shell:

```bash
conda run --no-capture-output -n patternad_env python --version
```

## Data

Preprocessed datasets are stored under:

```text
dataset/anomaly_detect/data
```

Dataset metadata is stored in:

```text
dataset/anomaly_detect/DETECT_META.csv
```

The code supports both the original long format with `date,data,cols` and the added wide format with `date,<variables>,label`, including `.csv.gz` files.

## Core Model

Model entry:

```text
ts_benchmark/baselines/PatternAD/PatternAD.py
```

Scoring implementation:

```text
ts_benchmark/baselines/PatternAD/utils/pattern_scoring.py
```

### Leakage-free context

The mask is chosen before context construction. Rolling mean/std use only visible values and explicit valid counts; trend uses differences whose two endpoints are visible; masked high-frequency entries are zeroed. The true target and the learned mask token are therefore excluded from the target's context.

### Complementary conditional scoring

Inference partitions the complete `(time, variable)` grid into deterministic complementary masks. Each cell is hidden and scored in exactly one pass, and runtime validation requires coverage to equal one everywhere. The final point score averages all variables, but every variable prediction was produced while that target was hidden.

### Distribution heads and losses

All variants use the same `3 * D` output head so MSE, Gaussian, and Student-t ablations have the same parameter count. Available modes are:

```text
mse       -> conditional mean + masked squared residual
gaussian  -> conditional mean/scale + masked Gaussian NLL
student_t -> conditional mean/scale/df + masked Student-t NLL
```

For MSE, training uses `masked MSE + reconstruction_full_loss_weight * full mean-MSE`. For Gaussian and Student-t, it uses `masked NLL + reconstruction_full_loss_weight * full mean-MSE`. The auxiliary full-window term is intentionally mean-MSE, not full NLL, so visible values cannot drive the predicted scale toward zero by simple copying.

The current Gaussian development path also uses target-blind visible scale to normalize the reconstruction input and de-normalize the conditional mean. A dedicated transition head predicts normalized transition mean and scale correction around a target-blind visible-difference scale prior, and receives a masked transition-NLL auxiliary loss. Transition diagnostics are persisted, but transition surprisal is not yet combined with the primary level-tail anomaly score.

At inference, probabilistic variants default to `pattern_score_mode="tail_probability"`: `-log` of the conditional two-sided tail probability of the absolute standardized residual. Density NLL remains available as an explicit ablation, but its `log(scale)` normalization term is not used as cross-regime anomaly rarity.

Legacy aggregate and reliability-weighted scorers remain available only as explicit ablation paths.

For the factorial context control, `use_context_conditioning=false` does not remove the context modules. It sends a learned dataset-level constant through the same `context_proj` and FiLM path as the visible-context variant. Thus A00/A01 differ from A10/A11 in dynamic context information rather than in the presence of the conditioning route.

## Strict Evaluation Protocol

`config/unfixed_detect_label_multi_config.json` now selects `evaluation_protocol="train_calibration"`. The protocol reserves the temporal tail of official train as an independent calibration split and leaves a gap of `seq_len - 1` points by default between model fit data and calibration data. Test scores and labels do not determine the threshold.

The threshold is a finite-sample calibration-only empirical/conformal-style quantile. Because overlapping time-series scores are not exchangeable, this is not presented as a strict conformal coverage guarantee. The strict factorial manifest predeclares `anomaly_ratios=[1.0]`; different ratios remain separate experiments in the leaderboard and are never collapsed by taking the best test metric. The strict summarizer accepts only threshold-independent ranking/VUS metrics and verifies that repeated score-metric values agree across any ratio rows.

Historical behavior is retained only behind the explicit `evaluation_protocol="legacy_test_contaminated"`. That path is for reproduction and can still depend on test scores and choose a test-metric maximum across ratios. Its output is test-label oracle evidence and must not be reported as an unbiased result.

The strategy now applies the requested seed before each model-series pair is constructed. Factorial runs record the dataset, variant, seed, exact source/config/data hashes, complete expected grid, and attempt number. Resume and summary both fail if the frozen identity changes or the grid is incomplete.

Strict attempts also persist `benchmark.log` and machine-readable model diagnostics. The detailed artifact and `run_metadata.json` must agree on epoch train/validation losses, best epoch, runtime, ordered calibration/test score calls, and Gaussian scale statistics. The runner rejects missing/non-finite diagnostics and calibration scale boundary fractions at or above the frozen 1% limit; test boundary fractions are report-only. The summarizer exposes these fields in `entity_seed_run_diagnostics.csv`.

## Strict Experiment Workflow

The current design is documented in [EXPERIMENT_PLAN_DRAFT.md](EXPERIMENT_PLAN_DRAFT.md). Executable manifests and tooling are in [config/patternad](config/patternad) and [scripts/patternad](scripts/patternad/README.md). The Gaussian objective is consistently defined as masked Gaussian NLL plus full mean-MSE.

Run the focused tests first:

```bash
conda run --no-capture-output -n patternad_env \
  python -m unittest discover -s tests -p 'test_*.py'
```

Inspect the first four-cell smoke command without executing it:

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/run_factorial_ablation.py \
  --group smoke --dataset Weather --variant A00 A10 A01 A11 \
  --seeds 2021 --gpus 0 --run-name p0_weather --dry-run
```

Remove `--dry-run` after checking the resolved dataset and command. Summarize a completed experiment with:

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/summarize_factorial.py \
  --input result/patternad_strict/p0_weather
```

The primary matrix holds complementary masking fixed and isolates the two research factors:

```text
A00: context off + MSE
A10: context on  + MSE
A01: context off + Gaussian training NLL + conditional tail score
A11: context on  + Gaussian training NLL + conditional tail score
```

`B00/B11` are unmasked diagnostics. Student-t is implemented but deferred until Gaussian establishes that conditional-scale modeling is useful.

## Convenience Dataset Scripts

Run one of the existing multivariate datasets:

```bash
sh ./scripts/multivariate_detection/detect_label/Weather_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/Weather_script/PatternAD_raw.sh
```

The `_raw` script replaces dynamic visible context with the learned constant control, disables conditional scoring, and uses raw residual scoring. Because it changes two factors at once, it is useful for historical continuity and smoke checks but is not a fair isolation of the context effect. Use `A00/A10/A01/A11` for the primary attribution experiment.

## Added Dataset Scripts

These datasets have been adapted locally. The factorial runner validates every referenced data and text file before it creates a run directory.

MetroPT-3 is the lightweight real-industrial smoke-test dataset:

```bash
sh ./scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD_raw.sh
```

HAI21 and SMD have dev/full split scripts. The default scripts are lightweight dev runs:

```bash
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD_raw.sh

sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD_raw.sh
```

The legacy full HAI21/SMD convenience scripts are explicit:

```bash
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD_full.sh
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD_raw_full.sh

sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD_full.sh
sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD_raw_full.sh
```

These convenience scripts reproduce the older confounded PatternAD/PatternAD_raw comparison. They are not inputs to the current factorial main table or locked confirmation; use `scripts/patternad/run_factorial_ablation.py` for those experiments.

## Evaluation Scalability

The VUS evaluator uses an exact sparse implementation for long multivariate datasets. It preserves the original RangeAUC-volume definition while avoiding repeated full-sequence scans for every window and threshold. `VUS_ROC` and `VUS_PR` also share the same volume computation.

This change affects evaluation runtime only; it does not change PatternAD reconstruction, anomaly scores, threshold ratios, or metric definitions. See [PATTERN_AD_MODIFICATION_LOG.md](PATTERN_AD_MODIFICATION_LOG.md) for the equivalence tests and MetroPT3 benchmark.

## Checkpoint and Result Compatibility

Current MSE/Gaussian/Student-t variants share a `3 * D` output head, while older checkpoints used a `D`-wide mean head. Current C0/C1 paths both traverse `context_proj`, FiLM, and `pre_encoder_norm`; C0 uses a learned constant and C1 uses visible dynamic context. Older context-off runs skipped conditioning and normalization. Old checkpoints are therefore not load-compatible in general, and old PatternAD/PatternAD_raw result tables are not directly comparable with new factorial results. Rerun every compared cell under the same current code, seed protocol, strict calibration protocol, and manifest.

## Notes

Current local-added datasets:

```text
MetroPT3: single 15-variable industrial dataset, lightweight smoke test.
HAI21: 3 industrial process parts, stronger motivation dataset but heavier.
SMD: 28 machine entities, standard benchmark supplement; stored as .csv.gz.
```

See [PATTERN_AD_MODIFICATION_LOG.md](PATTERN_AD_MODIFICATION_LOG.md) for implementation notes, result interpretation, and dataset adaptation history.
