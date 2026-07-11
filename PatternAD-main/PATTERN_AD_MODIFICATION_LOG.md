# PatternAD Modification Log

Date: 2026-07-11

## Current Goal

Build a multivariate time-series anomaly detector around the following motivation:

```text
The same reconstruction residual can have different anomaly meaning under different temporal and structural contexts; the model should estimate conditional residual uncertainty, not only a conditional mean.
```

The model is no longer treated as an optional switch on top of the previous LLM-prompt baseline. The outer benchmark class is now `PatternAD`, and its scoring path is multivariate by design.

## Main Design

`PatternAD` now uses a context-conditioned denoising reconstruction backbone. Each time step is encoded as a full multivariate state, and mask-aware local scale, trend, high-frequency activity, and mask structure are injected into the reconstruction backbone through FiLM-style conditioning before the Transformer encoder. Training masks both individual variable points and whole variable traces inside a window, forcing the model to recover missing values from temporal context and cross-variable evidence.

Inference now uses complementary conditional reconstruction. Deterministic mask passes partition the full time-variable grid, so every target is hidden and scored exactly once. The compatibility default remains MSE/raw residual. Gaussian and Student-t conditional distributions are opt-in and use conditional two-sided tail surprisal by default; density NLL remains an explicit diagnostic mode. The strict first-stage factorial experiment uses Gaussian before considering Student-t robustness.

The previous post-hoc aggregate scorer and reliability-weighted scorer remain in `pattern_scoring.py` only for explicit ablation. They are not the current default path because the raw-control experiments showed that handcrafted score aggregation/reweighting is weaker than improving the reconstruction process directly.

This makes the implementation match direction A: not anomaly type classification, not relation-graph prototype learning, and not text-topology alignment. The key object is the residual's meaning under local dynamics, implemented as a conditional residual distribution rather than a collection of handcrafted post-hoc weights.

## Files Changed

- `ts_benchmark/baselines/PatternAD/PatternAD.py`
  - Rewritten as a multivariate-only `PatternAD` wrapper.
  - Replaced the previous DeepSeek prompt-encoding backbone with a joint multivariate Transformer reconstructor.
  - Added masked-input reconstruction during training through `train_mask_ratio`.
  - Added whole-variable masking through `train_variable_mask_ratio`, forcing reconstruction from other variables and temporal context.
  - Removed single-variable `detect_fit`, `detect_score`, and `detect_label` paths from this class.
  - Removed the `use_pattern_aware_scoring` compatibility switch. Context-conditioned reconstruction plus conditional masked raw residual is now the model's default path.
  - Rejects data with one or fewer variables.
  - Uses the best validation checkpoint to fit the scorer on training-normal windows.

- `ts_benchmark/baselines/PatternAD/utils/pattern_scoring.py`
  - Defaults to raw reconstruction residual.
  - Supports mask-aware scoring, so inference residual is averaged only over conditionally hidden variables.
  - Retains scale, trend, shift, frequency, sync, aggregate, and reliability-weighted logic only for explicit legacy ablations.

- `ts_benchmark/baselines/PatternAD/__init__.py`
  - Exposes `PatternAD` as the package model entry.

- `ts_benchmark/baselines/PatternAD/models/` and `ts_benchmark/baselines/PatternAD/layers/`
  - Removed old DeepSeek/prompt-specific model and layer files because the current model uses a compact in-file multivariate reconstructor.

- `ts_benchmark/baselines/utils.py`
  - Removed eager DeepSeek tokenizer loading from the multivariate data path.
  - `MultiSegLoader` now returns tiny placeholder text tensors because PatternAD does not consume text.

- `scripts/multivariate_detection/detect_label/*_script/PatternAD.sh`
  - Updated model name to `PatternAD.PatternAD`.
  - Updated save paths to `label/<dataset>_PatternAD`.
  - Replaced old prompt-backbone hyperparameters with lightweight multivariate reconstruction defaults:
    `batch_size=64`, `d_model=128`, `d_ff=256`, `e_layers=2`, `num_epochs=30`,
    `train_mask_ratio=0.25`, `train_variable_mask_ratio=0.15`.

- `scripts/multivariate_detection/detect_label/*_script/PatternAD_raw.sh`
  - Added the no-context raw-control experiment for the current model.
  - Uses the same `PatternAD.PatternAD` wrapper, replaces dynamic visible context with the learned constant control, and disables conditional scoring.
  - Sets `pattern_score_components=["raw"]`, `pattern_score_aggregation="mean"`, `pattern_score_use_calibration=false`, and `pattern_score_mode="raw"`.
  - Saves results to `label/<dataset>_PatternAD_raw`.

- `ts_benchmark/evaluation/strategy/anomaly_detect.py`
  - Disabled verbose per-ratio result printing by default; set `verbose_result=true` in strategy config to restore it.
  - Removed unused pickle/base64 serialization of actual and inference arrays because the result columns are intentionally left empty.
  - Cached score-based metrics (`auc_roc`, `auc_pr`, `R_AUC_ROC`, `R_AUC_PR`, `VUS_ROC`, `VUS_PR`) once per dataset instead of recomputing them for every anomaly ratio.
  - Fixed the cached-metric call path by passing `score_metric_cache` in `execute`, `multi_execute`, and `mmd_execute`; the missing argument caused raw-control scripts to finish with empty metric rows.

- `scripts/univariate_detection/`
  - Removed the whole univariate experiment directory to avoid exposing irrelevant single-variable entry points.

- `README.md`
  - Updated the project description and quickstart to match PatternAD.

## Scoring Components

The compatibility default score is `raw`: mean squared reconstruction residual from complementary conditional predictions. Gaussian and Student-t use density NLL only when `pattern_score_mode="nll"`; selecting a non-MSE distribution without an explicit score mode switches to conditional two-sided tail surprisal. The following components are retained only as legacy ablation utilities in `pattern_scoring.py`:

- `scale`: residual normalized by local rolling variance.
- `trend`: residual between rolling trend estimates of the input and reconstruction.
- `shift`: residual between local left/right shift patterns of the input and reconstruction.
- `freq`: high-pass residual after removing rolling trend.
- `sync`: cross-variable concentration of residuals.

These components should not be treated as the main contribution unless an explicit ablation enables `pattern_score_mode="aggregate"` or `pattern_score_mode="reliability_weighted"`.

## Legacy Component Calibration

The scorer fits on training-normal reconstruction outputs after model training:

```text
train true windows + train reconstructed windows
-> component scores
-> per-component median/MAD statistics
```

Test windows use these training statistics only. The number of calibration windows is capped by `pattern_score_max_fit_windows` to prevent the scorer from becoming heavy. This component calibration is unrelated to the strict label-threshold calibration protocol added on 2026-07-11.

## Config

The current default model-related hyperparameters are:

```json
{
  "backbone_name": "ContextConditionedReconstructor",
  "train_mask_ratio": 0.25,
  "train_variable_mask_ratio": 0.15,
  "reconstruction_full_loss_weight": 0.1,
  "use_context_conditioning": true,
  "context_window": 7,
  "context_film_strength": 0.2,
  "use_conditional_scoring": true,
  "score_mask_ratio": 0.35,
  "reconstruction_distribution": "mse",
  "distribution_min_scale": 0.001,
  "distribution_max_scale": 100.0,
  "distribution_init_scale": 1.0,
  "student_t_df": 4.0,
  "student_t_learn_df": true,
  "student_t_max_df": 100.0,
  "pattern_score_components": ["raw"],
  "pattern_score_aggregation": "mean",
  "pattern_score_use_calibration": false,
  "pattern_score_mode": "raw",
  "pattern_score_max_fit_windows": 20000
}
```

Legacy scorer hyperparameters such as `pattern_score_local_window`, `pattern_score_trend_window`, `pattern_score_top_k`, and reliability-weight settings remain available only for ablation.

## Removed From This Model

- Single-variable anomaly detection entry points.
- Optional compatibility switch for pattern-aware scoring.
- Relation graph construction.
- Text-topology alignment.
- Relation prototype / static graph assumptions.
- LLM zero-shot anomaly scoring.
- Per-window text tokenization and LLM prompt encoding.

## Suggested Experiments

Primary comparison after the current revision:

```text
A00: context off + MSE + complementary conditional masks
A10: context on  + MSE + complementary conditional masks
A01: context off + Gaussian NLL + complementary conditional masks
A11: context on  + Gaussian NLL + complementary conditional masks
B00/B11: unmasked diagnostics only
```

Recommended first server order:

```text
P0 tests and Weather one-seed A00/A10/A01/A11 smoke
P1 synthetic mechanism suite through the dedicated generator/evaluator; benchmark registration is optional
P2 motivation group, three paired seeds
P3 robustness group after candidate selection is frozen
P4 locked confirmation group once, with five paired seeds
```

Recommended metrics:

```text
AUC-PR
AUC-ROC
VUS-PR
VUS-ROC
event-level F1
point-adjusted F1
```

## 2026-07-10 Result Analysis And Scorer Revision

### Workspace Note

There are two working copies in the current workflow:

```text
Server workspace: /media/h3c/users/wangyueyang1/cxy/PatternAD-main
Local downloaded workspace: E:\cxy-exp\cxy-exp\PatternAD-main
```

The local copy was used for result analysis and code modification. The updated local code should be synchronized to the server with Git before running the next experiment there.

### Raw-Control Result

The completed experiment compared the default PatternAD scorer against the raw-control script on seven multivariate datasets. Metrics were read from the detailed `*.tar.gz` result CSV files. For label metrics, the best value over the 42 anomaly-ratio rows was used. For score-ranking metrics, the reported values are the per-dataset score metrics in the detailed CSV. The best-over-42 label metrics are test-label oracle values and are retained here only as historical development evidence; they must not be used as unbiased results.

```text
              F1      Adjust-F1  Aff-F    AUC-ROC  AUC-PR  R-AUC-ROC  R-AUC-PR  VUS-ROC  VUS-PR
PatternAD     0.3634  0.5664     0.7339   0.6571   0.3303  0.6280     0.3342    0.6241   0.3352
Raw-control   0.3921  0.6313     0.7518   0.7275   0.3897  0.6798     0.3763    0.6794   0.3775
Delta         -0.0287 -0.0649    -0.0179  -0.0704  -0.0594 -0.0518    -0.0421   -0.0553  -0.0423
```

Interpretation: the first pattern-aware scorer was not experimentally supported. It treated raw, scale-normalized, trend, shift, frequency, and sync scores as independent anomaly evidence and aggregated them with top-k. This degraded ranking quality on most datasets. The strongest negative example was Weather, where raw-control reached much better AUC/VUS metrics. Energy and MSDS were the only clearly favorable or near-favorable cases for the old scorer.

### Revision Made After Analysis

This intermediate revision introduced reliability-weighted residual scoring. A later seven-dataset experiment showed that it repaired part of the old aggregate scorer but still lost to raw-control overall. It is therefore superseded by the current context-conditioned reconstruction revision documented below.

### Next Server Experiment

After syncing this local copy to the server, the next comparison should use the current context-conditioned reconstruction revision:

```text
PatternAD     = context-conditioned reconstruction + conditional masked raw residual
PatternAD_raw = no context conditioning + no conditional scoring + raw residual
```

Recommended first runs:

```text
sh ./scripts/multivariate_detection/detect_label/Weather_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/Weather_script/PatternAD_raw.sh
sh ./scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD_raw.sh
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD_raw.sh
sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD_raw.sh
```

The next result should be judged primarily by AUC-PR, AUC-ROC, VUS-PR, VUS-ROC, event-level F1, and point-adjusted F1. If the current model still loses to raw-control, the next change should continue improving the conditional reconstruction mechanism rather than returning to post-hoc handcrafted score weights.
## 2026-07-10 Dataset Adaptation: SMD

SMD has been downloaded from the OmniAnomaly repository and adapted into the current multivariate benchmark format.

Files added or updated:

```text
dataset/anomaly_detect/data/SMD_machine-*.csv.gz
dataset/anomaly_detect/data/SMD_text.csv
dataset/anomaly_detect/DETECT_META.csv
scripts/multivariate_detection/detect_label/SMD_script/PatternAD.sh
scripts/multivariate_detection/detect_label/SMD_script/PatternAD_raw.sh
ts_benchmark/data/utils.py
```

Adaptation details:

```text
entities: 28
variables per entity: 38
format: date, channel1, ..., channel38, label; stored as .csv.gz
metadata dataset_name: SMD
main result path: label/SMD_PatternAD
raw-control result path: label/SMD_PatternAD_raw
```

`ts_benchmark/data/utils.py` now supports wide CSV files without a `cols` column. Existing long-format files with `date,data,cols` remain supported.

Quick audit summary:

```text
total length: 1,416,825
mean test anomaly ratio: 4.21%
amp_auc_mean: 0.782
amp_auc range: 0.539 - 0.989
mean train_vol_p90/p10: 6.99
```

Temporary download files were removed after conversion. Telemanom was checked for SMAP/MSL, but the repository does not package raw train/test arrays; it requires Kaggle data access for `patrickfleith/nasa-anomaly-detection-dataset-smap-msl`. SMAP/MSL were therefore not adapted in this local copy.

Next server commands after syncing:

```text
sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD_raw.sh
```
## 2026-07-10 Dataset Adaptation: HAI 21.03

HAI 21.03 has been downloaded from the official HAI repository and adapted into the current multivariate benchmark format. This dataset is a stronger fit for the PatternAD motivation than SMD because simple amplitude separation is weak while normal-stage local volatility varies substantially.

Files added or updated:

```text
dataset/anomaly_detect/data/HAI21_part1.csv.gz
dataset/anomaly_detect/data/HAI21_part2.csv.gz
dataset/anomaly_detect/data/HAI21_part3.csv.gz
dataset/anomaly_detect/data/HAI21_text.csv
dataset/anomaly_detect/DETECT_META.csv
scripts/multivariate_detection/detect_label/HAI21_script/PatternAD.sh
scripts/multivariate_detection/detect_label/HAI21_script/PatternAD_raw.sh
```

Adaptation details:

```text
HAI21_part1.csv.gz = train1 + test1
HAI21_part2.csv.gz = train2 + test2
HAI21_part3.csv.gz = train3 + test3 + test4 + test5
variables: 79
label: attack
format: date, <79 process variables>, label
main result path: label/HAI21_PatternAD
raw-control result path: label/HAI21_PatternAD_raw
source: https://github.com/icsdataset/hai
```

Quick audit summary:

```text
file              T       D   train    anom%   seg_n  seg_med  amp_auc_mean  amp_auc_max  train_vol_p90/p10
HAI21_part1       259202  79  216001   1.46    5      98       0.452         0.416        7.95
HAI21_part2       345602  79  226801   2.90    20     151      0.538         0.529        8.11
HAI21_part3       718804  79  478801   2.03    25     203      0.633         0.571        7.77
```

Temporary HAI download files were removed after conversion. Only the adapted `.csv.gz` files, metadata, scripts, and text placeholder remain.

Next server commands after syncing:

```text
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD_raw.sh
```
## 2026-07-10 Dataset Adaptation: MetroPT-3

MetroPT-3 has been downloaded from the UCI Machine Learning Repository and adapted into the current multivariate benchmark format. It is useful as a lightweight real-industrial development dataset, but it should not be treated as the strongest motivation dataset because simple amplitude scoring is already very strong.

Files added or updated:

```text
dataset/anomaly_detect/data/MetroPT3.csv.gz
dataset/anomaly_detect/data/MetroPT3_text.csv
dataset/anomaly_detect/DETECT_META.csv
scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD.sh
scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD_raw.sh
```

Adaptation details:

```text
variables: 15
label: official failure intervals converted to point-level labels
train split: first month before 2020-03-01
format: date, <15 sensor variables>, label
main result path: label/MetroPT3_PatternAD
raw-control result path: label/MetroPT3_PatternAD_raw
source: https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset
```

Quick audit summary:

```text
file       T        D   train    anom%   seg_n  seg_med  amp_auc_mean  amp_auc_max  train_vol_p90/p10
MetroPT3   1516948  15  214850   2.30    4      5511     0.945         0.881        17.45
```

Interpretation: MetroPT-3 is computationally attractive and realistic, but not ideal as the core PatternAD motivation dataset because `amp_auc_mean=0.945` indicates that raw amplitude deviation is already highly discriminative. Use it for smoke tests, runtime checks, and lightweight multivariate-context experiments. Use HAI21 as the stronger motivation candidate.

Next server commands after syncing:

```text
sh ./scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD_raw.sh
```
## 2026-07-10 Runtime Tuning For Large Added Datasets

HAI21 and SMD were reorganized for practical experimentation. Full data remain available, but default scripts now run smaller dev experiments.

SMD storage change:

```text
SMD_machine-*.csv -> SMD_machine-*.csv.gz
old uncompressed SMD CSV files were removed
compressed SMD size: about 103 MB, down from about 409 MB
DETECT_META.csv now points to the .csv.gz files
```

Script convention:

```text
PatternAD.sh           = dev experiment
PatternAD_raw.sh       = dev raw-control experiment
PatternAD_full.sh      = full experiment
PatternAD_raw_full.sh  = full raw-control experiment
```

HAI21 dev/full split:

```text
HAI21 dev:  HAI21_part1 only, num_epochs=10, d_model=64, d_ff=128
HAI21 full: HAI21_part1 + HAI21_part2 + HAI21_part3, num_epochs=30, d_model=128, d_ff=256
```

SMD dev/full split:

```text
SMD dev machines:
- SMD_machine-3-2.csv.gz   low amplitude separability, amp_auc_mean=0.543
- SMD_machine-2-2.csv.gz   high normal volatility, amp_auc_mean=0.633, train_vol_p90/p10=20.48
- SMD_machine-3-9.csv.gz   medium amplitude separability and high volatility, amp_auc_mean=0.720
- SMD_machine-1-1.csv.gz   common medium-difficulty machine, amp_auc_mean=0.735
- SMD_machine-2-8.csv.gz   high amplitude-separability sanity check, amp_auc_mean=0.989

SMD full: all 28 SMD_machine-*.csv.gz files, num_epochs=30, d_model=128, d_ff=256
```

Recommended server order:

```text
sh ./scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD_raw.sh
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD_raw.sh
sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD.sh
sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD_raw.sh

# only after dev results are useful:
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD_full.sh
sh ./scripts/multivariate_detection/detect_label/HAI21_script/PatternAD_raw_full.sh
sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD_full.sh
sh ./scripts/multivariate_detection/detect_label/SMD_script/PatternAD_raw_full.sh
```
## 2026-07-10 Latest Seven-Dataset Result Analysis

The latest completed server run covers the original seven multivariate datasets only. The locally added MetroPT3/HAI21/SMD datasets have not been synced to the server yet and are not included in this result.

Comparison setup:

```text
PatternAD: latest reliability-weighted scorer
Raw-control: same reconstruction backbone, raw residual scoring only
Datasets: Weather, Genesis, Energy, Daphnet, SKAB, GECCO, MSDS
```

Mean results over the seven datasets:

```text
metric      PatternAD   Raw-control   Delta
F1          0.3813      0.3921       -0.0109
Adjust-F1   0.6428      0.6313       +0.0115
Aff-F       0.7317      0.7518       -0.0201
AUC-ROC     0.7109      0.7275       -0.0166
AUC-PR      0.3692      0.3897       -0.0205
R-AUC-ROC   0.6709      0.6798       -0.0089
R-AUC-PR    0.3645      0.3763       -0.0118
VUS-ROC     0.6704      0.6794       -0.0090
VUS-PR      0.3660      0.3775       -0.0115
```

Per-dataset deltas for the main ranking/event metrics:

```text
dataset   F1       AdjF1    AffF     AUC-ROC  AUC-PR   VUS-ROC  VUS-PR
Weather  -0.0376  -0.0295  -0.0149  -0.0449  -0.0516  -0.0574  -0.0522
Genesis  -0.0102  +0.0000  +0.0001  +0.0176  -0.0020  +0.0088  +0.0002
Energy   +0.0039  +0.0054  -0.0035  +0.0019  +0.0004  -0.0001  +0.0010
Daphnet  -0.0063  +0.0009  -0.0112  -0.0184  -0.0078  -0.0204  -0.0161
SKAB     -0.0090  +0.1045  +0.0032  -0.0463  -0.0706  -0.0147  -0.0235
GECCO    -0.0194  -0.0034  -0.1188  -0.0361  -0.0179  -0.0201  -0.0032
MSDS     +0.0024  +0.0025  +0.0045  +0.0104  +0.0060  +0.0410  +0.0135
```

Interpretation:

```text
1. Reliability weighting is better than the previous aggregate PatternAD scorer, especially on Weather, Genesis, and Daphnet.
2. It still does not beat raw residual scoring overall. Raw-control is stronger on most ranking metrics, including AUC-PR and VUS-PR.
3. The main positive signals are Energy and MSDS, with small gains, and SKAB Adjust-F1, but SKAB ranking metrics degrade.
4. Weather remains the clearest failure case: all major metrics drop against raw-control.
```

Comparison against the previous aggregate PatternAD scorer:

```text
metric      mean delta, latest reliability-weighted minus old aggregate
F1          +0.0179
Adjust-F1   +0.0764
AUC-ROC     +0.0538
AUC-PR      +0.0389
VUS-ROC     +0.0463
VUS-PR      +0.0308
```

Decision:

The latest scorer is a valid repair of the old aggregate scorer, but it is not experimentally sufficient as the main contribution. The current evidence says post-hoc residual reweighting is too weak and too unstable. The next model change should move context into reconstruction or learned calibration, rather than adding more handcrafted score weights.

## 2026-07-10 Context-Conditioned Reconstruction Revision

This revision moves the direction from post-hoc residual scoring into reconstruction itself.

Reason for the change:

```text
Old aggregate scorer: clearly worse than raw-control.
Reliability-weighted scorer: better than old aggregate scorer, but still weaker than raw-control on most ranking metrics.
Conclusion: context should not mainly reweight the final score; it should help the model reconstruct the value that is actually reasonable under the current dynamic state.
```

Code changes:

```text
ts_benchmark/baselines/PatternAD/PatternAD.py
- The backbone is now a context-conditioned denoising reconstructor.
- Local mean, local std, trend, high-frequency residual, and mask structure are encoded as context features.
- Context features modulate the Transformer input through FiLM-style gamma/beta conditioning.
- Training keeps random point masking and whole-variable masking.
- Inference uses deterministic conditional masking and scores only hidden variables.

ts_benchmark/baselines/PatternAD/utils/pattern_scoring.py
- Default scoring is now raw residual.
- If a score mask is supplied, raw residual is averaged only over hidden variables.
- Legacy aggregate and reliability-weighted scoring remain only for explicit unmasked ablations.

scripts/multivariate_detection/detect_label/*_script/PatternAD_raw.sh
- Raw-control scripts now disable context conditioning and conditional scoring.
- This makes raw-control a true no-context reconstruction baseline for the current model.
```

Current default comparison:

```text
PatternAD     = context-conditioned reconstruction + conditional masked raw residual
PatternAD_raw = no context conditioning + no conditional scoring + raw residual
```

Validation status in the local workspace:

```text
The canonical server environment is the global Conda environment
`patternad_env` at
`/media/h3c/users/wangyueyang1/.env/envs/patternad_env` (Python 3.8.20), invoked
through `/media/h3c/users/shared_app/miniconda3/bin/conda`. All PatternAD tests
and experiments should use this environment rather than searching for or
creating a similarly named replacement.
Focused model/protocol unit tests pass under that environment.
One-epoch MSE and Gaussian fit-to-score smoke tests return finite, length-aligned scores.
```


## 2026-07-11 Exact VUS Scalability Revision

Large added datasets exposed a scalability defect in the shared VUS evaluator. This was an evaluation implementation issue, not a PatternAD model or scoring issue.

Observed case:

```text
MetroPT3 test length: 1,302,098
anomaly events: 4
median anomaly length: 5,511
VUS window count: 11,023
old work: windows x 250 thresholds x full sequence scans
symptom: model fitting/scoring completed, then evaluation appeared to hang
```

PatternAD-side code changes:

```text
ts_benchmark/evaluation/metrics/vus_metrics.py
- RangeAUC_volume now uses an exact sparse implementation.
- Score thresholds are prepared once.
- Weighted true positives are accumulated only over anomaly ranges and their expanded support.
- The metric definition, thresholds, positive-range extension, ROC integration, and PR integration are unchanged.

ts_benchmark/evaluation/metrics/classification_metrics_label.py
- VUS_ROC and VUS_PR now share one generate_curve result for the same label/score arrays.
- This avoids computing the identical VUS volume twice.
```

Validation:

```text
Reference comparison: optimized implementation versus the Git HEAD implementation.
Test sizes: 50, 101, 500, and 2,000 points.
Test windows: 0, 1, 2, 3, 7, 15, and 30.
Included tied anomaly scores.
Result: all ROC/PR curves and final VUS_ROC/VUS_PR values matched within 1e-11.

MetroPT3 scale benchmark:
points: 1,302,098
windows: 11,023
optimized elapsed time: 4.52 seconds on the local benchmark machine
The benchmark used real MetroPT3 labels and deterministic random scores to isolate metric runtime.
```

Experiment interpretation:

```text
- Existing completed PatternAD VUS values remain semantically valid within their historical architecture and evaluation protocol; the VUS formula was not changed. They are not directly comparable with the 2026-07-11 factorial revision described below.
- New PatternAD and baseline runs must use the synchronized optimized evaluator for practical runtime.
- Do not compare runs that were interrupted before report generation; rerun those incomplete model-series pairs.
- Server paths and Python environments remain deployment-specific. The code uses repository-relative imports and the currently selected Python environment.
```

Server continuation after Git sync:

```bash
cd <server-workspace>/PatternAD-main
conda activate <server-environment>
sh ./scripts/multivariate_detection/detect_label/MetroPT3_script/PatternAD.sh
```

## 2026-07-11 Conditional Distribution And Strict Protocol Revision

This revision closes two gaps in the previous implementation: raw MSE could not distinguish equal residuals under different normal uncertainty, and the historical label/report path used test-contaminated thresholds and best-over-ratio aggregation.

### Leakage-Free Conditional Context

The documentation previously showed `describe_local_dynamics(x)` before masking. That pseudocode was misleading and has been corrected. The implemented order is:

```text
construct mask
replace hidden targets
compute local context with valid = not mask
predict hidden target distribution
score only the targets hidden in that pass
```

`JointMultivariateReconstructor._context_features` now uses masked rolling sums and valid counts. Local mean/std exclude hidden targets, trend differences are valid only when both endpoints are visible, and masked high-frequency entries are zero. Context is computed from observed inputs rather than from the learned `mask_token`.

### Complementary Mask Coverage

The old inference implementation scored one deterministic subset. `_complementary_score_masks` now partitions every `(time, variable)` cell across K passes, and `_predict_for_scoring` assembles the hidden predictions into one complete tensor. Coverage is checked at runtime and must equal one at every cell.

For the strict manifest, `score_mask_ratio=1/3`, so the A cells use three passes. For very low-dimensional inputs, the number of groups is capped by `D`; for example, `D=2` uses two passes and still covers each cell exactly once.

### Conditional Distribution Heads

Supported values of `reconstruction_distribution` are:

```text
mse
gaussian
student_t
```

All variants use the same `3 * D` final output head. MSE consumes only the mean output, while Gaussian consumes mean/scale and Student-t consumes mean/scale/df. Positive scale uses softplus plus a configurable floor and maximum; Student-t df is constrained above 2 and can be learned or fixed.

The training losses are:

```text
MSE:
masked MSE(mean, target) + lambda * full-window MSE(mean, target)

Gaussian / Student-t:
masked NLL(mean, scale[, df], target) + lambda * full-window MSE(mean, target)
```

The auxiliary term is deliberately full mean-MSE, not full NLL. Visible positions can be copied; applying NLL there would let the decoder reduce loss by collapsing its scale without learning hidden-target uncertainty. NLL scores are aggregated only from predictions made while their target was hidden.

The repository default remains `reconstruction_distribution="mse"` and raw scoring for compatibility. Gaussian/Student-t are opt-in; a non-MSE distribution selects NLL automatically unless `pattern_score_mode` is explicitly supplied. The first strict factorial uses Gaussian as D1. Student-t is implemented but deferred until conditional-scale modeling has evidence.

### Fair Architecture Controls And Compatibility

The unified `3 * D` output head keeps MSE/Gaussian/Student-t parameter counts equal. C0 sends a learned dataset-level constant through the same `context_proj`, FiLM, and `pre_encoder_norm` route used by C1; C1 replaces that constant with target-blind visible context.

These changes create a hard comparison boundary:

```text
old output head: D
current output head: 3 * D
old context-off path: skipped context projection, FiLM, and pre_encoder_norm
current C0 path: learned constant through context projection, FiLM, and pre_encoder_norm
```

Old checkpoints are generally not load-compatible because the output-head shapes differ. Old PatternAD/PatternAD_raw results also cannot be placed directly beside the new A00/A10/A01/A11 results: the context-off computation graph changed, and the strict threshold/seed protocol changed. Rerun all compared cells with the same current code and manifest.

### Strict Temporal Calibration

The new `train_calibration` evaluation protocol is explicit in `config/unfixed_detect_label_multi_config.json` and the factorial manifest. For label evaluation:

```text
official train prefix -> model fit input
gap                  -> default seq_len - 1 points
official train tail  -> independent calibration input
official test        -> evaluation only
```

The default calibration fraction is 20%. The threshold uses a finite-sample calibration-only empirical/conformal-style order statistic; overlapping time-series scores do not provide the exchangeability required for a strict conformal coverage guarantee:

```text
alpha = anomaly_ratio / 100
k = ceil((n_cal + 1) * (1 - alpha))
threshold = kth calibration score
```

If `k > n_cal`, the threshold is `+inf`. Test scores and labels do not enter threshold selection. The factorial manifest predeclares one 1% ratio. The non-legacy leaderboard keeps ratio rows separate instead of applying `aggfunc=max`, and the strict summarizer never optimizes across test ratios.

Strict calibration now also validates that every official-train label is finite and zero; contaminated train splits fail instead of being silently treated as normal. All 39 entities in the frozen smoke, motivation, robustness, and confirmation groups pass this label audit. The runner's attempt-number parser avoids Python 3.9-only string APIs, so `--resume` remains compatible with the documented Python 3.8 environment.

`legacy_test_contaminated` remains an explicit compatibility protocol. It can still call model label APIs that combine train/test scores, and the legacy leaderboard can still take the test-metric maximum. It must be labeled test-label oracle and is not an acceptable main protocol.

### Seed Fix

`AnomalyDetect.execute`, `multi_execute`, and `mmd_execute` now pass the current scalar or per-series strategy seed to `fix_random_seed` before constructing the model. Previously, the no-argument call silently reused the default seed 2021, so nominal multi-seed runs could initialize identically. PatternAD now uses dedicated seeded generators for training-mask draws and DataLoader shuffle; the strict matrix fixes dropout to zero so C0/C1 do not desynchronize paired random schedules.

### Reproducible Experiment Tools

The current strict guide and executable files are:

```text
EXPERIMENT_PLAN_DRAFT.md
config/patternad/factorial_ablation.json
config/patternad/dataset_groups.json
scripts/patternad/run_factorial_ablation.py
scripts/patternad/summarize_factorial.py
scripts/patternad/bootstrap_factorial.py
scripts/patternad/generate_contextual_synthetic.py
scripts/patternad/evaluate_contextual_mechanisms.py
scripts/patternad/README.md
config/patternad/synthetic_suite.json
```

The experiment plan now uses the implemented objective consistently: masked Gaussian NLL plus full mean-MSE, not full NLL.

The runner creates a frozen `run_plan.json` containing the complete identity grid plus critical source, benchmark-config, data/text, manifest, and dataset-config hashes. Resume refuses config drift. Locked confirmation additionally requires a clean worktree and the full predeclared dataset/variant/seed grid. The summarizer rejects missing, failed, unexpected, or hash-mismatched cells instead of silently aggregating only successful runs. Single-row static text descriptions are reused across fit/calibration/test without materializing millions of duplicate strings.

The synthetic suite now provides 20 predeclared switching-VAR generator seeds, with separate development and untouched confirmation groups. Its same-deviation and gradual/abrupt pairs use equal injected deviations in their compared windows; dependency breaks preserve each affected channel's event marginal while maximizing ground-truth conditional surprise over a seeded, fixed candidate set. Context OOD is explicitly a negative control for conditional-only scores. FPR pools exclude event spans plus a `seq_len - 1` overlap guard. The bootstrap tool reports deterministic paired intervals and fails closed when provenance, balance, replicate count, or stage-specific sampling assumptions are insufficient.

Run focused correctness tests:

```bash
conda run --no-capture-output -n patternad_env \
  python -m unittest discover -s tests -p 'test_*.py'
```

Dry-run the first four A cells:

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/run_factorial_ablation.py \
  --group smoke --dataset Weather --variant A00 A10 A01 A11 \
  --seeds 2021 --gpus 0 --run-name p0_weather --dry-run
```

Remove `--dry-run` to execute. Summarize completed detailed artifacts without a test-oracle maximum:

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/summarize_factorial.py \
  --input result/patternad_strict/p0_weather
```

The primary comparison is now `A00/A10/A01/A11`, not PatternAD versus `PatternAD_raw`, because the latter changes both context conditioning and conditional scoring. `B00/B11` remain unmasked diagnostics and cannot support the target-blind conditional-density claim by themselves.

### P0 Diagnostic Persistence

The first strict Weather P0 completed all A00/A10/A01/A11 cells and produced a complete score summary, but its artifact could not audit the planned Gaussian scale-boundary or training-curve checks. The result remains valid as a pipeline smoke and is not rewritten.

PatternAD now records epoch train/validation loss, best epoch, early-stop state, parameter count, fit/scorer time, and ordered calibration/test score calls. Probabilistic calls additionally record scale min/max/mean/std, finite counts, and lower/upper boundary counts and fractions. These diagnostics enter the detailed CSV, are validated against the frozen distribution/score mode, and are copied into `run_metadata.json`; the runner also persists combined stdout/stderr in `benchmark.log`. Missing diagnostics, non-finite values, phase overwrite, or a calibration scale boundary fraction at or above 1% prevent completion and resume. Test boundary fractions are retained for disclosure but never drive this gate. The strict summarizer verifies detail/metadata equality and writes `entity_seed_run_diagnostics.csv`.

The diagnostic rerun `p0_weather_diag_v2` completed A01/A11 on 2026-07-11. Both calibration and test calls reported zero lower/upper scale-boundary hits. A01 stopped after 12 epochs with best epoch 9; A11 stopped after 7 epochs with best epoch 4. Their six threshold-independent metrics exactly matched the corresponding first-P0 values, providing a same-seed persistence check. P0 is therefore closed; no further Weather reruns are required before P1.

## 2026-07-11 Visible-Context Scale-Prior Prototype

Model development is now prioritized over expanding the formal experiment matrix. The previous Gaussian head received local scale only through a generic context MLP and was otherwise free to learn `sigma` from the shared decoder. That did not structurally encode the core mechanism that equal residuals should use different reference scales in quiet and volatile contexts.

The probabilistic head now uses:

```text
sigma_prior = dataset_scale * local_visible_std ** context_scale_prior_mix
sigma       = sigma_prior * exp(limit * tanh(learned_log_correction))
```

`local_visible_std` excludes masked targets. C0 uses the same learned dataset-scale parameter but no dynamic local prior. The default prior mix is 0.5 rather than 1.0: local dynamics affect the reference scale, but anomalous neighboring points cannot fully dictate `sigma`. The decoder correction is bounded in log space. `use_context_scale_prior=false` retains the previous softplus head for a direct development comparison.

New unit contracts cover target blindness, larger initial scale in a volatile visible context, regime invariance of the C0 prior, and equal parameter counts. The old Weather P0 predates this prototype and is now pipeline evidence only. A partially launched generator-seed-3101 pilot was stopped and deleted; its sole completed A00 result belonged to the old model and had matched-ordering rate 0.2, so it was not retained as a comparable artifact. Deterministic synthetic input artifacts remain reusable.

One-epoch CPU diagnostics on generator seed 3101 showed that the scale prior improved A11 versus A01 in macro AP (`0.1054` versus `0.0819`) and maximum normal-regime FPR gap (`0.0045` versus `0.0190`). Same-deviation ordering improved from `0/3` to `1/3`, but abrupt-versus-gradual ordering remained `0/2`. These are development signals, not reported results.

An attempted transition-dependent scale suppression used the target-blind local slope to reduce `sigma` around rapid directed changes. It recovered one abrupt/gradual pair, but reduced same-deviation ordering to `0/3`, lowered macro AP to `0.0942`, and raised the calibration threshold from `5.80` to `12.83`. This coupling was rejected: `context_transition_scale_suppression` now defaults to `0.0`. The useful part, replacing adjacent-only trend with a visible-point local linear-regression slope that bridges masked gaps, remains as a context feature and will be evaluated separately.

The cleaned current-code one-epoch pair is `result/patternad_synthetic/dev_local_slope_v1/{A01,A11}`. A11 versus A01 has macro AP `0.1053` versus `0.0819` and maximum regime FPR gap `0.0045` versus `0.0190`; overall ordering remains `0.2`. This supports retaining the scale prior for further development, but it does not support expanding datasets or claiming that transition semantics are solved. The next useful experiment is a full-epoch A01/A11 run on generator seed 3101 only; broader grids remain deferred.

The requested 30-epoch generator-seed-3101 run completed in `result/patternad_synthetic/dev_local_slope_full`. A11 improved macro AP from A01's `0.0835` to `0.0931`; AP increased on all four generated mechanisms, and maximum regime FPR gap decreased from `0.00542` to `0.00450`. However, A11 matched ordering fell from `1/5` to `0/5`: all three equal-deviation quiet events remained below their volatile counterparts, and both abrupt events remained below gradual references. Dependency-break AP remained below prevalence (`0.0979` versus `0.1563`). The current model therefore improves coarse ranking/calibration but fails the defining residual-semantics contract. Do not expand seeds or real datasets. The next model-development step is to persist raw residual, standardized residual, predicted scale, and log-scale components so mean and scale failures can be separated before changing the architecture.

Score decomposition showed that A11 predicted the intended scale direction: quiet-event scale was about `0.65-0.73`, versus `0.97-1.09` for matched volatile events. Standardized squared residual correctly ordered two of three same-deviation pairs, but Gaussian density NLL reversed them because its `log(sigma)` normalization penalized the wider normal regime. Density is not the same estimand as conditional tail rarity across heteroscedastic contexts.

Probabilistic scoring now defaults to two-sided tail surprisal while training remains masked NLL. A one-epoch generator-seed-3101 sanity check increased A11 macro AP from the NLL version's `0.1053` to `0.1224`, same-deviation AP from `0.0976` to `0.1573`, and matched ordering from `1/5` to `2/5`; maximum regime FPR gap remained `0.00450`. A01 macro AP was `0.0815`. Both abrupt/gradual pairs and dependency-break detectability remain unresolved, so the next full run is limited to current A01/A11 on seed 3101.

The 30-epoch tail-probability pair completed in `dev_tail_probability_full`. A11 achieved macro AP `0.1180` versus A01 `0.0829`, same-deviation AP `0.1435` versus `0.0418`, and matched ordering `2/5` versus `1/5`; its normal-regime FPR gap was `0.00450`. Both abrupt/gradual pairs still failed and dependency-break AP remained below prevalence (`0.0989` versus `0.1563`). Component diagnostics localized the remaining same-deviation failure: volatile pair-1 raw squared residual was `3.57`, versus `1.28` in quiet context, while the predicted scale direction was already correct. The next prototype therefore normalizes the reconstruction input and conditional mean by target-blind visible local scale; transition handling remains separate.

Target-blind scale normalization produced a positive one-epoch signal: A11 macro AP increased from the tail-only prototype's `0.1224` to `0.1286`, same-deviation AP from `0.1573` to `0.1898`, and the failed pair-1 standardized-residual margin narrowed from `-1.316` to `-0.878`. Ordering remained `2/5`, so normalization is retained but not sufficient.

Transition diagnostics showed that the conditional-mean transition residual itself ordered both abrupt/gradual pairs correctly, while a scale synthesized from adjacent point sigmas reduced this to one pair. The unused third Gaussian output block now predicts `transition_scale`, trained by a masked transition NLL auxiliary objective with weight `0.1`; no parameters were added. A one-epoch A11 check kept macro AP at `0.1285` and made transition standardized-residual margins positive for both pairs (`+0.0830`, `+0.0374`). The transition branch is not yet part of the final anomaly score. A full A01/A11 run is required before selecting a joint level/transition tail combination.

The 30-epoch derived-transition run completed in `dev_transition_likelihood_full`. A11 retained level macro AP `0.1254`, same-deviation ordering `2/3`, and transition standardized-residual ordering `2/2` with margins `+0.1083` and `+0.0438`. However, its learned transition scale was nearly constant (`1.22-1.25`) across contexts. Unconditionally adding transition tail surprisal destroyed same-deviation ordering. Two deterministic gates were rejected: normalized local slope and a left/right change-point statistic had similar event means in abrupt, gradual, and volatile contexts and did not protect regime calibration.

The transition scale now has its own target-blind visible-difference prior, analogous to the level-scale prior. A separate transition head directly predicts normalized `delta_mu` and transition-scale correction instead of subtracting two independently reconstructed level means. All variants instantiate the same head. One-epoch A11 retained macro AP `0.1284` and same-deviation `2/3`; transition standardized-residual ordering remained `2/2`, although the second margin was only `+0.0073`. The head is not combined into the main score until a full run confirms this margin.

The 30-epoch explicit-transition-head run retained level macro AP `0.1262` and same-deviation `2/3`. Raw transition residual stayed `2/2`, but transition standardized residual finished at `1/2`; the second margin was `-0.0014`, effectively tied but not stable. Raising transition-loss weight from `0.1` to `0.5` did not improve the one-epoch margin (`+0.0055` versus `+0.0073`) and slightly reduced macro AP. Unconditional joint scoring, normalized-slope gating, and left/right change-point gating were all rejected. During P1-v1 the transition head remained auxiliary/diagnostic, was absent from the anomaly score, and its `0.1` auxiliary loss was applied only to D1 cells.

## 2026-07-12 P1 Result And Context-Stratified Tail Calibration

The complete P1-v1 crossed grid finished all 120 identities without failures. A11 improved macro AP over A00 by `0.063709` (95% crossed-bootstrap CI `[0.052872, 0.075587]`) and improved A11-A01 matched ordering by `0.186667` (`[0.066667, 0.293333]`). The registered regime-FPR criterion failed: the relative reduction against A00 was only `11.46%`, below `25%`, with a CI spanning zero. A11 also reduced abrupt/gradual ordering to `0.116667`; dependency-break AP remained below prevalence. Direction A therefore has a supported same-deviation mechanism but cannot advance to the real-data matrix in its v1 form.

The failed transition family is now closed. Its auxiliary loss is zero in every formal cell, removing a D1/D0 training-objective confound. P1-v2 replaces the uncalibrated theoretical Gaussian tail with a normal-only context-stratified empirical tail. Target-blind predicted log-scale defines four quantile bins; each bin's empirical survival probability is shrunk toward the global ECDF, with undersized bins falling back completely to the global reference. The map is fitted only on masked residuals from the dedicated normal score-reference segment inside model fit; outer temporal calibration data, test values, and all labels are excluded. Scores remain monotone in absolute standardized residual within each bin and finite for values beyond the reference maximum.

Expanded P1-v1 cells were consolidated from more than 1,300 files into `result/patternad_synthetic/p1_contextual_dev_v1/p1_raw_cells_20260712.tar.gz` (SHA256 `5b1be39f8942251289c7561f5ceac80362f4f0587d944e2cac68b8b5c6f77d34`). The strict summary, run plan, and frozen inputs remain unpacked. Regenerable seeds 3102-3110, obsolete single-seed prototype trees, old Weather P0 expansions, stale label results, and caches were removed. Runtime `result/` and generated synthetic seed directories are now ignored so future experiments do not dirty locked-run provenance.

## 2026-07-12 Disjoint Empirical-Tail Reference Revision

The first P1-v2 implementation used normal residuals from the optimization split to fit its predicted-scale-stratified ECDF. Although labels and outer calibration/test data were excluded, those residuals came from the segment used for gradient updates and could be optimistically narrow or regime-dependent. That weakens the intended claim of calibrated conditional tail rarity.

The model-fit prefix is now split in temporal order into three disjoint normal segments, with `seq_len - 1` points removed at both internal boundaries:

```text
optimization train  80% of usable model-fit points
early stopping      10%
score reference     10%
```

The scaler is fitted on optimization train only. The score-reference segment alone fits the normal-only contextual-tail ECDF; validation is only for checkpoint selection, and the outer train-calibration tail remains threshold-only. All A/B cells use the same split, so this change does not add a context/distribution confound. Diagnostics now persist the segment sizes, fractions, gap, and reference source. Unit coverage verifies non-overlap, exact window gaps, and invalid fraction rejection.

The stale tracked synthetic fixture `artifacts/patternad_synthetic/contextual_v1/seed_3101` was removed. Its recorded generator hash did not match the current generator, so the fail-closed P1 runner rejected it. The P1-v2-holdout runner regenerates all development fixtures from the frozen current generator instead of silently mixing incompatible synthetic inputs.

## 2026-07-12 P1-v2-Holdout Result And Causal Innovation Diagnostic

The complete P1-v2-holdout grid passed all provenance and balance checks: 10 generator seeds × 3 model seeds × A00/A10/A01/A11 = 120 identities. Its frozen Git commit is `20c857e`; it matches the current critical model, manifest, runner, and summarizer sources.

```text
A11 - A00 macro AP: +0.034939, 95% crossed-bootstrap CI [0.026281, 0.044365]
A11 - A01 matched ordering: +0.006667, CI [-0.100000, 0.100000]  -> fail
relative maximum regime-FPR-gap reduction vs A00: 5.40%, CI spans zero -> fail
A11 - A00 dependency-break AP: +0.007662, CI [0.005228, 0.010457] -> pass
```

The failure is structurally informative. Dynamic Gaussian conditioning strongly improves the equal-deviation subproblem (`A11-A01` same-deviation ordering `+0.4556`), but it degrades slow-drift versus abrupt-shift ordering (`-0.6667`). Thus a bidirectional masked reconstructor can use post-transition observations to explain an abrupt shift. The holdout ECDF revision was statistically necessary, but it cannot repair this information-set mismatch. No real-data matrix may start from P1-v2.

The new development-only branch is a causal innovation head. Its GRU receives `x_{<t}` through a one-step right shift and predicts a Gaussian distribution for `x_t`; perturbing `x_t` or future values cannot change the output at `t`. The head is trained on complete past observations by `reconstruction_causal_innovation_loss_weight`, is instantiated in every distribution variant for parameter-count fairness, and is disabled (`0.0`) in all formal A/B cells. At inference it exports raw and standardized innovation residuals as score components but does not alter the primary level-tail score. The first test is one full-epoch A11 run on generator 3101/model seed 2021 with weight `1.0`; inspect its component ordering before freezing any combination or expanded grid.

## 2026-07-12 Causal Level-Innovation Result And Delta-Innovation Revision

The single full-epoch A11 causal level-innovation diagnostic completed at
`result/patternad_synthetic/dev_causal_innovation`. Its primary score was
intentionally unchanged (`macro AP 0.104447`, matched ordering `2/5`). The
diagnostic component did not supply the missing mechanism: standardized causal
innovation got `0/3` same-deviation orderings and `1/2` abrupt-versus-gradual
orderings, with margins `-0.2571` and `+0.4072` for the latter. Its predicted
causal scale was approximately constant at `1.0`. The level-innovation branch
therefore stops here: no multi-seed grid, score combination, P2, or real-data
run is authorized from it.

The replacement remains development-only and causal, but moves the target to
innovation space. A shared past-only GRU predicts `x_t - x_{t-1}`; a separate
head emits its Gaussian mean and a bounded scale correction. The scale prior is
the rolling RMS of `x_j - x_{j-1}` for `j < t` only, with a positive floor.
Thus an abrupt onset is a direct prediction error while an already volatile
normal regime receives a larger scale without reading the present target or
future observations. Tests perturbing `x_t` and all future values verify that
both delta mean and delta scale at `t` remain unchanged.

The branch is instantiated for every variant but is disabled in formal cells:

```text
reconstruction_causal_delta_innovation_loss_weight = 0.0
use_causal_delta_innovation_diagnostics = false
```

One A11 development run only is next, with
`reconstruction_causal_delta_innovation_loss_weight=1.0`. Inspect
`causal_delta_innovation_standardized_squared_residual`; it must put both
abrupt/gradual pairs in the predicted order before any multi-seed expansion.
The evaluator now records resolved causal-diagnostic flags in
`score_run_metadata.json`, avoiding the earlier ambiguity where an enabled
loss implied a diagnostic branch but the raw command override did not list it.
