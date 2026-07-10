# PatternAD Modification Log

Date: 2026-07-05

## Current Goal

Build a multivariate time-series anomaly detector around the following motivation:

```text
The same reconstruction residual can have different anomaly meaning under different temporal and structural contexts.
```

The model is no longer treated as an optional switch on top of the previous LLM-prompt baseline. The outer benchmark class is now `PatternAD`, and its scoring path is multivariate by design.

## Main Design

`PatternAD` now uses a context-conditioned denoising reconstruction backbone. Each time step is encoded as a full multivariate state, and local scale, trend, high-frequency activity, and mask structure are injected into the reconstruction backbone through FiLM-style conditioning before the Transformer encoder. Training masks both individual variable points and whole variable traces inside a window, forcing the model to recover missing values from temporal context and cross-variable evidence.

The default inference path also uses conditional reconstruction: a deterministic subset of variables is hidden at each time step, and the anomaly score is the raw reconstruction residual on those hidden positions only. This makes the local dynamic context contribute to reconstruction quality itself, rather than multiplying the anomaly score after reconstruction.

The previous post-hoc aggregate scorer and reliability-weighted scorer remain in `pattern_scoring.py` only for explicit ablation. They are not the current default path because the raw-control experiments showed that handcrafted score aggregation/reweighting is weaker than improving the reconstruction process directly.

This makes the implementation match direction A: not anomaly type classification, not relation-graph prototype learning, and not text-topology alignment. The key object is the residual's meaning under local dynamics, implemented by making reconstruction conditional on that local dynamic state.

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
  - Uses the same `PatternAD.PatternAD` wrapper but disables context conditioning and conditional scoring.
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

The current default score is `raw`: mean squared reconstruction residual over conditionally hidden variables. The following components are retained only as legacy ablation utilities in `pattern_scoring.py`:

- `scale`: residual normalized by local rolling variance.
- `trend`: residual between rolling trend estimates of the input and reconstruction.
- `shift`: residual between local left/right shift patterns of the input and reconstruction.
- `freq`: high-pass residual after removing rolling trend.
- `sync`: cross-variable concentration of residuals.

These components should not be treated as the main contribution unless an explicit ablation enables `pattern_score_mode="aggregate"` or `pattern_score_mode="reliability_weighted"`.

## Calibration

The scorer fits on training-normal reconstruction outputs after model training:

```text
train true windows + train reconstructed windows
-> component scores
-> per-component median/MAD statistics
```

Test windows use these training statistics only. The number of calibration windows is capped by `pattern_score_max_fit_windows` to prevent the scorer from becoming heavy.

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

Primary comparison after this revision:

```text
PatternAD: context-conditioned reconstruction + conditional masked raw residual
PatternAD_raw: no context conditioning + no conditional scoring + raw residual
```

Recommended first server order:

```text
Weather PatternAD / PatternAD_raw
MetroPT3 PatternAD / PatternAD_raw
HAI21 dev PatternAD / PatternAD_raw
SMD dev PatternAD / PatternAD_raw
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

The completed experiment compared the default PatternAD scorer against the raw-control script on seven multivariate datasets. Metrics were read from the detailed `*.tar.gz` result CSV files. For label metrics, the best value over the 42 anomaly-ratio rows was used. For score-ranking metrics, the reported values are the per-dataset score metrics in the detailed CSV.

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

Interpretation: MetroPT-3 is computationally attractive and realistic, but not ideal as the core PatternAD motivation dataset because `amp_auc_mean=0.945` indicates that raw amplitude deviation is already highly discriminative. Use it for smoke tests, runtime checks, and lightweight multimodal-context experiments. Use HAI21 as the stronger motivation dataset.

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
Syntax check passed for PatternAD.py and pattern_scoring.py by Python compile().
A real forward smoke test was not run locally because this downloaded workspace does not have torch installed.
Run the smoke test on the server environment after syncing with Git.
```
