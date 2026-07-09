# PatternAD Modification Log

Date: 2026-07-05

## Current Goal

Build a multivariate time-series anomaly detector around the following motivation:

```text
The same reconstruction residual can have different anomaly meaning under different temporal and structural contexts.
```

The model is no longer treated as an optional switch on top of the previous LLM-prompt baseline. The outer benchmark class is now `PatternAD`, and its scoring path is multivariate by design.

## Main Design

`PatternAD` now uses a lightweight joint-variable reconstruction backbone instead of the previous LLM prompt encoder. Each time step is encoded as a full multivariate state, so the reconstructor can use cross-variable context instead of processing every channel independently. After the best checkpoint is selected, it calibrates a pattern-aware residual scorer on training-normal windows. At inference, reconstruction residuals are not used directly. They are converted into several context-aware evidence scores, calibrated by train-set median/MAD statistics, and then aggregated into the final anomaly score.

This makes the implementation match direction A: not anomaly type classification, not relation-graph prototype learning, and not text-topology alignment. The key object is the residual's meaning under local dynamics.

## Files Changed

- `ts_benchmark/baselines/PatternAD/PatternAD.py`
  - Rewritten as a multivariate-only `PatternAD` wrapper.
  - Replaced the previous DeepSeek prompt-encoding backbone with a joint multivariate Transformer reconstructor.
  - Added masked-input reconstruction during training through `train_mask_ratio`.
  - Added whole-variable masking through `train_variable_mask_ratio`, forcing reconstruction from other variables and temporal context.
  - Removed single-variable `detect_fit`, `detect_score`, and `detect_label` paths from this class.
  - Removed the `use_pattern_aware_scoring` compatibility switch. Pattern-aware scoring is now the model's default scoring mechanism.
  - Rejects data with one or fewer variables.
  - Uses the best validation checkpoint to fit the scorer on training-normal windows.

- `ts_benchmark/baselines/PatternAD/utils/pattern_scoring.py`
  - Implements the post-hoc pattern-aware scorer.
  - Computes residual evidence under raw, scale-normalized, trend, shift, high-frequency, and cross-variable concentration views.
  - Calibrates each evidence component with training median/MAD.
  - Aggregates calibrated evidence with `topk`, `mean`, `max`, or `logsumexp`.

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
  - Added the minimal control experiment for the current scoring module.
  - Uses the same `PatternAD.PatternAD` model and the same joint multivariate reconstructor.
  - Sets `pattern_score_components=["raw"]`, `pattern_score_aggregation="mean"`, and `pattern_score_use_calibration=false`.
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

- `raw`: mean squared reconstruction residual over variables.
- `scale`: residual normalized by local rolling variance, so equal residual magnitudes are treated differently in stable and volatile regions.
- `trend`: residual between rolling trend estimates of the input and reconstruction.
- `shift`: residual between local left/right shift patterns of the input and reconstruction.
- `freq`: high-pass residual after removing rolling trend, highlighting local oscillatory disagreement.
- `sync`: cross-variable concentration of residuals, intended only for multivariate settings.

## Calibration

The scorer fits on training-normal reconstruction outputs after model training:

```text
train true windows + train reconstructed windows
-> component scores
-> per-component median/MAD statistics
```

Test windows use these training statistics only. The number of calibration windows is capped by `pattern_score_max_fit_windows` to prevent the scorer from becoming heavy.

## Config

The new scoring-related hyperparameters are:

```json
{
  "pattern_score_components": ["raw", "scale", "trend", "shift", "freq", "sync"],
  "backbone_name": "JointMultivariateTransformer",
  "train_mask_ratio": 0.25,
  "train_variable_mask_ratio": 0.15,
  "reconstruction_full_loss_weight": 0.1,
  "pattern_score_local_window": 5,
  "pattern_score_trend_window": 7,
  "pattern_score_aggregation": "topk",
  "pattern_score_top_k": 2,
  "pattern_score_logsumexp_tau": 1.0,
  "pattern_score_eps": 1e-6,
  "pattern_score_use_calibration": true,
  "pattern_score_max_fit_windows": 20000
}
```

## Removed From This Model

- Single-variable anomaly detection entry points.
- Optional compatibility switch for pattern-aware scoring.
- Relation graph construction.
- Text-topology alignment.
- Relation prototype / static graph assumptions.
- LLM zero-shot anomaly scoring.
- Per-window text tokenization and LLM prompt encoding.

## Suggested Experiments

Primary comparison:

```text
Joint multivariate reconstruction + raw residual scoring
PatternAD pattern-aware residual scoring
```

Ablations:

```text
raw only
raw + scale
raw + scale + trend/shift
raw + scale + trend/shift + freq
all components
topk vs mean vs max vs logsumexp aggregation
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
