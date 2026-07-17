# Fixed Reconstruction-Error Gate v0

## Scope

This experiment is a pre-registered synthetic mechanism gate for the fixed post-hoc CATCH reconstruction-error decomposition. It does not test a new encoder, a new CATCH model, or a complete detection-oriented decomposition model.

For each seed, original CATCH is trained once on a 7,680-point normal series using its existing 80/20 internal split. Its one checkpoint is then reused for all six anomaly types. The scorer reads `original_score`, `time_score`, `frequency_score`, `slow_score`, `fast_score`, and fixed equal-weight `fusion_score` from one reconstruction forward per non-overlapping window.

## Fixed Design

- Training seeds: `20260717`, `20260718`, `20260719`; scoring seed: `20260717`.
- Four variables, one-second `DatetimeIndex`, normal training length 7,680, validation length 1,536, and test length 3,072.
- Original CATCH: `seq_len=192`, `patch_size=16`, `patch_stride=8`, `inference_patch_size=32`, `inference_patch_stride=1`, `num_epochs=3`, `batch_size=128`; all other settings use the original defaults.
- Fixed moving-average window: `W=15`, derived by the protocol rule.
- The normal generator, all anomaly amplitudes, event lengths, and event positions are fixed in `decomp_gate_v0_data.py` before training.

## Anomalies

The six fixed categories are `level_shift`, `slope_change`, `spike`, `variance_increase`, `periodic_amplitude`, and `periodic_phase`. Every category has at least 12 labeled anomaly points. All categories for one seed begin with the same stored normal test baseline and differ only by their specified injection.

## Evaluation And Gate

Evaluation uses raw pointwise AUC-PR, no point adjustment, no threshold search, and no event expansion. The bootstrap samples the 18 `(seed, anomaly_type)` units, never individual time points. The result directory stores all score arrays, per-seed/type metrics, branch response matrices, bootstrap output, and an explicit PASS/FAIL/NOT_EVALUABLE status for every protocol condition.

A passing gate means only that fixed low/high reconstruction-error bands show reproducible differential and complementary evidence on this synthetic suite. A failing gate stops dynamic fusion, weight search, adaptive cutoff, and router additions for this same post-hoc scoring scheme; it does not negate independently pre-registered model-side decomposition studies.
