# Frozen-cache recovery audit, 2026-10-24

Baseline: `e506b3f36e78f9636557a2da6983609150deced7`. This operation did not
run Stage 1, backward, optimizer updates, GMM fitting, Flow, recalibration, or
formal ablations.

## Pinned source outcomes

| Task | Train cache | Validation | Calibration | Test | Outcome |
|---|---|---|---|---|---|
| PSM | missing, no exact backup | exact hash | exact hash | exact hash | blocked |
| MSL | exact hash | exact hash | exact hash | exact hash | already complete; native validation and actual select_gmm source loader passed |
| Creditcard | missing, no exact backup | exact hash | exact hash | missing, no exact backup | blocked |
| CICIDS | missing, no exact backup | exact hash | exact hash | exact hash | blocked |

No missing cache was installed this turn. Each blocked source remains missing
exactly the same splits; no declared hash was changed. Manifest SHA256 before
and after was identical for all four sources. Existing split hashes recorded
in each `result.json` are unchanged before/after.

## Backup search

Search stayed within task-specific MNRF ViewProbe score/analysis/review roots
and known result source/backup roots. The current existing roots found for this
run were `result/score/MNRF-ViewProbe`,
`result/analysis/MNRF-ViewProbe-Ablation`,
and `result/review/MNRF-ViewProbe`. Five ZIP/TAR archives were indexed. PSM
train, Creditcard train/test, and CICIDS train produced no matching archive
member; the candidate file/member byte hashes did not match the immutable
manifest digest. Search counts and paths are in each task's `result.json`.

## Bounded forward diagnosis

For PSM, Creditcard and CICIDS, validation and calibration caches were replayed
at the first batch, midpoint batch, last full batch, and true tail batch. This
gives 24 split/batch cases and 96 per-view comparisons across S/E/P/R. Every
per-view comparison matched exactly: maximum absolute difference
`0`, non-equal elements `0`. This only establishes equality for
those sampled batches, not for the missing train/test files.

Training receipts record Python 3.8.20, PyTorch 2.4.1+cu121, NumPy 1.24.4,
pandas 1.5.3, scikit-learn 1.3.2, CUDA runtime 12.1, requested `cuda:0`, and an
NVIDIA A800-SXM4-80GB. They bind the original input/config and relevant code
digests. They do not record Torch intra/inter-op thread counts, matmul
precision, TF32, or cuDNN determinism; those historical settings remain
unknown. Current bounded diagnostics used one Torch intra-op thread,
`float32_matmul_precision=highest`, TF32 disabled and cuDNN enabled, but these
current settings are not evidence of the original training settings.

The missing original train/test feature files are unavailable, so no exact
`.npy` header/data comparison or element-wise error locations can be stated.
Because bounded replay found no concrete numerical-path difference, no further
full-split forward was attempted and no cache was installed. A future exact
backup may still be restored with `--backup-only`; alternatively a separately
authorized targeted forward can be considered after a cause is established.

## Fresh single-seed matrix inventory

The new immutable plan is
`branch-ablation-20261024T054610Z-1900565/plan.json`. Read-only summary says
72/72 positions present in the plan, 5 reused, 40 pending, 27 blocked, 0
conflicts, 0 complete datasets of 8. It did not execute any missing cell. The
summary status is failed/blocked, not complete.

The compact review package contains only plan, tables, summary and bounded
point evidence. It does not include encoders, GMM files, feature caches or
full variable-NLL arrays.
