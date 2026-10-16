# MNRF-Lite V2

This is an independent detector under `ts_benchmark/baselines/mnrf_lite_v2`.
It does not change original CATCH, TypeFusion v1/v2, or their scripts. The
implementation follows the fixed first-version design below; it is not a
performance claim.

## Model Architecture

Input is standardized per variable using statistics fitted only on the internal
training subset. There is no window normalization. A shared point embedding
consumes value, valid adjacent difference, observation mask, and difference
mask. A six-block depthwise-separable causal TCN (dilations 1, 2, 4, 8, 16,
32; two kernel-3 layers per block; width 32) is applied independently to each
variable. Each layer has left-only padding and per-time LayerNorm.

The four variable-preserving views are concatenated in `[S,E,P,R]` order to
make 25 coordinates per variable: current state (1), causal evolution (8),
last-32-point pattern (8), and cross-variable relation (8). Evolution reads
only the preceding 127 samples. Relation source tokens are built independently
per variable and target self-attention is masked before a single cross-variable
attention operation. The target relation therefore receives no self source.

Features are standardized with training-cache statistics. The density model is
four alternating attention-conditioned affine coupling blocks. Mechanism A is
`(S,R)` and B is `(E,P)`; blocks update B, A, B, A. The 25-D coordinate path is
not compressed. The sole detector score is the four-noise mean Gaussian-base
negative log likelihood after log-determinant correction.

## Training

The single benchmark invocation performs all steps. The adapter uses the
dataset's official train/test boundary and splits only the training segment:
after two purges of `L-1` points, remaining points are assigned 80/10/10 to
train, validation, and calibration. Windows never cross these boundaries.

Stage 1 randomly trains one of E/P/R per batch using a temporary shared
conditional Gaussian decoder. P masks eight consecutive values before point
embedding and recomputes valid differences. Early stopping uses the equally
weighted validation macro-NLL over E/P/R; the best encoder is restored.
The encoder is then frozen, kept in eval mode, and its complete unmasked
features are streamed into chunked `.npy` memmaps. The feature standardizer is
fit only on train features.

Stage 2 trains only the four coupling blocks with AdamW. It restores the best
fixed-four-noise validation checkpoint. The validation segment estimates each
variable's score median and MAD scale. An independent calibration segment
sets the 0.995 quantile threshold. Test labels are not used for training,
selection, or threshold choice. Defaults are seed 2021, L=128, pattern length
32, width 32, batch 64, stage limits 50/30 epochs, patience 5, learning rate
0.001, weight decay 0.0001, feature scale floor 0.05, and training noise
standard deviation 0.01.

## Inference and Scoring

Each stride-one complete test window produces one score at its end timestamp.
The first `L-1` test samples are invalid and excluded from evaluation; they are
not padded or assigned fabricated scores. `results/*_aligned_scores.npz`
records the score, prediction, aligned time indices, original valid mask,
labels, and per-variable diagnostic values. Metrics include AUROC, average
precision (AUPRC), calibrated-threshold precision/recall/F1, and normal false
alarm rate. A one-class valid test segment has undefined AUC and the run is
marked failed by the shell validator rather than treated as a valid result.

## Configuration and Limitations

The dedicated config is `config/mnrf_lite_v2_unfixed_detect_score.json`; the
dedicated `mnrf_valid_detect` strategy preserves the complete-window index and
does not use the generic anomaly strategy's padding behavior. The individual
task scripts select `large_detect` or `external_detect` explicitly. Defaults
can be overridden with `GPU_ID`, `BATCH_SIZE`, `SEED`, `STAGE1_EPOCHS`,
`STAGE2_EPOCHS`, `CACHE_CHUNK_SIZE`, and `MNRF_SAVE_PATH` relative to the
repository's `result/` directory.
Set `GPU_ID=cpu` to force CPU, or provide a physical GPU index to select it via
`CUDA_VISIBLE_DEVICES`.

The current supported list is based on metadata, available CSVs, D>=2, and
whether the 80/10/10 split can provide at least one complete L=128 window in
each segment. Internal tasks: ASD_dataset_1..12, CICIDS, CalIt2, Creditcard,
GECCO, Genesis, MSL, NYC, PSM, SMAP, SMD, SWAT. External tasks: BATADAL,
HAI20_07, MTSB_METRO, MTSB_OPPORTUNITY_01..13, MTSB_SWAN_SF, MetroPT3.
MTSB_OCCUPANCY_01 and MTSB_OCCUPANCY_02 are not supported under the frozen
split because their train segments are too short after the purge gaps.

Before a formal run, the preflight checks paths, metadata, feature count,
split feasibility, and configuration without training or creating a cache:

```bash
python scripts/mnrf_lite_v2_preflight.py --task PSM --data-source large_detect
```

Suggested first formal command (not run during implementation):

```bash
bash ./scripts/multivariate_detection/detect_score/PSM_script/MNRFLiteV2.sh
```

From another directory, using the registered global Conda environment name
`catch_env`:

```bash
conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFLiteV2.sh
```

All task commands are listed in [MNRF_LITE_V2_COMMANDS.md](MNRF_LITE_V2_COMMANDS.md).
Each run stores reports, logs, checkpoints, feature caches, deployment bundle,
metadata, and aligned test output in
`result/score/MNRF-Lite-V2/<task>/run-<UTC>-<PID>/`. Smoke tests use only small
synthetic data; no formal task was trained as part of implementation.
