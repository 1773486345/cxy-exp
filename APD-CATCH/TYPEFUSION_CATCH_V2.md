# TypeFusion-CATCH v2

TypeFusion-CATCH v2 is an independent, single-stage model. It keeps the
benchmark entry point `typefusion_catch_v2.TypeFusionCATCHV2`, result model name
`TypeFusionCATCHV2`, and result directory `result/score/TypeFusion-CATCH-V2/`.
The original CATCH and TypeFusion-CATCH v1 directories remain read-only.

## Research Mapping

Challenge one is handled by four non-exclusive normality tasks: state,
evolution, pattern, and relation. A compound intervention may supervise two
types at once; there is no clean/single/compound classifier and no one-hot
routing.

Challenge two is handled by four typed EvidenceAdapters, a sufficient
log-sum-exp path, six pairwise relation tokens, and a short temporal relation
mixer. The scorer receives only typed tokens and evidence logits. It has no
raw-input or context bypass.

The model decomposes normality conditions and detection tasks, not the raw
signal into additive components.

## Shared Frontend

`SharedRepresentationFrontend` has two shared parameter groups.

The time path is Linear(C to d_model), depthwise temporal Conv1d(kernel 3),
pointwise Conv1d, GELU, and per-time LayerNorm. It returns `h_time [B,T,D]`
for the State branch only.

The frequency path applies RevIN, FFT, real/imaginary channels, frequency
patching, patch embedding, and a channel-aware Transformer independently inside
each frequency patch. It returns `h_freq [B,P,C,D]`. There is no cross-patch
global mixing, reconstruction head, Flatten Head, dc loss, mask optimizer, or
standalone score.

Pattern masked views are created in the time domain before calling this same
frequency frontend. A complete unmasked frequency representation is never
used to predict its own target patch.

## Four Normality Tests

State consumes `h_time`, computes differentiable soft prototype assignments,
prototype context, weighted distance, compactness, commitment, and usage
regularization. Usage is `assignment.mean(dim=(batch,time))`, so prototype
usage has gradient.

Evolution consumes only StandardScaler-space `x`. It right-shifts the input and
uses causal depthwise/pointwise blocks with LayerNorm. Prediction at t reads
`x[:t]`; t=0 raw error, evidence, and task contribution are zero.

Pattern performs even-mask and odd-mask completion. Target points are zeroed
before patch encoding, positional embeddings distinguish patches, and the two
predictions are overlap-added. Time completion and a fixed 0.1 local frequency
term form its raw error.

Relation masks deterministic channel groups before channel mixing. It runs
group by group and batch chunk by batch chunk, retains only selected-channel
predictions, and uses activation checkpointing with `use_reentrant=False` and
`preserve_rng_state=True`. Attention rows are bounded by 2048 and no
`[B,G,T,C,D]` tensor is created.

Every branch returns `z [B,T,branch_dim]`, `raw_error [B,T]`,
`evidence_logit [B,T]`, `softplus(evidence_logit)`, and a task loss.

## Interventions and Losses

Training uses a persistent `torch.Generator`; it is not re-seeded on each
batch. Validation uses `seed + global_sample_index`. Scenarios are fixed at
25% clean, 50% single strong, 12.5% compound strong, and 12.5% compound weak.
State, evolution, pattern, and relation interventions preserve their respective
semantics. Weak compound views share the exact sampled interval and parameters
with the compound view. Batch size one uses a nonzero time roll as donor.

The fixed objective is:

`L = 1.0 L_task + 0.5 L_evidence + 0.5 L_responsibility + 1.0 L_score
+ 0.25 L_score_rank + 0.1 L_clean_score + 0.25 L_synergy`.

Responsibility compares target types only against non-target types; two target
types in a compound do not compete. Losses and scores are checked explicitly
for finite values and raise `FloatingPointError` instead of masking NaN/Inf.

## Joint Score

Each independent adapter maps `concat(z, log1p(raw_error), evidence_logit)`
to one typed token. The scorer forms four branch tokens and six pair tokens,
processes ten tokens per time point with a small Transformer, then applies a
depthwise/pointwise temporal relation mixer with kernel 5. The relation
correction is `2.0 * tanh(relation_delta_raw)`.

The only formal score is:

`joint_score = softplus(joint_logit)`

The adapter returns `scores, scores` after the existing window-to-point
expansion. Branch errors, sufficient logits, relation deltas, thresholds,
labels, and score weights are not benchmark outputs.

## Training and Configuration

There is one optimizer and one single-stage training loop. `num_epochs` equals
the corresponding CATCH task configuration, while StandardScaler is fitted
once on the train split. Validation comes only from that split. `train_label`
is accepted solely for benchmark interface compatibility and is never used.

The v2-only fixed settings are seed 2021, state memory 32, top-k 4,
prototype temperature 1.0, branch dimension 128, temporal layers 3, joint
dimension 128, two joint layers, four joint heads, relation groups 4, maximum
relation attention rows 2048, sufficient temperature 1.0, correction cap 2.0,
responsibility margin 0.2, score margin 0.5, synergy margin 0.2, and the loss
weights shown above. They are uniform across tasks.

## Prepared Commands

These are prepared only and have not been run. Run one task at a time, starting
with PSM. Do not start them in the background, in parallel, or through a runner.
Use `GPU_ID` explicitly; after a real OOM, set `BATCH_SIZE` manually and
arrange a paired baseline rerun before comparing. No performance conclusion is
made here.

```text
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_1_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_2_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_3_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_4_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_5_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_6_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_7_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_8_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_9_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_10_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_11_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/ASD_dataset_12_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/CICIDS_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/SWAT_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/GECCO_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/Genesis_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/CalIt2_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/Creditcard_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/MSL_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/NYC_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/PSM_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/SMAP_script/TypeFusionCATCHV2.sh
GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/SMD_script/TypeFusionCATCHV2.sh
```

## Correctness Checks

```text
python -m compileall ts_benchmark/baselines/typefusion_catch_v2 scripts
python -m unittest discover -s ts_benchmark/baselines/typefusion_catch_v2/tests -p 'test_*.py'
bash -n scripts/multivariate_detection/detect_score/PSM_script/TypeFusionCATCHV2.sh
git diff --check
git diff -- ts_benchmark/baselines/catch/
git diff -- ts_benchmark/baselines/typefusion_catch/
```

No formal training, benchmark, ablation, result summary, external validation,
or claim of superiority over CATCH is part of this implementation round.
