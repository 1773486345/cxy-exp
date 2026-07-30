# TypeFusion-CATCH v2

TypeFusion-CATCH v2 is an independent model under
`ts_benchmark/baselines/typefusion_catch_v2`.  The existing CATCH and v1
directories are frozen.  v2 addresses two separate design challenges:

1. anomaly-type priors become four non-exclusive normality tests;
2. typed evidence is scored jointly through explicit pairwise relations.

No scenario classifier, branch score voting, threshold calibration, or result
selection logic is part of the model.

## Data Flow

StandardScaler windows first pass through Phase A, which composes the original
CATCH adapter and trains its unchanged reconstruction, frequency, dc-loss,
optimizer, scheduler and early-stopping path.  The best original CATCH state is
loaded by `CATCHAnchor`, then every anchor parameter is frozen.  The anchor
reconstruction alone is encoded by `AnchorContextEncoder`; raw input never has
an independent path to the joint scorer.

Phase B runs label-free interventions on standardized windows and trains the
context encoder, four normality branches, four independent EvidenceAdapters and
`RelationAwareJointScorer`.  The four branches are:

- State: local state encoder plus learnable normal prototypes and top-k distance;
- Evolution: right-shifted causal convolution with per-time LayerNorm and a
  prediction of `x[t]` from `x[:t]` only;
- Pattern: two complementary even/odd masked patch completion passes with
  overlap-add;
- Relation: deterministic channel-group masking and conditional reconstruction,
  evaluated group by group to bound memory.

Each branch returns `z [B,T,D]`, `raw_error [B,T]`, `evidence_logit [B,T]`,
`evidence=softplus(evidence_logit)`, and a branch task loss.  EvidenceAdapters
concatenate `z`, `log1p(raw_error)` and evidence logits, add independent type
embeddings, and apply FiLM from the anchor context.

## Interventions

`TypeInterventionGenerator` uses only a `torch.Generator` derived from seed
2021.  Scenarios are fixed at 25% clean, 50% single-strong, 12.5% compound-
strong and 12.5% compound-weak.  State interventions use interval offsets;
evolution uses an aligned all-channel donor successor; pattern uses one shared
patch permutation; relation replaces only selected channels after per-channel
mean/std matching.  Weak compound samples retain both targets and masks and
provide the three views needed by the synergy loss.  Validation interventions
are derived from `seed + sample_index` and do not read labels.

## Relation-Aware Joint Scorer

For every time point, the scorer receives four branch tokens and six unordered
pair tokens: `(state,evolution)`, `(state,pattern)`, `(state,relation)`,
`(evolution,pattern)`, `(evolution,relation)`, `(pattern,relation)`.  Pair MLPs
consume `u_i`, `u_j`, `abs(u_i-u_j)` and `u_i*u_j`, then receive anchor-context
FiLM.  A small Transformer processes the ten tokens.  The sufficient path is a
fixed-temperature log-sum-exp of the four independent branch heads, preserving
strong evidence from a single branch.  The relation correction is bounded by
`2.0 * tanh(relation_delta_raw)` and is added to the sufficient logit.  No
branch weights are returned or used.

The only formal score is:

`joint_score = softplus(joint_logit)`

`detect_score` returns this score twice in the benchmark adapter's required
`(scores, scores)` format after the existing window-to-point expansion.  CATCH
reconstruction, branch errors, intervention masks and type targets are never
part of the formal score.

## Losses and Phases

Phase A uses the original CATCH loss and training protocol.  Phase B uses the
fixed `loss_version=typefusion_catch_v2_joint_score_v1` objective:

`L = 1.0 L_task + 0.5 L_evidence + 0.5 L_responsibility + 1.0 L_score`

`  + 0.25 L_score_rank + 0.1 L_clean_score + 0.25 L_synergy`.

`L_task` is the sum of the four clean normality tasks.  Evidence combines clean
zero targets and positive intervention masks.  Responsibility uses a fixed
0.2 non-exclusive margin.  Joint score uses class-balanced BCE with a capped
positive weight, a 0.5 region margin, clean-score suppression and a 0.2 weak
compound synergy margin.  No test label is read by either phase.

The adapter records `anchor_optimizer_steps`, `type_optimizer_steps`,
`completed_phases`, `anchor_best_validation_loss` and
`type_best_validation_loss`.  v2 deliberately makes no claim that its total
training budget equals CATCH.

## Configuration

The 23 prepared scripts inherit each task's public CATCH values for sequence,
patch, model width, heads, dropout, batch size, learning rate, patience and
epochs.  v2-only fixed values are `state_memory_size=32`, `state_topk=4`,
`branch_dim=128`, `joint_dim=128`, `joint_layers=2`, `joint_heads=4`,
`relation_mask_groups=4`, `sufficient_temperature=1.0`,
`relation_correction_cap=2.0`, `responsibility_margin=0.2`, `score_margin=0.5`,
`synergy_margin=0.2`, and seed `2021`.  Type training epochs equal the CATCH
epoch count for that task.  The public model name is `TypeFusionCATCHV2` and
the import path is `typefusion_catch_v2.TypeFusionCATCHV2`.

## Prepared Commands

The following commands are prepared only; none has been run.  Run one task at a
time, starting with PSM, and inspect its result before continuing.  Do not
background or parallelize these commands.  Set `GPU_ID` explicitly; after a
real OOM, set `BATCH_SIZE` manually and arrange a paired CATCH rerun before any
comparison.

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

Only random CPU tensor smoke and unit tests are intended in this implementation
round:

```text
python -m compileall ts_benchmark/baselines/typefusion_catch_v2 scripts
python -m unittest discover -s ts_benchmark/baselines/typefusion_catch_v2/tests -p 'test_*.py'
bash -n scripts/multivariate_detection/detect_score/PSM_script/TypeFusionCATCHV2.sh
git diff --check
git diff -- ts_benchmark/baselines/catch/
git diff -- ts_benchmark/baselines/typefusion_catch/
```

No formal benchmark, 23-task run, ablation, result summary, external
validation, or performance claim is made by this v2 implementation.
