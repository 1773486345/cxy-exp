# TypeFusion-CATCH v1

## Research Questions

TypeFusion-CATCH addresses two fixed questions on normal-only multivariate
time-series training data:

1. How can detection branches become specialised for distinct anomaly
   mechanisms instead of learning four copies of the same reconstruction?
2. When branch evidence is consistent, complementary, or conflicting at one
   time point, how can a single final anomaly score be produced without a
   score ensemble?

The model treats state, temporal evolution, segment pattern, and multivariate
relation as overlapping anomaly priors.  They are not labels and no one-hot
anomaly classifier is present.  One event may therefore activate several
branches.

## Architecture

`X_standardized -> RevIN -> FFT/CATCH-style frequency patching/CATCH-style cross-channel Transformer + local temporal
stem -> {State, Evolution, Pattern, Relation} -> EvidenceAdapter ->
BranchFusionTransformer -> leave-one-branch-out normal tokens ->
JointNormalDecoder -> X_hat_joint`

TypeFusion-CATCH is based on CATCH's frequency patching, cross-channel modelling
ideas and anomaly-detection protocol, then reimplements the shared frequency
representation, type-specialised branches, conflict fusion and joint normal
reconstruction.  The shared encoder uses RevIN, full FFT, CATCH-style frequency
patching and a CATCH-style cross-channel Transformer; it is not the original
complete CATCH Trans_C/backbone, does not use its learned channel mask generator,
and does not retain its large Flatten heads.  Its compact depthwise-separable
temporal path prevents short state and local evolution information from being
discarded by frequency processing.  It is not a fifth expert and has no anomaly
score.

The four branch tasks are deliberately different:

- State: combines local temporal patches with shared temporal tokens, reads a
  top-k sparse set of learnable normal-state prototypes, then decodes only from
  that normal memory.  It targets spikes, bounds and level/state shifts.
- Evolution: directly right-shifts the adapter's train-data-StandardScaler
  input before a causal TCN.  It never reads full-window RevIN mean or variance.
  Every causal block uses LayerNorm across the hidden dimension independently
  at each time point, never BatchNorm or any time-aggregating normalisation.
  Its output at time `t` has no path from `X[t]` or later inputs, so it predicts
  normal evolution, changes, timing shifts and transitions from history alone.
- Pattern: uses shared CATCH-style channel-frequency tokens, randomly masks
  frequency patches during branch training, applies a frequency Transformer,
  decodes complex spectral patches, overlap-adds them and applies IFFT.  It
  targets shape, frequency, phase, periodic and collective segment changes.
- Relation: expands channel-mask groups along the batch dimension.  It predicts
  a channel only from the group execution in which that channel is zero-masked;
  channel attention and temporal mixing then reconstruct it from other channels
  and their context.  It targets conditional dependence and synchrony changes.

Each branch returns a normal representation `z_k`, a normal reconstruction or
prediction `x_hat_k`, and a dense evidence map.  State, Pattern and Relation
operate in RevIN space; Evolution uses the train-data StandardScaler space for
its target, prediction and evidence to preserve the causal boundary.

## Conflict-Aware Fusion

Four independent `EvidenceAdapter` instances convert `(z_k, e_k, x_hat_k)` to
local patch tokens `[B, P, 4, D]`.  They include branch-type embeddings and
learned patch positions.  Evidence is summarised inside each local patch, never
collapsed to one window-level score.

During fusion training, one or two branch types are replaced by a learned mask
token.  The branch-axis Transformer predicts the held-out normal token from
the remaining branch tokens and shallow temporal context.  The target token is
stop-gradient normal training evidence.

During validation, the same batched four-way LOO forward used for inference is
also the deterministic mask objective: only each view's masked branch token is
compared with its stop-gradient observed token.  Therefore Fusion/Finetune
validation includes a real `lambda_mask * L_branch_mask` term rather than a
zero placeholder caused by `model.eval()`.

At inference, four leave-one-branch-out views are concatenated as a `B*4`
batch.  A single BranchFusionTransformer pass predicts state from the other
three, evolution from the other three, and so on.  The selected predictions
form `Q_normal`.  This asks what normal evidence each branch should have given
the other mechanisms; it does not emit scores, weights, votes, gates, or branch
binary decisions.

`JointNormalDecoder` accepts only `Q_normal` plus learned decoder patch queries.
It performs branch cross-attention, a light patch-temporal decoder and
overlap-add reconstruction.  It cannot read raw `X`, any raw branch score, or
uncorrected branch token.  The sole official output is:

`total_score = |X - X_hat_joint|`

The benchmark adapter averages this pointwise map only across channels, using
CATCH's existing window/point flattening contract.  Branch maps and
`branch_conflict_map` are diagnostics only.  There is no mean, max, fixed or
learned score weighting, voting, threshold calibration, or post-hoc gate.

## Loss And Training

State, Pattern, Relation and Joint losses use the RevIN-normalised training
space.  Evolution loss uses the train-data StandardScaler space by design:

`L = L_state + L_evolution + L_pattern_time + L_relation + 0.1 L_pattern_freq + 0.1 L_branch_mask + L_joint`

The actual stage objective is explicit:

- `branch_pretrain`: the four specialised branch objectives and frequency loss;
- `fusion_train`: `0.1 L_branch_mask + L_joint`, with shared stem and branches
  frozen and adapters/fusion/decoder trainable;
- `joint_finetune`: the full fixed loss, with fusion components plus only the
  late shared projections and final branch layers unfrozen.

`detect_fit` defaults to `fit_mode="three_stage"` and creates one model instance
for `branch_pretrain -> fusion_train -> joint_finetune`.  Every stage creates an
optimizer from only its currently trainable parameters, saves its best complete
`state_dict`, restores that state, then starts the next stage from it.  The
StandardScaler is fitted once before Stage 1 and reused thereafter.  Explicit
`branch_pretrain_epochs`, `fusion_train_epochs`, and `joint_finetune_epochs`
are debug-only controls.  The formal default is
`training_budget_mode="equal_total_steps"`: with
`catch_train_epochs=3`, TypeFusion allocates no more than the corresponding
CATCH optimizer updates across all stages.  It assigns floor(total/3) updates
to Stage 1 and Stage 2, with the remainder assigned to Stage 3; all stages need
at least one update.  Validation and checkpoint restoration never count as
optimizer updates.  `debug_stage_epochs` is an explicit non-formal smoke mode.

Stage 1 skips EvidenceAdapter, BranchFusionTransformer and JointNormalDecoder
entirely because its loss contains only specialised branch terms.  Fusion,
Finetune and `detect_score` always execute the complete joint path.  Stage 3
uses the fixed `joint_finetune_lr_scale=0.1`, so its Adam learning rate is
`config.lr * 0.1`; the first two stages use `config.lr`.

`fit_mode="single_stage"` is retained for debugging only.  Its Fusion/Finetune
calls require a prior complete checkpoint and the already fitted scaler via
`detect_fit(..., previous_checkpoint=..., previous_scaler=...)` or
`load_stage_checkpoint`; otherwise the adapter raises instead of freezing
random branches.

## Formal Reproducibility Protocol

The formal TypeFusion-CATCH default is `seed=2021`, matching the CATCH benchmark
protocol.  Every formal command also passes `--seed 2021`, and its TypeFusion
model hyperparameters explicitly include `"seed": 2021`, because `detect_fit`
resets the model-level Python, NumPy, PyTorch CPU and available CUDA random
states.  cuDNN deterministic mode is enabled and cuDNN benchmarking is disabled
when fitting.

Formal runs use `fit_mode="three_stage"` with
`training_budget_mode="equal_total_steps"`.  The total number of optimizer
updates over the three stages equals the reference CATCH updates from the same
train loader and `catch_train_epochs`; checkpoint restoration and validation do
not add updates.  Stage 3 uses a fixed learning rate of `0.1 * lr`.  Repeated
CPU runs with the same configuration are covered by a small-data deterministic
test, but passing that test is a reproducibility property rather than evidence
that the model is effective.  There is currently no real-data performance
conclusion.

## CATCH Relationship

Retained from CATCH: benchmark defaults for sequence length, patch size/stride,
batch size, Adam learning rate, RevIN/FFT principles, CATCH-style frequency
patching and cross-channel Transformer principles, and the point-level
`detect_score` window protocol.

Modified: reconstruction heads are blockwise patch decoders and overlap-add,
not the original repeated high-dimensional Flatten heads.  Channel masking is
deterministic-group conditional reconstruction rather than CATCH's learned
channel adjacency mask.  Frequency reconstruction is a dedicated masked
pattern branch.

Not used: CATCH's direct single-path reconstruction score, frequency side score
addition, and its score-time threshold procedure.  The original
`ts_benchmark/baselines/catch/` code remains untouched.

## Parameters

The parameter count depends on the observed variable count `c_in`, which the
benchmark sets from the train frame.  With the CATCH-compatible default
configuration and `c_in=4`, TypeFusion-CATCH has `1,498,789` parameters:
shared stem `229,632`; State `70,592`; Evolution `69,508`; Pattern `402,208`;
Relation `150,657`; four evidence adapters `83,456`; BranchFusionTransformer
`282,752`; JointNormalDecoder `209,984`.  Under the same `c_in=4` CATCH default
configuration, original CATCH has `210,879,520` parameters.  These counts are
an architecture audit only.  The reduction is not evidence of performance,
speed, generalisation, or successful lightweight design; prior work indicates
that an extreme parameter reduction may severely damage performance, so real
data results are required.  Branches are separate module instances:
prototype memory/state decoder, causal TCN, frequency Transformer/decoder,
and masked relation encoder/decoder have no shared parameter objects.  The
shared CATCH stem is intentionally shared before branching.

## Current Limitations

- The model has implementation, random-tensor and small-DataFrame continuity
  validation only; no formal training or benchmark performance claim is made.
- The original causal test covered only `CausalEvolutionBranch`.  The current
  full-model train/eval tests also traverse `TypeFusionCATCHModel` and
  `SharedCatchStem`, verifying that target and future changes cannot alter the
  Evolution prediction at the target position.  Full-model train/eval relation
  masking tests likewise verify that a masked target value cannot affect its
  selected relation reconstruction.
- Frequency patches use right padding only when the configured CATCH grid does
  not exactly cover `seq_len`; the output is crop-restored to `seq_len`.
- Relation reconstruction uses `relation_mask_groups` simultaneous group views;
  more groups increase batch-axis activation memory.
- Evolution predictions at the first point use the fixed zero history token,
  because no earlier observation is available within a window.
- The present adapter supports score-based benchmark strategies through
  `detect_score`; it deliberately does not introduce a new label thresholding
  interface.
