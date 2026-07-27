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

`X_standardized -> RevIN -> FFT/frequency patching/Trans_C channel fusion + local temporal
stem -> {State, Evolution, Pattern, Relation} -> EvidenceAdapter ->
BranchFusionTransformer -> leave-one-branch-out normal tokens ->
JointNormalDecoder -> X_hat_joint`

The shared encoder is information-preserving only.  It retains CATCH's RevIN,
full FFT, frequency patches and cross-channel Transformer idea.  Its compact
depthwise-separable temporal path prevents short state and local evolution
information from being discarded by frequency processing.  It is not a fifth
expert and has no anomaly score.

The four branch tasks are deliberately different:

- State: combines local temporal patches with shared temporal tokens, reads a
  top-k sparse set of learnable normal-state prototypes, then decodes only from
  that normal memory.  It targets spikes, bounds and level/state shifts.
- Evolution: directly right-shifts the adapter's train-data-StandardScaler
  input before a causal TCN.  It never reads full-window RevIN mean or variance.
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
default to three each and are never data-set or test-result dependent.

`fit_mode="single_stage"` is retained for debugging only.  Its Fusion/Finetune
calls require a prior complete checkpoint and the already fitted scaler via
`detect_fit(..., previous_checkpoint=..., previous_scaler=...)` or
`load_stage_checkpoint`; otherwise the adapter raises instead of freezing
random branches.

## CATCH Relationship

Retained from CATCH: defaults for sequence length, patch size/stride, model
width/depth, batch size, epochs, Adam learning rate, RevIN, FFT, frequency
patching, channel-fusion Transformer principles, and the point-level
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
an architecture audit, not a performance comparison.  Branches are separate module instances:
prototype memory/state decoder, causal TCN, frequency Transformer/decoder,
and masked relation encoder/decoder have no shared parameter objects.  The
shared CATCH stem is intentionally shared before branching.

## Current Limitations

- The model has implementation, random-tensor and small-DataFrame continuity
  validation only; no formal training or benchmark performance claim is made.
- The original causal test covered only `CausalEvolutionBranch`.  The current
  full-model test also traverses `TypeFusionCATCHModel` and `SharedCatchStem`,
  verifying that target and future changes cannot alter the Evolution prediction
  at the target position.
- Frequency patches use right padding only when the configured CATCH grid does
  not exactly cover `seq_len`; the output is crop-restored to `seq_len`.
- Relation reconstruction uses `relation_mask_groups` simultaneous group views;
  more groups increase batch-axis activation memory.
- Evolution predictions at the first point use the fixed zero history token,
  because no earlier observation is available within a window.
- The present adapter supports score-based benchmark strategies through
  `detect_score`; it deliberately does not introduce a new label thresholding
  interface.
