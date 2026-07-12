# Direction A2 Development Protocol

Status: paused after four independently frozen A2-v2 model routes. The A2
question and its v2 falsification contract remain archived as negative
evidence; no fifth score model is authorized on that same task.

## Claim Boundary

A2 asks whether an observed local trajectory is compatible with normal
evolution from its event-pre observable state. It does not claim that anomaly
semantics must be a point residual, a tail probability, a reconstruction
error, a graph discrepancy, or a learned embedding distance.

The closed A-v1 claim was narrower: one conditional residual/innovation tail
could jointly explain normal volatility context and abrupt onset. A2 neither
reopens nor retunes that claim. Its archived evidence remains at
[`archive/direction_a/`](archive/direction_a/README.md).

## Task Contract

For an event boundary `t`, pre-event history `H`, and judged horizon `L`:

```text
P_t = X[t-H : t)       event-pre observable state
Y_t = X[t : t+L]       observed trajectory or state transition to judge
```

The A2 output is a compatibility assessment for `(P_t, Y_t)`. It may later be
implemented as a conditional trajectory likelihood, transition energy,
contrastive compatibility score, state-machine transition test, or another
pre-registered method. A point residual alone cannot establish the A2 claim.

The pre-event encoder/routing state must be a function of `P_t` only. A model
may inspect `Y_t` only in the branch that evaluates the candidate trajectory;
it must not use `Y_t` to construct the state against which that same trajectory
is judged.

When normal compatibility has observable heterogeneity, its reference tail and
outer threshold may be stratified only by a predeclared `P_t` reliability
feature. A2-M1 uses event-pre multichannel increment RMS with two equal-count
optimization-normal bins. It does not use B scores, hidden generator regime,
event role, onset metadata, or any `Y_t` value to route calibration.

## Required Synthetic Contract

The first A2 suite must be generated before choosing a model and must expose
generator-only certificates. Every paired episode uses an identical `P_t` and
identical endpoint `X[t+L-1]` within tolerance. Its distinction is trajectory
shape or transition compatibility, not terminal magnitude.

| Role | Construction | Required purpose |
| --- | --- | --- |
| Normal scheduled transition | A cue-conditioned, normal-rate state change from `P_t` to endpoint `e` | Normal positive control |
| Incompatible timing transition | Same `P_t`, endpoint `e`, profile-increment summaries, and transition magnitude, but a cue-incompatible onset | Primary A2 event |
| Normal coordinated transition | Normal target and driver evolution from one common state transition | Normal multivariate control |
| Unsupported transition | Same target endpoint or trajectory magnitude, but a transition incompatible with normal coordination | Secondary semantic case |
| No-event normal control | Normal continuation with no state transition | FPR control |

The generator must record pair ID, source state, endpoint tie error, trajectory
distance, normal transition support, and all seeds. Generator-only latent state
may validate the construction but must never be passed to the model,
normalizer, calibration map, threshold, or inference routing.

The timing cue is an explicit raw `P_t` signal, not hidden generator state: the
signed change in its declared channel over the final cue interval must recover
its two timing modes on every predeclared generator seed. At the same time, cue
alone and onset alone must each predict the primary normal/incompatible label
only at chance; the label resides in their compatibility relation.

The generated normal stream is time-partitioned before any future model choice:
`optimization=768`, guard `35`, `validation=384`, guard `35`, `reference=512`,
guard `35`, and `outer_calibration=384`. The guard is exactly `H + L - 1`.
Normal transition references may only originate in the reference range; a
future model cannot use reference or outer-calibration windows for fitting.
Each split also receives its own normal-trajectory bank containing scheduled,
coordinated, and no-event controls. This is necessary because normal evolution
is multimodal even when `P_t` is the same. A future candidate may fit only on
the optimization split's ordinary normal windows and transition bank, select
only on the validation equivalents, establish normal references only on the
reference equivalents, and set its outer threshold only on the
outer-calibration equivalents. The paired counterfactual episodes are held-out
test material.

## Pre-Registered Gates

No model is authorized until the suite itself passes its construction checks.
After model selection is frozen, the following gates apply per generator seed:

| Gate | Requirement |
| --- | --- |
| Event-pre isolation | Replacing any value in `Y_t` does not change the pre-event state/routing representation, up to `1e-7` |
| Cue observability | The predeclared raw `P_t` cue rule recovers timing mode with 100% accuracy and positive margin on every generator seed |
| Endpoint tie | Paired scheduled/incompatible-timing endpoints and prescribed pre-event windows tie up to `1e-7` |
| Primary ordering | Cue-incompatible trajectory outranks matched normal scheduled trajectory in at least `14/16` pairs with a predeclared positive median margin |
| Endpoint and trajectory-summary control | The observed paired windows, not merely an injected component, tie in endpoint, maximum increment, increment L1/L2, and onset marginal distribution; an A2 success cannot be attributed to a point residual, global slope, energy, or time-only onset rule |
| Normal transitions | Scheduled/coordinated/no-event normal controls each remain below their outer-normal threshold in at least `14/16` pairs |
| Background normal | FPR is at most `0.10` in every predeclared observable normal stratum; maximum stratum gap is at most `0.05` |
| Non-trivial normal skill | The selected model beats a predeclared history/trajectory-mean baseline on normal held-out trajectory prediction or compatibility ranking |
| Split provenance | Optimisation, validation, reference, outer calibration, and test are time-disjoint with an explicit `H + L - 1` guard; test scores/labels are absent from fitting and thresholding |

Exact pair counts and margin are provisional only until the generator length,
event horizon, and normal process are jointly frozen. They must be recorded
before the first model run and cannot be reduced after observing a model score.

## A2-M1 Implementation

A2-M1 is a GRU-conditioned mixture trajectory model. The GRU encoder consumes
only `P_t`; it outputs a mixture of full `L x D` future trajectories, per-mode
variance, and mixture weights. The A2 score is the candidate `Y_t`'s
full-trajectory mixture negative log likelihood, followed by a reference-only
upper-tail map. This is a conditional compatibility model, not a point residual
detector.

The model is implemented at
`ts_benchmark/baselines/A2TransitionCompatibility/` and run through
`scripts/a2/run_transition_compatibility.py`. It receives no role, cue mode,
onset, generator regime, or other generator metadata. Its four normal inputs
are time-isolated: ordinary normal windows plus the normal-transition bank for
optimization, validation, reference, and outer calibration respectively.

The initial development run at contract seed `5101`, model seed `6101`, and
outer alpha `0.10` passed event-pre isolation, primary timing ordering
(`16/16` positive pairs after ordinary-normal coverage was added), secondary
coordination ordering (`16/16`), background-normal FPR, and normal prediction
skill. It did not pass the no-event control (`13/16` below threshold, where
the gate is `14/16`). This is a failed development gate, not evidence for an
A2 claim. The next run must freeze any calibration decision and use new
contract seeds; no real benchmark or B comparison is authorized yet.

M1b was the separate alpha-`0.05` calibration variant in
`config/a2/trajectory_gru_m1b_alpha05.json`. Its frozen four-seed confirmation
is complete at `result/a2/m1b_confirmation_v1/`: contract seeds `5102..5105`
pair with model seeds `6202..6205`, and **0/4** runs pass all gates. Primary
positive-pair counts were `8/16`, `8/16`, `9/16`, and `16/16`; coordination
counts were `0/16`, `1/16`, `1/16`, and `16/16`. The fourth run also has an
observable-stratum background FPR of `21.17%` in one bin (gap `21.17` pp),
above the `10%` / `5` pp limits. The other three runs fail the paired-ordering
gates. Raw trajectory NLL has the same pairwise directions as the calibrated
tail score, and no primary pair crosses a reliability bin, so this is neither a
tail-map saturation artifact nor a routing artifact.

M1 is therefore frozen as a failed **conditional mixture-density
implementation** of A2. Do not rerun M1 with more seeds, a different alpha,
mixture count, GRU width, or the unconditional M1 ablation. This does not close
the A2 question: it rejects only the premise that normal pairwise compatibility
will be learned robustly as full-trajectory conditional density under this
contract.

## A2-M2: Contrastive Compatibility Energy

M2 is a separately named model family, not an M1 calibration edit. A
pre-event GRU encodes `P_t`; a second GRU encodes only the candidate future's
within-horizon increments `Y[t+1]-Y[t]`, which removes direct absolute level
and endpoint access. It trains on normal matched pairs `(P_t, Y_t)` and uses
other normal windows in the same minibatch as mismatched-pair negatives. Its
score is the event-pre/future contrastive energy, then the same reference-only,
event-pre-reliability-stratified tail and outer threshold as M1. An auxiliary
normal forecast head is used only for the existing normal-skill gate.

M2 cannot consume role, cue mode, onset, generator regime, endpoint metadata,
or any B output. It must first be evaluated only as a development run at
contract seed `5101`, model seed `6301`, with
`config/a2/contrastive_energy_m2_v1.json` and
`scripts/a2/run_contrastive_compatibility.py`. It may advance to a new frozen
multi-seed confirmation only if that development run passes every existing A2
gate. Its required control is
`config/a2/contrastive_energy_unconditional_m2_v1.json`, which replaces the
event-pre state with a learned constant while retaining the future-increment
encoder and every split/calibration rule. A real-data benchmark or B comparison
is not authorized before that confirmation.

The M2 development result is complete at
`result/a2/m2_development_seed6301/` and passes every gate: primary and
coordination ordering are both `16/16` with positive median tail margins
`4.4482` and `5.7071`; scheduled/coordinated/no-event controls are
`16/16`, `16/16`, and `14/16` below threshold; background FPR is `2.43%` and
`1.41%` across its two observable strata (gap `1.02` pp); the normal-forecast
MAE improves `10.53%` over persistence; and changing `Y_t` leaves the encoded
event-pre state exactly unchanged. The raw contrastive energy has the same
`16/16` primary and coordination directions as the calibrated tail, and all
primary pairs remain in the same reliability bin. The construction audit also
passes: cue-only and onset-only rules each remain at chance while their joint
mapping is exact.

The frozen event-pre ablation is complete at
`result/a2/m2_event_pre_control_seed6301.json`. Its only configuration change
was `model.condition_on_event_pre: true -> false`, with the same contract hash
and all remaining configuration fields fixed. The conditioned M2 remains a
complete pass; the unconditional control fails primary timing at `8/16` positive
pairs with median tail margin `-0.0670` (and fails overall). The automatic
decision artifact therefore passes its preregistered criterion. M2's
development signal depends on the event-pre/future relation, rather than on the
candidate future increments alone.

`scripts/a2/analyze_m2_ablation.py` enforces this decision: the contract hash
and all configuration fields other than experiment ID and that condition flag
must be identical, the conditioned result must retain every gate, and the
unconditional result must fail the **primary timing ordering** gate. This
criterion was fixed before the unconditional result was inspected.

### A2-v1 Confirmation Invalidation

The first M2 confirmation attempt must not be interpreted. Its v1 contract
used an **additive** raw cue: a fixed signed ramp was added to stochastic
background history. Consequently, cue observability was not guaranteed for
every predeclared generator seed. The runner wrote valid-looking artifacts for
`5106/6302` and `5107/6303`, then stopped before `5108` after that contract
failed its own cue-observability audit. The partial directory is retained as
`result/a2/m2_confirmation_v1/` with
`INCOMPLETE_INVALID_CONFIRMATION.json`; it is debugging/provenance only and
must not be pooled, reported, or read as an M2 confirmation result.

This is a **contract-generation defect**, not an M2 success or failure. The
v1 conditioned development and ablation results remain valid only for their
single v1 contract seed; they cannot establish cross-seed robustness because
the v1 contract itself is not seed-stable.

### A2-v2 Seed-Stable Contract

`config/a2/transition_contract_v2.json` replaces the additive cue with an
anchored overwrite over the declared cue interval. The predeclared raw rule,
last-minus-first cue-channel difference, is now exactly the configured
`-1.2/+1.2` up to its declared `5e-7` float32 amplitude tolerance for every
generated episode. The generator and audit record the maximum amplitude error
in addition to cue accuracy and margin. Contract-only preflight audits have passed for v2 development seed
`5110` and frozen future confirmation seeds `5111..5114`, each with cue
accuracy `1.0`, margin approximately `1.2`, and no violations.

Changing the cue encoding changes the data-generating contract, so M2-v1
model evidence does not transfer to v2. The required sequence is therefore:

1. Run M2-v2 development with contract seed `5110`, model seed `6310`, using
   `config/a2/transition_contract_v2.json` and
   `config/a2/contrastive_energy_m2_v2.json`.
2. If and only if it passes every existing gate, run its same-contract
   unconditional control with
   `config/a2/contrastive_energy_unconditional_m2_v2.json`, then evaluate it
   through `scripts/a2/analyze_m2_ablation.py`.
3. If and only if that v2 ablation passes, run the frozen confirmation at
   `config/a2/m2_v2_confirmation_v1.json`: contract seeds `5111..5114` pair
   with model seeds `6311..6314` through `scripts/a2/run_m2_confirmation.py`.

M2-v2 development is complete at
`result/a2/m2_v2_development_seed6310/` and passes every gate. Primary timing
and coordination ordering are both `16/16` with median tail margins `0.5487`
and `5.3209`; scheduled/coordinated/no-event controls are `15/16`, `16/16`,
and `14/16` below threshold; background FPR is `2.71%` and `5.48%` across its
two observable strata (gap `2.77` pp); and normal forecast MAE improves
`10.10%` over persistence. Event-pre isolation is exactly zero. Raw
contrastive energy has the same `16/16` directions as the tail score for both
primary and coordination pairs, and no primary pair crosses a reliability bin.
The embedded v2 contract audit is a full pass, including exact cue observability
within the declared float32 tolerance. M2-v2 may proceed to its matching
unconditional control, but not yet to confirmation.

The frozen M2-v2 event-pre control is complete at
`result/a2/m2_v2_event_pre_control_seed6310.json`. Its configuration and
contract hash exactly match the conditioned run except for
`model.condition_on_event_pre: true -> false`. The conditioned model remains a
complete pass; the unconditional control has **0/16** positive primary pairs,
median tail margin `-0.1353`, and fails overall. Its raw contrastive energy
has the same `0/16` primary direction, and no primary pair crosses a
reliability bin. The automatic ablation gate passes. This establishes, within
the v2 contract, that M2's timing signal is not available from candidate-future
increments alone.

### A2-M2-v2 Confirmation Result and Closure

M2-v2's frozen confirmation is complete at
`result/a2/m2_v2_confirmation_v1/`. All four contract seeds passed preflight:
each has cue accuracy `1.0`, margin approximately `1.2`, maximum cue-amplitude
error below the declared `5e-7` tolerance, and no construction violations. The
result is nevertheless **2/4 complete gate passes**, not a confirmation.

The relation signal itself is stable: every run has `16/16` positive primary
timing pairs and `16/16` positive coordination pairs, with positive median
tail margins. The failure is normal-score calibration/control stability:

| Contract/model seed | Failed gate | Evidence |
| --- | --- | --- |
| `5111/6311` | Background normal | high-reliability FPR `12.48%`; stratum gap `11.86` pp |
| `5112/6312` | Background normal and normal controls | FPR `14.05%` / `10.24%`; no-event normal control only `10/16` below threshold |
| `5113/6313` | None | complete pass |
| `5114/6314` | None | complete pass |

The two failed runs are not contract failures and not tail-map routing
artifacts: the paired raw contrastive energies have the same `16/16` ordering
directions as their tail scores, and every primary pair stays within one
reliability bin. Therefore M2-v2 demonstrates an event-pre/future relation
signal on this synthetic contract, but it does **not** demonstrate a stable
calibrated anomaly detector under the required normal-control gates.

M2 is closed as the A2 contrastive-energy route. Do not sweep outer alpha,
reliability-bin count, network width, temperature, forecast weight, more seeds,
or post-hoc threshold rules; those would be responses to this observed failure,
not confirmation. The result does not close the broader A2 question that
event-pre state can change the meaning of a future transition. It does close
the present claim that this normal-pair contrastive score plus a stratified
empirical tail establishes that question as a reliable anomaly detector.

The confirmation runner correctly audited every frozen contract seed before it
created an output directory and required the exact v2 ablation artifact. No
real-data result, B comparison, or M2 retune is authorized from this result.
Any future A2 work must be a separately named model hypothesis with a new
frozen candidate/configuration and must not be described as an M2 repair.

## A2-M3: Discrete Transition-Code Compatibility

M3 was a new A2 candidate rather than a modification of M2. Its
hypothesis is that normal future trajectories can be represented by a small
set of within-horizon transition codes, and that the event-pre state predicts
which normal code is allowed. A candidate is anomalous when its nearest normal
transition code is improbable under its event-pre state or lies far from all
normal code support.

The model observes only `P_t = X[t-H:t)` and a candidate future
`Y_t = X[t:t+L)`. The event-pre GRU produces code probabilities. A separate
GRU encodes internal candidate increments `Y[:,1:] - Y[:,:-1]`; its nearest
learned normal code defines the candidate label and normal-support distance.
The score is code surprisal plus normalized code-support distance. It neither
uses residuals, episode role, onset, cue mode, regime, driver identity, nor
generator metadata. A small forecast head is retained only for the existing
normal-skill gate, not as the anomaly score.

This differs from M2 materially: M2 evaluated a continuous event-pre/future
pair energy and used two volatility strata for tail calibration. M3 evaluates
a finite conditional code prediction plus distance to a finite normal support
set and uses one global reference stratum. Normal heterogeneity is represented
inside the codebook, not repaired afterwards by score bins. `M3` additionally
requires every one of its five predetermined codes to have at least eight
optimization windows; a collapsed codebook is a failed run even if other gates
pass.

The development configuration is frozen at contract/model seeds `5115/6320`:
`config/a2/transition_contract_v2.json` with a seed override and
`config/a2/transition_code_m3_v1.json`. Its unconditional control differs only
in `model.condition_on_event_pre: true -> false` and is frozen at
`config/a2/transition_code_unconditional_m3_v1.json`. Development may advance
only if every shared A2 gate and the code-coverage gate passes. Then, and only
then, run that matching unconditional control and
`scripts/a2/analyze_m3_ablation.py`; it passes only if the unconditional model
retains code coverage but fails primary timing ordering. Fresh confirmation
seeds are reserved as contract/model pairs `5116/6321` through `5119/6324`.

M3's frozen development run completed at
`result/a2/m3_v1_development_seed6320/` on contract/model seeds `5115/6320`.
The contract preflight passed: cue observability accuracy was `1.0`, cue margin
was approximately `1.2`, maximum cue-amplitude error was `1.07e-7`, and there
were no construction violations. The model did not pass all gates. Its five
optimization code occupancies were `[1021, 0, 0, 0, 0]`, so the required
minimum occupancy of eight failed for four codes; primary timing ordering was
only `8/16` positive pairs with median tail margin `-0.01143`. This is a
semantic collapse of the proposed finite-code representation, not a contract,
calibration, or implementation failure: event-pre isolation, normal skill,
all three normal controls, and background FPR (`4.55%`) passed; secondary
coordination ordering was `16/16` with median margin `6.1067`.

M3 is closed without hyperparameter sweeping. Do not run its unconditional
control, confirmation seeds, or code-count/loss/initialization/threshold
variants. The retained configuration, runner, and single result directory are
negative evidence and provenance only; no real-data or B comparison is
authorized from this route.

## A2-M4: Landmark And Direction Compatibility

M4 was a separately frozen A2 candidate. It tested an explicit structural
claim rather than learning a density, continuous pair energy, or latent code:
conditional on the event-pre state, a normal future has support for **where**
its strongest within-horizon change occurs and **which cross-channel direction**
that change takes. A candidate is scored by the surprisal of its change landmark
among nearby normal event-pre states plus its angular mismatch from normal
directions at that same landmark.

For a normalized candidate future `Y`, M4 computes internal increments
`d_i = Y[i+1] - Y[i]`, takes `tau = argmax_i ||d_i||`, and normalizes `d_tau`
to a direction. Its event-pre feature is only the final `P_t` state and the
last seven internal increments in `P_t`. Neighbors are retrieved only from the
time-disjoint reference-normal split. No role, cue mode, onset, regime, driver
identity, generator metadata, learned trajectory decoder, contrastive pair
embedding, learned codebook, residual score, or post-hoc reliability bin is an
input. The one global outer calibration is unchanged.

The frozen configuration is `neighbor_count=32`, `state_increment_length=8`,
`landmark_smoothing=1`, `direction_weight=1`, and `outer_alpha=0.05` in
`config/a2/landmark_direction_m4_v1.json`. Its sole development run is contract
and model seeds `5120/6330`, using the v2 contract with a seed override. The
matching unconditional control is
`config/a2/landmark_direction_unconditional_m4_v1.json`; it differs only in
`condition_on_event_pre: true -> false`. If and only if development passes all
shared A2 gates, run that control and `scripts/a2/analyze_m4_ablation.py`; the
control must fail primary ordering. Only then may `scripts/a2/run_m4_confirmation.py`
run the already frozen pairs `5121/6331` through `5124/6334` from
`config/a2/m4_confirmation_v1.json`; that file also records the distinct
development contract seed `5120` required to verify the ablation provenance.

M4 closes without a sweep if development fails any gate, the event-pre ablation
does not establish dependency, or frozen confirmation later has any failed run.
Do not respond to such a result by changing neighbor count, event-pre length,
landmark rule, direction weighting, calibration threshold, or seed list.

M4 development completed at `result/a2/m4_v1_development_seed6330/` on
contract/model seeds `5120/6330`. Every gate passed: primary timing was `14/16`
with median tail margin `1.5986`; secondary coordination was `15/16` with
median margin `0.5618`; all normal controls were `16/16` below threshold;
background FPR was `4.95%`; normal forecast MAE improved `5.96%` over
persistence; and event-pre isolation was zero. The reference support contained
all 11 possible internal-change landmarks, with counts from `35` to `87`.

Its matched event-pre ablation completed at
`result/a2/m4_v1_event_pre_control_seed6330.json`. It uses the identical
contract hash and every identical model field except
`condition_on_event_pre: true -> false`. The conditioned development run still
passes every gate, while the unconditional control has only `5/16` positive
primary pairs with median tail margin `0`; the frozen ablation decision passes.
The unconditional control's scheduled normal control also degrades to `10/16`
below threshold. Thus M4's development signal is not available from candidate
future landmarks and directions alone.

M4's frozen confirmation completed at
`result/a2/m4_v1_confirmation_v1/`. All four v2 contracts passed preflight
(cue observability accuracy `1.0`, margin about `1.2`, amplitude errors below
the declared `5e-7` tolerance, no violations), but only **1/4** runs passed all
gates. The failures are not background-calibration failures: global background
FPRs were `3.03%`, `4.45%`, `7.89%`, and `2.43%`, and all normal-control gates
passed. They are primary structural-ordering failures:

| Contract/model seed | Complete pass | Primary pairs | Secondary pairs | Failed gates |
| --- | --- | --- | --- | --- |
| `5121/6331` | yes | `14/16` | `14/16` | none |
| `5122/6332` | no | `11/16` | `15/16` | primary ordering |
| `5123/6333` | no | `12/16` | `11/16` | primary and secondary ordering |
| `5124/6334` | no | `13/16` | `14/16` | primary ordering |

The same deficiency exists in M4's raw landmark-direction score, before its
reference tail map: primary raw directions are `14/16`, `12/16`, `12/16`, and
`13/16` across these runs. M4 therefore has an event-pre-dependent development
effect but does not establish stable structural compatibility detection. M4 is
closed. Do not run new M4 seeds or tune neighbor count, state length, landmark
definition, direction weighting, tail rule, or threshold.

At this boundary, the current A2-v2 detector-development branch is paused:
M1 full-trajectory density, M2 continuous pair energy, M3 finite transition
codes, and M4 explicit landmark/direction support have each failed their own
frozen confirmation criterion. This does not negate the broader Direction A
motivation or the contract's observable relation (the contract audit remains
valid). It does mean that a fifth score proposal on this contract is not
authorized. Further Direction A work requires a separately justified task or
data contract, stated before selecting another detector family.

## Model Selection Boundary

The first implementation decision must compare a small number of complete
task-level candidates against the same suite, splits, parameter budget, and
gates. Candidate families may include:

```text
conditional trajectory forecasting or state-space models
segment-level masked/denoising transition models
event-pre state encoders with conditional compatibility energy
contrastive normal-transition representations
```

Architecture does not constitute the contribution. A candidate cannot advance
because it improves a generic reconstruction metric alone; it must pass the
matched endpoint/trajectory and normal-control gates. Frequency, graph, text,
or another external modality may be introduced only as a separately named
evidence source with strictly event-pre availability and a no-modality control.

## Relationship To B

B1's isolated temporal/cross repairs are retained baselines, not an A2
implementation. `R_T`, `R_C`, and `D` may be reported alongside A2 only after
their causal timestamps and aggregation are frozen. They must not be silently
folded into an A2 representation or threshold.

The generator specification, shortcut-control audit, A2-M1 implementation, and
tests now exist at
`config/a2/transition_contract_v1.json`,
`scripts/a2/generate_transition_contract.py`, and
`scripts/a2/audit_transition_contract.py`. The audit checks that endpoint and
observed-trajectory increment summaries tie, onset marginals tie, and only the cue-to-onset
mapping reverses in the primary pair. It additionally certifies that cue-only
and onset-only majority rules are each at chance, while their conditional
mapping determines the generated primary label. Construction and model tests are
in `tests/test_a2_transition_contract.py` and `tests/test_a2_transition_model.py`.
The contract and A2-M1 remain synthetic-development artifacts; no real
benchmark, B continuation, or A2 transfer claim is authorized yet.
