# Direction A2 Pre-Model Protocol

Status: active definition. This document defines the question and the
falsification contract before selecting an A2 model family or running an A2
training job.

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

## Required Synthetic Contract

The first A2 suite must be generated before choosing a model and must expose
generator-only certificates. Every paired episode uses an identical `P_t` and
identical endpoint `X[t+L-1]` within tolerance. Its distinction is trajectory
shape or transition compatibility, not terminal magnitude.

| Role | Construction | Required purpose |
| --- | --- | --- |
| Normal gradual transition | A normal, temporally distributed state change from `P_t` to endpoint `e` | Normal positive control |
| Incompatible abrupt transition | Same `P_t` and endpoint `e`, but a jump/trajectory unavailable under the normal transition process | Primary A2 event |
| Normal coordinated transition | Normal target and driver evolution from one common state transition | Normal multivariate control |
| Unsupported transition | Same target endpoint or trajectory magnitude, but a transition incompatible with normal coordination | Secondary semantic case |
| No-event normal control | Normal continuation with no state transition | FPR control |

The generator must record pair ID, source state, endpoint tie error, trajectory
distance, normal transition support, and all seeds. Generator-only latent state
may validate the construction but must never be passed to the model,
normalizer, calibration map, threshold, or inference routing.

## Pre-Registered Gates

No model is authorized until the suite itself passes its construction checks.
After model selection is frozen, the following gates apply per generator seed:

| Gate | Requirement |
| --- | --- |
| Event-pre isolation | Replacing any value in `Y_t` does not change the pre-event state/routing representation, up to `1e-7` |
| Endpoint tie | Paired gradual/abrupt endpoints and prescribed pre-event windows tie up to `1e-7` |
| Primary ordering | Incompatible abrupt trajectory outranks matched normal gradual trajectory in at least `14/16` pairs with a predeclared positive median margin |
| Point-residual control | Endpoint residual/magnitude is tied; an A2 success cannot be attributed to a larger terminal deviation |
| Normal transitions | Gradual/coordinated/no-event normal controls each remain below their outer-normal threshold in at least `14/16` pairs |
| Background normal | FPR is at most `0.10` in every predeclared observable normal stratum; maximum stratum gap is at most `0.05` |
| Non-trivial normal skill | The selected model beats a predeclared history/trajectory-mean baseline on normal held-out trajectory prediction or compatibility ranking |
| Split provenance | Optimisation, validation, reference, outer calibration, and test are time-disjoint with an explicit `H + L - 1` guard; test scores/labels are absent from fitting and thresholding |

Exact pair counts and margin are provisional only until the generator length,
event horizon, and normal process are jointly frozen. They must be recorded
before the first model run and cannot be reduced after observing a model score.

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

The generator specification and generator-only unit tests now exist at
`config/a2/transition_contract_v1.json`,
`scripts/a2/generate_transition_contract.py`, and
`tests/test_a2_transition_contract.py`. The contract is the current review
artifact; no real benchmark, B continuation, or A2 neural model run is
authorized before it is explicitly frozen.
