# Direction A3: Trigger-Response Graph Model

Status: A3-G1, A3-G2, and A3-G3 are closed after their single frozen
development runs. G3's GPU result is retained at
`result/a3/a3_g3_development_seed8301_gpu/`: all paired structure, normal-event
control, and past-only gates pass, but ordinary-background FPR is `12.55%`
(`92/733`), above the frozen `10%` gate. Direction A returns to model design;
there is no authorized A3 training run.

## Model Claim

An event is not merely an arbitrary pair of an event-pre window and a future
window. In many multivariate processes, an observable pre-event trigger
permits a structured response: selected variables respond, with a particular
delay and signed direction. A3 asks whether the **trigger-to-response graph**
observed in a candidate window is one normal mechanisms allow.

For an event-pre window `P_t` and future window `Y_t`, define a fixed,
observable response extractor:

```text
E(Y_t)_j = (active_j, onset_j, direction_j), for every channel j
```

`active_j` says whether channel `j` has a transition in the judged horizon;
`onset_j` is its first supported transition position; and `direction_j` is its
signed transition direction. These are extracted from raw values and fixed
before model fitting. They are not generator metadata, labels, learned codes,
or a reconstruction residual.

The A3 model is a factorized trigger-response graph decoder:

```text
P_t --past encoder--> h_t --graph decoder--> p(active_j, onset_j, direction_j | P_t)
Y_t --fixed extractor--> E(Y_t)
```

For each channel, the decoder predicts response presence, delay, and direction
from the full raw event-pre state. Candidate evidence is then evaluated as
three separately retained mechanism quantities:

```text
activation surprisal:  -log p(active_j | P_t)
delay surprisal:       -log p(onset_j | active_j, P_t)
direction surprisal:   -log p(direction_j | active_j, P_t)
```

The diagnostic result is the channel-by-component table, rather than a learned
weighted fusion into one opaque scalar. A later detection rule may use a
predeclared component appropriate to a particular violation type, but it may
not select or tune a combination after test results are observed.

## Why This Is Not A2-M5

A2 asks whether a full future trajectory is compatible with an event-pre
state, and its v2 contract deliberately ties endpoints and timing marginals.
M1 through M4 tested four ways of reducing that pair to one compatibility
score: conditional density, contrastive energy, discrete future codes, and
nearest-normal landmark/direction support. Those results close a fifth scoring
variant on that same task.

A3 changes the object that is modeled. It explicitly represents a transition
as a directed, multichannel response mechanism. The primary output is not a
trajectory NLL, pair energy, prototype assignment, or nearest-neighbor score;
it is a predicted response graph with observable node states. Its required
data construction contains an observed trigger and multiple normal response
modes, which A2 intentionally did not require.

This is therefore a different, falsifiable claim:

```text
Normal multivariate events have a stable, event-pre-observable
trigger-to-response structure that can be recovered without anomaly labels.
```

If it fails, the conclusion is about that structured-mechanism hypothesis, not
about Direction A's broader dynamic-semantic motivation.

## Frozen A3-v1 Synthetic Task

The first implementation will use six raw channels: a source-like channel,
one target-like channel, three possible responding channels, and a nuisance
channel. Channel names and generator roles are audit-only and are never passed
to the model. The model always receives all raw channels in `P_t` and predicts
tokens for all channels in `Y_t`.

There are two normal response modes. Each has a different observable trigger
in the end of `P_t`, and a different multichannel activation/delay/direction
pattern in `Y_t`. Normal no-trigger windows have no induced response. The
suite contains these paired cases:

| Pair | Same raw event-pre state? | Future construction | Purpose |
| --- | --- | --- | --- |
| Normal routed response / misrouted response | Yes | Normal uses the trigger's response mode; the paired candidate uses the other normal mode | Primary: wrong delay and propagation despite a normal-looking future mode |
| Normal routed response / partial propagation | Yes | Target follows the triggered mode; selected supporting channels follow the opposite mode or are absent | Secondary: response graph is broken while the main target response remains plausible |
| Normal no-trigger / untriggered response | Yes | One has no response; the other carries a normal response pattern | Activation control |

The primary pairs are balanced across the two modes. Thus trigger alone and
the complete future response mode alone each predict normal versus misrouted
at chance; the relation is necessary. The generator must audit this result
from raw values before any model is fitted. The first response extractor is
fixed to increment-energy thresholding, first supported onset, and signed
increment aggregation. Its thresholds are model configuration, not learned
from paired anomaly examples.

Ordinary normal windows and normal event windows are time-partitioned into
optimization, validation, reference, and outer-calibration regions with an
`H + L - 1` guard. Only normal windows may train the graph decoder or set
component reference tails. Paired mechanism violations are test-only.

## A3-G1 Model

`A3TriggerResponseGraph` contains:

1. A past-only GRU encoder over `P_t`.
2. Per-channel activation heads, then conditional onset and direction heads.
3. A fixed future token extractor used for normal self-supervision and
   candidate evaluation.
4. Reference-only calibration for each retained component, without an
   after-the-fact fusion weight.

Training uses normal windows only. It minimizes the token likelihood of the
fixed observable responses from those windows. Since tokens are fixed and
channelwise, this cannot fail through an unidentifiable learned codebook of
the kind observed in A2-M3. Since the output separates presence, delay, and
direction, it cannot hide a broken propagation pattern inside one global
landmark as A2-M4 did.

The unconditional ablation replaces `h_t` with a learned constant while
leaving the token extractor, decoder heads, normal splits, and calibration
fixed. It must fail the misrouted-response relation gate for A3-G1 to support
the claim that the past trigger matters.

The normal-only fitting set includes both ordinary normal continuation windows
and normal routed/no-trigger event windows from each frozen split. The response
likelihood is the **sum** of channelwise node terms for each component; an
unexpected activation in several nodes must accumulate rather than disappear
through an across-channel average. The implementation retains the component
terms before this aggregation for diagnosis.

### A3-G1 Development Result

The only A3-G1 development run is complete at
`result/a3/a3_g1_development_seed8101_implfix1/`, with frozen contract/model
seeds `7101/8101`. The construction audit passes: trigger-only and
response-mode-only primary classification are both exactly `0.5`, all `16/16`
raw trigger-response relations hold, and all `16/16` partial-propagation pairs
tie the target trajectory. Event-pre isolation is exactly zero.

The graph decoder recovers the three paired relations in that development
suite: misrouted delay is `15/16` with median tail margin `+3.6076`; partial
propagation direction is `15/16` with `+5.4589`; and untriggered activation is
`15/16` with `+3.0776`. This does **not** establish A3-G1, because its normal
controls fail: routed normal responses are below the delay threshold in only
`10/16` pairs. Its ordinary-background activation FPR is `10.23%`, above the
predeclared `10%` limit (delay and direction are `6.68%` and `6.82%`).

A3-G1 is consequently closed as a **factorized GRU graph-decoder route**. It
does not receive an event-pre ablation, confirmation seeds, a threshold edit,
larger network, alternative optimization, or a real-data run. Its failure is
not evidence against the A3 mechanism question: the paired relation scores
are strong, but independent node heads do not provide stable joint normal
graph calibration.

An earlier G1 engineering probe omitted ordinary normal continuation windows
from fitting/calibration and averaged node terms. It was invalid for model
decisions and has been removed; it is not included in the G1 conclusion above.

## Candidate A3-G2: Observable Graph Grammar

G2 is proposed as a distinct model family, not a G1 capacity/calibration
variant. Its hypothesis is that a normal trigger selects a **joint response
grammar**, rather than independent channel node distributions. A fixed raw
event-pre extractor accepts only a terminal cue interval that is sufficiently
linear and has sufficient signed amplitude; it selects the strongest qualifying
channel and otherwise emits a no-trigger state.
The model does not receive a declared source channel, cue mode, or generator
role. It then learns a smoothed normal conditional distribution over complete
observable response-graph signatures:

```text
q(P_t) = (inferred trigger channel, sign, no-trigger flag)
G(Y_t) = joint vector of each channel's (active, onset, direction)
p(G(Y_t) | q(P_t))
```

The score is the conditional grammar surprisal of the whole response graph;
diagnostic output still decomposes which nodes disagree with the accepted
normal grammar. This differs from A2-M3 because both `q` and `G` are fixed,
raw observable states, not a learned future codebook; it differs from G1
because it preserves dependencies among response nodes instead of summing
independent heads.

G2 is implemented under
`ts_benchmark/baselines/A3TriggerResponse/A3ObservableGraphGrammar.py`, with
its raw audit in `scripts/a3/audit_observable_graph_grammar.py` and its runner
in `scripts/a3/run_observable_graph_grammar.py`. The autoregressive decoder
conditions each channel node on the preceding fixed observable node tokens, so
it represents a joint graph grammar rather than G1's conditionally independent
heads. It has no learned future codebook.

The fixed trigger extractor and raw graph representation passed CPU regression
and preflight audits on contract seeds `7101..7105`: each audit retains
trigger-only and response-mode-only primary accuracy at `0.5`, recovers every
routed trigger, and rejects ordinary normal continuation windows. Its single
frozen GPU development run is complete at
`result/a3/a3_g2_development_seed8201/` with contract/model seeds `7101/8201`.
All three structure relations pass at `16/16`: misrouted median margin
`+4.1744`, partial-propagation `+1.5141`, and untriggered response `+2.8729`.
Normal routed/no-trigger controls also pass at `15/16` and `14/16`. However,
ordinary-background FPR is `11.60%`, above the predeclared `10%` limit. G2 is
therefore closed. Do not run an ablation, confirmation, retune, threshold
change, or real-data experiment for G2.

## Candidate A3-G3: Counterfactual Effect-Graph Grammar

G3 is a separate response representation, not a G2 calibration or capacity
variant. G1 and G2 encode raw future displacement as a response node. Their
paired mechanism gates succeed, but ordinary stochastic movement can itself
enter that raw graph and inflate the background FPR. G3 represents a response
as an effect beyond the normal continuation predictable before the boundary.

It fits two normal-only, past-only objects: a ridge continuation map from the
complete raw `P_t`, and a fixed-state normal response template. Both use only
time-disjoint normal optimization windows:

```text
q(P_t)       = fixed raw trigger state
B(P_t)       = ridge_continuation(P_t) + normal_template[q(P_t)]
R(P_t, Y_t)  = Y_t - B(P_t)
G_eff        = fixed (active, onset, direction) graph of R(P_t, Y_t)
p(G_eff | q(P_t)) = autoregressive joint effect-graph grammar
```

The final evidence is complete effect-graph surprisal. G3 neither thresholds
nor fuses scalar forecast residual magnitude; terminal effect norm is
diagnostic-only, while the scored object preserves onset and sign topology.
The continuation and template cannot read `Y_t` at inference. Thus normal
background and normal routed responses should map to near-null effect graphs,
directly addressing G1/G2's shared background-calibration failure.

The frozen implementation is `A3CounterfactualEffectGraph.py`, config is
`config/a3/counterfactual_effect_graph_g3_v1.json`, raw audit is
`scripts/a3/audit_counterfactual_effect_graph.py`, and runner is
`scripts/a3/run_counterfactual_effect_graph.py`. It fixes the contract token
threshold at `0.60`, contract/model seeds at `7101/8301`, ridge penalty,
grammar capacity, and calibration alpha before development. CPU smoke verified
raw audit, fitting/reference/outer-calibration, all gate computations, and
that replacing `Y_t` changes neither trigger state nor counterfactual
baseline. It is not a development result.

### A3-G3 Development Result

The one frozen GPU development run is complete at
`result/a3/a3_g3_development_seed8301_gpu/`, using contract/model seeds
`7101/8301` and the exact frozen configuration recorded in its summary and
checkpoint. The raw audit passes; past-only trigger and baseline differences
are both exactly zero. The three paired effect-graph gates pass at `16/16`:
misrouted median tail margin `+4.0422`, partial-propagation `+1.9176`, and
untriggered response `+3.7939`. Normal routed/no-trigger controls are also
within gate at `16/16` and `15/16`.

However, ordinary-background FPR is `92/733 = 12.55%`, exceeding the frozen
`10%` maximum. Thus the counterfactual effect representation did not repair
the shared calibration failure: it preserves mechanism separation but cannot
stably distinguish ordinary background effects under this contract. G3 is
closed. Do not run an ablation, confirmation, retune, threshold edit, feature
probe, or real-data experiment for G3.

## Current Model Work

The current work is model design, not dataset preparation or a chain of small
experiments. The independent-background FPR protocol in
[`A3_BACKGROUND_CALIBRATION_V2.md`](A3_BACKGROUND_CALIBRATION_V2.md) is now
frozen and CPU-audited; it is an evaluation contract, not a fourth detector.
The route-identifiability successor contract in
[`A3_ROUTE_IDENTIFIABILITY_V2.md`](A3_ROUTE_IDENTIFIABILITY_V2.md) now resolves
the measured A3-v1 route/background collinearity without reopening closed
models. N1's background-nulling mechanism proposal is in
[`A3_N1_BACKGROUND_NULLING_PROPOSAL.md`](A3_N1_BACKGROUND_NULLING_PROPOSAL.md);
its normal-only preflight, CPU smoke, frozen GPU development, and past-free
trigger control all pass. The control drops the primary relation from `16/16`
to `13/16`, establishing its necessary event-pre dependency. The only
permitted next action is the pre-registered four-pair CUDA confirmation in
`config/a3/background_nulling_n1_confirmation_v1.json`.

Working discipline is deliberately proportionate, rather than either a large
test campaign or an unchecked one-shot run:

1. One new mechanism-level model claim, one implementation, and one frozen
   development run.
2. Before that run, retain only the checks that decide whether the model result
   is interpretable: raw mechanism construction, past-only state isolation,
   and one CPU end-to-end smoke through fitting and calibration.
3. A failed development run closes that model. A passing one receives its
   necessary event-pre ablation, then a confirmation decision. There are no
   feature sweeps, threshold searches, or exploratory seed probes in between.

This keeps the work concrete: every model has enough evidence to be trusted or
rejected, while avoiding a proliferation of small score-level experiments.

The invalid first G1 engineering output and the closed G1 implementation have
been removed. Corrected G1, G2, and G3 results remain negative evidence.

## Pre-Registered Gates

Before model development, the generator audit must establish all of these from
raw data only:

1. paired event-pre equality, token-extractor determinism, and time-split
   provenance;
2. primary trigger-only and response-mode-only label prediction at chance,
   while their raw relation is exact;
3. normal routed, no-trigger, and background windows have valid response
   tokens;
4. the target response is tied in the partial-propagation pair, so a
   target-only score cannot solve the secondary case.

For a frozen model, required gates are:

1. replacing `Y_t` cannot change the past encoding;
2. misrouted responses exceed their matched normal response in the predeclared
   delay/propagation component in at least `14/16` pairs with positive median
   margin;
3. partial-propagation responses exceed their normal pair in the predeclared
   activation/direction component in at least `14/16` pairs with positive
   median margin;
4. normal routed, no-trigger, and ordinary background controls respect each
   component's reference-only threshold, with per-component FPR at most `0.10`;
5. the past-free ablation fails the primary relation gate; and
6. a development pass advances once to four frozen contract/model seed pairs.

No real benchmark, multi-modal claim, Direction B input, score fusion sweep,
or model-capacity retuning is authorized until the synthetic development,
ablation, and frozen confirmation sequence has completed.

## Implementation Boundary

All A3 artifacts belong only in these locations:

```text
config/a3/
scripts/a3/
ts_benchmark/baselines/A3TriggerResponse/
result/a3/
research/direction_a/A3_EXPERIMENT_PLAN.md
```

This keeps A3 separate from A2 and from Direction B. The immediate work is to
state a new mechanism claim under the frozen independent-background contract;
it is not detector training, a calibration sweep, or a dataset-cleaning task.
