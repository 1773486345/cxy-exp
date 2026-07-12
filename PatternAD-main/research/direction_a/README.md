# Direction A: A2 and A3

A2 studied whether an observed future trajectory is compatible with its
event-pre observable state. It is independent of the closed A-v1 conditional
residual route and of the B repair models. Its v2 detector-development branch
is paused after four model routes; no fifth A2 compatibility score is open.

| Area | Location |
| --- | --- |
| Active protocol and A2 model decisions | [`A2_EXPERIMENT_PLAN.md`](A2_EXPERIMENT_PLAN.md) |
| Contract and model configs | [`../../config/a2/`](../../config/a2/) |
| Generator, audit, and model runners | [`../../scripts/a2/`](../../scripts/a2/) |
| Contract and model tests | [`../../tests/test_a2_transition_contract.py`](../../tests/test_a2_transition_contract.py), [`../../tests/test_a2_transition_model.py`](../../tests/test_a2_transition_model.py) |
| A2 model implementations | `../../ts_benchmark/baselines/A2TransitionCompatibility/` |
| A2 model runners | `../../scripts/a2/run_transition_compatibility.py`, `../../scripts/a2/run_contrastive_compatibility.py`, `../../scripts/a2/run_transition_code_compatibility.py` |
| Future artifacts | `../../result/a2/` |

A2-M1's four-seed confirmation failed (`0/4` complete passes). M2 passed
single-seed development and its event-pre ablation, but its valid A2-v2
four-seed confirmation passed only `2/4` complete gates because normal-score
calibration was unstable. M2 is closed as an A2 model route. The partial v1
confirmation is separately invalid because that earlier additive-cue contract
was not seed-stable. M3's one frozen development run also failed: its finite
transition codebook collapsed to one occupied code and primary timing ordering
was only `8/16`; M3 is closed without a sweep. M4, the explicitly defined
landmark-and-direction support candidate, passed development and its event-pre
ablation but passed only `1/4` frozen confirmation runs, so it is also closed.
The A2-v2 detector-development branch is paused rather than opening a fifth
score variant. No A2 route has a real-data result or transfer claim.

A3 is the active separate Direction A model task: it models an observable
event-pre trigger and the joint multichannel response it permits. Its protocol
is [`A3_EXPERIMENT_PLAN.md`](A3_EXPERIMENT_PLAN.md); implementation is isolated
under `../../config/a3/`, `../../scripts/a3/`,
`../../ts_benchmark/baselines/A3TriggerResponse/`, and `../../result/a3/`.
A3-G1, a factorized GRU decoder for activity/delay/direction graph nodes,
closed at its single frozen development run: all three paired mechanism gates
were `15/16`, but routed-normal delay control was only `10/16` and activation
background FPR was `10.23%`. It does not advance to ablation or confirmation.
G2, the independently specified observable joint graph grammar, completed its
only frozen GPU development run. Its three paired structure gates are all
`16/16` and normal controls pass, but background FPR is `11.60%`, above the
frozen `10%` gate. G2 is closed without ablation, confirmation, retuning, or a
real-data claim. G3, the counterfactual effect-graph grammar, also completed
its single frozen GPU run. It passes raw audit, all three paired relations
(`16/16` each), normal-event controls, and past-only isolation, but fails the
ordinary-background FPR gate at `92/733 = 12.55%`. G3 is closed without
ablation, confirmation, retuning, or a real-data claim. See the concise record
in [`A3_EXPERIMENT_PLAN.md`](A3_EXPERIMENT_PLAN.md).

Before another A3 model is considered, the independent-background FPR protocol
in [`A3_BACKGROUND_CALIBRATION_V2.md`](A3_BACKGROUND_CALIBRATION_V2.md) replaces
A3-v1's overlapping background windows with 2,048 independent, regime-balanced
blocks and pooled/per-regime one-sided Wilson gates. It is contract-only work,
not a detector result or a way to reopen G1/G2/G3.

A further route-identifiability audit found that A3-v1's injected route had
absolute cosine `0.99688` with its normal latent loading. The successor
[`A3_ROUTE_IDENTIFIABILITY_V2.md`](A3_ROUTE_IDENTIFIABILITY_V2.md) makes the
route orthogonal while retaining the paired relation and independent-background
protocol. N1 background-nulling is the first proposal under that new contract;
its all-channel normal-only preflight, CPU smoke, and frozen GPU development
pass. Its one required past-free trigger control is now pending user execution.
