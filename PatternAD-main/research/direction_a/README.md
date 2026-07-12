# Direction A2

A2 studies whether an observed future trajectory is compatible with its
event-pre observable state. It is independent of the closed A-v1 conditional
residual route and of the B repair models.

| Area | Location |
| --- | --- |
| Active protocol, M1 result, and M2 design | [`A2_EXPERIMENT_PLAN.md`](A2_EXPERIMENT_PLAN.md) |
| Contract and model configs | [`../../config/a2/`](../../config/a2/) |
| Generator, audit, and model runners | [`../../scripts/a2/`](../../scripts/a2/) |
| Contract and model tests | [`../../tests/test_a2_transition_contract.py`](../../tests/test_a2_transition_contract.py), [`../../tests/test_a2_transition_model.py`](../../tests/test_a2_transition_model.py) |
| A2 model implementations | `../../ts_benchmark/baselines/A2TransitionCompatibility/` |
| A2 model runners | `../../scripts/a2/run_transition_compatibility.py`, `../../scripts/a2/run_contrastive_compatibility.py` |
| Future artifacts | `../../result/a2/` |

A2-M1's four-seed confirmation failed (`0/4` complete passes). M2 passed
single-seed development and its event-pre ablation, but its valid A2-v2
four-seed confirmation passed only `2/4` complete gates because normal-score
calibration was unstable. M2 is closed as an A2 model route. The partial v1
confirmation is separately invalid because that earlier additive-cue contract
was not seed-stable. Neither route has a real-data result or transfer claim.
