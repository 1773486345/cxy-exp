# Direction A Archive

A-v1 is closed as of 2026-07-12. Its conditional residual-semantics claim
failed the predeclared paired mechanism gates and must not be presented as
evidence for Direction A2 or as the active PatternAD research path.

This directory retains the smallest useful reproduction record after the raw
development outputs were removed.

| Preserved item | Purpose |
| --- | --- |
| `p1_v2_holdout/` | P1-v2 run plan, frozen config inputs, generator logs, and aggregate gates/paired statistics. |
| `final_state_tail/` | Complete final causal state-tail diagnostic, including score arrays and evaluation JSON. |
| `source_snapshot/` | Snapshot of the uncommitted A-specific implementation, evaluator, tests, configs, and A documentation before active documentation moved to Direction B. |
| `legacy_entrypoints/` | The 24 former `PatternAD*.sh` convenience commands removed from the active script tree. |

The final state-tail diagnostic used generator seed `3101`, model seed `2021`,
and a 30-epoch budget. It passed normal FPR-gap control (`0.01873`) but failed
both primary ordering requirements: same-deviation `1/3` and abrupt-vs-gradual
`0/2`. P1-v2 also failed its paired A11-A01 and regime-FPR-improvement gates.

Removed after this archive was made:

- `result/patternad_synthetic/p1_contextual_calibrated_v2_holdout/` raw cell
  outputs (27 MB);
- `artifacts/patternad_synthetic/contextual_v1/` regenerated P1 inputs
  (23 MB);
- the previous live `dev_causal_delta_state_tail/` result copy.

The frozen inputs and source snapshot are sufficient to inspect the failed
protocol or regenerate its raw artifacts. The active A2 pre-model protocol is
[`../../research/direction_a/A2_EXPERIMENT_PLAN.md`](../../research/direction_a/A2_EXPERIMENT_PLAN.md);
B1 remains at
[`../../research/direction_b/B1_EXPERIMENT_PLAN.md`](../../research/direction_b/B1_EXPERIMENT_PLAN.md).
