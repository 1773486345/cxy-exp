# Gate v0 Interrupted Runs

This is an external audit record. It does not alter either interrupted directory and neither run is eligible for Gate statistics, partial metric selection, checkpoint reuse, or run-ID reuse.

## `decomp_gate_v0_20260717_cpu_01`

- Git commit: `9df9db758075513118d16a680f6812e580170d76`.
- Dirty working tree: staged Gate v0 report/data/runner/tests; two untracked legacy `CATCH-master` MSL result files; and untracked `CATCH-master/result/score/CATCH/SMAP/` (the complete verbatim value remains in `manifest.json`).
- Start timestamp: `2026-07-17T15:37:12` from `run.log`; the initial manifest timestamp is retained in that directory.
- Last manifest state: `running`.
- Last visible log: `seed=20260717: one original CATCH training run`.
- Present: `manifest.json`, `config.json`, `run.log`, empty `checkpoints/`, and empty `scores/`.
- Missing: checkpoint, checkpoint hash, normalization statistics, continuous scores, metrics, branch response matrix, bootstrap, and final Gate decision.
- Interruption point: seed `20260717`, original CATCH training.
- Cause: external execution-environment cleanup. A process terminated by `SIGKILL` cannot update its manifest.

## `decomp_gate_v0_20260717_cpu_02`

- Git commit: `9df9db758075513118d16a680f6812e580170d76`.
- Dirty working tree: the first run's entries plus untracked `APD-CATCH/result/decomposition_study_v0/` (the complete verbatim value remains in `manifest.json`).
- Start timestamp: `2026-07-17T15:42:23` from `run.log`; the initial manifest timestamp is retained in that directory.
- Last manifest state: `running`.
- Last visible log: `seed=20260717: one original CATCH training run`.
- Present: `manifest.json`, `config.json`, `run.log`, empty `checkpoints/`, and empty `scores/`.
- Missing: checkpoint, checkpoint hash, normalization statistics, continuous scores, metrics, branch response matrix, bootstrap, and final Gate decision.
- Interruption point: seed `20260717`, original CATCH training.
- Cause: external execution-environment cleanup. A process terminated by `SIGKILL` cannot update its manifest.

## Scientific Status

Both runs are `NOT_EVALUABLE`. They contribute zero `(seed, anomaly_type)` units to Gate v0. No partial result may be selected, and neither run ID may be reused. This is an infrastructure record, not a Gate failure and not evidence for or against the fixed reconstruction-error bands.
