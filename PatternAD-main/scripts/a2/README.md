# Direction A2 Contract Tools

`generate_transition_contract.py` is a generator-only tool. It produces the
matched event-pre/endpoint trajectory contract defined in
[`../../research/direction_a/A2_EXPERIMENT_PLAN.md`](../../research/direction_a/A2_EXPERIMENT_PLAN.md); it does not fit a
model, derive a score, calibrate a threshold, or make a detection claim.

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a2/generate_transition_contract.py \
  --config config/a2/transition_contract_v1.json \
  --output-dir result/a2/transition_contract_v1_seed5101
```

The output records normal streams, a normal cue-conditioned transition
reference bank, all episode windows, exact pair ties, and generator-only
support certificates. It is the construction artifact used by the separate
A2-M1 model runner below.
The normal stream persists its frozen optimization, validation, reference, and
outer-calibration ranges. Each range has its own normal bank with scheduled,
coordinated, and no-event trajectories; reference windows remain restricted to
the reference range. A2 models also receive all ordinary-normal windows that
fit wholly inside their assigned range.

Audit the construction without producing an A2 score or result artifact:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a2/audit_transition_contract.py \
  --config config/a2/transition_contract_v1.json
```

The audit rejects hidden/unobservable cues, endpoint, observed-trajectory-summary,
global-onset, cue-only, and coordination shortcuts. It does not assess a detector.

`run_transition_compatibility.py` is the separate A2-M1 model runner. It fits a
GRU-conditioned mixture over whole future trajectories and scores a candidate
trajectory by mixture negative log likelihood. It never accepts role, cue-mode,
or onset metadata as model input.

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a2/run_transition_compatibility.py \
  --contract-config config/a2/transition_contract_v1.json \
  --experiment-config config/a2/trajectory_gru_v1.json \
  --output-dir result/a2/a2_m1_seed6101
```

M1b's fixed four-seed confirmation is complete at
`result/a2/m1b_confirmation_v1/` and failed (`0/4` complete passes). Its
trajectory-density route is frozen; it is not an invitation to sweep alpha or
GRU settings.

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a2/run_m1b_confirmation.py \
  --output-root result/a2/m1b_confirmation_v1 \
  --torch-threads 1
```

`run_contrastive_compatibility.py` is the separate A2-M2 development runner.
It learns whether a normal candidate future's *within-horizon increments* are
compatible with `P_t`, from matched normal pairs and mismatched normal-pair
negatives only. It does not reuse M1's trajectory-density score.

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a2/run_contrastive_compatibility.py \
  --contract-config config/a2/transition_contract_v1.json \
  --experiment-config config/a2/contrastive_energy_m2_v1.json \
  --output-dir result/a2/m2_development_seed6301 \
  --torch-threads 1
```

M2 is development-only until one run passes every existing gate. Its
no-event-pre control is deliberately separate at
`config/a2/contrastive_energy_unconditional_m2_v1.json`.

The conditioned M2 development run at contract/model seed `5101/6301` passed
all gates. Run its frozen event-pre control next; only the condition flag is
different, and the output directory is separate from both M1b and M2
development artifacts:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a2/run_contrastive_compatibility.py \
  --contract-config config/a2/transition_contract_v1.json \
  --experiment-config config/a2/contrastive_energy_unconditional_m2_v1.json \
  --output-dir result/a2/m2_unconditional_control_seed6301 \
  --torch-threads 1

/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a2/analyze_m2_ablation.py \
  --conditioned-summary result/a2/m2_development_seed6301/summary.json \
  --unconditional-summary result/a2/m2_unconditional_control_seed6301/summary.json \
  --output result/a2/m2_event_pre_control_seed6301.json
```

The ablation passes only when conditioned M2 still passes every gate and the
unconditional control fails the primary timing-ordering gate. It passed at
`result/a2/m2_event_pre_control_seed6301.json`: removing `P_t` reduced primary
ordering to `8/16` with negative median margin. This single-seed v1 ablation
would have authorized v1 confirmation, but its later cross-seed contract audit
revealed that the v1 additive cue was not seed-stable.

The former `m2_confirmation_v1` directory is invalid and incomplete: its v1
additive cue failed the contract audit on a later frozen seed. It is retained
with an explicit marker for provenance only. Do not rerun or interpret it.

A2-v2 uses an anchored raw cue whose declared raw difference is stable to its
`5e-7` float32 tolerance for every seed. Because that
changes the contract, rerun M2 development first:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a2/run_contrastive_compatibility.py \
  --contract-config config/a2/transition_contract_v2.json \
  --experiment-config config/a2/contrastive_energy_m2_v2.json \
  --output-dir result/a2/m2_v2_development_seed6310 \
  --torch-threads 1
```

Only if v2 development passes every gate, run the matching unconditional
control and `analyze_m2_ablation.py` with the two v2 summaries. If that v2
ablation passes, confirmation is predeclared with fresh seeds `5111..5114`:

M2-v2 development passed all gates at
`result/a2/m2_v2_development_seed6310/`. Run the only authorized next step:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a2/run_contrastive_compatibility.py \
  --contract-config config/a2/transition_contract_v2.json \
  --experiment-config config/a2/contrastive_energy_unconditional_m2_v2.json \
  --output-dir result/a2/m2_v2_unconditional_control_seed6310 \
  --torch-threads 1

/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a2/analyze_m2_ablation.py \
  --conditioned-summary result/a2/m2_v2_development_seed6310/summary.json \
  --unconditional-summary result/a2/m2_v2_unconditional_control_seed6310/summary.json \
  --output result/a2/m2_v2_event_pre_control_seed6310.json
```

M2-v2's preflighted confirmation is complete at
`result/a2/m2_v2_confirmation_v1/`: only `2/4` runs passed every gate. All
four runs preserve `16/16` primary and coordination ordering, but normal-score
calibration failed on two seeds: one has `12.48%` high-stratum FPR and the
other has `14.05%` / `10.24%` FPR plus only `10/16` no-event controls below
threshold. M2 is closed. Do not rerun this command with calibration or model
variants; the artifacts are retained as negative evidence.

`run_transition_code_compatibility.py` was the separate A2-M3 development
runner. It predicted a finite normal within-horizon transition code from the
event-pre state, then scored a candidate with code surprisal plus distance to
normal code support. It used one global calibration stratum; the codebook,
rather than post-hoc bins, represented normal transition heterogeneity. The
five-code support coverage requirement was an additional mandatory gate.

Its only authorized development run used fresh contract/model seeds `5115/6320`:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a2/run_transition_code_compatibility.py \
  --contract-config config/a2/transition_contract_v2.json \
  --contract-seed 5115 \
  --experiment-config config/a2/transition_code_m3_v1.json \
  --output-dir result/a2/m3_v1_development_seed6320 \
  --torch-threads 1
```

The run completed at `result/a2/m3_v1_development_seed6320/` and failed: the
codebook occupancy was `[1021, 0, 0, 0, 0]` against a minimum of eight for each
code, and primary timing ordering was `8/16` with median tail margin `-0.01143`.
Its secondary coordination ordering, normal controls, global background FPR,
and normal-skill gates passed, but that does not override the failed primary
and code-coverage gates. M3 is closed. Do not run the unconditional control,
fresh confirmation, or any M3 variant; retain the runner and result only as
negative evidence.

`run_landmark_compatibility.py` is the distinct A2-M4 development runner. It
uses the event-pre terminal state plus seven event-pre increments to retrieve
time-disjoint normal reference neighbors. A candidate score combines the
surprisal of its strongest future-increment landmark with its direction mismatch
from matching-landmark neighbors. It is not an M1 density, M2 energy, or M3
codebook score; it uses one global calibration stratum.

The only authorized first M4 command is the frozen development run:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a2/run_landmark_compatibility.py \
  --contract-config config/a2/transition_contract_v2.json \
  --contract-seed 5120 \
  --experiment-config config/a2/landmark_direction_m4_v1.json \
  --output-dir result/a2/m4_v1_development_seed6330 \
  --torch-threads 1
```

The matching unconditional control and the four confirmation pairs are already
frozen. Development passed and its matched event-pre ablation passed: removing
the event-pre state reduced primary ordering to `5/16`. The four preflighted
confirmation runs are complete at `result/a2/m4_v1_confirmation_v1/`, but only
`1/4` passed all gates. Background FPR and normal controls pass on every seed;
primary structural ordering fails on three seeds at `11/16`, `12/16`, and
`13/16`, and the raw landmark-direction score has the same limitation. M4 is
closed. Do not run new M4 seeds or variants.
