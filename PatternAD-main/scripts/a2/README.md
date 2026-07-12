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
