# Direction A3 Tools

Direction A3 is separate from A2 compatibility scoring and Direction B repair.
It models an observable event-pre trigger and the multichannel response graph
that normal mechanisms permit. Its full design and decision record is in
[`../../research/direction_a/A3_EXPERIMENT_PLAN.md`](../../research/direction_a/A3_EXPERIMENT_PLAN.md).

`generate_trigger_response_contract.py` and
`audit_trigger_response_contract.py` define and audit the shared A3-v1 raw
mechanism suite. They do not train a detector. The contract has two balanced
normal trigger-response modes plus misrouted, partial-propagation, and
untriggered counterfactuals. The raw audit verifies that a trigger alone and a
response mode alone each predict the primary label only at chance.

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a3/audit_trigger_response_contract.py \
  --config config/a3/trigger_response_contract_v1.json
```

G1's factorized graph decoder was closed after its only valid development run
at `result/a3/a3_g1_development_seed8101_implfix1/`: normal routed-response
delay control and activation FPR failed. Its source and invalid engineering
probe have been removed; do not recreate or rerun it.

`audit_observable_graph_grammar.py` is the model-independent audit for A3-G2.
G2's trigger state is a fixed raw terminal-cue shape: a channel must be both
sufficiently linear across the final cue interval and have sufficient signed
amplitude. The audit checks this extractor, raw trigger/response relation, and
ordinary-normal rejection before any G2 model is fitted.

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a3/audit_observable_graph_grammar.py \
  --contract-config config/a3/trigger_response_contract_v1.json \
  --experiment-config config/a3/observable_graph_grammar_g2_v1.json \
  --contract-seed 7101
```

G2's one frozen GPU development run is complete at
`result/a3/a3_g2_development_seed8201/`. Its three paired graph relations are
all `16/16` and normal controls pass, but its ordinary-background FPR is
`11.60%`, above the frozen `10%` limit. G2 is closed. Do not rerun it, add an
ablation, or change its model/calibration settings.

G3, the counterfactual effect-graph grammar, completed its one frozen GPU run
at `result/a3/a3_g3_development_seed8301_gpu/`. Its raw audit, past-only
isolation, all three paired relation gates (`16/16` each), and normal event
controls pass. Ordinary-background FPR is `92/733 = 12.55%`, above the frozen
`10%` limit, so G3 is closed. Its source, config, checkpoint, and summary are
retained as negative evidence. Do not rerun, edit, ablate, confirm, retune, or
apply G3 to real data.

`independent_background_calibration_v2.json` is a separate A3-v2 evaluation
contract. It creates 2,048 independent, regime-balanced ordinary-background
blocks and freezes pooled plus per-regime one-sided Wilson FPR bounds. It is
not a detector and must not be used to re-score closed G1/G2/G3 routes.

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a3/audit_independent_background_contract.py \
  --base-contract-config config/a3/trigger_response_contract_v1.json \
  --protocol-config config/a3/independent_background_calibration_v2.json
```

`trigger_response_route_identifiability_v2.json` is a successor task contract,
not a rerun of G1/G2/G3. It removes the measured route/background-factor
collinearity while retaining A3's paired relation. Audit it with:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a3/audit_route_identifiability_contract.py \
  --contract-config config/a3/trigger_response_route_identifiability_v2.json \
  --background-protocol config/a3/independent_background_route_v2.json
```

N1's development and the required past-free trigger control are complete. The
control has the same hashes and settings as development except
`condition_on_event_pre=false`; its primary gate drops from `16/16` to `13/16`,
so the ablation decision passes. The only authorized work is the four frozen
CUDA confirmation pairs in `background_nulling_n1_confirmation_v1.json`.
Run them sequentially as foreground jobs; each result directory is fixed and
will refuse an overwrite:

```bash
GPU_ID=0  # Set this only after checking that the selected GPU has usable headroom.
for pair_index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$GPU_ID" timeout --foreground --signal=TERM --kill-after=30s 1800s \
    /media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
    scripts/a3/run_background_nulling_confirmation_pair.py \
    --pair-index "$pair_index" \
    --torch-threads 1
done
```

After all four commands write their summaries, make the one no-overwrite
confirmation decision:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a3/analyze_background_nulling_confirmation.py \
  --confirmation-config config/a3/background_nulling_n1_confirmation_v1.json \
  --output result/a3/a3_n1_confirmation_v1.json
```
