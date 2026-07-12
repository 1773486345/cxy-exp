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

The output records normal streams, all episode windows, exact pair ties, and
generator-only support certificates. It is the sole A2 artifact authorized at
this stage.
