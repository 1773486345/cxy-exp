# A3-v2 Independent Background Calibration Contract

Status: CPU contract audit passes. This is an evaluation-protocol decision,
not a detector, a G4 result, or a reopening of A3-v1 G1/G2/G3.

## Motivation

A3-v1 evaluated ordinary-background FPR on 733 overlapping `H + L` windows
from one 768-point normal stream. G1, G2, and G3 all separated paired response
mechanisms but failed the same background gate. Overlap makes the effective
sample size and uncertainty of that background estimate unclear, so a future
model cannot claim stable calibration using that population alone.

The frozen A3-v2 protocol changes only the population used to evaluate a
future model's ordinary-background false alarms. It retains A3-v1's normal
process equations, history/horizon, trigger extractor, and operating target.
It does not change any prior result, threshold, model, or anomaly pair.

## Independent Population

The base contract is `a3_trigger_response_contract_v1`. For each declared
normal noise scale, A3-v2 spawns 1,024 distinct `SeedSequence` children. Each
child generates one stationary fixed-regime trajectory through a 96-step
burn-in plus exactly one `H + L = 36` window. No two evaluation windows share
a raw stream, a time range, or a PRNG child stream.

```text
regime 0: 1,024 independent blocks, scale 0.80
regime 1: 1,024 independent blocks, scale 1.35
pooled:   2,048 independent blocks
```

The audit checks finite shape, exact regime balance, unique source IDs, unique
spawn keys, unique raw windows, and that the frozen raw trigger extractor does
not accept any ordinary event-pre block. Its successful CPU audit reports all
2,048 checks unique and zero fixed-trigger false acceptances.

## Frozen FPR Rule

The target remains `FPR <= 0.10`; it is not relaxed because G3 observed
`12.55%`. A future detector's scores must be produced without reading this
background evaluation population. The decision then requires both:

1. The pooled one-sided 95% Wilson upper bound is at most `0.10`.
2. Each fixed-noise-regime one-sided 95% Wilson upper bound is at most `0.10`.

With 2,048 pooled blocks, this permits at most 182 exceedances (`8.89%`)
because its upper bound is `9.98%`. The per-regime rule is evaluated separately
over 1,024 blocks. This headroom is intentional: an observed point FPR of
exactly 10% is not evidence that the underlying FPR is at most 10%.

## Decision Boundary

This protocol is a prerequisite for, not evidence for, a future detector. G1,
G2, and G3 remain closed and must not be rescored on A3-v2. No GPU run is
authorized by this contract. A future model must first state a mechanism that
separates ordinary stochastic innovations from triggered response effects, then
freeze its own optimization/calibration partitions and evaluate once against
this independent background population.

Run the contract audit only with:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a3/audit_independent_background_contract.py \
  --base-contract-config config/a3/trigger_response_contract_v1.json \
  --protocol-config config/a3/independent_background_calibration_v2.json
```
