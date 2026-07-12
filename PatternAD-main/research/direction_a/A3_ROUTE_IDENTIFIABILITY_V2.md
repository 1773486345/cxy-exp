# A3-v2 Route-Identifiability Contract

Status: CPU contract audit passes. This is a new task contract, not a detector
result and not a rerun of closed A3-v1 G1/G2/G3.

## Why A3-v1 Was Confounded

On its four response channels, A3-v1's normal latent loading is
`[0.88, 0.61, -0.56, 0.43]`. Its injected response route is
`[1.00, 0.78, -0.70, 0.42]`. Their absolute cosine is `0.99688`.

Consequently, an ordinary change in the normal latent state naturally creates
almost the same routed future graph as an induced event. This explains why
three distinct A3-v1 graph representations separated matched pairs but could
not stabilize ordinary-background calibration. It is an identifiability defect
of the task for a route-based claim, not evidence that every route model is
false.

## Frozen Successor Contract

The successor keeps the A3 process, splits, cues, response amplitude, ramp
duration, normal modes, paired counterfactuals, and response-mode balancing.
Only the response route changes to:

```text
r0 = [ 1.0, -1.0,  1.0,  0.6744186046511628]
r1 = -r0
```

For the same background loading, `dot(loading, r0) = 0` up to floating-point
precision and the measured absolute cosine is `2.33e-17`. Every route component
has magnitude at least `0.60`, so no weak component is hidden beneath the
existing response-token energy scale. A normal response mode and the opposite
misrouted mode remain marginally balanced; the trigger/future relation remains
necessary for the primary label.

The raw audit confirms `16` exact primary relations, target ties for all `16`
partial-propagation pairs, and chance (`0.5`) prediction from trigger alone or
future mode alone. Its paired independent-background protocol has 2,048 unique
fixed-regime normal blocks, zero fixed-trigger acceptances, and the frozen
pooled/per-regime Wilson decision rule.

## Boundary

The route change is not a post-hoc threshold adjustment: it repairs a measured
background/response collinearity while preserving the counterfactual relation
that defines A3. It cannot revalidate G1/G2/G3. A future detector must have a
new mechanism claim and be fitted only after raw preflight passes.

Audit it with:

```bash
/media/h3c/users/wangyueyang1/.env/envs/patternad_env/bin/python -B \
  scripts/a3/audit_route_identifiability_contract.py \
  --contract-config config/a3/trigger_response_route_identifiability_v2.json \
  --background-protocol config/a3/independent_background_route_v2.json
```
