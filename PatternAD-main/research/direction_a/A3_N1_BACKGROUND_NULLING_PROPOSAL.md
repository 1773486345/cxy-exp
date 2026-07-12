# A3-N1: Background-Nulling Route Graph

Status: raw preflight, one full CPU fitting/calibration smoke, frozen GPU
development, and the necessary past-free trigger ablation all pass. N1 now has
one remaining authorized stage: four frozen CUDA confirmation pairs.

## Claim

Ordinary multivariate background motion is dominated by a normal common-factor
innovation subspace, whereas a triggered response is a route in the orthogonal
complement. A detector should remove the normal background direction before it
forms a response graph, rather than treating raw endpoint displacement or a
conditional point-forecast residual as the response itself.

N1 estimates one all-channel background direction from normal optimization
increments only. It projects every future raw increment onto that direction's
orthogonal complement, then forms a joint trigger-conditioned route graph from
the projected trajectory. Generator response-channel roles are audit-only and
are not supplied to N1:

```text
B = first PCA direction of normal optimization increments
Delta Y_perp = (I - B B^T) Delta Y
q(P_t)       = fixed event-pre trigger state
G_perp       = fixed route graph extracted from Delta Y_perp
p(G_perp | q(P_t)) = future N1 joint route model
```

The final future detector would score a joint route graph, not a scalar
residual tail. This differs from G3: G3 subtracts a learned point continuation
and template, whereas N1 removes a normal stochastic subspace. It is only
meaningful on the route-identifiable contract: A3-v1's induced route was nearly
parallel to the very direction N1 would remove.

## Frozen Raw Preflight

Before detector code, the normal-only PCA preflight must verify all of these:

1. The first fitted all-channel normal increment factor has absolute cosine at
   least `0.90` with the known generator loading. The generator loading is
   audit-only and is never supplied to N1.
2. The declared induced route retains at least `0.95` of its norm after the
   fitted background projection.
3. The A3-v2 route contract and its independent background protocol pass their
   raw audits unchanged.
4. No anomaly label, paired candidate, or background evaluation block is used
   to fit the factor, choose its rank, choose a projection threshold, or change
   the route extractor.

The frozen preflight configuration is
`config/a3/background_nulling_n1_v1.json`. It passes with all-channel factor
alignment `0.99974` and route retention `0.99993`. The implementation in
`A3BackgroundNullingRouteGraph.py` uses this factor, a fixed raw trigger state,
and an autoregressive joint grammar over the projected all-channel route graph.
Its full CPU fitting/calibration smoke completes all paired, control, and 2,048
independent-background gate calculations without writing a result directory.

The development configuration is
`config/a3/background_nulling_n1_development_v1.json`: factor rank one,
all-channel projection, normal optimization-only factor fitting, and outer
calibration alpha `0.05`. The smaller alpha is fixed before development so the
independent 95% upper-bound FPR gate has statistical headroom; it is not a
post-hoc adjustment to G1/G2/G3.

The frozen development gates are past-only trigger isolation; at least `14/16`
positive matched margins with positive median for each of misrouting, partial
propagation, and untriggered response; normal routed/no-trigger controls; and
both pooled and per-regime independent-background one-sided Wilson upper bounds
at most `0.10`. A failure closes N1. Only a complete pass permits a frozen
past-free trigger ablation and then confirmation; it never authorizes a sweep,
G1/G2/G3 reuse, or a real-data study.

### N1 Development Result

The frozen GPU development run at
`result/a3/a3_n1_background_nulling_seed8401_gpu/` passes every gate. Primary,
partial-propagation, and untriggered margins are all positive in `16/16`
pairs, with medians `+4.2281`, `+4.2281`, and `+6.4329`. Normal routed and
no-trigger controls are each below threshold in `15/16` episodes. On the new
independent background population, pooled FPR is `84/2048 = 4.10%` with a
one-sided 95% upper bound `4.89%`; regime bounds are `1.13%` and `9.10%`, both
below the frozen `10%` limit. Checkpoint hashes, fitted factor, and all configs
match the summary exactly.

### N1 Past-Free Trigger Control

The GPU control at
`result/a3/a3_n1_past_free_control_seed8401_gpu/` uses the same seed, contract,
normal data, factor rank, calibration, and model settings as development; its
only change is `condition_on_event_pre=false`. Its primary misrouted route gate
falls to `13/16` positive pairs (below the frozen `14/16` minimum), with median
tail margin `+0.2644`, versus development's `16/16` and `+4.2281`. The required
ablation analysis at `result/a3/a3_n1_past_free_ablation.json` passes with no
configuration-hash violations. This establishes that N1's development result
depends on the available event-pre state; it is not a test-time future-state
substitute.

### Frozen Confirmation

`config/a3/background_nulling_n1_confirmation_v1.json` pre-registers exactly
four new contract/model pairs: `7202/8402`, `7203/8403`, `7204/8404`, and
`7205/8405`. Each retains the frozen development configuration, including
rank-one all-channel background nulling, event-pre conditioning, outer alpha
`0.05`, and the independent-background protocol; only the contract and model
random seeds change. `run_background_nulling_confirmation_pair.py` selects one
pair from that file and refuses its already-existing result directory. The
confirmation analyzer requires all four summaries to be CUDA runs with their
exact expected hashes and complete gate passes. Any incomplete pass closes N1;
the sequence does not authorize a seed replacement, retune, extra ablation, or
real-data study.
