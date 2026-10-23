# MNRF Frozen-View Ablation Summary

Batch: `branch-ablation-20261023T173741Z-2`

Batch status: `failed`. Coverage: 0/8 complete datasets, 4/72 validated cells. Missing combinations are not filled with zero. `long_table.csv` is independently recalculated from each saved NLL and matched source labels. `paired_changes.csv` contains same-task, same-seed point and anomaly-segment additions/losses; bounded examples are in `change_cases.json`. `paired_point_scores.npz` contains only cells validated in this summary invocation.

For E/P/R, addition is `metric(S+branch)-metric(S)` and deletion is `metric(FULL)-metric(FULL-minus-branch)`. Positive deletion means removal hurt. S has deletion only. AP is average precision, not Aff-F. Segment hits mean at least one predicted point and are diagnostic only.

Head seeds vary GMM initialization on frozen features; they are not encoder seeds. These results do not prove causal representation contribution, paper-protocol equivalence, or superiority to CATCH. Full caches, encoders and GMM files stay in their server source runs.
