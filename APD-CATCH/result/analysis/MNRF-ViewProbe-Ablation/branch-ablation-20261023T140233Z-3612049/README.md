# MNRF Frozen-View Ablation Summary

Batch: `branch-ablation-20261023T140233Z-3612049`

This package keeps one row for every planned dataset/view/head-seed cell. Missing combinations are not filled with zero. `long_table.csv` is independently recalculated from each saved score vector and matched source labels. `paired_changes.csv` contains same-task, same-seed point and anomaly-segment deltas. `paired_point_scores.npz` contains scored vectors, labels, global/local coordinates, validity masks and thresholds for complete or reused cells only.

For E/P/R, addition is `metric(S+branch)-metric(S)` and deletion is `metric(FULL)-metric(FULL-minus-branch)`. Positive deletion means removal hurt. S has deletion only. AP is average precision, not Aff-F. Segment hits mean at least one predicted point and are diagnostic only.

Head seeds vary GMM initialization on frozen features; they are not encoder seeds. These results do not prove causal representation contribution, paper-protocol equivalence, or superiority to CATCH. Full caches, encoders and GMM files stay in their server source runs.
