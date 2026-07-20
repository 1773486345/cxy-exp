# Decomposition Applicability Report

This report is read-only: it does not train, infer, rescore, or invoke a benchmark runner.

## Fixed Formal Sources
- Explicit frozen source mapping used for 23/23 execution tasks; no archive was selected by mtime.
- GECCO CATCH uses CATCH_RSA_GECCO seq_len=192: PR 0.409311912, ROC 0.963459932; MSD uses the seq_len=192 total_score: PR 0.406495071, ROC 0.964381743.
- Source/config conflicts: none.
- Required fields not persisted by the frozen log/JSON sources: PSM, Genesis, GECCO, CalIt2, NYC, MSL; these are reported as partial verification, not silently substituted.

## ASD Loader Parity
- ASD parity exact match: 12/12; overall status: pass.
- Analysis labels and formal loader labels are compared for shape, dtype, anomaly count, element equality, and SHA-256 checksum.

## Drift Correction
```text
    descriptor  maximum_absolute_difference  tasks_with_rank_change  maximum_rank_change
    mean_drift                     0.263386                      19                    7
variance_drift                     0.211355                      17                    7
```
- mean_drift and variance_drift now use per-channel normalized changes before averaging adjacent-window pairs; legacy values are retained only for this comparison.

## Performance Groups
- Task PR groups before GECCO fair-source correction: {'neutral': 9, 'gain': 9, 'loss': 5}; after correction: {'neutral': 10, 'gain': 9, 'loss': 4}.
- ASD paper-level values are equal-weight macro means of its 12 execution tasks.
```text
paper_dataset  task_count  delta_auc_pr pr_group  delta_auc_roc roc_group
          PSM           1     -0.006205  neutral      -0.000287   neutral
      Genesis           1      0.044783     gain       0.012738      gain
        GECCO           1     -0.002817  neutral       0.000922   neutral
       CalIt2           1      0.011448     gain       0.011403      gain
          NYC           1     -0.017561     loss      -0.010377      loss
          MSL           1      0.008438  neutral      -0.001087   neutral
          ASD          12      0.013523     gain       0.016754      gain
       CICIDS           1     -0.000567  neutral      -0.070580      loss
   Creditcard           1      0.003627  neutral      -0.010000      loss
         SMAP           1     -0.015018     loss      -0.054415      loss
          SMD           1      0.000585  neutral      -0.013466      loss
         SWAT           1     -0.017759     loss      -0.015934      loss
```

## Candidate Correlations
```text
                     descriptor        target  task_rho  paper_rho  rho_without_asd_group  task_any_loo_sign_flip  paper_any_loo_sign_flip  task_any_group_loo_sign_flip  gain_median  loss_or_neutral_median  median_direction_consistent  qualified_for_pr  qualified_for_roc
                  channel_count  delta_auc_pr -0.245864  -0.150613              -0.068337                   False                     True                         False    19.000000               22.000000                         True             False              False
                  channel_count delta_auc_roc -0.531729  -0.616463              -0.619592                   False                    False                         False    19.000000               19.000000                        False             False              False
                     mean_drift  delta_auc_pr  0.515810   0.300699               0.090909                   False                    False                         False     0.516713                0.314910                         True             False              False
                     mean_drift delta_auc_roc  0.589921   0.706294               0.627273                   False                    False                         False     0.502579                0.361576                         True             False               True
                 variance_drift  delta_auc_pr  0.380435   0.223776              -0.009091                   False                     True                          True     0.405602                0.292440                         True             False              False
                 variance_drift delta_auc_roc  0.414032   0.153846              -0.090909                   False                     True                          True     0.399980                0.296224                         True             False              False
low_frequency_energy_ratio_mean  delta_auc_pr  0.096838  -0.195804              -0.418182                   False                     True                          True     0.741004                0.665404                         True             False              False
low_frequency_energy_ratio_mean delta_auc_roc  0.406126   0.419580               0.309091                   False                    False                         False     0.734633                0.678911                         True             False               True
               spectral_entropy  delta_auc_pr  0.068182   0.181818               0.309091                    True                    False                          True     0.324626                0.325347                        False             False              False
               spectral_entropy delta_auc_roc -0.190711   0.041958               0.136364                   False                     True                          True     0.324626                0.325347                         True             False              False
         periodicity_top3_ratio  delta_auc_pr  0.150198  -0.013986              -0.190909                   False                     True                          True     0.702703                0.556849                         True             False              False
         periodicity_top3_ratio delta_auc_roc  0.468379   0.531469               0.454545                   False                    False                         False     0.700986                0.626577                         True             False               True
              correlation_drift  delta_auc_pr -0.251976  -0.286713              -0.209091                   False                    False                         False     2.809017                4.085565                         True             False              False
              correlation_drift delta_auc_roc -0.591897  -0.594406              -0.554545                   False                    False                         False     2.444237                3.329085                         True             False               True
          trend_energy_over_raw  delta_auc_pr  0.054348  -0.160839              -0.227273                    True                    False                          True     0.685275                0.686963                        False             False              False
          trend_energy_over_raw delta_auc_roc  0.197628   0.258741               0.263636                   False                    False                         False     0.685554                0.675314                         True             False              False
       residual_energy_over_raw  delta_auc_pr -0.070158   0.251748               0.354545                    True                    False                          True     0.214326                0.197113                        False             False              False
       residual_energy_over_raw delta_auc_roc -0.331028  -0.258741              -0.263636                   False                    False                         False     0.185478                0.217012                         True             False              False
     trend_over_residual_energy  delta_auc_pr  0.103755  -0.181818              -0.263636                    True                     True                          True     3.197351                3.617010                        False             False              False
     trend_over_residual_energy delta_auc_roc  0.311265   0.335664               0.363636                   False                    False                         False     3.696154                3.173748                         True             False              False
```
- Row LOO evaluates every removed execution/paper record. Grouped LOO removes all ASD subsets together and each remaining paper dataset once.

## Score Diagnostics
- CATCH continuous scores are absent from frozen formal archives; score-vector Spearman remains N/A without prohibited rescoring.

## Conclusion
- A: mean_drift is a candidate association for Delta AUC-ROC; low_frequency_energy_ratio_mean is a candidate association for Delta AUC-ROC; periodicity_top3_ratio is a candidate association for Delta AUC-ROC; correlation_drift is a candidate association for Delta AUC-ROC.
- Any qualified association is descriptive, non-causal, not a direct model-selection rule, and not externally validated on independent datasets.
