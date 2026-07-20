# Decomposition Applicability Report

This report is read-only: it does not call a training, scoring, or benchmark entry point.

## Formal Performance Coverage
- Formal MSD delta available for 23/23 execution tasks.
- PSM, Genesis, and GECCO use the terminal formal total-screen JSON line in their MSD logs.
- ASD paper-level values are equal-weight macro means of its 12 execution tasks.

## Descriptor Method
- Raw descriptors use normal training prefixes and non-overlapping formal seq_len windows.
- Spectral quantities are per-channel training-window summaries; correlation drift is N/A for one channel.
- Decomposition uses replicate-padded moving averages and verifies trend + residual = input.

## Task-Level Performance
```text
          task  catch_auc_pr  msd_auc_pr  delta_auc_pr pr_group  catch_auc_roc  msd_auc_roc  delta_auc_roc roc_group           msd_source_kind
           PSM      0.436646    0.430441     -0.006205  neutral       0.647548     0.647262      -0.000287   neutral   formal total-screen log
       Genesis      0.265986    0.310769      0.044783     gain       0.973541     0.986279       0.012738      gain   formal total-screen log
         GECCO      0.417423    0.406495     -0.010928     loss       0.969981     0.964382      -0.005599   neutral   formal total-screen log
        CalIt2      0.113310    0.124758      0.011448     gain       0.837335     0.848738       0.011403      gain  formal total-screen JSON
           NYC      0.080123    0.062562     -0.017561     loss       0.818435     0.808057      -0.010377      loss  formal total-screen JSON
           MSL      0.165589    0.174027      0.008438  neutral       0.661964     0.660877      -0.001087   neutral  formal total-screen JSON
 ASD_dataset_1      0.124849    0.125786      0.000938  neutral       0.497176     0.503276       0.006100   neutral formal by-dataset archive
 ASD_dataset_2      0.473872    0.488184      0.014312     gain       0.962085     0.966326       0.004242   neutral formal by-dataset archive
 ASD_dataset_3      0.220059    0.259443      0.039385     gain       0.806264     0.830167       0.023903      gain formal by-dataset archive
 ASD_dataset_4      0.154110    0.197228      0.043117     gain       0.840161     0.876350       0.036190      gain formal by-dataset archive
 ASD_dataset_5      0.266653    0.291259      0.024606     gain       0.943584     0.950720       0.007135   neutral formal by-dataset archive
 ASD_dataset_6      0.207689    0.197167     -0.010522     loss       0.842839     0.827221      -0.015618      loss formal by-dataset archive
 ASD_dataset_7      0.071784    0.088437      0.016654     gain       0.814284     0.877634       0.063350      gain formal by-dataset archive
 ASD_dataset_8      0.145445    0.148187      0.002743  neutral       0.861920     0.866163       0.004243   neutral formal by-dataset archive
 ASD_dataset_9      0.314735    0.308531     -0.006203  neutral       0.877143     0.882751       0.005607   neutral formal by-dataset archive
ASD_dataset_10      0.379742    0.393949      0.014207     gain       0.857467     0.866833       0.009367   neutral formal by-dataset archive
ASD_dataset_11      0.212069    0.242624      0.030555     gain       0.796359     0.820564       0.024205      gain formal by-dataset archive
ASD_dataset_12      0.199880    0.192361     -0.007519  neutral       0.784465     0.816794       0.032329      gain formal by-dataset archive
        CICIDS      0.002173    0.001606     -0.000567  neutral       0.790505     0.719925      -0.070580      loss formal by-dataset archive
    Creditcard      0.100479    0.104107      0.003627  neutral       0.957814     0.947814      -0.010000      loss formal by-dataset archive
          SMAP      0.128537    0.113519     -0.015018     loss       0.497478     0.443063      -0.054415      loss formal by-dataset archive
           SMD      0.172337    0.172922      0.000585  neutral       0.810861     0.797395      -0.013466      loss formal by-dataset archive
          SWAT      0.158745    0.140985     -0.017759     loss       0.343745     0.327812      -0.015934      loss formal by-dataset archive
```

## Paper-Level Performance
```text
paper_dataset  task_count  catch_auc_pr  msd_auc_pr  delta_auc_pr  catch_auc_roc  msd_auc_roc  delta_auc_roc
          PSM           1      0.436646    0.430441     -0.006205       0.647548     0.647262      -0.000287
      Genesis           1      0.265986    0.310769      0.044783       0.973541     0.986279       0.012738
        GECCO           1      0.417423    0.406495     -0.010928       0.969981     0.964382      -0.005599
       CalIt2           1      0.113310    0.124758      0.011448       0.837335     0.848738       0.011403
          NYC           1      0.080123    0.062562     -0.017561       0.818435     0.808057      -0.010377
          MSL           1      0.165589    0.174027      0.008438       0.661964     0.660877      -0.001087
          ASD          12      0.230907    0.244430      0.013523       0.823645     0.840400       0.016754
       CICIDS           1      0.002173    0.001606     -0.000567       0.790505     0.719925      -0.070580
   Creditcard           1      0.100479    0.104107      0.003627       0.957814     0.947814      -0.010000
         SMAP           1      0.128537    0.113519     -0.015018       0.497478     0.443063      -0.054415
          SMD           1      0.172337    0.172922      0.000585       0.810861     0.797395      -0.013466
         SWAT           1      0.158745    0.140985     -0.017759       0.343745     0.327812      -0.015934
```

## Score Diagnostics
- CATCH continuous scores are not stored in the formal archives, so CATCH-versus-MSD Spearman score correlation is N/A without a rerun.
- MSD quantile diagnostics are available only where a frozen `*_scores.npz` exists; component AUCs from formal logs are still retained.

## Descriptor Correlations
```text
level                      descriptor        target  spearman_rho  n most_influential_leave_one_out  max_leave_one_out_shift  leave_one_out_sign_flip
 task                   channel_count  delta_auc_pr     -0.206398 23                            NYC                 0.147669                    False
 task                   channel_count delta_auc_roc     -0.495996 23                            NYC                 0.112827                    False
 task                      mean_drift  delta_auc_pr      0.481225 23                  ASD_dataset_9                 0.082863                    False
 task                      mean_drift delta_auc_roc      0.634387 23                           SMAP                 0.043196                    False
 task                  variance_drift  delta_auc_pr      0.520751 23                  ASD_dataset_6                 0.112225                    False
 task                  variance_drift delta_auc_roc      0.465415 23                  ASD_dataset_6                 0.141587                    False
 task low_frequency_energy_ratio_mean  delta_auc_pr      0.125494 23                            NYC                 0.144975                    False
 task low_frequency_energy_ratio_mean delta_auc_roc      0.402174 23                            NYC                 0.115613                    False
 task                spectral_entropy  delta_auc_pr      0.030632 23                           SWAT                 0.097826                     True
 task                spectral_entropy delta_auc_roc     -0.209486 23                         CICIDS                 0.106155                    False
 task          periodicity_top3_ratio  delta_auc_pr      0.169960 23                            NYC                 0.146810                    False
 task          periodicity_top3_ratio delta_auc_roc      0.460474 23                            NYC                 0.112648                    False
 task               correlation_drift  delta_auc_pr     -0.250988 23                            NYC                 0.140316                    False
 task               correlation_drift delta_auc_roc     -0.577075 23                            NYC                 0.110107                    False
 task           trend_energy_over_raw  delta_auc_pr      0.033597 23                           SMAP                 0.101920                     True
 task           trend_energy_over_raw delta_auc_roc      0.181818 23                           SMAP                 0.107849                    False
 task        residual_energy_over_raw  delta_auc_pr     -0.064229 23                  ASD_dataset_4                 0.111095                     True
 task        residual_energy_over_raw delta_auc_roc     -0.323123 23                           SWAT                 0.091897                    False
 task      trend_over_residual_energy  delta_auc_pr      0.087945 23                  ASD_dataset_4                 0.107708                     True
 task      trend_over_residual_energy delta_auc_roc      0.297431 23                           SMAP                 0.091333                    False
paper                   channel_count  delta_auc_pr     -0.126095 12                         CalIt2                 0.198988                     True
paper                   channel_count delta_auc_roc     -0.535903 12                         CICIDS                 0.139546                    False
paper                      mean_drift  delta_auc_pr      0.265734 12                        Genesis                 0.220280                    False
paper                      mean_drift delta_auc_roc      0.587413 12                           SWAT                 0.158042                    False
paper                  variance_drift  delta_auc_pr      0.601399 12                           SWAT                 0.180420                    False
paper                  variance_drift delta_auc_roc      0.321678 12                        Genesis                 0.194406                    False
paper low_frequency_energy_ratio_mean  delta_auc_pr     -0.174825 12                            NYC                 0.220280                     True
paper low_frequency_energy_ratio_mean delta_auc_roc      0.356643 12                            NYC                 0.152448                    False
paper                spectral_entropy  delta_auc_pr      0.181818 12                            MSL                 0.200000                    False
paper                spectral_entropy delta_auc_roc     -0.013986 12                         CICIDS                 0.250350                     True
paper          periodicity_top3_ratio  delta_auc_pr     -0.013986 12                            NYC                 0.232168                     True
paper          periodicity_top3_ratio delta_auc_roc      0.475524 12                            NYC                 0.160839                    False
paper               correlation_drift  delta_auc_pr     -0.265734 12                            NYC                 0.216084                    False
paper               correlation_drift delta_auc_roc     -0.517483 12                            MSL                 0.173427                    False
paper           trend_energy_over_raw  delta_auc_pr     -0.146853 12                           SMAP                 0.180420                    False
paper           trend_energy_over_raw delta_auc_roc      0.223776 12                           SMAP                 0.169231                    False
paper        residual_energy_over_raw  delta_auc_pr      0.216783 12                           SWAT                 0.171329                    False
paper        residual_energy_over_raw delta_auc_roc     -0.244755 12                           SWAT                 0.127972                    False
paper      trend_over_residual_energy  delta_auc_pr     -0.174825 12                           SWAT                 0.183916                     True
paper      trend_over_residual_energy delta_auc_roc      0.286713 12                           SMAP                 0.150350                    False
```

## Paper Baseline Reference
- No reliable structured original CATCH paper table was present locally; paper-reported values remain N/A rather than being substituted by current reruns.

## Conclusion
- A: at least one descriptor meets the numerical cross-level screen: variance_drift
- This is descriptive evidence from frozen runs, not a model-selection rule or a new implementation proposal.
