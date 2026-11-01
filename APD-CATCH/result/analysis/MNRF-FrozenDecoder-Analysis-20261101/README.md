# Frozen E/R decoder result analysis

This report is a read-only analysis of the eight already completed frozen E/R diagnostics. It does not train, run an encoder, fit a GMM, or recalibrate a source.

## Evidence and limits

Each task has 13 diagnostic rows: saved Full GMM reference; E/R decoder, latent-train-mean intervention, persistence, and train-target-mean references, with squared-residual and/or NLL scores as saved by the diagnostic. The analysis reuses saved residuals, per-variable NLL, centers/scales, thresholds and coordinates. It does not treat these scores as one common detector.

The diagnostic manifests bind these outputs to the following source runs:

| Dataset | Source Run | Variables | Test points | Anomaly points |
|---|---|---:|---:|---:|
| PSM | `full_gmm-full-20261024T115119Z-3976714` | 25 | 87650 | 24378 |
| MSL | `full_gmm-full-20261021T112147Z-2470335` | 55 | 73538 | 7766 |
| GECCO | `full_gmm-full-20261020T204951Z-1642355` | 9 | 69166 | 730 |
| Genesis | `full_gmm-full-20261020T205005Z-1645522` | 18 | 12425 | 50 |
| NYC | `full_gmm-full-20261020T211341Z-1931786` | 3 | 4225 | 99 |
| CalIt2 | `full_gmm-full-20261021T112212Z-2474004` | 2 | 2329 | 70 |
| Creditcard | `full_gmm-full-20261024T115131Z-3977864` | 29 | 142213 | 221 |
| CICIDS | `full_gmm-full-20261024T115157Z-3980319` | 72 | 84925 | 74 |

## Main findings

- **PSM:** R residual AP/AUC=0.5294058926815113/0.7470662574420426; R NLL AP/AUC=0.29743356532318355/0.6056665034686077; E residual AP/AUC=0.3864684109487426/0.6491956192616228. These are diagnostic rankings, not claims of a new detector.
- **MSL:** R residual AP/AUC=0.13001339993208763/0.5550962921896789; R NLL AP/AUC=0.14847075141831118/0.5460096338079796; E residual AP/AUC=0.12489310225392836/0.5563819183287777. These are diagnostic rankings, not claims of a new detector.
- **GECCO:** R residual AP/AUC=0.3037308963650342/0.870483281249875; R NLL AP/AUC=0.12003903832847536/0.6915533421086554; E residual AP/AUC=0.2748937999329405/0.8382698023230584. These are diagnostic rankings, not claims of a new detector.
- **Genesis:** R residual AP/AUC=0.09083072896326719/0.8506941414141412; R NLL AP/AUC=0.03979385758246183/0.8645963636363636; E residual AP/AUC=0.018536673002974496/0.5826812121212122. These are diagnostic rankings, not claims of a new detector.
- **NYC:** R residual AP/AUC=0.025300691221053575/0.5336373918535819; R NLL AP/AUC=0.022941939150560666/0.5193990315173058; E residual AP/AUC=0.021496735238467207/0.4367279190352385. These are diagnostic rankings, not claims of a new detector.
- **CalIt2:** R residual AP/AUC=0.07663467748230145/0.78193258711187; R NLL AP/AUC=0.047999462681015445/0.7173148675140708; E residual AP/AUC=0.08873109172410025/0.8115284892177324. These are diagnostic rankings, not claims of a new detector.
- **Creditcard:** R residual AP/AUC=0.3561095993755127/0.954259611592419; R NLL AP/AUC=0.00175561137245161/0.590763733040597; E residual AP/AUC=0.022875611097572073/0.9403701986651978. These are diagnostic rankings, not claims of a new detector.
- **CICIDS:** R residual AP/AUC=0.0012620138624273546/0.6799414999966555; R NLL AP/AUC=0.001632175143873298/0.7201302314677526; E residual AP/AUC=0.0012325994011649687/0.6740241319680573. These are diagnostic rankings, not claims of a new detector.

## Creditcard residual/NLL reversal

For R decoder, squared-residual AP is **0.356110** while NLL AP is **0.001756**. The residual score uses the decoder mean error; NLL additionally standardizes the per-variable Gaussian likelihood and therefore can change both variable ranking and the final max aggregation. The top anomaly-point max contributors are recorded in `max_attribution.csv`. Pointwise sigma is not saved, so this report cannot claim that a particular anomaly was hidden by high predicted variance; a minimal follow-up would save per-point sigma, decoder mean and target for a fixed deterministic sample.

The saved per-variable ranking shows that R squared residual is led by V14 (AP 0.772511) and V17 (0.707879), while R NLL drops to V14 0.141851 and V17 0.058146. The max attribution changes from V14/V17 = 0.471/0.371 of anomaly points for squared residual to V17/V14 = 0.543/0.335 for NLL. At the fixed thresholds, squared residual has TP/FP/FN = 184/1167/37, whereas NLL has 0/222/221. This localizes the reversal to per-variable NLL scaling and the final variable-max/threshold chain, but not to a proven pointwise-variance cause.

## MSL target and condition fields

MSL is analyzed as 55 input fields and 27 source sequences are not reconstructed by this diagnostic into separate retrained models. `col_0` is reported separately in `variable_metrics.csv`; `col_37`, `col_51` and `col_25` are retained as ordinary fields in the full attribution table. A large residual contribution does not by itself mean a field controls the max score; use `max_attribution.csv` for the latter.

For `col_0`, E residual/NLL are AUC/AP 0.527862/0.201526 and 0.538552/0.210619; R residual/NLL are 0.482959/0.155729 and 0.477443/0.153374. The R residual max attribution for anomaly points is col_2 0.483 and col_0 0.210; R NLL is led by col_30 0.464 and col_0 0.211. The inspected condition fields remain weak individually: R residual AP is 0.128141 (col_25), 0.127757 (col_37), and 0.128047 (col_51), while their R NLL AP is 0.109768, 0.109436, and 0.109820. These are field-level clues on the old concatenated 55-column source, not a 27-sequence boundary-aware retraining result.

## NYC event profile

NYC retains the source event boundaries, global coordinates and unscored gaps. `nyc_event_profile.csv` reports target-variable absolute residual/NLL near event starts, interiors and ends. The passenger/calendar role interpretation is a diagnostic annotation, not evidence that calendar fields are causal or that lower prediction error proves anomaly information.

The source has one scored anomaly event (99 labelled points; one scored segment). In the saved phase table, E absolute residual median is 0.162 at the start, 0.060 through the interior, and 0.270 at the end; R is 0.213, 0.133, and 0.854 respectively. This is compatible with a predictor that follows a sustained change after the transition, but the single event and three-channel role annotation are insufficient to establish a causal calendar/flow explanation. E residual produces one TP and 12 FP; R residual and NLL produce zero TP and 31 FP.

## Full reference and alarm context

The source Full GMM references are recorded in `full_reference.csv` with the actual `test_nll.npy`, `calibration_nll.npy`, calibration threshold/quantile, source identity, feature fingerprint, fit-endpoint hash, fixed-noise hash and cache hash. The saved Full AUC/AP are: PSM 0.744926/0.437281, MSL 0.602445/0.159006, GECCO 0.442452/0.223303, Genesis 0.899568/0.046805, NYC 0.691339/0.044089, CalIt2 0.713040/0.053971, Creditcard 0.920188/0.051007, CICIDS 0.538825/0.000917. These are the original Full GMM scores, not E/R decoder scores.

Across the saved diagnostics, residual ranking and fixed-threshold alarms often disagree. GECCO R residual reaches AUC/AP 0.870483/0.303731 but has FPR 0.111476; Genesis R residual reaches 0.850694/0.090831 with FPR 0.026424; PSM R residual reaches 0.747066/0.529406 with FPR 0.234322. CalIt2 E residual (AUC/AP 0.811528/0.088731) beats persistence AP 0.064834 but remains a low-recall, high-FPR alarm (1 TP, 37 FP). CICIDS E/R residuals do not beat persistence AP 0.001635, and no E/R NLL alarm has a true positive. These comparisons are diagnostic and use each saved score's own historical threshold; they are not a new threshold selection.

## Task-specific interpretation

- PSM and GECCO show R squared-residual ranking above their simple static/persistence references in the saved metrics, but fixed-threshold alarms and event coverage remain separate outcomes.
- Genesis R residual improves ranking over persistence and target-mean references while its fixed threshold still has substantial false positives; ranking gain is not alarm calibration gain.
- CalIt2 is the historical modulo-10 representation. The separate RawCount report exists and shows a RawCount Full rerun plus CATCH/IF/PCA, but no RawCount S-only result; it must not be substituted into the old source comparison.
- CICIDS has no verified entity/session identity in the frozen input; row-adjacent event interpretations remain limited.

## RawCount CalIt2 status

RawCount output: `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031`. Full and CATCH/IF/PCA metrics are present; S-only is absent. The exact manual S-only command, which has not been run in this round, is:

```bash
python scripts/run_mnrf_branch_ablation.py --task CalIt2 --mode execute --source-dir result/score/MNRF-ViewProbe-RawCount/CalIt2/rawcount-full-20261031T20261031T092320Z-1093827 --views S
```

It should use the RawCount source only if a new isolated head run is authorized.

## Next P diagnostic

Use a fixed 8-step masking rule on raw input windows, mask only the P target positions before the encoder, retain the original E/R/S/R roles, and re-encode the masked input. Compare a P-masked encoder against persistence and a fixed train-mean recovery using the same validation median/MAD and calibration q=0.995. Use the source test coordinates, segment IDs and no-cross-boundary windows; budget at most 256 train, 128 validation, 128 calibration and 512 test diagnostic samples initially. Existing source validation, axis binding, chunked cache writing and evaluation helpers can be reused, but the current full P cache cannot stand in for the masked representation. This round did not run it.

## Output files

`summary.csv` contains aggregate E/R diagnostic rows; `reference_metrics.csv` preserves all 13 rows per task, including the latent-mean intervention and simple references; `full_reference.csv` records the source Full GMM NLL, calibration arrays, axes and binding hashes; `variable_distributions.csv` contains normal/anomaly per-variable residual/NLL statistics; `variable_metrics.csv` contains per-variable and aggregate ranking metrics; `max_attribution.csv` contains tie-split max attribution for normal/anomaly/TP/FP/FN; `scale_diagnostics.csv` records MAD floors; `nyc_event_profile.csv` contains bounded event phase rows; `source_bindings.csv` and `analysis_config.json` record source identities, layouts, protocol fields and RawCount status.

