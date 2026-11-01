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

For R decoder, squared-residual AP is **0.356110** while NLL AP is **0.001756**. The ranking deterioration is already present at the single-variable auxiliary-NLL layer; a positive median/MAD affine standardization cannot change a variable's ordering. Cross-variable scaling and the final max aggregation can further change the overall result. The top anomaly-point max contributors are recorded in `max_attribution.csv`. Pointwise sigma is not saved, so this report does not claim that particular anomalies were hidden by high predicted variance; a fixed-sample export of per-point mu/sigma would be required.

## E-branch reference comparison

E residual scores are compared separately with persistence and the training-target-mean reference in `reference_metrics.csv`. GECCO, the historical modulo-10 CalIt2 source, and Creditcard have E residual AP above persistence but not above the target-mean reference; this is not evidence that historical conditions add detection information. Small differences are reported as such. A stronger conditional increment requires consistent improvement over both simple references under the same coordinates and threshold protocol.

| Dataset | E decoder AP | E persistence AP | E target-mean AP | E latent-mean AP | R decoder AP | R persistence AP | R target-mean AP | R latent-mean AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PSM | 0.3864684109487426 | 0.37901239314995094 | 0.37474388222333577 | 0.3739591541401812 | 0.5294058926815113 | 0.37901239314995094 | 0.37474388222333577 | 0.37304692967495656 |
| MSL | 0.12489310225392836 | 0.12008100004806371 | 0.12306269537218295 | 0.12313428078763616 | 0.13001339993208763 | 0.12008100004806371 | 0.12306269537218295 | 0.12048611763255399 |
| GECCO | 0.2748937999329405 | 0.028799373028349076 | 0.27516558566234023 | 0.29818935803497176 | 0.3037308963650342 | 0.028799373028349076 | 0.27516558566234023 | 0.2555603034512376 |
| Genesis | 0.018536673002974496 | 0.007139677825014861 | 0.02495639886519836 | 0.024595275377390584 | 0.09083072896326719 | 0.007139677825014861 | 0.02495639886519836 | 0.02475830547143218 |
| NYC | 0.021496735238467207 | 0.022420947179246246 | 0.03877616251990683 | 0.042011678071957194 | 0.025300691221053575 | 0.022420947179246246 | 0.03877616251990683 | 0.03835649889187045 |
| CalIt2 | 0.08873109172410025 | 0.06483403623506631 | 0.09109413959702396 | 0.09329577345519346 | 0.07663467748230145 | 0.06483403623506631 | 0.09109413959702396 | 0.09329577345519346 |
| Creditcard | 0.022875611097572073 | 0.00851153923326723 | 0.022918339613947042 | 0.022439387524304882 | 0.3561095993755127 | 0.00851153923326723 | 0.022918339613947042 | 0.01839937616835035 |
| CICIDS | 0.0012325994011649687 | 0.001634693782862798 | 0.00177295802977299 | 0.0017647285868185507 | 0.0012620138624273546 | 0.001634693782862798 | 0.00177295802977299 | 0.001732776536625158 |

## MSL target and condition fields

MSL is analyzed as 55 input fields and 27 source sequences are not reconstructed by this diagnostic into separate retrained models. `col_0` is reported separately in `variable_metrics.csv`; `col_37`, `col_51` and `col_25` are retained as ordinary fields in the full attribution table. A large residual contribution does not by itself mean a field controls the max score; use `max_attribution.csv` for the latter.

## NYC event profile

NYC retains the source event boundaries, global coordinates and unscored gaps. `nyc_event_profile.csv` reports the original target-variable rows. `nyc_event_neighborhood.csv` adds a pre-fixed k=8 summary for event_first8, event_last8, remaining event_interior, and same-segment consecutive normal_before8/normal_after8 points when available. This source has one scored anomaly event, so the neighborhoods describe that event only; they do not establish a general persistence mechanism. Passenger/calendar role interpretation remains a diagnostic annotation, not evidence that calendar fields are causal or that lower prediction error proves anomaly information.

## Task-specific interpretation

- PSM and GECCO show R squared-residual ranking above their simple static/persistence references in the saved metrics, but fixed-threshold alarms and event coverage remain separate outcomes.
- Genesis R residual improves ranking over persistence and target-mean references while its fixed threshold still has substantial false positives; ranking gain is not alarm calibration gain.
- CalIt2 is the historical modulo-10 representation. The separate RawCount report exists and shows a RawCount Full rerun plus CATCH/IF/PCA, but no RawCount S-only result; it must not be substituted into the old source comparison.
- CICIDS has no verified entity/session identity in the frozen input; row-adjacent event interpretations remain limited.

## RawCount CalIt2 status

RawCount output: `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031`. Search roots were `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/score/MNRF-ViewProbe-RawCount`, `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/score/MNRF-ViewProbe-SOnly`, `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/score/MNRF-ViewProbe-Components`, and `/media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031`; 0 manifest candidate(s) were inspected and 0 same-source S-only candidate(s) passed the binding/status checks. No same-source completed RawCount S-only output was verified; the available report is RawCount Full plus CATCH/IF/PCA.
The report contains RawCount Full plus CATCH/IF/PCA metrics, but this read-only analysis did not execute an S-only fit. The accurate future entry is `run_mnrf_view_probe.py --mode select_gmm --views S` with the fixed RawCount `--source-dir`, explicit `--data-root`, CPU and a unique output directory; the complete command is recorded in `analysis_config.json`.

## Next P diagnostic

Use the training-time 32-step local window with a continuous 8-step P mask, the saved valid flag, relative-position input and the original auxiliary decoder. Apply the identical deterministic mask rule to every split without using test labels. Persistence must read the unmasked history only, never the held-out masked truth. Map each recovered position back to its original global coordinate and segment; aggregate overlaps by a fixed rule (for example mean over valid reconstructions), and retain only common valid points. Treat 256/128/128/512 as a process-validation budget, not a formal performance sample. With small calibration sets, q=.995 is unstable; do not publish a comparative claim without enough positives and a predeclared budget. Existing axis validation and chunked I/O can be reused, but the full P cache cannot stand in for a masked representation. This round did not run it.

## Output files

`summary.csv` contains aggregate E/R diagnostic rows; `reference_metrics.csv` preserves the complete 13-row distinction between Full GMM, decoder, latent-mean intervention, persistence and train-target-mean references; `full_reference.csv` records the source Full GMM NLL, calibration arrays, axes and binding hashes; `variable_distributions.csv` contains normal/anomaly per-variable residual/NLL statistics; `variable_metrics.csv` contains corrected per-variable and aggregate ranking metrics; `max_attribution.csv` contains tie-split max attribution for normal/anomaly/TP/FP/FN; `scale_diagnostics.csv` records MAD floors; `nyc_event_profile.csv` and `nyc_event_neighborhood.csv` contain bounded event rows and fixed-k summaries; `source_bindings.csv` and `analysis_config.json` record source identities, layouts, protocol fields and RawCount status.

