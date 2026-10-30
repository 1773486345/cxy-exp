# Frozen Full conditional-score diagnostic

This is a fixed post-hoc CPU diagnostic using the eight registered Full sources. It does not fit a GMM, encode, retrain, change feature/cache identities, or tune on test labels.

## Scoring

The original Full rows reuse saved scores and thresholds after native source validation. The two diagnostic rows replay saved GMMs against float32-normalized features plus the original four fixed noise vectors, widening to float64 only at likelihood evaluation. The replay is checked against saved validation/calibration/test per-variable NLL (`rtol=2e-5`, `atol=2e-5`).

For the responsibility-aware rows, raw variables are scored only when role is evidenced as an observation; NYC's time-of-day and day-of-week columns are explicitly exogenous. Unknown MSL column mappings and other unknown responsibilities remain scored. Conditional density is computed directly as a posterior-weighted conditional Gaussian mixture: `p(S,P | E,R)`, with S/P dimensions `[0,1) U [9,17)` and E/R context dimensions `[1,9) U [17,25)`. No subtraction of two large NLLs is used.

Each diagnostic method gets its own validation median/MAD and calibration 0.995 linear quantile; variable aggregation remains max and prediction uses strict `score > threshold`. Metrics include AUC/AP, point metrics, recall at fixed FPR budgets, and `affiliation_f_valid_segment_macro`; this segment macro is not the paper's original Affiliation-F.

## Field responsibilities

| Dataset | Field-role evidence | Scoring responsibility in this diagnostic | Sequence/entity evidence |
|---|---|---|---|
| PSM | The source is the pooled server-metrics benchmark (25 channels), but this frozen conversion names channels `feature_N`; a channel dictionary is not present. | All 25 retained as observations; individual metric meanings unknown. | Timestamped aggregate series; host/application entities are not identified in this frozen source. |
| MSL | Telemanom defines 27 telemetry channels with command context, but this run has 55 `col_N` inputs and no local column-to-source map or raw channel arrays. | All 55 retained as observations; mapping and telemetry/context split unknown. | Official 27 test sequence ranges reproduced in sorted channel order; corresponding train per-channel lengths unavailable. |
| GECCO | Named inputs include water-quality measurements such as pH, Redox, and conductivity/turbidity-related fields; no independent exogenous mapping is present. | All 9 retained as observations; fields are not discarded based on names. | Environmental sensor stream; no separate entity/session key in the frozen source. |
| Genesis | Local names identify `MotorData.*` process measurements and discrete status-like channels; no source dictionary proves status fields are external conditions. | All 18 retained as observations, including low-cardinality status channels. | Process sequence identity beyond the frozen continuous axis is unavailable. |
| NYC | Every row matches the upstream three-column file: passenger_count is passenger flow; timeofday and dayofweek encode time context. | passenger_count is the target observation; the two evidenced time fields are conditions, retained as context but excluded from role-aware target aggregation. | Half-hour timestamps map row-for-row; no entity ID. |
| CalIt2 | UCI documents flow ID 7 as outflow and 9 as inflow, with half-hour count aggregates. Original raw counts range 0–62; local values equal `count % 10`. | Both lossy-converted streams remain target observations; no condition is inferred from low cardinality. | Real half-hour time index; flow directions are separate variables, not entity IDs. |
| Creditcard | ULB schema contains transactions with PCA-anonymized V1–V28 and Amount; original Time is absent from these 29 model inputs. | All 29 retained as transaction observations; PCA components are not interpretable field-by-field. | Transaction row order is not a per-card sequence: card/customer IDs are absent. |
| CICIDS | Inputs are flow-statistic fields; the frozen conversion drops identifiers/time/session keys. Full-axis inventory finds 10 constant and 4 exact-duplicate channels. | All 72 retained, including constants and duplicates; no silent feature removal. | Flow records do not establish a continuous physical series or entity boundaries. |

`field_responsibilities.csv` records full-axis range, cardinality, constant status, exact duplicates, and whether a field role is evidenced. Sources: [PSM/Time-Series-Library](https://github.com/thuml/Time-Series-Library), [Telemanom](https://github.com/khundman/telemanom), [GECCO water-quality channel descriptions](https://www.mdpi.com/2073-4441/18/6/726), [UCI CalIt2](https://archive.ics.uci.edu/dataset/156/calit2), [ULB credit-card schema](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), and [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html). NYC is directly matched to the upstream file supplied in the task. For PSM, Genesis, Creditcard and CICIDS, exact frozen CSV provenance/transform lineage beyond the local loader is incomplete; semantics are not claimed as a per-column ground truth. Low cardinality alone never makes a field exogenous. No constant or duplicate field was removed.

The current CalIt2 representation is lossy modulo-10 bucketing; no score-only change can restore count magnitude. The conditional comparison diagnoses the frozen representation, not a corrected raw-count model.

## Sequence limits

The MSL report gives 27 test sequence boundaries and common-coordinate rescoring after excluding windows whose 192-row receptive field crosses a test sequence boundary. This is only a test boundary sensitivity check: the frozen encoder/heads were trained under the original concatenated axis, so it is not an independently retrained segmented experiment. Per-channel training sequence lengths are not present locally, so no training-side boundary correction is claimed. NYC's `nyc_original_row_time.csv` preserves row-to-source-time mapping. Creditcard and CICIDS lack entity IDs in the frozen source, so within-entity continuity cannot be reconstructed; especially CICIDS consists of flow records, for which row adjacency is not established as a physical time series.

## Source evidence

See `source_evidence.json`, `field_responsibilities.csv`, and the MSL/NYC evidence CSVs. A failed dataset remains explicitly failed and is not silently omitted.

Completed datasets: 8/8. No model weights, feature caches, or per-window NLL matrices are copied here.
