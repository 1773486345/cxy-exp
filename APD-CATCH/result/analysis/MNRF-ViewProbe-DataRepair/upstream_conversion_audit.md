# Upstream-to-CSV conversion audit

This is a provenance audit, not a cache or metric re-audit. A `verified`
entry means the original values and ordering were independently matched;
`partial` means the local transformation is known but the upstream bytes or
producer are not present; `unknown` means a column dictionary or lossless
producer record is unavailable. No labels were used to clean data or alter a
split.

| task | raw-to-CSV evidence | lossless? | field responsibility | limitation |
|---|---|---|---|---|
| PSM | Local server-metrics CSV is parsed losslessly by the long-table loader; no upstream per-column export is retained. | unknown | all `feature_N` remain observations | source producer/channel dictionary absent |
| MSL | 27 Telemanom MSL train/test arrays, sorted by channel, concatenate to the local 55-column matrix; shape/dtype-aware hash matches `5fe9f96f...b9074a5a`. | verified | `col_0` telemetry target candidate; other columns retained as command/context | Kaggle credentials unavailable; public mirror used |
| GECCO | Local GECCO channel names and numeric values load without lossy parsing; independent raw export is not stored in this workspace. | partial | all named water-quality fields remain observations | source-to-column byte lineage missing |
| Genesis | Local `MotorData.*` numeric channels parse losslessly; no checked-in producer transform or status dictionary. | partial | all 18 channels remain observations | status-like fields are not proven exogenous |
| NYC | Local three fields match the upstream taxi file row-for-row; time-of-day/day-of-week are derived context fields. | verified | passenger flow observation; time fields context | no entity identifier |
| CalIt2 (historical) | UCI counts map exactly to historical values modulo 10; the conversion producer is not in the repository. | lossy | both flows are observations | magnitude information was discarded |
| CalIt2 RawCount | UCI `CalIt2.data` is parsed as integer counts; flow 9 max 62 and flow 7 max 54; timestamps and historical binary labels are preserved in an independent root. | verified | both flow directions observations | labels remain the benchmark label axis |
| Creditcard | Local `Time,V1..V28,Amount`-style numeric table parses losslessly; raw CSV producer and dropped transaction identifiers are not retained. | partial | all retained numeric fields observations | no transaction/entity continuity |
| CICIDS | Local flow-statistic columns parse losslessly from the benchmark CSV; upstream capture/session conversion is not retained. | partial | all retained numeric fields observations | no session/entity boundary in frozen file |

The detailed MSL and CalIt2 evidence is in `msl-raw-mapping-20261031.json`,
`msl_protocol.md`, and `calit2-rawcount-20261031/data_root/RAWCOUNT_PROTOCOL.json`.
