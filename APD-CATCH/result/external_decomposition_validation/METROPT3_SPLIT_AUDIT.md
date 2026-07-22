# MetroPT-3 Split Audit

## Official Source

- Archive: `dataset/external_validation/raw/metropt3/metropt+3+dataset.zip`
- Archive member: `MetroPT3(AirCompressor).csv`
- Bytes: `218381995`
- SHA-256: `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`
- Timestamp column and parse format: `timestamp`, `%Y-%m-%d %H:%M:%S`
- Raw rows: `1516948`
- Invalid timestamps: `0`
- Duplicate timestamps: `0`
- Original timestamp order monotonic: `True`
- Stable-sorted time range: `2020-02-01 00:00:00` to `2020-09-01 03:59:50`
- Adjacent interval seconds (min / median / mode / max): `8.0` / `10.0` / `10.0` / `172918.0`

## Complete-Month Definitions

The previous loader selected only the first observed month and required its final timestamp to reach that month's final theoretical second. This is definition **D**: a stricter range-edge condition than calendar coverage, and it does not search later calendar months.

- **A**: a month is not the truncated first or last coverage month, and has at least one observation on both its first and last natural day.
- **B**: definition A plus at least one observation on every natural day.
- **C**: definition B plus every theoretical sample at the inferred modal interval.

First satisfying month: A = `2020-03`, B = `2020-03`, C = `NONE`.

The formal loader uses definition A. With `2020-03` selected, train is March observations only and test starts at `2020-04-01 00:00:00`; the preceding truncated February observations are excluded from both formal train and formal test. It does not fill missing timestamps, resample, select an arbitrary 30-day interval, or inspect test labels to choose the month. The detailed monthly evidence is in [metropt3_calendar_coverage.csv](metropt3_calendar_coverage.csv).
