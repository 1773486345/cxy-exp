# APD-CATCH

This working tree is rebuilt from the original CATCH project. It contains no
custom research model. The added tasks retain original CATCH commands and the
standard upstream baseline commands that were run for those tasks.

## Data

The original anomaly-detection data remains under `dataset/anomaly_detect`.
The added external tasks are registered in
`dataset/external_validation/EXTERNAL_DETECT_META.csv`:

- HAI 20.07
- BATADAL
- MetroPT-3
- mTSBench OPPORTUNITY (13 tasks)
- mTSBench Occupancy (2 tasks)
- mTSBench Metro
- mTSBench SWAN-SF

Prepared external CSV files contain a timestamp, numeric feature columns, and
a binary label. The downloader and preparation commands are in
`scripts/data_preparation/external_validation`.

## Run CATCH

Run an original bundled CATCH task, for example:

```bash
sh ./scripts/multivariate_detection/detect_score/SMD_script/CATCH.sh
```

Run CATCH for an added external task, for example:

```bash
sh ./scripts/multivariate_detection/detect_score/HAI20_07_script/CATCH.sh
```

Each added task has independent commands in
`scripts/multivariate_detection/detect_score/<task>_script/`. They use the
unchanged score configuration and the `external_detect` data source.

## Retained Baselines

The retained external reports and their original `tar.gz` artifacts are under
`result/score/external_validation/<task>/<baseline>/`. They cover CATCH and
the standard baseline runs that emitted valid metrics for the added tasks.
Every retained report records a finite AUC-ROC value. The historical
configuration reported AUC-ROC only, so no AUC-PR value is stored for these
runs.

Future model development starts from this clean CATCH baseline.
