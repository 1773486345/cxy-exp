# APD-CATCH

This working tree retains the original CATCH project and its task commands.
The independent MNRF-Lite V2 implementation is documented below and does not
modify the original CATCH baseline or its scripts.

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

## MNRF-Lite V2

An independent two-stage detector is available as `MNRFLiteV2`. It keeps the
official train/test boundary, performs purged train/validation/calibration
splits, and evaluates only complete stride-one windows. For example, from the
repository root:

```bash
bash ./scripts/multivariate_detection/detect_score/PSM_script/MNRFLiteV2.sh
```

Use `GPU_ID=0` for the first already-visible GPU or `GPU_ID=cpu` for explicit
CPU mode; existing `CUDA_VISIBLE_DEVICES` restrictions are preserved. Override
`BATCH_SIZE`, `ENCODING_BATCH_SIZE`, stage epoch limits, the I/O-only cache
chunk size, `EVOLUTION_DIM`, `PATTERN_DIM`, `RELATION_DIM`, and `MNRF_SAVE_PATH`
explicitly. The S/E/P/R defaults are `1/8/8/8`; S remains the direct scalar
observation. Check per-task continuity and
post-purge windows without training:

```bash
conda run --no-capture-output -n catch_env python scripts/mnrf_lite_v2_preflight.py --task PSM --data-source large_detect
```

See [MNRF_LITE_V2.md](MNRF_LITE_V2.md) for the model, training, scoring
protocol, temporal cadence/timezone contract, epoch-boundary resume semantics,
and per-task preflight status. Normal cadence is fitted from the training
subset and frozen for validation/test. No performance conclusion is implied by
the implementation or smoke tests.
