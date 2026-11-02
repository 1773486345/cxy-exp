# MSL Boundary × Scoring-Object Diagnostic

This is a single-task, separately stored controlled comparison. It does not
modify the original MSL CSV, metadata, or Full run.

The verified mapping is
`result/analysis/MNRF-ViewProbe-DataRepair/msl-raw-mapping-20261031.json`.
It records 27 sorted telemetry channels and their train/test row counts. The
current loader-reconstructed float32 feature matrix has the mapping's expected
SHA256 `5fe9f96f80e632f01640f3a59f65e295ec300e084b6700d571120a22b9074a5a`;
the source CSV SHA256 is checked again before the run. A non-feature
`entity_id` is attached after loading: `train:<channel>` and
`test:<channel>`. The existing window protocol uses that identity in addition
to its frozen timestamp cadence, so no window can cross a channel or the
train/test concatenation boundary. All 55 `col_0..col_54` values and the
original labels, order, internal purged split rules and seed are retained.

The new Full run uses the existing MSL values: seed 2021, seq_len 192, batch
and encoding batch 128, 50 Stage1 epochs, patience 5, learning rate 0.001,
weight decay 0.0001, 32-wide hidden state, S/E/P/R dimensions 1/8/8/8,
full-covariance 4-component GMM, reg 0.001, n_init 1, max_iter 100, 20,000
fit-window cap, fixed noise std 0.01/seed 20261013, feature floor 0.05,
score MAD floor 0.001 and calibration q=0.995. Only the sequence-boundary
handling is different from the historical Full run.

After a manually launched new run completes, the read-only comparison
validates both complete GMM runs and derives both score objects from each
run's saved validation, calibration and test per-variable NLL:

1. `all_55_max`: max across all 55 standardized variable NLLs.
2. `col_0_only`: standardized `col_0` NLL; the other 54 fields remain model
   inputs and are not removed from training or the fitted GMM.

Each score object gets its own calibration threshold from the same run's
calibration endpoints using linear q=0.995 and prediction `score > threshold`.
The four cells are old concatenated/all-55, old concatenated/col_0, new
sequence-bounded/all-55 and new sequence-bounded/col_0. They are evaluated on
the exact intersection of old and new valid global test endpoints. No score is
filled or interpolated. Event metrics use the mapped 27 test-channel
boundaries. The historical 0.159/0.279 AP values are not copied into the new
comparison; the tool recomputes all four cells from saved NLL on the common
coordinates.

Manual entry, from the project root after activating `catch_env`:

```bash
bash scripts/multivariate_detection/detect_score/MSL_script/MNRFBoundaryScoreObject.sh
```

The new source is placed in
`result/score/MNRF-ViewProbe-MSL-Boundary/MSL/<unique-run-id>/`. The four-cell
table and binding record are placed in
`result/analysis/MNRF-ViewProbe-MSL-Boundary/<same-run-id>/`. The launcher
refuses existing output paths. This diagnostic does not run CATCH or any other
baseline.
