# MNRF frozen-cache recovery

This report covers the four pinned Full sources. Only missing feature cache files were considered; no Stage 1, backward, optimizer, GMM fit, Flow, recalibration, or formal ablation was started. A generated file was installed only after its `.npy` byte SHA256 matched the existing manifest.

- MSL: train and test installed with exact manifest hashes; native validator passed.
- PSM: test installed; train forward output hash differed and was rejected.
- CICIDS: test installed; train forward output hash differed and was rejected.
- Creditcard: train and test forward output hashes differed and were rejected.

The source files, manifests, checkpoints, GMM, calibration, NLL and results were not rewritten. See `recovery_report.json` and each task's `result.json` for hashes, shapes, timings and failure diagnostics.
