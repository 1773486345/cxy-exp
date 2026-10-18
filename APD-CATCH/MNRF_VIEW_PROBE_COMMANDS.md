# MNRF View Probe commands

Commands are prepared only; none were launched as a formal experiment. Run
one task, one seed and one head manually. The first PSM command creates the
stage-one encoder, frozen caches, GMM, calibration and result from scratch.
All output paths are independent from MNRF-Lite V2.

Read-only PSM preflight:

```bash
conda run --no-capture-output -n catch_env python /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/mnrf_lite_v2_preflight.py --task PSM --data-source large_detect
```

Initial full four-view probe (`1/8/8/8`):

```bash
GPU_ID=0 VIEW_PROBE_MODE=full_gmm conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
```

The same independent entry is prepared for the two confirmation tasks (run
only after their ordinary read-only preflights):

```bash
GPU_ID=0 VIEW_PROBE_MODE=full_gmm conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/MSL_script/MNRFViewProbe.sh
GPU_ID=0 VIEW_PROBE_MODE=full_gmm conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/GECCO_script/MNRFViewProbe.sh
```

Stage-one training and cache export only:

```bash
GPU_ID=0 VIEW_PROBE_MODE=stage1 VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/stage1-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
```

Same-initialization random four-view GMM (independent output):

```bash
GPU_ID=0 VIEW_PROBE_MODE=random_gmm VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/random-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
```

Reuse an explicit stage-one source for full views (replace the placeholder with
the exact completed stage-one directory):

```bash
GPU_ID=0 VIEW_PROBE_MODE=select_gmm VIEW_PROBE_VIEWS=S,E,P,R VIEW_PROBE_SOURCE_DIR=result/score/MNRF-ViewProbe/PSM/stage1-2021 VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/full-from-source-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
```

S-only and one-branch deletion probes use the same explicit source and each
write a new head/calibration:

```bash
GPU_ID=0 VIEW_PROBE_MODE=s_only VIEW_PROBE_VIEWS=S VIEW_PROBE_SOURCE_DIR="/absolute/path/to/STAGE1_RUN" VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/S-only-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
GPU_ID=0 VIEW_PROBE_MODE=select_gmm VIEW_PROBE_VIEWS=S,P,R VIEW_PROBE_SOURCE_DIR="/absolute/path/to/STAGE1_RUN" VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/drop-E-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
GPU_ID=0 VIEW_PROBE_MODE=select_gmm VIEW_PROBE_VIEWS=S,E,R VIEW_PROBE_SOURCE_DIR="/absolute/path/to/STAGE1_RUN" VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/drop-P-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
GPU_ID=0 VIEW_PROBE_MODE=select_gmm VIEW_PROBE_VIEWS=S,E,P VIEW_PROBE_SOURCE_DIR="/absolute/path/to/STAGE1_RUN" VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/drop-R-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
GPU_ID=0 VIEW_PROBE_MODE=select_gmm VIEW_PROBE_VIEWS=E,P,R VIEW_PROBE_SOURCE_DIR="/absolute/path/to/STAGE1_RUN" VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/drop-S-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
```

Existing Flow control, same source and full layout only:

```bash
GPU_ID=0 VIEW_PROBE_MODE=flow VIEW_PROBE_VIEWS=S,E,P,R VIEW_PROBE_SOURCE_DIR="/absolute/path/to/STAGE1_RUN" VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/flow-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
```

An unequal-dimension probe keeps the same data and protocol but creates a new
encoder/cache identity; it is not a rescaling of an existing 1/8/8/8 run:

```bash
GPU_ID=0 EVOLUTION_DIM=3 PATTERN_DIM=5 RELATION_DIM=7 VIEW_PROBE_MODE=full_gmm VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/full-1-3-5-7-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
```

Use `GPU_ID=cpu`, `BATCH_SIZE`, and `ENCODING_BATCH_SIZE` explicitly when the
shared GPU is busy. No command performs automatic retries or batch search.

After manual runs, create a read-only summary without selecting a best run:

```bash
conda run --no-capture-output -n catch_env python /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/summarize_mnrf_view_probe.py --root /media/h3c/users/wangyueyang1/cxy/APD-CATCH/result/score/MNRF-ViewProbe
```

Validate one completed run without modifying it:

```bash
conda run --no-capture-output -n catch_env python /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/validate_mnrf_view_probe.py /absolute/path/to/run
```

Resume an explicitly interrupted stage-one run in its existing directory:

```bash
GPU_ID=cpu VIEW_PROBE_MODE=stage1 VIEW_PROBE_RESUME_FROM=/absolute/path/to/run VIEW_PROBE_OUTPUT=/absolute/path/to/run conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
```

Compare a probe and an existing CATCH artifact on explicit common coordinates
(the labels file is a read-only full-coordinate `.npy`):

```bash
conda run --no-capture-output -n catch_env python /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/evaluate_mnrf_view_probe_common.py --probe /absolute/probe/results/aligned_scores.npz --catch /absolute/catch/aligned_scores.npz --labels /absolute/labels.npy --output /absolute/comparison.json
```

Only a registered execution manifest can opt into strict CATCH-common; an
unregistered pair is reported as `index_intersection_reference`:

```bash
conda run --no-capture-output -n catch_env python /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/register_mnrf_catch_common.py --probe /absolute/probe/results/aligned_scores.npz --catch /absolute/catch/aligned_scores.npz --labels /absolute/labels.npy --train-indices /absolute/train.npy --validation-indices /absolute/validation.npy --calibration-indices /absolute/calibration.npy --training-evidence /absolute/catch_training_evidence.json --calibration-evidence /absolute/catch_calibration_evidence.json --output /absolute/catch_common_manifest.json
```

For a registered comparison, add the three split files and
`--common-manifest /absolute/catch_common_manifest.json`; invalid evidence or
a threshold different from the calibration artifact exits nonzero. Add
`--reference-mode` only for an intentionally unregistered
`index_intersection_reference` comparison.

The explicit-split CATCH-common adapter (short synthetic or one manually
selected task; this command is prepared, not run by this repository change):

```bash
conda run --no-capture-output -n catch_env python /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/run_mnrf_catch_common.py --data-csv /absolute/task.csv --train-indices /absolute/common-contract/catch_train_indices.npy --validation-indices /absolute/common-contract/catch_validation_indices.npy --calibration-indices /absolute/common-contract/catch_calibration_indices.npy --train-rows /absolute/common-contract/train_rows.npy --common-contract /absolute/common-contract/common_contract.json --task-name TASK.csv --output /absolute/catch-common-run
```

The adapter defaults match the CATCH score path (`score_lambda=0.05`,
inference patch size 32, stride 1); override them only as an explicitly named
protocol variant. The endpoint arrays must be purged, mutually window-disjoint
coordinates. `--train-rows` must contain exactly every raw row in the Probe
internal training range `[train_start, train_stop)`, exported from that run's
`probe_manifest.json` `source_identity.ranges.train`; it is not a window-row
union and includes short valid regions that cannot form a training window.
Strict registration verifies that exact row sequence against the completed
Probe run and requires the versioned v2 common contract. Use separate endpoint
files for train, validation, and calibration; an uncontracted run is
reference-only. The contract freezes the Probe cadence and continuous
segments, so CATCH does not re-estimate cadence from endpoint coverage.

Generate those frozen files from one validated Probe run and the exact source
CSV (the exporter is read-only with respect to both inputs):

```bash
conda run --no-capture-output -n catch_env python /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/export_mnrf_view_probe_common.py --probe-run "/absolute/path/to/validated/PSM/probe-run" --data-csv "/absolute/path/to/PSM.csv" --output-dir "/absolute/path/to/PSM/common-contract" --catch-seq-len 192
```

The exporter verifies the raw feature values, variable order, canonical time
coordinates and all entity/session boundary columns before writing both the
Probe endpoint files and the independently derived `catch_*_indices.npy`
files, plus `train_rows.npy`, `labels.npy`, and `common_contract.json`. It
refuses to overwrite an existing export directory. Use the CATCH files and
contract for the CATCH adapter; use the Probe files for Probe evidence. Do not
hand-author split or label arrays. If a split is shorter than the requested
CATCH context, export fails before creating a usable contract.

To resume a ViewProbe run, pass its exact existing directory through
`VIEW_PROBE_RESUME_FROM` and use the same data, layout, source and GMM/Flow
configuration. A rejected resume is read-only; it does not rewrite the old
run. Flow resumes an in-progress checkpoint at its next epoch, while a
completed Flow checkpoint only finalizes calibration and scoring. Before either
case writes, the checkpoint's model state, optimizer moments, RNG states and
frozen statistics are validated strictly; damaged or cross-run artifacts are
rejected without creating logs or temporary files. Stage-one, GMM and Flow
artifacts must not be mixed across runs. For timestamped common evaluation,
timezone display names may differ after CSV parsing only when the canonical
`timestamp_utc_ns` sequence is identical; actual time changes and aware/naive
semantic changes remain errors.
