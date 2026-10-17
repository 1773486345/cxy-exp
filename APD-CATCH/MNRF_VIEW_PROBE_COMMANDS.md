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
GPU_ID=0 VIEW_PROBE_MODE=s_only VIEW_PROBE_VIEWS=S VIEW_PROBE_SOURCE_DIR=<STAGE1_RUN> VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/S-only-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
GPU_ID=0 VIEW_PROBE_MODE=select_gmm VIEW_PROBE_VIEWS=S,P,R VIEW_PROBE_SOURCE_DIR=<STAGE1_RUN> VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/drop-E-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
GPU_ID=0 VIEW_PROBE_MODE=select_gmm VIEW_PROBE_VIEWS=S,E,R VIEW_PROBE_SOURCE_DIR=<STAGE1_RUN> VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/drop-P-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
GPU_ID=0 VIEW_PROBE_MODE=select_gmm VIEW_PROBE_VIEWS=S,E,P VIEW_PROBE_SOURCE_DIR=<STAGE1_RUN> VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/drop-R-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
GPU_ID=0 VIEW_PROBE_MODE=select_gmm VIEW_PROBE_VIEWS=E,P,R VIEW_PROBE_SOURCE_DIR=<STAGE1_RUN> VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/drop-S-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
```

Existing Flow control, same source and full layout only:

```bash
GPU_ID=0 VIEW_PROBE_MODE=flow VIEW_PROBE_VIEWS=S,E,P,R VIEW_PROBE_SOURCE_DIR=<STAGE1_RUN> VIEW_PROBE_OUTPUT=result/score/MNRF-ViewProbe/PSM/flow-2021 conda run --no-capture-output -n catch_env bash /media/h3c/users/wangyueyang1/cxy/APD-CATCH/scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh
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
