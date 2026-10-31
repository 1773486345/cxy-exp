# Proposed common comparison protocol

## Protocol identity

Use the exact registered Full source for each of PSM, MSL, GECCO, Genesis, NYC, CalIt2, Creditcard, and CICIDS. Record raw CSV and metadata hashes, source identity, variable order, train/validation/calibration/test endpoint hashes, and the code/environment used. Do not replace the modulo-10 CalIt2 source silently; RawCount is a separate data version. Keep paper values, CATCH-master original runs, and APD-CATCH common runs in separate protocol classes.

## Data and boundaries

Preserve original row order, labels, and known train-side contamination. Use the Full source's frozen train/validation/calibration/test ranges and valid masks. MSL evaluation must carry the 27 official test sequence ranges and exclude windows crossing a boundary for the boundary-sensitivity diagnostic; this does not claim that the existing encoder was retrained with segmented boundaries. NYC retains the row-to-time mapping and treats passenger count as the observed signal with evidenced calendar fields as context. Creditcard and CICIDS have no verified entity/session IDs in the frozen input; do not invent entity segments.

## Native models and information flow

Each method keeps its native detector, head, loss, and score definition. Reconstruction methods score reconstruction error; discrepancy methods retain their association/discrepancy term; density and classic detectors retain their native decision score. A common causal endpoint rule would change native full-window methods, so native and causal-adapted variants must be named separately. No test label or hidden label-derived cleaning is allowed. Overlap, cold start, tail, and missing-score behavior is recorded per adapter and then mapped to the common valid point set; no implicit zero fill may create an evaluated point. The native CATCH mode=thre example is a full non-overlap window flattened to all positions, whereas MNRF is an endpoint score; this is an alignment difference, not a generic no-future assertion.

## Fitting and thresholding

Fit on the prescribed training range only. Validation may control early stopping. Use the independent calibration range for one model-specific q=0.995 threshold and strict `score > threshold`; never choose a threshold, anomaly ratio, hyperparameter, or report row using test labels. If an original method's historical protocol uses train+test percentile thresholds or an anomaly-ratio sweep, report it only in the original-protocol table and do not call it common-protocol evidence.

## Metrics and reporting

Compute ROC-AUC and average precision on the identical valid test points. Compute point precision/recall/F1/FPR and TP/FP/FN/TN at each method's own independent threshold. Compute Affiliation precision/recall/F with the same contiguous valid intervals, raw time/entity boundaries, and no point adjustment; name segment macro explicitly when used. Preserve no-score anomaly segments as `valid_points=0` rather than deleting them. Report coverage and all boundary exclusions.

## What can and cannot be unified

The scored coordinates, labels, threshold source, comparison symbol, metric definitions, and provenance can be unified. Native model heads, reconstruction/discrepancy formulas, PCA/TODS versus sklearn implementations, Merlion versions, context-field handling, and historical train+test thresholding cannot be made equivalent without changing the method. Therefore the final report must retain three labeled tables: paper reference, original-protocol reproduction, and common-protocol result. Missing artifacts remain unknown; they are not imputed from another method or dataset.

## Required evidence before execution

Resolve the anomaly-ratio selection rule from the existing report-generation path and obtain an explicit boundary decision for Creditcard/CICIDS. The smallest useful execution is a frozen auxiliary-decoder/score-head diagnostic on the registered sources plus the corrected CalIt2 RawCount comparison. Preserve the paper table as a separate reference and expand only after threshold semantics and native-versus-causal alignment are documented; this is not a default eight-dataset x 16-method rerun.
