# MNRF Full GMM benchmark evaluation

This batch evaluates the eight registered historical `MNRF-Full-GMM` sources against their official test axes. It is not a Full Flow result, a common false-alarm-budget comparison, or a baseline rerun. Each source keeps its saved standardization, max aggregation, calibration threshold (`q=0.995`) and strict `score > threshold` rule. ROC-AUC/AP and point metrics use the benchmark metric implementations. Benchmark Affiliation-F is computed only where the full official test axis is scored; no point adjustment is applied.

| Dataset | Official scored | Coverage | ROC-AUC | AP | Benchmark Affiliation-F | Point F1 | Normal FPR | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PSM | 87841/87841 | 100% | 0.74555 | 0.43724 | 0.72633 | 0.52125 | 27.27% | complete |
| MSL | 73538/73729 | 99.74% |  |  |  |  |  | blocked: cross-channel prefix |
| GECCO | 69261/69261 | 100% | 0.44322 | 0.22330 | 0.28822 | 0.37728 | 0.102% | complete |
| Genesis | 12616/12616 | 100% | 0.89781 | 0.04589 | 0.85282 | 0.10022 | 3.07% | complete |
| NYC | 4416/4416 | 100% | 0.69547 | 0.04317 | 0.67850 | 0 | 0.162% | complete |
| CalIt2 (legacy modulo-10) | 2520/2520 | 100% | 0.71402 | 0.05324 | 0.60876 | 0.03468 | 3.92% | complete |
| Creditcard | 142213/142404 | 99.87% |  |  |  |  |  | blocked: entity/session unproven |
| CICIDS | 84925/85116 | 99.78% |  |  |  |  |  | blocked: entity/session unproven |

The three blocked rows retain valid-endpoint ROC-AUC/AP, point metrics and FPR in `main_results.csv` under `valid_endpoint_*`. Those subset values are diagnostics only and are deliberately blank in the full-axis metric columns. Their missing 191-point prefixes cannot be filled without crossing the known MSL channel boundary or assuming undocumented Creditcard/CICIDS entity continuity.

Cold-start prefixes were produced for PSM (191), GECCO (95), Genesis (191), NYC (191), and legacy CalIt2 (191). Before prefix inference, each run reproduced the original saved endpoint scores; a bounded sample of newly loaded frozen-encoder features was compared against source caches. Old valid scores, coordinates, labels, threshold and source bytes were preserved. The additions used the frozen encoder, fixed GMM, source normalization and fixed noise only; no Stage1 training, GMM fitting, recalibration, or baseline execution occurred. CalIt2 here is the historical modulo-10 data source, not RawCount.

## Files and evidence

- `main_results.csv`: one row per registered source, with official-axis metrics only when complete.
- Full-axis score bundles for the five completed tasks are under `result/score/MNRF-Comparison-Adapter/<TASK>/benchmark-main-20261103T200322Z-851688/`; each `adapter_manifest.json` binds its `results/aligned_scores.npz` to the fixed source. Strict evaluator JSON files remain in this analysis batch as `scores/<TASK>-evaluation.json`.
- `manifest.json`: pinned source plan, source manifest/score hashes, evaluation protocol, execution boundary and environment provenance.
- `cleanup_log.csv`: exact generated-report directories removed and their pre-cleanup `du` byte counts.

The 24 common-protocol baseline artifacts remain in `result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/` and their inventory in `result/analysis/MNRF-Comparison-Adapter-20261102/`. They are separate implementations and remain common-protocol diagnostics, not replacements for the paper's 16-method table. The retained historical integrated diagnosis is `result/analysis/MNRF-Comparison-Diagnosis-20261102T125000Z/`. Corrected-data experiments remain separate: RawCount CalIt2 under `result/analysis/MNRF-ViewProbe-DataRepair/`, and MSL segmentation diagnostics in the corresponding MSL boundary experiment. Neither replaces the historical Full source in this table.

No baseline result was changed. This report does not claim that all eight MNRF rows are complete or that q=.995 gives the same false-alarm budget as other methods' native thresholds.
