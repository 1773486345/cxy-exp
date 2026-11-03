# MNRF Full GMM benchmark evaluation

This batch evaluates the eight registered historical `MNRF-Full-GMM` sources against their official test axes. It is not a Full Flow result, a common false-alarm-budget comparison, or a baseline rerun. Each source keeps its saved standardization, max aggregation, calibration threshold (`q=0.995`) and strict `score > threshold` rule. ROC-AUC/AP and point metrics use the benchmark metric implementations. Benchmark Affiliation-F is computed only where the full official test axis is scored; no point adjustment is applied.

## Paper comparison

`paper_comparison.csv` is the eight-dataset side-by-side table. Its 16 method pairs of `*_paper_auc_roc` and `*_paper_affiliation_f` come from the fixed ICLR 2025 Table 2 fields in `result/analysis/MNRF-Baseline-Investigation-20261031/baseline_settings.csv` (one `detect_score` record per method/dataset); the MNRF columns come only from this batch's `main_results.csv`. Paper values are not replaced with local archive or common-protocol scores. MNRF alarm metrics use each Full source's saved q=.995 threshold; they are not a matched false-alarm-budget comparison. MSL and CICIDS now have full-axis metrics under masked cold-start; Creditcard full-axis columns remain blank because the endpoint reproduction gate failed. The original paper AUC and Affiliation-F definitions remain separate columns and are only compared to MNRF's corresponding metrics.

| Dataset | Official scored | Coverage | ROC-AUC | AP | Benchmark Affiliation-F | Point F1 | Normal FPR | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PSM | 87841/87841 | 100% | 0.74555 | 0.43724 | 0.72633 | 0.52125 | 27.27% | complete |
| MSL | 73729/73729 | 100% | 0.60150 | 0.15768 | 0.53440 | 0.12174 | 2.31% | complete: masked C-1 history |
| GECCO | 69261/69261 | 100% | 0.44322 | 0.22330 | 0.28822 | 0.37728 | 0.102% | complete |
| Genesis | 12616/12616 | 100% | 0.89781 | 0.04589 | 0.85282 | 0.10022 | 3.07% | complete |
| NYC | 4416/4416 | 100% | 0.69547 | 0.04317 | 0.67850 | 0 | 0.162% | complete |
| CalIt2 (legacy modulo-10) | 2520/2520 | 100% | 0.71402 | 0.05324 | 0.60876 | 0.03468 | 3.92% | complete |
| Creditcard | 142213/142404 | 99.87% |  |  |  |  |  | blocked: endpoint reproduction tolerance |
| CICIDS | 85116/85116 | 100% | 0.56687 | 0.00460 | 0.21287 | 0.00595 | 1.88% | complete: masked current-only history |

Creditcard remains a blocked row: its conservative masked rule was prepared, but the newly loaded frozen encoder differed from the saved cache at a bounded endpoint check (`max_abs=6.318092346191406e-06` versus the fixed `rtol=1e-5, atol=1e-6` check), so no prefix score was published. Its valid-endpoint ROC-AUC/AP and point metrics remain under `valid_endpoint_*` and are not full-axis results. MSL and CICIDS now have full-axis rows from the explicit masked rules below; these are not the old unmasked history rule.

Cold-start prefixes were produced for PSM (191), GECCO (95), Genesis (191), NYC (191), legacy CalIt2 (191), MSL (191), and CICIDS (191). MSL uses only C-1 test-sequence history; it never borrows the T-9 train tail. CICIDS uses the current observation with all history marked missing because entity/session continuity is unproven. Creditcard used the same conservative rule but was rejected before publication by the endpoint reproducibility gate. Every published bundle preserves old valid scores, coordinates, labels and the frozen threshold; masked bundles also carry `prefix_history_valid` and `prefix_history_global_indices`. For these two masked additions, old points were checked against bound source artifact hashes, coordinates, labels, masks and threshold, and a bounded set of loaded-encoder endpoints was reproduced; the adapter does not claim a second full GMM scoring pass over all old points. The prefix scores used the frozen encoder, fixed GMM, source normalization and fixed noise only; no Stage1 training, GMM fitting, recalibration, or baseline execution occurred. CalIt2 here is the historical modulo-10 data source, not RawCount.

## Files and evidence

- `main_results.csv`: one row per registered source, with official-axis metrics only when complete.
- Full-axis score bundles for the seven completed tasks are under `result/score/MNRF-Comparison-Adapter/<TASK>/benchmark-main-20261103T200322Z-851688/` (masked bundles for MSL and CICIDS use the `-masked-history` sibling directory); each `adapter_manifest.json` binds its `results/aligned_scores.npz` to the fixed source. Strict evaluator JSON files remain in this analysis batch as `scores/<TASK>-evaluation.json` and `*-masked-evaluation.json`.
- `manifest.json`: pinned source plan, source manifest/score hashes, evaluation protocol, execution boundary and environment provenance.
- `cleanup_log.csv`: exact generated-report directories removed and their pre-cleanup `du` byte counts.

The 24 common-protocol baseline artifacts remain in `result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/` and their inventory in `result/analysis/MNRF-Comparison-Adapter-20261102/`. They are separate implementations and remain common-protocol diagnostics, not replacements for the paper's 16-method table. The retained historical integrated diagnosis is `result/analysis/MNRF-Comparison-Diagnosis-20261102T125000Z/`. Corrected-data experiments remain separate: RawCount CalIt2 under `result/analysis/MNRF-ViewProbe-DataRepair/`, and MSL segmentation diagnostics in the corresponding MSL boundary experiment. Neither replaces the historical Full source in this table.

No baseline result was changed. This report does not claim that all eight MNRF rows are complete or that q=.995 gives the same false-alarm budget as other methods' native thresholds.
