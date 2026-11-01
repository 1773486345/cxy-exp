#!/usr/bin/env python
"""Generate the read-only MNRF/baseline comparison and dataset diagnosis.

The generator consumes the already audited settings and frozen decoder CSVs.
It does not load a model, fit a head, recalibrate, or run a benchmark.
"""

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "result/analysis/MNRF-Baseline-Investigation-20261031"
FROZEN = ROOT / "result/analysis/MNRF-FrozenDecoder-Analysis-20261102-final"
TASKS = ("PSM", "MSL", "GECCO", "Genesis", "NYC", "CalIt2", "Creditcard", "CICIDS")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value(row, key, default=""):
    return row.get(key, default) if row else default


def f(row, key):
    raw = value(row, key)
    return None if raw in ("", None, "null", "None") else float(raw)


def fmt(number, digits=6):
    return "unknown" if number is None else f"{number:.{digits}f}"


def rows_for(rows, task, view=None, variant=None, score=None):
    return [row for row in rows if row.get("dataset") == task
            and (view is None or row.get("view") == view)
            and (variant is None or row.get("variant") == variant)
            and (score is None or row.get("score") == score)]


def first(rows, task, view=None, variant=None, score=None):
    matches = rows_for(rows, task, view, variant, score)
    return matches[0] if matches else None


def metric_line(row):
    return (f"AUC={fmt(f(row, 'auc_roc'))}, AP={fmt(f(row, 'average_precision'))}, "
            f"P/R/F1={fmt(f(row, 'precision'))}/{fmt(f(row, 'recall'))}/{fmt(f(row, 'f1'))}, "
            f"FPR={fmt(f(row, 'fpr'))}, TP/FP/FN/TN={value(row, 'tp')}/{value(row, 'fp')}/"
            f"{value(row, 'fn')}/{value(row, 'tn')}, "
            f"segments hit/scored={value(row, 'anomaly_segments_hit')}/{value(row, 'anomaly_segments_scored')}")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def alignment_rows(settings, full_rows):
    result = []
    for row in settings:
        task = row.get("dataset")
        full = first(full_rows, task, "FULL", "saved_full_gmm_reference", "saved_original_aggregate")
        result.append({
            "dataset": task, "method": row.get("method"), "entry_mode": row.get("entry_mode"),
            "paper_auc_roc": row.get("paper_auc_roc"), "paper_affiliation_f": row.get("paper_affiliation_f"),
            "baseline_entry": row.get("entry_script"), "effective_window": row.get("effective_window"),
            "effective_aggregation": row.get("effective_aggregation"),
            "effective_threshold": row.get("effective_threshold"),
            "baseline_evidence": row.get("evidence_status"),
            "mnrf_source_run_id": value(full, "source_run_id"),
            "mnrf_common_status": "common-protocol Full available" if full else "unknown",
            "mnrf_auc_roc": value(full, "auc_roc"),
            "mnrf_average_precision": value(full, "average_precision"),
            "mnrf_precision": value(full, "precision"),
            "mnrf_recall": value(full, "recall"),
            "mnrf_f1": value(full, "f1"),
            "mnrf_fpr": value(full, "fpr"),
            "mnrf_affiliation_f": value(full, "affiliation_f_valid_segment_macro"),
            "alignment_judgement": "conditional: ranking can compare only on identical valid points; alarm metrics remain protocol-separated",
            "difference_summary": "native baseline head/loss/window/threshold are retained; see effective_* fields",
            "comparison_class": "paper reference; native baseline vs MNRF common diagnostic",
        })
    for task in TASKS:
        full = first(full_rows, task, "FULL", "saved_full_gmm_reference", "saved_original_aggregate")
        result.append({
            "dataset": task, "method": "MNRF Full GMM", "entry_mode": "common_protocol",
            "paper_auc_roc": "", "paper_affiliation_f": "", "baseline_entry": "",
            "effective_window": "source manifest seq_len", "effective_aggregation": "max((NLL-center)/scale, axis=variables)",
            "effective_threshold": "calibration q=0.995; score > threshold", "baseline_evidence": "frozen source and saved score",
            "mnrf_source_run_id": value(full, "source_run_id"),
            "mnrf_common_status": "available" if full else "blocked",
            "mnrf_auc_roc": value(full, "auc_roc"),
            "mnrf_average_precision": value(full, "average_precision"),
            "mnrf_precision": value(full, "precision"),
            "mnrf_recall": value(full, "recall"),
            "mnrf_f1": value(full, "f1"),
            "mnrf_fpr": value(full, "fpr"),
            "mnrf_affiliation_f": value(full, "affiliation_f_valid_segment_macro"),
            "alignment_judgement": "common coordinates and source binding verified for the saved Full GMM diagnostic",
            "difference_summary": "MNRF Full GMM probe, validation median/MAD and independent calibration q=0.995",
            "comparison_class": "common-protocol diagnostic; not paper value",
        })
    return result


def protocol_report(out):
    text = """# MNRF and Baseline Comparison Protocol

## Scope and evidence classes

This report accepts the confirmed baseline premise: the baseline settings and paper-table results are valid. Absence of baseline result files in this checkout is not treated as a failed experiment. The numeric paper reference is the ICLR 2025 conference-paper Table 2; `baseline_settings.csv` records the server entry points and effective settings for both `detect_score` and `detect_label`. CATCH-master archive rows, where available, are original-protocol records. APD-CATCH rows are common-protocol diagnostics. These three classes are never merged into one score.

## What produces the MNRF table score

The table's MNRF score is the saved Full GMM probe, not the complete MNRF-Lite V2 flow and not the frozen E/R decoder diagnostics. `views.py:44-83` constructs per-window causal state/evolution/pattern/relation views and concatenates the layout. The complete model's optional coupling head is `fusion.py:63-125` (`AlternatingAffineFlow`); its training NLL is implemented in `training.py:327-334` and stage-2 fitting in `training.py:358-460`. No Full table row in this report silently substitutes that flow.

For the Full GMM probe, `probe.py:1128-1354` fits one GMM per original variable on the selected 25 feature coordinates and the fixed fit-endpoint sample. `probe.py:1359-1393` scores each variable with the saved model and averages the four fixed-noise evaluations. `probe.py:1395-1425` computes validation median and MAD scale, then the calibration 0.995 quantile. `probe.py:1427-1455` aggregates the normalized variable scores by a maximum and predicts with strict `score > threshold`. Thus, for endpoint t and variable c:

`nll[t,c] = mean_j(-log p_gmm((feature[t,c,selected] - center_feature[c]) / scale_feature[c] + noise_j))`;

`z[t,c] = (nll[t,c] - median(validation_nll[c])) / max(1.4826*MAD(validation_nll[c]), score_mad_floor)`;

`score[t] = max_c z[t,c]`, and `threshold = quantile(max_c z_calibration[t,c], 0.995)`.

The E/R decoder rows in `MNRF-FrozenDecoder-Analysis-20261102-final/reference_metrics.csv` use the auxiliary decoder's squared residual or Gaussian NLL, with their own validation median/MAD and calibration threshold. They are diagnostics of conditional readout quality, not the table's Full detector. The coupling flow is a separate complete-model design path; the Full table does not prove it was used.

## Alignment decisions

The directly comparable MNRF row is the common-protocol Full GMM row when the baseline comparison uses the same source version, labels, valid test coordinates and metric definition. Its ranking metrics (ROC-AUC and average precision) can be compared after exact coordinate alignment; changing a threshold cannot change either ranking metric. Point P/R/F1/FPR and Affiliation metrics must remain separate when a baseline uses a different threshold source, point adjustment, flattened full-window coverage, or segment definition.

MNRF currently uses frozen source train/validation/calibration/test endpoints, no second split, preserved labels and strict endpoint `score > threshold`. Its standardization statistics are fit on the frozen feature-training cache; the alarm threshold is independent calibration q=.995. The baseline settings table shows that native CATCH and several transformer adapters use full non-overlapping windows flattened to positions, while label paths use their own train+test percentile/anomaly-ratio conventions. Those native methods retain their heads, losses, contexts and aggregation. Equal hidden widths, learning rates, epochs, or a forced GMM head are not fairness requirements.

For a fair next comparison, report three side-by-side blocks: (1) paper Table 2 values, (2) original-protocol archive values with their native threshold/post-processing, and (3) common-protocol values on the shared valid coordinates. Do not relabel a common calibrated MNRF alarm as the paper's Aff-F. If a baseline's historical point score uses point adjustment or test-derived anomaly ratio, retain that fact in its original-protocol column rather than silently converting it.

## Executable comparison settings

For MNRF, use the registered Full source for each task, preserve its input hashes, split/end-point hashes, variable order and valid mask, and report the saved Full GMM score as the primary common diagnostic. Use the source's native sequence length and variables, CPU/GPU only as an execution resource, and no test-label tuning. Report ranking metrics on the exact valid test points. Report alarms with the independent calibration threshold and strict `score > threshold`, while putting native baseline alarms in their own protocol column. Affiliation precision/recall/F must use the same valid contiguous intervals and boundaries; no point adjustment or gap concatenation is allowed.

The resulting comparison is fair at the data identity, evaluated coordinates, provenance and metric-definition layers while preserving each method's native detector. It is not a claim that all heads or threshold protocols are mathematically identical.

## Evidence limits

The paper/reference values and server settings are confirmed by the existing investigation artifacts. Local common MNRF rows are saved and validated. Historical multi-anomaly-ratio selection rules for some native label paths remain unknown; this does not block the independent common diagnostic. Creditcard and CICIDS entity/session continuity is not recoverable from the frozen input and remains a boundary limitation.
"""
    (out / "comparison_protocol.md").write_text(text, encoding="utf-8")


def diagnosis_report(out, full_rows, refs, attrs, scales, neighborhoods):
    findings = {
        "PSM": ("Full ranking is useful (" + metric_line(first(full_rows, "PSM", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "). "
                "R squared-residual is a stronger diagnostic ranking than E, but its calibrated alarm still has high FPR; E/R residuals do not explain the Full score by themselves. "
                "Evidence-supported adjustment: inspect calibration upper tail and max-variable attribution before changing the head. A fixed-threshold recalibration on the same scores is the smallest test; it cannot be used to claim an AUC/AP change."),
        "MSL": ("Full is moderate in ranking and low in recall at its fixed threshold (" + metric_line(first(full_rows, "MSL", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "). "
                "E/R residual diagnostics are close to simple references, while the prior audit records 27 telemetry sequences concatenated into 55 fields. The sequence-boundary effect and mixed target/condition roles are supported concerns, not proven causal causes. "
                "Minimum test: score the existing frozen outputs on non-cross-sequence coordinates, then separately report col_0; a boundary-aware retrain would be a later experiment."),
        "GECCO": ("The Full GMM ranking is poor while E/R residual rankings are high (Full " + metric_line(first(full_rows, "GECCO", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "; R residual " + metric_line(first(refs, "GECCO", "R", "decoder", "squared_residual")) + "). "
                  "This separates a density/aggregation/calibration issue from a lack of information in the frozen readout; calibration contamination is possible because the source audit records labeled anomalies in fitting ranges, but it is not proved as the sole cause. "
                  "Minimum test: replay the same cached NLL with a fixed non-max diagnostic only for analysis, and compare calibration-tail composition without selecting a test-optimal threshold."),
        "Genesis": ("Full has strong AUC but weak AP and alarm precision (" + metric_line(first(full_rows, "Genesis", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "). "
                    "R residual improves over persistence in ranking but its fixed threshold still produces false positives; only three scored anomaly segments limit generalization. "
                    "Minimum test: inspect normal high-score variable attribution and calibration-tail windows; any threshold change must be a predeclared calibration protocol, not a test optimization."),
        "NYC": ("Full ranking is above random but its fixed alarm has no true positives (" + metric_line(first(full_rows, "NYC", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "). "
                "The frozen E/R report has one scored event and k=8 neighborhood summaries; lower prediction error in calendar/target readout does not establish anomaly information. Passenger target and calendar context must be kept distinct. "
                "Minimum test: one predeclared target-only/context-only scoring comparison on the same event boundaries, explicitly marked as single-event evidence."),
        "CalIt2": ("The historical modulo-10 source has low Full AP (" + metric_line(first(full_rows, "CalIt2", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "). "
                   "The existing RawCount report is a separate data version and showed improved MNRF/CATCH/IF/PCA ranking, but no RawCount S-only result was verified; it must not be mixed into the old eight-task average. E/R residuals exceed persistence in some rankings but do not constitute a branch gain over the target-mean reference. "
                   "Minimum test: compare old and RawCount Full under the same coordinate/calibration contract, then run the already prepared RawCount S-only head if needed."),
        "Creditcard": ("Full ranking is high but AP and calibrated precision are low relative to its AUC (" + metric_line(first(full_rows, "Creditcard", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "). "
                       "R squared-residual AP is 0.356110 while R auxiliary NLL AP is 0.001756; the ranking loss is already present at the single-variable NLL layer, and cross-variable scaling/max aggregation can worsen it. Pointwise mu/sigma are not saved, so high-variance masking is not established. "
                       "Minimum test: export fixed pointwise mu/sigma for a predeclared sample and compare per-variable residual/NLL ranks without choosing variables from test AP."),
        "CICIDS": ("Full ranking is near random and the fixed alarm has no true positives (" + metric_line(first(full_rows, "CICIDS", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "). "
                   "E/R residuals are also weak at the alarm threshold, although their ranking can exceed Full. The source lacks verified entity/session continuity, so row adjacency and event interpretation remain unknown. "
                   "Minimum test: obtain a source-level session/flow boundary mapping; until then, treat boundary-sensitive Affiliation and window interpretation as limited rather than changing the score."),
    }
    lines = ["# Dataset Diagnosis", "", "This is a read-only synthesis of the existing Full GMM and frozen E/R CSVs. No model, encoder, decoder, GMM or calibration was run. All values below retain the source dataset version and valid-score protocol.", "", "## Cross-dataset score snapshot", "", "| Dataset | Full AUC | Full AP | Full P/R/F1 | Full FPR | R residual AP/AUC | E residual AP/AUC |", "|---|---:|---:|---|---:|---:|---:|"]
    for task in TASKS:
        full = first(full_rows, task, "FULL", "saved_full_gmm_reference", "saved_original_aggregate")
        rrow = first(refs, task, "R", "decoder", "squared_residual")
        erow = first(refs, task, "E", "decoder", "squared_residual")
        lines.append(f"| {task} | {fmt(f(full,'auc_roc'))} | {fmt(f(full,'average_precision'))} | {fmt(f(full,'precision'))}/{fmt(f(full,'recall'))}/{fmt(f(full,'f1'))} | {fmt(f(full,'fpr'))} | {fmt(f(rrow,'average_precision'))}/{fmt(f(rrow,'auc_roc'))} | {fmt(f(erow,'average_precision'))}/{fmt(f(erow,'auc_roc'))} |")
    lines += ["", "## Evidence and adjustment by dataset", ""]
    for task in TASKS:
        full = first(full_rows, task, "FULL", "saved_full_gmm_reference", "saved_original_aggregate")
        lines += [f"### {task}", "", findings[task], "",
                  f"Evidence record: `reference_metrics.csv` Full row with source `{value(full, 'source_run_id')}`; supporting files are `variable_metrics.csv`, `max_attribution.csv`, and `scale_diagnostics.csv` in the frozen analysis directory.",
                  "", "Evidence classification: the metric values and saved attribution/floor counts are directly recorded; causal explanations below are labelled as supported concerns or hypotheses and are not treated as proven.", ""]
        scale_rows = [row for row in scales if row.get("dataset") == task and row.get("view") in ("E", "R")]
        if scale_rows:
            floors = ", ".join(f"{row['view']}/{row['score']}: {row['scale_floor_active_count']} of {row['variable_count']} scales at floor" for row in scale_rows)
            lines.append(f"Scale evidence: {floors}.")
            lines.append("")
        top = [row for row in attrs if row.get("dataset") == task and row.get("view") == "R" and row.get("score") == "squared_residual" and row.get("population") in ("normal", "anomaly")]
        top.sort(key=lambda row: -float(row.get("max_attribution_share") or 0))
        if top:
            shown = ", ".join(f"{row['population']}:{row['variable']} {float(row['max_attribution_share']):.3f}" for row in top[:4])
            lines.append(f"R max attribution evidence (largest shares): {shown}.")
            lines.append("")
        if task == "NYC":
            event_rows = [row for row in neighborhoods if row.get("dataset") == task]
            lines.append(f"NYC fixed-neighborhood rows available: {len(event_rows)}; this is one scored event and is not a general mechanism test.")
            lines.append("")
    lines += ["## Interpretation boundary", "", "A residual AP above persistence is only a comparison to that reference; it is not proof of conditional information gain. The target-mean and latent-mean references are also required. Lower prediction error is not equivalent to better anomaly ranking. Full GMM, E/R decoder residual/NLL, and any complete coupling-flow score are separate heads and must not be pooled.", "", "## Up to three next validations", "", "1. Freeze the common coordinates and calibration contract, then run one predeclared score-head replay on GECCO/MSL to separate density/aggregation effects from representation information.", "2. Run the corrected CalIt2 RawCount Full versus the existing modulo-10 Full, then the prepared RawCount S-only head only if the Full comparison warrants it.", "3. Export fixed pointwise decoder mean/sigma for Creditcard and a boundary-aware MSL/CICIDS coordinate sample; this directly tests variance ranking and sequence/session assumptions without changing the production head.", ""]
    (out / "dataset_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    baseline_settings = read_csv(BASELINE / "baseline_settings.csv")
    refs = read_csv(FROZEN / "reference_metrics.csv")
    full_rows = [row for row in refs if row.get("view") == "FULL"
                 and row.get("variant") == "saved_full_gmm_reference"
                 and row.get("score") == "saved_original_aggregate"]
    attrs = read_csv(FROZEN / "max_attribution.csv")
    scales = read_csv(FROZEN / "scale_diagnostics.csv")
    neighborhoods = read_csv(FROZEN / "nyc_event_neighborhood.csv")
    write_csv(args.output_dir / "comparison_alignment.csv", alignment_rows(baseline_settings, full_rows))
    protocol_report(args.output_dir)
    diagnosis_report(args.output_dir, full_rows, refs, attrs, scales, neighborhoods)
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except (OSError, subprocess.CalledProcessError):
        git_head, git_status = None, None
    metadata = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "git_head": git_head, "git_status_short": git_status,
                "baseline_settings": str((BASELINE / "baseline_settings.csv").resolve()),
                "frozen_analysis": str(FROZEN.resolve()), "datasets": list(TASKS),
                "baseline_rows": len(baseline_settings), "full_rows": len(full_rows),
                "model_execution": False, "encoder_forward": False, "new_head_fit": False,
                "new_calibration": False}
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "baseline_rows": len(baseline_settings),
                      "full_rows": len(full_rows), "diagnostic_rows": len(refs), "model_execution": False}, indent=2))


if __name__ == "__main__":
    main()
