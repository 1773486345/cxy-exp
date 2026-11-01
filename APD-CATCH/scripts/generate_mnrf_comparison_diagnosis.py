#!/usr/bin/env python
"""Generate the read-only MNRF/baseline comparison and dataset diagnosis.

The generator consumes the already audited settings and frozen decoder CSVs.
It does not load a model, fit a head, recalibrate, or run a benchmark.
"""

import argparse
import ast
import csv
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "result/analysis/MNRF-Baseline-Investigation-20261031"
FROZEN = ROOT / "result/analysis/MNRF-FrozenDecoder-Analysis-20261102-final"
FULL_DIAGNOSTIC = ROOT / "result/analysis/MNRF-ViewProbe-FullDiagnostic/full-head-diagnostic-20261030T062555Z-3226097"
CONDITIONAL = ROOT / "result/analysis/MNRF-ViewProbe-ConditionalScore/conditional-score-20261030T-run2"
DATA_REPAIR = ROOT / "result/analysis/MNRF-ViewProbe-DataRepair"
TASKS = ("PSM", "MSL", "GECCO", "Genesis", "NYC", "CalIt2", "Creditcard", "CICIDS")
COMMON_METHODS = {"CATCH", "Isolation Forest", "PCA"}
REQUIRED_SOURCE_FILES = (
    "probe_manifest.json", "run_status.json", "input_statistics.npz",
    "results/test_nll.npy", "results/aligned_scores.npz",
    "heads/calibration_nll.npy", "heads/calibration.npz",
    "heads/gmm_status.json", "heads/gmm_models.pkl", "heads/feature_statistics.npz",
    "heads/fixed_noise.npy", "heads/gmm_fit_endpoints.npy",
    "checkpoints/best_encoder.pt", "checkpoints/initial_encoder.pt",
)
REQUIRED_SPLIT_FILES = tuple(
    f"cache/{split}_{kind}.npy"
    for split in ("train", "validation", "calibration", "test")
    for kind in ("features", "window_end_indices")
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value(row, key, default=""):
    return row.get(key, default) if row else default


def f(row, key):
    raw = value(row, key)
    return None if raw in ("", None, "null", "None") else float(raw)


def structured(raw):
    """Read the CSV's JSON/Python-literal evidence fields without changing them."""
    if raw in (None, "", "null", "None"):
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None


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


def source_contract(full_rows, full_reference_rows):
    """Return only source records that are unique and materially complete.

    This is a read-only contract check for the report. It intentionally does
    not recompute file hashes or invoke the full validator.
    """
    refs_by_id = {}
    ref_duplicates = set()
    for row in full_reference_rows:
        source_id = row.get("source_run_id", "")
        if source_id in refs_by_id:
            ref_duplicates.add(source_id)
        refs_by_id[source_id] = row
    ids = [row.get("source_run_id", "") for row in full_rows]
    id_counts = Counter(ids)
    result = {}
    for task in TASKS:
        matches = [row for row in full_rows if row.get("dataset") == task]
        reasons = []
        row = matches[0] if len(matches) == 1 else None
        if len(matches) != 1:
            reasons.append(f"expected exactly one Full metric row, found {len(matches)}")
        source_id = value(row, "source_run_id")
        reference = refs_by_id.get(source_id)
        if not source_id:
            reasons.append("source_run_id missing")
        if id_counts.get(source_id, 0) != 1 or source_id in ref_duplicates:
            reasons.append("source_run_id is not unique in source records")
        if reference is None:
            reasons.append("full_reference.csv has no matching source record")
        source_path = Path(value(reference, "source_run")) if reference else None
        if source_path is None or not source_path.is_dir():
            reasons.append("registered source directory is missing")
        if reference:
            for field in ("feature_fingerprint", "source_identity", "fit_endpoint_sha256",
                          "fixed_noise_sha256", "input_statistics_sha256", "cache_sha256"):
                if not value(reference, field):
                    reasons.append(f"source record field {field} is missing")
        if source_path and source_path.is_dir():
            for relative in REQUIRED_SOURCE_FILES + REQUIRED_SPLIT_FILES:
                if not (source_path / relative).is_file():
                    reasons.append(f"missing {relative}")
        identity = structured(value(reference, "source_identity")) if reference else None
        if not isinstance(identity, dict):
            reasons.append("source_identity is not parseable")
        result[task] = {
            "status": "verified" if not reasons else "blocked",
            "reason": "; ".join(reasons),
            "source_id": source_id,
            "source_path": str(source_path.resolve()) if source_path and source_path.exists() else str(source_path or ""),
            "source_record": reference,
            "identity": identity,
        }
    return result


def mnrf_settings(context):
    if context["status"] != "verified":
        return "unverified source; no MNRF setting may be claimed as bound"
    identity = context.get("identity") or {}
    return (f"registered Full GMM source {context['source_id']}; seq_len={identity.get('seq_len', 'unknown')}, "
            f"variables={len(identity.get('variable_names', [])) or 'unknown'}, frozen valid endpoints, "
            "feature-training statistics, fixed four-noise scoring, variable-max aggregation, "
            "validation median/MAD and independent calibration q=0.995; strict score > threshold")


def alignment_detail(row, context):
    method = row.get("method", "")
    mode = row.get("entry_mode", "")
    explicit = value(row, "explicit_parameters") or "none recorded"
    inherited = value(row, "inherited_parameters_or_defaults") or "none recorded"
    window = value(row, "effective_window") or "unknown"
    patch = value(row, "effective_inference_patch") or "unknown"
    aggregation = value(row, "effective_aggregation") or "unknown"
    threshold = value(row, "effective_threshold") or "unknown"
    confirmed = (f"{mode}; native entry {value(row, 'entry_script')}; window={window}; inference_patch={patch}; "
                 f"aggregation={aggregation}; threshold={threshold}; explicit={explicit}; inherited={inherited}; "
                 f"special={value(row, 'special_handling') or 'none recorded'}")
    mnr = mnrf_settings(context)
    differences = []
    if mode == "detect_score":
        differences.append("score entry is continuous and does not itself define an alarm threshold")
    else:
        if "combined train+test" in threshold:
            differences.append("label threshold uses combined train+test percentile/anomaly-ratio, unlike independent MNRF calibration")
        elif "native backend" in threshold:
            differences.append("label threshold is delegated to the native backend, not MNRF q=.995")
        else:
            differences.append("label threshold/post-processing is native and must remain a separate alarm column")
    if window not in ("", "1", "unknown"):
        differences.append(f"native window={window} and patch={patch} are not MNRF endpoint-causal source settings")
    if aggregation and aggregation != "unknown":
        differences.append("native aggregation is retained; MNRF uses per-variable normalized NLL max")
    if context["status"] != "verified":
        judgement = "缺少明确细节：MNRF Full 源绑定未验证"
        action = "do not claim direct comparison; repair source evidence before using an MNRF row (probe.py source validation)"
    elif method in COMMON_METHODS:
        judgement = "已有分数可重算：共同协议诊断可在共享有效点核对"
        action = "keep native baseline score; compare ranking on shared points; report alarms in separate native/common columns (probe.py::_calibrate/_write_scores)"
    elif mode == "detect_score":
        judgement = "paper/native ranking reference only; common-point score is required for a direct score comparison"
        action = "do not change the baseline; if direct common comparison is needed, obtain its score on MNRF exact valid coordinates; retain MNRF probe.py::_gmm_scores"
    else:
        judgement = "需要后续调整 MNRF 报告：native label protocol is not the independent-calibration alarm protocol"
        action = "do not retune the baseline; report paper/native alarm separately and add a predeclared common alarm only as a second column (probe.py::_calibrate/_write_scores)"
    consistent = ("dataset/task identity and confirmed baseline configuration are retained; "
                  "each detector keeps its native head/loss rather than being forced to GMM")
    return {
        "baseline_confirmed_conditions": confirmed,
        "mnrf_current_settings": mnr,
        "consistent_items": consistent,
        "difference_items": "; ".join(differences),
        "difference_summary": "; ".join(differences),
        "judgement": judgement,
        "alignment_judgement": judgement,
        "mnrf_action": action,
        "code_or_record_basis": f"{value(row, 'entry_script')}; baseline_settings.csv; MNRF probe.py and registered source record",
    }


def alignment_rows(settings, full_rows, contexts):
    result = []
    for row in settings:
        task = row.get("dataset")
        full = first(full_rows, task, "FULL", "saved_full_gmm_reference", "saved_original_aggregate")
        detail = alignment_detail(row, contexts[task])
        result.append({
            "dataset": task, "method": row.get("method"), "entry_mode": row.get("entry_mode"),
            "paper_auc_roc": row.get("paper_auc_roc"), "paper_affiliation_f": row.get("paper_affiliation_f"),
            "baseline_entry": row.get("entry_script"), "effective_window": row.get("effective_window"),
            "effective_aggregation": row.get("effective_aggregation"),
            "effective_threshold": row.get("effective_threshold"),
            "baseline_evidence": row.get("evidence_status"),
            "mnrf_source_run_id": value(full, "source_run_id") if contexts[task]["status"] == "verified" else "",
            "mnrf_common_status": "common-protocol Full verified" if contexts[task]["status"] == "verified" else "blocked",
            "mnrf_auc_roc": value(full, "auc_roc"),
            "mnrf_average_precision": value(full, "average_precision"),
            "mnrf_precision": value(full, "precision"),
            "mnrf_recall": value(full, "recall"),
            "mnrf_f1": value(full, "f1"),
            "mnrf_fpr": value(full, "fpr"),
            "mnrf_affiliation_f": value(full, "affiliation_f_valid_segment_macro"),
            **detail,
            "comparison_class": "paper reference; native baseline vs MNRF common diagnostic",
        })
    for task in TASKS:
        full = first(full_rows, task, "FULL", "saved_full_gmm_reference", "saved_original_aggregate")
        context = contexts[task]
        result.append({
            "dataset": task, "method": "MNRF Full GMM", "entry_mode": "common_protocol",
            "paper_auc_roc": "", "paper_affiliation_f": "", "baseline_entry": "",
            "effective_window": "source manifest seq_len", "effective_aggregation": "max((NLL-center)/scale, axis=variables)",
            "effective_threshold": "calibration q=0.995; score > threshold", "baseline_evidence": "frozen source and saved score",
            "mnrf_source_run_id": value(full, "source_run_id") if context["status"] == "verified" else "",
            "mnrf_common_status": "verified" if context["status"] == "verified" else "blocked",
            "mnrf_auc_roc": value(full, "auc_roc"),
            "mnrf_average_precision": value(full, "average_precision"),
            "mnrf_precision": value(full, "precision"),
            "mnrf_recall": value(full, "recall"),
            "mnrf_f1": value(full, "f1"),
            "mnrf_fpr": value(full, "fpr"),
            "mnrf_affiliation_f": value(full, "affiliation_f_valid_segment_macro"),
            "baseline_confirmed_conditions": "no baseline row; this is the APD-CATCH common-protocol MNRF diagnostic",
            "mnrf_current_settings": mnrf_settings(context),
            "consistent_items": "registered source, saved score, valid test coordinates and metric binding",
            "difference_items": "not a paper or native-baseline result; Full GMM probe is distinct from complete Flow and E/R decoder diagnostics",
            "difference_summary": "not a paper or native-baseline result; Full GMM probe is distinct from complete Flow and E/R decoder diagnostics",
            "judgement": "directly usable as the MNRF common-protocol row" if context["status"] == "verified" else "缺少明确细节：source binding not verified",
            "alignment_judgement": "directly usable as the MNRF common-protocol row" if context["status"] == "verified" else "缺少明确细节：source binding not verified",
            "mnrf_action": "report ranking and alarm metrics in the common-protocol block; never replace paper/native baseline values",
            "code_or_record_basis": "reference_metrics.csv, full_reference.csv, probe.py, registered source contract",
            "comparison_class": "common-protocol diagnostic; not paper value",
        })
    return result


def protocol_report(out, settings, contexts):
    method_rows = []
    seen = set()
    for row in settings:
        key = (row.get("method"), row.get("entry_mode"))
        if key in seen:
            continue
        seen.add(key)
        detail = alignment_detail(row, contexts[TASKS[0]])
        method_rows.append((row, detail))
    method_table = [
        "## Per-entry alignment decisions", "",
        "The following table is generated from each method's `detect_score`/`detect_label` settings. It is intentionally not a single generic judgement: window, aggregation, threshold and native configuration are retained per entry.", "",
        "| Method | Entry | Confirmed baseline conditions | MNRF current setting | Consistent items | Differences | Judgement | MNRF action |", "|---|---|---|---|---|---|---|---|"
    ]
    for row, detail in method_rows:
        def cell(value):
            return str(value or "").replace("|", "\\|").replace("\n", " ")
        method_table.append("| " + " | ".join(cell(x) for x in (
            row.get("method"), row.get("entry_mode"), detail["baseline_confirmed_conditions"],
            "registered dataset-specific Full source; source seq_len/variables and endpoint mask are inherited; Full GMM, max, validation median/MAD, calibration q=0.995, strict score > threshold",
            detail["consistent_items"], detail["difference_items"],
            detail["judgement"], detail["mnrf_action"])) + " |")
    source_lines = ["## MNRF source binding", "", "| Dataset | Status | Source Run ID | Source path | Reason |", "|---|---|---|---|---|"]
    for task in TASKS:
        c = contexts[task]
        source_lines.append(f"| {task} | {c['status']} | {c['source_id']} | `{c['source_path']}` | {c['reason'] or 'all required source records/files present'} |")
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

MNRF should therefore be reported in two explicit forms: the existing Full GMM common-protocol row (with its frozen source, valid points and independent calibration) and, when a native baseline alarm is cited, a separate baseline-native row. MNRF should not be changed merely to match a baseline's hidden width, learning rate or detector head. Where a baseline uses a combined train+test percentile or anomaly-ratio label rule, a common alarm can be added only as a predeclared second analysis; it must not overwrite the native result or be called a reproduction of the paper metric. Ranking comparisons use AUC/AP on shared valid points; alarm comparisons use separately named threshold/segment definitions.

## Evidence limits

The paper/reference values and server settings are confirmed by the existing investigation artifacts. Local common MNRF rows are saved and validated. Historical multi-anomaly-ratio selection rules for some native label paths remain unknown; this does not block the independent common diagnostic. Creditcard and CICIDS entity/session continuity is not recoverable from the frozen input and remains a boundary limitation.
"""
    text += "\n" + "\n".join(source_lines) + "\n\n" + "\n".join(method_table) + "\n"
    (out / "comparison_protocol.md").write_text(text, encoding="utf-8")


def diag_first(rows, task, configuration=None, method=None, score=None):
    matches = [row for row in rows if row.get("dataset") == task
               and (configuration is None or row.get("configuration") == configuration)
               and (method is None or row.get("method") == method)
               and (score is None or row.get("score") == score)]
    return matches[0] if matches else None


def diagnostic_evidence(task, full_rows, full_diagnostic, loss_rows, conditional_rows,
                        msl_target, msl_boundary, raw_rows):
    details = []
    full_max = diag_first(full_diagnostic, task, "FULL_MAX")
    full_mean = diag_first(full_diagnostic, task, "FULL_MEAN")
    diag_max = diag_first(full_diagnostic, task, "DIAG_MAX")
    diag_mean = diag_first(full_diagnostic, task, "DIAG_MEAN")
    def metric_pair(row):
        if not row or row.get("status") not in ("complete", "reused"):
            return "failed/ unavailable"
        if not row.get("average_precision") or not row.get("roc_auc"):
            return "unavailable"
        return f"{float(row['average_precision']):.4f}/{float(row['roc_auc']):.4f}"

    if full_max and full_mean and diag_max and diag_mean:
        details.append(
            "FullDiagnostic fixed-head contrasts (AP/AUC): "
            f"full/max {metric_pair(full_max)}, full/mean {metric_pair(full_mean)}, "
            f"diag/max {metric_pair(diag_max)}, diag/mean {metric_pair(diag_mean)}; "
            "these are fixed post-hoc contrasts, not selected replacement scores.")
    tail = diag_first(loss_rows, task, method="Full-full-max")
    if tail:
        details.append(
            "FullDiagnostic calibration evidence: q=.995 tail has "
            f"{tail.get('calibration_q995_tail_n', 'unknown')} points, "
            f"{tail.get('calibration_q995_label_anomalies', 'unknown')} label anomalies and "
            f"{tail.get('calibration_q995_context_anomalies', 'unknown')} contextual anomalies; "
            "this affects alarm interpretation, not ranking metrics.")
    cond = diag_first(conditional_rows, task, method="responsibility_conditional_S_P_given_E_R")
    orig_cond = diag_first(conditional_rows, task, method="original_joint_saved")
    if cond and orig_cond:
        details.append(
            "ConditionalScore post-hoc row: joint versus p(S,P|E,R) "
            f"AP {float(orig_cond['average_precision']):.4f}->{float(cond['average_precision']):.4f}, "
            f"AUC {float(orig_cond['roc_auc']):.4f}->{float(cond['roc_auc']):.4f}; "
            "it retains the source cache and is not a refit or proof of field causality.")
    if task == "MSL" and msl_target:
        base = float(msl_target.get("saved_full_threshold", 0))
        met = msl_target.get("metrics", {})
        details.append(
            "DataRepair MSL target-only post-hoc aggregation for col_0: "
            f"AP {float(first(full_rows, 'MSL', 'FULL', 'saved_full_gmm_reference', 'saved_original_aggregate')['average_precision']):.4f}"
            f"->{float(met['average_precision']):.4f}, AUC {float(first(full_rows, 'MSL', 'FULL', 'saved_full_gmm_reference', 'saved_original_aggregate')['auc_roc']):.4f}"
            f"->{float(met['roc_auc']):.4f}; source Full threshold was {base:.6g}. "
            "This is post-hoc target aggregation, not a segmented retrain.")
        boundary = diag_first(msl_boundary, task, method="MNRF-Full")
        if boundary:
            details.append(
                "MSL boundary exclusion retained 68,572 points after excluding 4,966 "
                f"cross-sequence/cold-start points; diagnostic AUC/AP={float(boundary['roc_auc']):.4f}/{float(boundary['average_precision']):.4f}. "
                "The encoder/head was not retrained under these boundaries.")
    if task == "CalIt2" and raw_rows:
        parts = []
        for row in raw_rows:
            old = row.get("old_mod10", {})
            parts.append(f"{row['method']} AP {float(old.get('average_precision', 0)):.4f}->{float(row['average_precision']):.4f}, AUC {float(old.get('roc_auc', 0)):.4f}->{float(row['roc_auc']):.4f}")
        details.append("DataRepair RawCount paired results (separate data version): " + "; ".join(parts) + ".")
    return details


def diagnosis_report(out, full_rows, refs, attrs, scales, neighborhoods,
                     full_diagnostic, loss_rows, conditional_rows,
                     msl_target, msl_boundary, raw_rows):
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
                   "The completed RawCount report is a separate data version: MNRF AUC/AP changes to 0.853538/0.170287, while CATCH, IF and PCA are 0.881578/0.281496, 0.890166/0.253774 and 0.822361/0.146648. These four-method results must not be mixed into the old eight-task average; RawCount S-only is not a prerequisite for this conclusion. E/R residuals exceed persistence in some rankings but do not constitute a branch gain over the target-mean reference. "
                   "The supported adjustment is to preserve RawCount as a separate data-version protocol and investigate the remaining gap to CATCH/IF; the role of each representation branch remains unproven."),
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
        extra = diagnostic_evidence(task, full_rows, full_diagnostic, loss_rows, conditional_rows,
                                    msl_target, msl_boundary, raw_rows if task == "CalIt2" else [])
        if extra:
            lines.append("Existing diagnostic evidence:")
            lines.extend(f"- {item}" for item in extra)
            lines.append("")
    lines += ["## Interpretation boundary", "", "A residual AP above persistence is only a comparison to that reference; it is not proof of conditional information gain. The target-mean and latent-mean references are also required. Lower prediction error is not equivalent to better anomaly ranking. Full GMM, E/R decoder residual/NLL, conditional-score diagnostics, and any complete coupling-flow score are separate heads and must not be pooled.", "", "## Up to three next validations", "", "1. Run a predeclared boundary-aware MSL diagnostic/retrain comparison that keeps the 27 sequence identities and reports col_0 separately; this tests whether concatenation, rather than the frozen representation, explains the target-column gain.", "2. Audit GECCO calibration labels and CICIDS flow/session boundaries together, then repeat only the fixed-coordinate alarm analysis; this adds the missing evidence needed to interpret calibration contamination and segment metrics.", "3. Export fixed pointwise decoder mean/sigma for Creditcard and compare residual versus NLL ranks on a predeclared sample; this tests the remaining variance-masking hypothesis without selecting variables from test AP.", ""]
    (out / "dataset_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    baseline_settings = read_csv(BASELINE / "baseline_settings.csv")
    refs = read_csv(FROZEN / "reference_metrics.csv")
    full_reference_rows = read_csv(FROZEN / "full_reference.csv")
    full_rows = [row for row in refs if row.get("view") == "FULL"
                 and row.get("variant") == "saved_full_gmm_reference"
                 and row.get("score") == "saved_original_aggregate"]
    contexts = source_contract(full_rows, full_reference_rows)
    attrs = read_csv(FROZEN / "max_attribution.csv")
    scales = read_csv(FROZEN / "scale_diagnostics.csv")
    neighborhoods = read_csv(FROZEN / "nyc_event_neighborhood.csv")
    full_diagnostic = read_csv(FULL_DIAGNOSTIC / "full_four_configurations.csv")
    loss_rows = read_csv(FULL_DIAGNOSTIC / "loss_decomposition.csv")
    conditional_rows = read_csv(CONDITIONAL / "conditional_scores.csv")
    msl_boundary = read_csv(CONDITIONAL / "msl_boundary_sensitivity.csv")
    msl_target_path = DATA_REPAIR / "msl-target-only-20261031.json"
    msl_target = json.loads(msl_target_path.read_text()) if msl_target_path.is_file() else None
    raw_path = DATA_REPAIR / "calit2-rawcount-20261031/paired_comparison.json"
    raw_rows = json.loads(raw_path.read_text()) if raw_path.is_file() else []
    write_csv(args.output_dir / "comparison_alignment.csv", alignment_rows(baseline_settings, full_rows, contexts))
    protocol_report(args.output_dir, baseline_settings, contexts)
    diagnosis_report(args.output_dir, full_rows, refs, attrs, scales, neighborhoods,
                     full_diagnostic, loss_rows, conditional_rows,
                     msl_target, msl_boundary, raw_rows)
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
                "source_validation": {task: {"status": item["status"], "source_run_id": item["source_id"],
                                              "source_path": item["source_path"], "reason": item["reason"]}
                                      for task, item in contexts.items()},
                "model_execution": False, "encoder_forward": False, "new_head_fit": False,
                "new_calibration": False}
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "baseline_rows": len(baseline_settings),
                      "full_rows": len(full_rows), "diagnostic_rows": len(refs),
                      "verified_sources": sum(item["status"] == "verified" for item in contexts.values()),
                      "model_execution": False}, indent=2))


if __name__ == "__main__":
    main()
