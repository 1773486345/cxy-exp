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
COMMON_PROTOCOL = ROOT / "result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/common_protocol_metrics.csv"
TASKS = ("PSM", "MSL", "GECCO", "Genesis", "NYC", "CalIt2", "Creditcard", "CICIDS")
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


def canonical(value):
    """Canonical form for comparing stored identity dictionaries."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def common_protocol_records():
    """Index actual common-protocol artifacts; do not infer them by name."""
    if not COMMON_PROTOCOL.is_file():
        return {}
    return {(row.get("dataset", ""), row.get("method", "")): row
            for row in read_csv(COMMON_PROTOCOL)}


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
        run_status = {}
        manifest = {}
        if source_path and source_path.is_dir():
            for relative in REQUIRED_SOURCE_FILES + REQUIRED_SPLIT_FILES:
                if not (source_path / relative).is_file():
                    reasons.append(f"missing {relative}")
            try:
                run_status = json.loads((source_path / "run_status.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                reasons.append("run_status.json is unreadable")
            try:
                manifest = json.loads((source_path / "probe_manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                reasons.append("probe_manifest.json is unreadable")
            if run_status.get("status") != "complete":
                reasons.append(f"run_status.status={run_status.get('status', 'missing')!r}, not complete")
        identity = structured(value(reference, "source_identity")) if reference else None
        if not isinstance(identity, dict):
            reasons.append("source_identity is not parseable")
        manifest_identity = manifest.get("source_identity") if isinstance(manifest, dict) else None
        if isinstance(identity, dict) and isinstance(manifest_identity, dict):
            if canonical(identity) != canonical(manifest_identity):
                reasons.append("manifest source_identity does not match registered source record")
        elif source_path and source_path.is_dir():
            reasons.append("manifest source_identity is missing or not parseable")
        if isinstance(manifest, dict) and manifest.get("config", {}).get("task_name"):
            if manifest["config"].get("task_name") != f"{task}.csv":
                reasons.append("manifest task_name does not match dataset")
        if isinstance(manifest, dict) and reference:
            if value(reference, "feature_fingerprint") and manifest.get("feature_fingerprint") != value(reference, "feature_fingerprint"):
                reasons.append("manifest feature_fingerprint does not match source record")
            if value(reference, "fixed_noise_sha256") and manifest.get("fixed_noise_sha256") != value(reference, "fixed_noise_sha256"):
                reasons.append("manifest fixed_noise_sha256 does not match source record")
        result[task] = {
            "status": "verified" if not reasons else "presence_checked",
            "presence_checked": True,
            "run_status": run_status.get("status", "missing"),
            "manifest_checked": bool(manifest),
            "validation_basis": "run_status complete + manifest/source_identity and registered binding match" if not reasons else "file presence checked; complete source binding not established",
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


def alignment_detail(row, context, common_record=None):
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
    common_method = {"CATCH": "CATCH-common", "Isolation Forest": "IsolationForest", "PCA": "PCA"}.get(method)
    common_status = "not_bound"
    common_binding = "no common-protocol artifact bound"
    if common_record:
        common_status = "source_id_match" if common_record.get("source_full_run_id") == context.get("source_id") else "source_id_mismatch"
        common_binding = (f"method={common_record.get('method')}; source_full_run_id={common_record.get('source_full_run_id')}; "
                          f"protocol={common_record.get('protocol')}; parameter_summary={common_record.get('parameter_summary')}")
    if context["status"] != "verified":
        judgement = "缺少明确细节：MNRF Full 源绑定未验证"
        action = "do not claim direct comparison; repair source evidence before using an MNRF row (probe.py source validation)"
    elif method == "PCA":
        judgement = "需要后续调整 MNRF 报告：native PyOD/TODS PCA 与 common sklearn residual PCA 非同一实现"
        action = "keep paper/native PCA separate; common residual-PCA evidence may be reported only in a named common-protocol column"
    elif method == "Isolation Forest":
        judgement = "需要后续调整 MNRF 报告：native Merlion IF 与 common sklearn IF 非同一实现"
        action = "keep paper/native Merlion score separate; common sklearn IF is an explicitly different diagnostic"
    elif method == "CATCH" and mode == "detect_score" and common_status == "source_id_match":
        judgement = "已有绑定的共同协议 CATCH 分数可重算/核对"
        action = "compare only the bound CATCH-common artifact on its exact coordinates; retain paper/native score and label rows separately"
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
        "common_protocol_binding": common_binding,
        "common_protocol_status": common_status,
        "code_or_record_basis": f"{value(row, 'entry_script')}; baseline_settings.csv; common_protocol_metrics.csv; MNRF probe.py and registered source record",
    }


def alignment_rows(settings, full_rows, contexts):
    result = []
    common_records = common_protocol_records()
    for row in settings:
        task = row.get("dataset")
        full = first(full_rows, task, "FULL", "saved_full_gmm_reference", "saved_original_aggregate")
        common_method = {"CATCH": "CATCH-common", "Isolation Forest": "IsolationForest", "PCA": "PCA"}.get(row.get("method"))
        detail = alignment_detail(row, contexts[task], common_records.get((task, common_method)))
        result.append({
            "dataset": task, "method": row.get("method"), "entry_mode": row.get("entry_mode"),
            "paper_auc_roc": row.get("paper_auc_roc"), "paper_affiliation_f": row.get("paper_affiliation_f"),
            "baseline_entry": row.get("entry_script"), "effective_window": row.get("effective_window"),
            "effective_aggregation": row.get("effective_aggregation"),
            "effective_threshold": row.get("effective_threshold"),
            "baseline_evidence": row.get("evidence_status"),
            "mnrf_source_run_id": value(full, "source_run_id") if contexts[task]["status"] == "verified" else "",
            "mnrf_common_status": "common-protocol Full verified" if contexts[task]["status"] == "verified" else "presence_checked_only",
            "mnrf_presence_checked": str(contexts[task].get("presence_checked", False)).lower(),
            "mnrf_auc_roc": value(full, "auc_roc"),
            "mnrf_average_precision": value(full, "average_precision"),
            "mnrf_precision": value(full, "precision"),
            "mnrf_recall": value(full, "recall"),
            "mnrf_f1": value(full, "f1"),
            "mnrf_fpr": value(full, "fpr"),
            "mnrf_affiliation_f": value(full, "affiliation_f_valid_segment_macro"),
            **detail,
            "common_protocol_status": detail.get("common_protocol_status", "not_bound"),
            "common_protocol_binding": detail.get("common_protocol_binding", "no common-protocol artifact bound"),
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
    common_records = common_protocol_records()
    for row in settings:
        task = row.get("dataset")
        common_method = {"CATCH": "CATCH-common", "Isolation Forest": "IsolationForest", "PCA": "PCA"}.get(row.get("method"))
        detail = alignment_detail(row, contexts[task], common_records.get((task, common_method)))
        method_rows.append((row, detail))
    method_table = [
        "## Per-entry alignment decisions", "",
        "The following table is generated from each method's `detect_score`/`detect_label` settings. It is intentionally not a single generic judgement: window, aggregation, threshold and native configuration are retained per entry.", "",
        "| Dataset | Method | Entry | Confirmed baseline conditions | MNRF current setting | Consistent items | Differences | Common binding | Judgement | MNRF action |", "|---|---|---|---|---|---|---|---|---|---|"
    ]
    for row, detail in method_rows:
        def cell(value):
            return str(value or "").replace("|", "\\|").replace("\n", " ")
        method_table.append("| " + " | ".join(cell(x) for x in (
            row.get("dataset"), row.get("method"), row.get("entry_mode"), detail["baseline_confirmed_conditions"],
            mnrf_settings(contexts[row.get("dataset")]),
            detail["consistent_items"], detail["difference_items"], detail["common_protocol_binding"],
            detail["judgement"], detail["mnrf_action"])) + " |")
    source_lines = ["## MNRF source binding", "", "| Dataset | Binding status | run_status | Source Run ID | Source path | Reason |", "|---|---|---|---|---|---|"]
    for task in TASKS:
        c = contexts[task]
        source_lines.append(f"| {task} | {c['status']} | {c['run_status']} | {c['source_id']} | `{c['source_path']}` | {c['reason'] or 'complete status and registered binding match'} |")
    text = """# MNRF and Baseline Comparison Protocol

## Formal comparison settings to use

For the already-confirmed 16-method paper table, retain each method's confirmed native data version, train/test split, target/condition fields, window context, score formula and native alarm rule. Do not replace a native result with a common-protocol score. For a directly comparable MNRF row, use the registered Full source's train/validation/calibration/test split, original row order and entity/segment boundaries; score only the source's valid test endpoints and preserve its invalid/cold-start mask. Report ROC-AUC and average precision on those exact points. Report point precision/recall/F1/FPR and Affiliation values in separate columns with their threshold and segment definitions.

MNRF's common alarm setting is the saved independent calibration set, q=0.995, strict `score > threshold`, with no test-label threshold selection or point adjustment. This is a common-protocol diagnostic, not a replacement for a baseline's native percentile, anomaly-ratio or backend threshold. The paper/native alarm is reported under its own protocol; a threshold change may affect alarm metrics but cannot change ROC-AUC/AP.

Existing MNRF aligned scores can be used to recalculate ranking and common-alarm metrics without new fitting: the eight registered Full GMM rows, their saved `aligned_scores.npz`, labels, masks and calibration records are the reusable set. Existing CATCH-common/IF/PCA artifacts can only be compared when their implementation, source Run ID, configuration and score coordinates are explicitly bound in `common_protocol_metrics.csv`. Native Merlion IF and PyOD/TODS PCA are not interchangeable with common sklearn IF or residual PCA. Other baseline methods have paper/native values but no bound common score in this checkout; they need a separately named adapter or score artifact before a direct common-coordinate comparison. The independent q=.995 MNRF alarm belongs in the common-protocol diagnostic block/appendix, never in the paper Table 2 or native-baseline column.

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
        "PSM": ("[已完成] Full ranking is useful (" + metric_line(first(full_rows, "PSM", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "). "
                "[已完成] FullDiagnostic 的协方差/max-mean 对照和正常高分变量归因已保存；R squared-residual 的排序较强但固定阈值 FPR 较高。 "
                "[尚未证实] E/R 是否解释 Full 的增益仍没有因果证据。 [真正新增验证] 只需在同一分数上预注册校准尾部审计，不改变 AUC/AP。"),
        "MSL": ("[已完成] Full 的排序和固定阈值指标为 " + metric_line(first(full_rows, "MSL", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + ". "
                "[已完成] 27 条序列边界排除诊断及 col_0 目标列事后聚合均已保存（目标列 AP 0.1590→0.2792）；这不是按边界重新训练。 "
                "[尚未证实] 拼接边界与目标/条件混合对训练表示的因果影响。 [真正新增验证] 如需确认，应进行保留序列身份的边界感知重训，而不是把已完成排除诊断写成待完成。"),
        "GECCO": ("[已完成] Full GMM 排序较弱而 E/R residual 排序较高（Full " + metric_line(first(full_rows, "GECCO", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "; R residual " + metric_line(first(refs, "GECCO", "R", "decoder", "squared_residual")) + "). "
                  "[已完成] FullDiagnostic 已比较 full/diag 与 max/mean，并保存校准上尾和变量归因。 [支持但未证实] 校准污染可能影响报警，不等于已证明是唯一原因。 [真正新增验证] 仅需预注册校准标签审计或边界控制。"),
        "Genesis": ("[已完成] Full has strong AUC but weak AP and alarm precision (" + metric_line(first(full_rows, "Genesis", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "). "
                    "FullDiagnostic 的正常高分变量归因、校准尾部和 R/persistence 对照已保存；仅三个 scored anomaly segments 限制外推。 "
                    "[尚未证实] 工况变化是否导致正常高分；[真正新增验证] 仅复核预注册的正常尾部窗口，不以测试标签改阈值。"),
        "NYC": ("[已完成] Full ranking is above random but its fixed alarm has no true positives (" + metric_line(first(full_rows, "NYC", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "). "
                "单一有评分事件的 k=8 neighborhood summaries 已保存，并区分客流目标与日历条件。 [尚未证实] 预测误差降低是否包含异常信息；[真正新增验证] 只可做预注册 target-only/context-only 诊断。"),
        "CalIt2": ("[已完成] The historical modulo-10 source has low Full AP (" + metric_line(first(full_rows, "CalIt2", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "). "
                   "The completed RawCount report is a separate data version: MNRF AUC/AP changes to 0.853538/0.170287, while CATCH, IF and PCA are 0.881578/0.281496, 0.890166/0.253774 and 0.822361/0.146648. These four-method results must not be mixed into the old eight-task average; RawCount S-only is not a prerequisite for this conclusion. E/R residuals exceed persistence in some rankings but do not constitute a branch gain over the target-mean reference. "
                   "[尚未证实] E/R 是否提供稳定 target-mean 以上增量；RawCount 与旧版本不得混合。"),
        "Creditcard": ("[已完成] Full ranking is high but AP and calibrated precision are low relative to its AUC (" + metric_line(first(full_rows, "Creditcard", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "). "
                       "R squared-residual AP is 0.356110 while R auxiliary NLL AP is 0.001756; the ranking loss is already present at the single-variable NLL layer, and cross-variable scaling/max aggregation can worsen it. Pointwise mu/sigma are not saved, so high-variance masking is not established. "
                       "[尚未证实] 没有逐点 mu/sigma，不能断言高 sigma 掩盖异常；[真正新增验证] 仅导出预注册 mu/sigma 样本。"),
        "CICIDS": ("[已完成] Full ranking is near random and the fixed alarm has no true positives (" + metric_line(first(full_rows, "CICIDS", "FULL", "saved_full_gmm_reference", "saved_original_aggregate")) + "). "
                   "E/R residuals are also weak at the alarm threshold, although their ranking can exceed Full. The source lacks verified entity/session continuity, so row adjacency and event interpretation remain unknown. "
                   "[尚未证实] E/R 与 Full 的差异是否来自表示或边界；源没有可验证 entity/session 连续性。 [真正新增验证] 获取 flow/session 边界映射，不改变评分。"),
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
                "source_validation": {task: {"status": item["status"], "presence_checked": item["presence_checked"],
                                              "run_status": item["run_status"], "validation_basis": item["validation_basis"],
                                              "source_run_id": item["source_id"], "source_path": item["source_path"], "reason": item["reason"]}
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
