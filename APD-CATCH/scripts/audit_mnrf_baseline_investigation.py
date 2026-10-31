#!/usr/bin/env python
"""Build a read-only MNRF baseline investigation from existing evidence.

This script never opens model code for execution and never runs a benchmark. It
only inventories checked-in scripts and already published reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path


DATASETS = ["PSM", "MSL", "GECCO", "Genesis", "NYC", "CalIt2", "Creditcard", "CICIDS"]
METHODS = [
    "CATCH", "ModernTCN", "iTransformer", "DualTF", "Anomaly Transformer",
    "DCdetector", "TimesNet", "PatchTST", "DLinear", "NLinear", "AutoEncoder",
    "OCSVM", "Isolation Forest", "PCA", "HBOS", "TFAD",
]

SOURCES = {
    "PSM": "full_gmm-full-20261024T115119Z-3976714",
    "MSL": "full_gmm-full-20261021T112147Z-2470335",
    "GECCO": "full_gmm-full-20261020T204951Z-1642355",
    "Genesis": "full_gmm-full-20261020T205005Z-1645522",
    "NYC": "full_gmm-full-20261020T211341Z-1931786",
    "CalIt2": "full_gmm-full-20261021T112212Z-2474004",
    "Creditcard": "full_gmm-full-20261024T115131Z-3977864",
    "CICIDS": "full_gmm-full-20261024T115157Z-3980319",
}

ARCHIVE_SCORE = {
    "PSM": "run-20260716T222700Z-660928-22488",
    "MSL": "run-20260716T213635Z-58595-28964",
    "GECCO": "run-20260716T213302Z-16971-16047",
    "Genesis": "run-20260716T213208Z-6088-16524",
    "NYC": "run-20260716T222643Z-657156-32728",
    "CalIt2": "run-20260715T183704Z-2374669-5874",
    "Creditcard": "run-20260716T202925Z-3426561-275",
    "CICIDS": "run-20260715T120722Z-1345491-4045",
}

ARCHIVE_LABEL = {
    "PSM": "run-20260716T223734Z-776688-6344",
    "MSL": "run-20260716T232646Z-1365820-1457",
    "GECCO": "run-20260716T213559Z-51278-24592",
    "Genesis": "run-20260716T221132Z-484571-23537",
    "NYC": "run-20260716T223043Z-701607-26288",
    "CalIt2": "run-20260715T184135Z-2437260-7989",
    "Creditcard": "run-20260717T085033Z-4154569-18693",
    "CICIDS": "run-20260715T133707Z-2321802-10349",
}


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def method_rows(root: Path):
    evidence = {
        "CATCH": ("reconstruction autoencoder with RevIN/FFT/patch transformer", "CATCHModel -> reconstruction [B,L,C]", "MSE reconstruction + dc_loss + frequency_loss", "mean((x-xhat)^2,axis=vars)+score_lambda*frequency_error", "overlapping windows folded by provider; score path returns window endpoints", "train+test score percentile for label path; external score strategy otherwise", "direct-code", "ts_benchmark/baselines/catch/CATCH.py:142-434; scripts/run_mnrf_catch_common.py:487-561"),
        "ModernTCN": ("reconstruction anomaly detector", "ModernTCN reconstruction", "reconstruction MSE", "mean squared reconstruction error", "window scores from self_impl/ModernTCN.py", "train+test percentile in detect_label", "direct-code", "ts_benchmark/baselines/self_impl/ModernTCN.py:136-380"),
        "iTransformer": ("transformer reconstruction adapter", "time_series_library model output xhat", "MSELoss", "mean((x-xhat)^2,axis=vars)", "mode=thre overlapping window scores", "train+test percentile in TransformerAdapter.detect_label", "direct-code", "ts_benchmark/baselines/time_series_library/adapters_for_transformers.py:614-816"),
        "DualTF": ("dual time/frequency reconstruction and discrepancy detector", "time and frequency heads", "reconstruction plus KL/discrepancy terms", "temporal reconstruction plus temperature-scaled discrepancy, folded over overlap", "overlapping windows averaged by evaluation counts", "ratio percentile in label path", "direct-code", "ts_benchmark/baselines/self_impl/DualTF.py:533-670"),
        "Anomaly Transformer": ("reconstruction plus association discrepancy", "reconstruction and series/prior attention", "reconstruction and association losses", "softmax(-(series_loss+prior_loss))*reconstruction error", "window scores; provider handles overlap/tail", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/self_impl/AnomalyTransformer.py:339-434"),
        "DCdetector": ("discrepancy detector", "series/prior outputs", "discrepancy/KL objective", "softmax(-(series_loss+prior_loss))", "window scores from detector strategy", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/self_impl/DCdetector.py:332-418"),
        "TimesNet": ("transformer reconstruction adapter", "TimesNet output xhat", "MSELoss", "mean squared reconstruction error", "mode=thre window scores", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/time_series_library/adapters_for_transformers.py:614-816; scripts/multivariate_detection/detect_score/PSM_script/TimesNet.sh"),
        "PatchTST": ("transformer reconstruction adapter", "PatchTST output xhat", "MSELoss", "mean squared reconstruction error", "mode=thre window scores", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/time_series_library/adapters_for_transformers.py:614-816; scripts/multivariate_detection/detect_score/PSM_script/PatchTST.sh"),
        "DLinear": ("linear reconstruction adapter", "DLinear output xhat", "MSELoss", "mean squared reconstruction error", "mode=thre window scores", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/time_series_library/adapters_for_transformers.py:614-816; scripts/multivariate_detection/detect_score/PSM_script/DLinear.sh"),
        "NLinear": ("linear reconstruction adapter", "NLinear output xhat", "MSELoss", "mean squared reconstruction error", "mode=thre window scores", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/time_series_library/adapters_for_transformers.py:614-816; scripts/multivariate_detection/detect_score/PSM_script/NLinear.sh"),
        "AutoEncoder": ("Merlion anomaly model adapter", "merlion.models.anomaly.AutoEncoder", "delegated to Merlion train", "delegated get_anomaly_score; higher score is abnormal", "adapter reindexes returned score and fills absent timestamps with 0", "delegated get_anomaly_label/benchmark strategy", "direct-code/current-version", "ts_benchmark/baselines/merlion/merlion_models.py:43-121; scripts/multivariate_detection/detect_score/PSM_script/AutoEncoder.sh"),
        "OCSVM": ("one-class density boundary", "TODS OCSVMSKI wrapping PyOD OCSVM", "unsupervised fit", "PyOD decision score, higher is abnormal", "row/window adapter output", "benchmark label strategy", "direct-code; runtime-unavailable", "ts_benchmark/baselines/tods/tods_models.py:58-118; ts_benchmark/baselines/tods/third_party/OCSVM_skinterface.py:54-208; scripts/multivariate_detection/detect_score/PSM_script/ocsvmski.sh"),
        "Isolation Forest": ("tree isolation outlier detector", "Merlion IsolationForest in legacy script; sklearn IsolationForest in common protocol", "unsupervised tree fit", "negative score_samples in common runner; adapter delegates legacy score", "row-aligned common contract; legacy adapter reindexes", "common q=.995; legacy backend label path", "direct-code/direct-record", "ts_benchmark/baselines/merlion/merlion_models.py:43-121; scripts/run_mnrf_common_baselines.py:121-176"),
        "PCA": ("reconstruction residual outlier detector", "TODS PCA detector in legacy script; sklearn residual PCA in common protocol", "unsupervised PCA fit", "reconstruction residual (common: mean squared error)", "common row-aligned endpoints; legacy TODS windows", "common q=.995; legacy backend strategy", "direct-code/direct-record", "ts_benchmark/baselines/tods/third_party/PCAODetector.py:61-228; scripts/run_mnrf_common_baselines.py:121-176; scripts/multivariate_detection/detect_score/PSM_script/pcaodetectorski.sh"),
        "HBOS": ("histogram density outlier detector", "TODS HBOS wrapping PyOD HBOS", "unsupervised histogram fit", "independent histogram negative log-density, higher abnormal", "row/window adapter output", "benchmark label strategy", "direct-code; runtime-unavailable", "ts_benchmark/baselines/tods/tods_models.py:58-118; ts_benchmark/baselines/tods/third_party/HBOS.py:53-131; scripts/multivariate_detection/detect_score/PSM_script/hbosski.sh"),
        "TFAD": ("window-level supervised anomaly classifier", "single sigmoid/logit per window folded to timestamps", "BCELoss on reduced window labels; validation helper contains unused MSE", "folded sigmoid probabilities averaged by overlap count", "suspect-window tail and overlap folding; missing scores remain provider-dependent", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/self_impl/TFAD.py:152-218,300-396"),
    }
    rows = []
    ratio_notes = {
        "CATCH": "CATCH.py exposes anomaly_ratio=[0.1,0.5,1,2,3,5,10,15,20,25]; detect_label evaluates each; the final paper-row selection is not inferred here.",
        "ModernTCN": "PSM script sets anomaly_ratio=1; other task scripts are literal per-task settings; evaluator can retain multiple returned ratios.",
        "iTransformer": "TransformerAdapter config contains an anomaly_ratio list; detect_label emits one result per ratio; report selection is external/unknown.",
        "DualTF": "Task scripts may provide fre_anormly_ratio; the benchmark strategy still supports multiple anomaly-ratio outputs; final selection is not inferred.",
        "Anomaly Transformer": "Benchmark strategy can evaluate configured anomaly_ratio candidates; no test-label selection is assumed without an archive record.",
        "DCdetector": "Benchmark strategy can evaluate configured anomaly_ratio candidates; no test-label selection is assumed without an archive record.",
        "TimesNet": "TransformerAdapter emits configured anomaly_ratio candidates; final report-row selection is not established by the entry script.",
        "PatchTST": "TransformerAdapter emits configured anomaly_ratio candidates; final report-row selection is not established by the entry script.",
        "DLinear": "TransformerAdapter emits configured anomaly_ratio candidates; final report-row selection is not established by the entry script.",
        "NLinear": "TransformerAdapter emits configured anomaly_ratio candidates; final report-row selection is not established by the entry script.",
        "AutoEncoder": "Merlion adapter delegates labels to the backend; candidate ratios/selection are not exposed by this wrapper.",
        "OCSVM": "TODS adapter returns score/label arrays; no local multi-ratio selection evidence.",
        "Isolation Forest": "Legacy Merlion label path is delegated; common protocol uses one q=.995 threshold and no anomaly-ratio sweep.",
        "PCA": "Legacy TODS label path is delegated; common protocol uses one q=.995 threshold and no anomaly-ratio sweep.",
        "HBOS": "TODS adapter returns score/label arrays; no local multi-ratio selection evidence.",
        "TFAD": "TFAD label path uses configured percentile candidates; final report-row selection is not established by the entry script.",
    }
    for method in METHODS:
        task, head, loss, score, aggregation, threshold, status, code = evidence[method]
        rows.append({
            "method": method,
            "actual_task": task,
            "output_head_and_shape": head,
            "training_loss_and_joint_training": loss,
            "score_formula_and_direction": score,
            "variable_window_time_aggregation": aggregation,
            "temporal_context_and_future_information": "past/current explicit windows; no future value in endpoint window; exact boundary behavior is adapter/provider specific",
            "cold_start_tail_padding_missing_scores": "provider pads/reindexes or omits endpoint scores; see strategy/provider evidence; not uniform across adapters",
            "threshold_and_test_label_use": threshold,
            "anomaly_ratio_candidates_and_selection": ratio_notes[method],
            "metric_entry_points": "ts_benchmark/evaluation/strategy/anomaly_detect.py and classification_metrics_label.py; common MNRF uses mnrf_valid_detect.py",
            "evidence_status": status,
            "code_evidence": code,
        })
    return rows


METHOD_SCRIPT = {
    "CATCH": "CATCH.sh",
    "ModernTCN": "ModernTCN.sh",
    "iTransformer": "iTransformer.sh",
    "DualTF": "DualTF.sh",
    "Anomaly Transformer": "AnomalyTransformer.sh",
    "DCdetector": "DCdetector.sh",
    "TimesNet": "TimesNet.sh",
    "PatchTST": "PatchTST.sh",
    "DLinear": "DLinear.sh",
    "NLinear": "NLinear.sh",
    "AutoEncoder": "AutoEncoder.sh",
    "Isolation Forest": "IsolationForest.sh",
    "OCSVM": "ocsvmski.sh",
    "HBOS": "hbosski.sh",
    "PCA": "pcaodetectorski.sh",
    "TFAD": "TFAD.sh",
}

PAPER_TABLE = {
    "PSM": {
        "ModernTCN": ("0.593", "0.825"), "iTransformer": ("0.592", "0.854"), "DualTF": ("0.600", "0.725"),
        "Anomaly Transformer": ("0.514", "0.710"), "DCdetector": ("0.501", "0.682"), "TimesNet": ("0.592", "0.842"),
        "PatchTST": ("0.586", "0.831"), "DLinear": ("0.580", "0.831"), "NLinear": ("0.585", "0.843"),
        "AutoEncoder": ("0.650", "0.707"), "OCSVM": ("0.619", "0.531"), "Isolation Forest": ("0.542", "0.620"),
        "PCA": ("0.648", "0.702"), "HBOS": ("0.620", "0.658"), "TFAD": ("0.500", "0.628"),
    },
    "GECCO": {
        "ModernTCN": ("0.952", "0.893"), "iTransformer": ("0.795", "0.839"), "DualTF": ("0.714", "0.701"),
        "Anomaly Transformer": ("0.516", "0.782"), "DCdetector": ("0.555", "0.687"), "TimesNet": ("0.954", "0.894"),
        "PatchTST": ("0.949", "0.906"), "DLinear": ("0.947", "0.893"), "NLinear": ("0.936", "0.882"),
        "AutoEncoder": ("0.769", "0.823"), "OCSVM": ("0.804", "0.666"), "Isolation Forest": ("0.619", "0.424"),
        "PCA": ("0.711", "0.785"), "HBOS": ("0.557", "0.708"), "TFAD": ("0.499", "0.627"),
    },
    "Creditcard": {
        "ModernTCN": ("0.957", "0.744"), "iTransformer": ("0.934", "0.713"), "DualTF": ("0.703", "0.663"),
        "Anomaly Transformer": ("0.552", "0.650"), "DCdetector": ("0.504", "0.632"), "TimesNet": ("0.957", "0.744"),
        "PatchTST": ("0.957", "0.746"), "DLinear": ("0.954", "0.738"), "NLinear": ("0.948", "0.742"),
        "AutoEncoder": ("0.909", "0.561"), "OCSVM": ("0.953", "0.714"), "Isolation Forest": ("0.860", "0.634"),
        "PCA": ("0.871", "0.710"), "HBOS": ("0.951", "0.695"), "TFAD": ("0.500", "0.600"),
    },
}


def _script_command(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if "run_benchmark.py" in line:
            return line.strip()
    return ""


def build_baseline_settings(root: Path):
    common_dir = root / "result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun"
    common = read_csv(common_dir / "common_protocol_metrics.csv")
    common_map = {(r.get("dataset"), r.get("method")): r for r in common}
    catch_summary = read_csv(root.parent / "CATCH-master/result/CATCH_reproduction_summary/physical_task_metrics.csv")
    catch_map = {r.get("physical_task"): r for r in catch_summary}
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            script = root / "scripts/multivariate_detection/detect_score" / (dataset + "_script") / METHOD_SCRIPT[method]
            command = _script_command(script) if script.exists() else ""
            reference = PAPER_TABLE.get(dataset, {}).get(method)
            if method == "CATCH" and dataset in catch_map:
                reference = (catch_map[dataset].get("reproduced_score_auc_roc", ""), catch_map[dataset].get("reproduced_label_affiliation_f", ""))
            common_method = {"Isolation Forest": "IsolationForest", "PCA": "PCA"}.get(method)
            common_row = common_map.get((dataset, common_method), {}) if common_method else {}
            if "--adapter" in command:
                special = "Native model through transformer_adapter; the command's seq_len/batch/model dimensions are dataset-specific."
            elif method == "CATCH":
                special = "Native CATCH score/label entry; dataset-specific model dimensions, window and epoch values are preserved verbatim in server_command."
            elif method in {"AutoEncoder", "Isolation Forest"}:
                special = "Native Merlion adapter; backend defaults are delegated to the installed server package unless command parameters override them."
            elif method in {"OCSVM", "PCA", "HBOS"}:
                special = "Native TODS/PyOD adapter; backend parameters are the literal command values (often empty JSON)."
            else:
                special = "Native self_impl entry; dataset-specific parameters are the literal command values."
            rows.append({
                "method": method,
                "dataset": dataset,
                "entry_script": str(script),
                "script_exists": str(script.exists()).lower(),
                "script_sha256": __import__("hashlib").sha256(script.read_bytes()).hexdigest() if script.exists() else "",
                "server_command": command,
                "implementation_or_adapter": "see --model-name in server_command; native entry, not common adapter",
                "actual_parameters": command.split("--model-hyper-params", 1)[1].split(" --gpus", 1)[0].strip() if "--model-hyper-params" in command else "",
                "data_file": dataset + ".csv",
                "special_handling": special,
                "paper_auc_roc": reference[0] if reference else ("common-only" if not common_row else ""),
                "paper_affiliation_f": reference[1] if reference else ("common-only" if not common_row else ""),
                "paper_reference_status": "CATCH paper/original-protocol archive reference" if method == "CATCH" and reference else ("CATCH Table 2 literature reference" if reference else "not supplied for this dataset"),
                "common_protocol_evidence": str(common_dir / "common_protocol_metrics.csv") if common_row else "",
                "common_protocol_source": common_row.get("source_full_run_id", ""),
                "local_run_required_for_this_investigation": "no",
                "evidence_status": "direct-code/settings; paper value available" if reference else "direct-code/settings; paper value unavailable",
            })
    return rows


def common_table(root: Path):
    path = root / "result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/common_protocol_metrics.csv"
    rows = read_csv(path)
    wanted = ["MNRF Full", "CATCH-common", "IsolationForest", "PCA"]
    out = ["| Dataset | MNRF Full AUC/AP | CATCH-common AUC/AP | IF AUC/AP | PCA AUC/AP |", "|---|---:|---:|---:|---:|"]
    by = {(r.get("dataset"), r.get("method")): r for r in rows}
    for dataset in DATASETS:
        cells = []
        for method in wanted:
            r = by.get((dataset, method), {})
            if not r:
                cells.append("not available")
            else:
                cells.append("{}/{}".format(r.get("roc_auc", ""), r.get("average_precision", "")))
        out.append("| {} | {} | {} | {} | {} |".format(dataset, *cells))
    return "\n".join(out)


def build_readme(root: Path, out: Path):
    package_versions = []
    for distribution in ("numpy", "pandas", "scikit-learn", "torch", "salesforce-merlion", "pyod"):
        try:
            package_versions.append("{}={}".format(distribution, metadata.version(distribution)))
        except metadata.PackageNotFoundError:
            package_versions.append("{}=not-installed".format(distribution))
    try:
        import tods  # noqa: F401
        package_versions.append("tods=importable")
    except Exception:
        package_versions.append("tods=not-importable")
    env = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "packages": ", ".join(package_versions),
    }
    table = common_table(root)
    return f"""# MNRF baseline investigation

This is a read-only investigation generated on {env['commit']} with `{env['executable']}` (Python {env['python']}). Current catch_env packages: {env['packages']}. It did not train, encode, fit a head, recalculate a cache, or modify an existing result. Evidence labels are `direct-code`, `direct-record`, `inference`, and `unknown`.

## Scientific conclusion and comparison boundary

The primary baseline reference in this investigation is the fixed CATCH paper table (15 non-CATCH methods) plus the CATCH paper/original-protocol archive summary. The server scripts and configurations are treated as the authoritative implementation settings; a method does not need a local checkpoint or successful run to be included in the settings table.

The existing eight-dataset APD-CATCH common contract (MNRF Full, CATCH-common, IsolationForest, and PCA) is reported separately as a diagnostic comparison. Its rows use the same frozen Full source, endpoint coordinates, independent calibration convention, and common metric entry point. This is a protocol comparison, not a claim that the historical implementations are identical.

Historical CATCH is not one jointly evaluated row. `CATCH-master/result/score/CATCH` and `.../label/CATCH` are separate runs; the archive combines score AUC from one run with Affiliation-F from another. Those values are useful as a paper/original-protocol reference, but cannot be joined with a common-protocol AP or point metric. The CATCH paper table is a third category and is not a rerun.

Most methods have no local eight-dataset result in the scoped APD-CATCH result roots. This is an evidence-availability fact, not a failed experiment. Their server scripts and code paths are still recorded for all eight datasets in `baseline_settings.csv`; the paper table remains the numerical reference. Historical TODS artifacts in CATCH-master are sparse and do not establish an eight-dataset matrix. `tods` is not importable in the current `catch_env`, while current Merlion is 1.3.1 and PyOD is 2.1.0; no historical environment recovery is attempted.

The available common-protocol snapshot is:

{table}

These rows support ranking and alarm comparisons only within the common contract. They do not establish a 16-method comparison, and they do not prove a representation or causal mechanism.

## Dataset-specific qualifications

- **CalIt2 (direct-record):** the old local input is a modulo-10 conversion of UCI counts. The RawCount rerun report records MNRF AUC/AP changing from 0.713040/0.053971 to 0.853538/0.170287, CATCH from 0.835812/0.106986 to 0.881578/0.281496, IF from 0.800190/0.101147 to 0.890166/0.253774, and PCA from 0.732786/0.080117 to 0.822361/0.146648. The RawCount report's sentence saying “PCA AP decreases slightly” contradicts its own numbers; the numbers imply an increase and the historical file is left unchanged.
- **MSL (direct-record/unknown):** 27 telemetry sequences have 55 input fields each. `col_0` is a candidate telemetry target and `col_1..54` are not proven exogenous conditions. The existing col_0 aggregation is a post-hoc diagnostic on concatenated data, not a segmented retraining result.
- **NYC (direct-record):** three fields are passenger count, time-of-day, and day-of-week. The field-role diagnostic excluded evidenced calendar fields from target aggregation and did not improve results; this does not prove that removing context is generally correct.
- **Creditcard and CICIDS (direct-record):** the frozen inputs do not retain card/customer or flow/session identifiers sufficient to reconstruct entity continuity. Row adjacency must not be presented as a verified per-entity time series.

## What is confirmed versus unknown

The code directly confirms reconstruction versus discrepancy heads, losses, score direction, provider window behavior, and label-mode percentile thresholding for the 16 methods; see `method_mechanisms.csv`. The server entry script, exact command, hyperparameters, special handling, and paper-reference availability for every method x dataset pair are in `baseline_settings.csv`. Existing common-run source/metric bindings and the separate CATCH archive score/label runs are diagnostic evidence only.

Unknowns include the final selection rule when a method emits several anomaly ratios and results for methods without a local run. Historical package archaeology is outside this round; the current `catch_env` version is recorded only to explain current server behavior. No test-label selection was inferred merely from an `anomaly_ratio` list; the code does show train+test score distributions in label-mode thresholding, which is a protocol difference from the independent q=.995 common calibration.

## Minimum next experiment

Use the eight registered Full sources and one fixed common contract. Preserve raw file hashes, original row order and known contamination; use the saved train/validation/calibration/test endpoints and valid masks; prohibit crossing verified MSL sequence boundaries in the evaluation diagnostic, while clearly separating that from a segmented retrain. Fit each method with its native head and loss on training data only, use validation only for early stopping, derive one independent q=.995 threshold from the calibration segment, and predict with strict `score > threshold`. Do not force GMM, PCA, or a common head onto methods whose native scores differ.

Evaluate ROC-AUC/AP on the exact common scored points; report point precision/recall/F1/FPR and Affiliation precision/recall/F using the same valid contiguous segments, gaps, entity boundaries, and no point adjustment. Publish paper values, original-protocol archive values, and common-protocol values in separate tables. The unresolved choices before execution are (a) whether calendar/context fields are scored or only supplied as conditions for each native method, and (b) what explicit boundary contract is defensible for Creditcard/CICIDS without entity/session IDs.

No real benchmark was started in this investigation. No method is marked failed merely because a local result is absent.
"""


def build_comparison_protocol():
    return """# Proposed common comparison protocol

## Protocol identity

Use the exact registered Full source for each of PSM, MSL, GECCO, Genesis, NYC, CalIt2, Creditcard, and CICIDS. Record raw CSV and metadata hashes, source identity, variable order, train/validation/calibration/test endpoint hashes, and the code/environment used. Do not replace the modulo-10 CalIt2 source silently; RawCount is a separate data version. Keep paper values, CATCH-master original runs, and APD-CATCH common runs in separate protocol classes.

## Data and boundaries

Preserve original row order, labels, and known train-side contamination. Use the Full source's frozen train/validation/calibration/test ranges and valid masks. MSL evaluation must carry the 27 official test sequence ranges and exclude windows crossing a boundary for the boundary-sensitivity diagnostic; this does not claim that the existing encoder was retrained with segmented boundaries. NYC retains the row-to-time mapping and treats passenger count as the observed signal with evidenced calendar fields as context. Creditcard and CICIDS have no verified entity/session IDs in the frozen input; do not invent entity segments.

## Native models and information flow

Each method keeps its native detector, head, loss, and score definition. Reconstruction methods score reconstruction error; discrepancy methods retain their association/discrepancy term; density and classic detectors retain their native decision score. A window may use only the explicit past/current receptive field ending at the scored timestamp. No future value, test label, or hidden label-derived cleaning is allowed. Overlap, cold start, tail, and missing-score behavior is recorded per adapter and then mapped to the common valid point set; no implicit zero fill may create an evaluated point.

## Fitting and thresholding

Fit on the prescribed training range only. Validation may control early stopping. Use the independent calibration range for one model-specific q=0.995 threshold and strict `score > threshold`; never choose a threshold, anomaly ratio, hyperparameter, or report row using test labels. If an original method's historical protocol uses train+test percentile thresholds or an anomaly-ratio sweep, report it only in the original-protocol table and do not call it common-protocol evidence.

## Metrics and reporting

Compute ROC-AUC and average precision on the identical valid test points. Compute point precision/recall/F1/FPR and TP/FP/FN/TN at each method's own independent threshold. Compute Affiliation precision/recall/F with the same contiguous valid intervals, raw time/entity boundaries, and no point adjustment; name segment macro explicitly when used. Preserve no-score anomaly segments as `valid_points=0` rather than deleting them. Report coverage and all boundary exclusions.

## What can and cannot be unified

The scored coordinates, labels, threshold source, comparison symbol, metric definitions, and provenance can be unified. Native model heads, reconstruction/discrepancy formulas, PCA/TODS versus sklearn implementations, Merlion versions, context-field handling, and historical train+test thresholding cannot be made equivalent without changing the method. Therefore the final report must retain three labeled tables: paper reference, original-protocol reproduction, and common-protocol result. Missing artifacts remain unknown; they are not imputed from another method or dataset.

## Required evidence before execution

Resolve the anomaly-ratio selection rule from the existing report-generation path and obtain an explicit boundary decision for Creditcard/CICIDS. The smallest useful execution is one seed over the eight datasets for methods whose native server settings are fixed; preserve the paper table as a separate reference and expand only after threshold semantics are documented.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir or root / "result/analysis/MNRF-Baseline-Investigation-20261031"
    if out.exists() and any(out.iterdir()):
        if not args.overwrite or out.is_symlink():
            raise SystemExit("refusing to overwrite non-empty output directory: {}".format(out))
        for child in out.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "method_mechanisms.csv", method_rows(root), [
        "method", "actual_task", "output_head_and_shape", "training_loss_and_joint_training",
        "score_formula_and_direction", "variable_window_time_aggregation",
        "temporal_context_and_future_information", "cold_start_tail_padding_missing_scores",
        "threshold_and_test_label_use", "anomaly_ratio_candidates_and_selection", "metric_entry_points", "evidence_status", "code_evidence",
    ])
    write_csv(out / "baseline_settings.csv", build_baseline_settings(root), [
        "method", "dataset", "entry_script", "script_exists", "script_sha256", "server_command",
        "implementation_or_adapter", "actual_parameters", "data_file", "special_handling",
        "paper_auc_roc", "paper_affiliation_f", "paper_reference_status",
        "common_protocol_evidence", "common_protocol_source",
        "local_run_required_for_this_investigation", "evidence_status",
    ])
    (out / "README.md").write_text(build_readme(root, out), encoding="utf-8")
    (out / "comparison_protocol.md").write_text(build_comparison_protocol(), encoding="utf-8")
    old_protocol = out / "run_protocols.csv"
    if old_protocol.exists():
        old_protocol.unlink()
    print("output_dir={}".format(out.resolve()))
    print("method_rows=16")
    print("dataset_count=8")
    print("settings_rows=128 (16 methods x 8 datasets)")
    print("common_protocol_rows=8 x MNRF/CATCH/IF/PCA where present")
    print("environment=read-only; no benchmark execution")


if __name__ == "__main__":
    main()
