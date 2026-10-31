#!/usr/bin/env python
"""Build a read-only MNRF baseline investigation from existing evidence.

This script never opens model code for execution and never runs a benchmark. It
only inventories checked-in scripts and already published reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def method_rows(root: Path):
    evidence = {
        "CATCH": ("reconstruction autoencoder with RevIN/FFT/patch transformer", "CATCHModel -> temporal reconstruction [B,L,C] plus frequency reconstruction/CFM states", "MSE reconstruction + dc_loss + frequency_loss; model and mask module jointly optimized with separate Adam schedules", "mean((x-xhat)^2,axis=vars) + score_lambda*mean(frequency_error), higher is abnormal", "detect_score uses SegLoader mode=thre: non-overlap full windows, every within-window position is flattened; label threshold uses step-1 train/test scores but final predictions are mode=thre", "train+test score percentile for label path; external score strategy otherwise", "direct-code", "ts_benchmark/baselines/catch/CATCH.py:142-434; ts_benchmark/baselines/utils.py:261-295; scripts/run_mnrf_catch_common.py:487-561"),
        "ModernTCN": ("reconstruction anomaly detector", "ModernTCNModel -> output [B,L,C]; final projection Linear(d_model, patch_size)", "reconstruction MSE, jointly optimizing the ModernTCN model", "mean((x-xhat)^2,axis=vars), higher is abnormal", "detect_score flattens every position of non-overlap mode=thre windows; label threshold uses step-1 train/test windows but predicts on mode=thre", "train+test percentile in detect_label", "direct-code", "ts_benchmark/baselines/self_impl/ModernTCN/ModernTCN.py:136-380; ts_benchmark/baselines/self_impl/ModernTCN/models/ModernTCN_model.py"),
        "iTransformer": ("transformer reconstruction adapter", "iTransformer output projection Linear(d_model, seq_len) per channel, output [B,L,C]", "MSELoss, jointly optimizing the transformer and projection", "mean((x-xhat)^2,axis=vars), higher is abnormal", "mode=thre non-overlap full-window flatten for score; label threshold uses step-1 train/test then mode=thre prediction", "train+test percentile in TransformerAdapter.detect_label", "direct-code", "ts_benchmark/baselines/time_series_library/adapters_for_transformers.py:614-816; ts_benchmark/baselines/time_series_library/models/iTransformer.py"),
        "DualTF": ("dual time/frequency reconstruction and discrepancy detector", "separate time and frequency Anomaly Transformer heads", "time/frequency reconstruction plus association KL objectives, trained in separate stages", "time_score and fre_score are each normalized then added: score=normalization(time_score)+normalization(fre_score), higher is abnormal", "each branch folds overlapping windows to timestamps by evaluation counts; final score is aligned after the two normalized branches are added", "ratio percentile in label path", "direct-code", "ts_benchmark/baselines/self_impl/DualTF/DualTF.py:548-780,1059-1072"),
        "Anomaly Transformer": ("reconstruction plus association discrepancy", "AnomalyTransformer model returns reconstruction plus series/prior association tensors", "reconstruction and association discrepancy losses, jointly trained", "softmax(-(series_loss+prior_loss)) * reconstruction error, higher is abnormal", "window-level scores are flattened through the detector provider; exact tail coverage follows the provider", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/self_impl/Anomaly_trans/AnomalyTransformer.py:339-434"),
        "DCdetector": ("discrepancy/contrastive detector", "DCdetector returns series/prior association tensors and discrepancy scores", "contrastive/discrepancy objective rather than plain reconstruction", "softmax(-(series_loss+prior_loss)) association metric, higher is abnormal", "window scores are flattened through the detector provider; boundary/tail behavior follows SegLoader", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/self_impl/DCdetector/DCdetector.py:332-418"),
        "TimesNet": ("transformer reconstruction adapter", "TimesNet model output [B,L,C]", "MSELoss, jointly optimizing the model", "mean((x-xhat)^2,axis=vars), higher is abnormal", "mode=thre non-overlap full-window flatten for score; label threshold uses step-1 train/test then non-overlap prediction", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/time_series_library/adapters_for_transformers.py:614-816; scripts/multivariate_detection/detect_score/PSM_script/TimesNet.sh"),
        "PatchTST": ("transformer reconstruction adapter", "PatchTST model output [B,L,C]", "MSELoss, jointly optimizing the model", "mean((x-xhat)^2,axis=vars), higher is abnormal", "mode=thre non-overlap full-window flatten for score; label threshold uses step-1 train/test then non-overlap prediction", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/time_series_library/adapters_for_transformers.py:614-816; scripts/multivariate_detection/detect_score/PSM_script/PatchTST.sh"),
        "DLinear": ("linear reconstruction adapter", "DLinear model output [B,L,C]", "MSELoss, jointly optimizing the model", "mean((x-xhat)^2,axis=vars), higher is abnormal", "mode=thre non-overlap full-window flatten for score; label threshold uses step-1 train/test then non-overlap prediction", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/time_series_library/adapters_for_transformers.py:614-816; scripts/multivariate_detection/detect_score/PSM_script/DLinear.sh"),
        "NLinear": ("linear reconstruction adapter", "NLinear model output [B,L,C]", "MSELoss, jointly optimizing the model", "mean((x-xhat)^2,axis=vars), higher is abnormal", "mode=thre non-overlap full-window flatten for score; label threshold uses step-1 train/test then non-overlap prediction", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/time_series_library/adapters_for_transformers.py:614-816; scripts/multivariate_detection/detect_score/PSM_script/NLinear.sh"),
        "AutoEncoder": ("Merlion multivariate autoencoder", "Merlion AEModule: MLP encoder -> latent hidden_size -> mirrored MLP decoder; sequence input [k,C] is reconstructed and one scalar is returned per timestamp", "Adam minimizes the module reconstruction loss on unlabeled training windows; no labels are passed to the Merlion adapter", "Merlion model output is reconstruction loss/anomaly score, higher is abnormal; the wrapper returns get_anomaly_score", "sequence scores are indexed back to the requested timestamps; the project wrapper fills missing returned timestamps with 0", "Merlion backend threshold/post-process; no local anomaly-ratio sweep is exposed", "direct-code/current-version", "ts_benchmark/baselines/merlion/merlion_models.py:43-121; merlion/models/anomaly/autoencoder.py:64-180; scripts/multivariate_detection/detect_score/PSM_script/AutoEncoder.sh"),
        "OCSVM": ("one-class density boundary", "TODS OCSVMSKI wrapping the bundled PyOD OCSVM primitive; fit returns one score per adapter row/window", "unsupervised fit on unlabeled input", "PyOD OCSVM decision score, higher is abnormal", "row/window adapter output; exact temporal grouping is provider-defined", "benchmark label strategy", "direct-code; top-level runtime import unavailable", "ts_benchmark/baselines/tods/tods_models.py:18-50,96-118; ts_benchmark/baselines/tods/third_party/tods/sk_interface/detection_algorithm/OCSVM_skinterface.py:1-10; ts_benchmark/baselines/tods/third_party/tods/detection_algorithm/PyodOCSVM.py:128-208; scripts/multivariate_detection/detect_score/PSM_script/ocsvmski.sh"),
        "Isolation Forest": ("tree isolation outlier detector", "Merlion IsolationForest wraps sklearn IsolationForest with DifferenceTransform + Shingle(size=2,stride=1) by default; common protocol uses direct sklearn rows", "unsupervised tree fit; Merlion backend uses random_state=0", "legacy and common adapters negate sklearn score_samples so larger values are more abnormal", "legacy Merlion transform/shingle context and reindexing differ from common row-aligned scores", "common q=.995; legacy backend label path", "direct-code/direct-record/current-version", "ts_benchmark/baselines/merlion/merlion_models.py:43-121; merlion/models/anomaly/isolation_forest.py:25-106; scripts/run_mnrf_common_baselines.py:121-176"),
        "PCA": ("principal-component outlier detector", "project-local TODS PCAODetectorSKI wrapping PCAODetectorPrimitive; common protocol uses sklearn residual PCA", "unsupervised PCA fit", "PyOD 2.1.0 PCA score = sum_j cdist(X_scaled, selected_component_j) / selected_explained_variance_ratio_j; larger is more abnormal. This is not reconstruction MSE. The common sklearn diagnostic is a separate residual-MSE implementation.", "common row-aligned endpoints; legacy TODS windows", "common q=.995; legacy TODS threshold strategy", "direct-code/direct-record", "ts_benchmark/baselines/tods/tods_models.py:18-50,96-118; ts_benchmark/baselines/tods/third_party/tods/sk_interface/detection_algorithm/PCAODetector_skinterface.py:1-10; ts_benchmark/baselines/tods/third_party/tods/detection_algorithm/PCAODetect.py:151-220; catch_env pyod/models/pca.py:PCA.fit,decision_function; scripts/run_mnrf_common_baselines.py:121-176"),
        "HBOS": ("histogram density outlier detector", "project-local TODS HBOSSKI wrapping the bundled PyOD HBOS primitive", "unsupervised histogram fit", "independent histogram negative log-density/decision score, higher abnormal", "row/window adapter output; exact temporal grouping is provider-defined", "benchmark label strategy", "direct-code; top-level runtime import unavailable", "ts_benchmark/baselines/tods/tods_models.py:18-50,96-118; ts_benchmark/baselines/tods/third_party/tods/sk_interface/detection_algorithm/HBOS_skinterface.py:1-10; ts_benchmark/baselines/tods/third_party/tods/detection_algorithm/PyodHBOS.py:76-145; scripts/multivariate_detection/detect_score/PSM_script/hbosski.sh"),
        "TFAD": ("window-level supervised anomaly classifier with synthetic anomaly injection", "one sigmoid/logit per multivariate window, repeated over suspect-window tail", "BCELoss against reduced true/injected window labels; training uses input labels and configured spike injection", "sigmoid window probability folded/averaged over suspect tails, higher is abnormal", "unfolded windows with configured stride; only suspect tail is voted; detect_score converts NaN gaps to zero", "train+test percentile in label path", "direct-code", "ts_benchmark/baselines/self_impl/TFAD/TFAD.py:63-218,300-396; ts_benchmark/baselines/self_impl/TFAD/model/tfad_datamodule.py"),
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
    context_notes = {
        "CATCH": "Each score at position j in a mode=thre window [kL,(k+1)L) reads the whole window, so its maximum input index is (k+1)L-1; positions before that end therefore include later observations within the same window. This is offline window scoring, not a causal endpoint score.",
        "ModernTCN": "mode=thre score positions use the full [kL,(k+1)L) window; a position can read later values up to the window end. Label-mode threshold estimation additionally uses step-1 windows.",
        "iTransformer": "Same SegLoader mode=thre full-window context as the transformer adapter; score positions are not causal endpoints and can read later values inside the window.",
        "DualTF": "Time/frequency loaders use overlapping windows and fold each window over its covered timestamps; a timestamp can receive windows extending into its future. No claim of causal endpoint scoring.",
        "Anomaly Transformer": "The association/reconstruction window is evaluated as a full sequence; provider alignment can expose later values within that sequence to earlier reported positions.",
        "DCdetector": "The detector receives full windows; score alignment is provider-defined and not an endpoint-causal guarantee.",
        "TimesNet": "Full non-overlap mode=thre windows are scored at all within-window positions; the maximum read index is the window end, so earlier positions see later values in the same window.",
        "PatchTST": "Full non-overlap mode=thre windows are scored at all within-window positions; the maximum read index is the window end.",
        "DLinear": "Full non-overlap mode=thre windows are scored at all within-window positions; the maximum read index is the window end.",
        "NLinear": "Full non-overlap mode=thre windows are scored at all within-window positions; the maximum read index is the window end.",
        "AutoEncoder": "The adapter delegates temporal context to Merlion's AutoEncoder and reindexes its returned series; exact receptive field is not exposed in this wrapper.",
        "OCSVM": "The TODS adapter receives the tabular/window representation supplied by the benchmark; exact temporal receptive field and future access are backend/config dependent.",
        "Isolation Forest": "Legacy Merlion context is backend-defined; the common sklearn runner scores aligned rows from the frozen training contract and has no window look-ahead beyond those rows.",
        "PCA": "Legacy TODS context is backend/window-defined; common residual PCA scores aligned rows from the frozen training contract.",
        "HBOS": "The TODS adapter receives the benchmark's tabular/window representation; exact temporal context is backend/config dependent.",
        "TFAD": "A window logit is repeated over the suspect tail of the window; each voted timestamp can read through the window end. This is an offline window classifier, not causal endpoint scoring.",
    }
    cold_notes = {
        "CATCH": "SegLoader mode=thre uses full windows with no explicit padding; __len__=(N-L)//L+1, so a remainder shorter than L is omitted. Missing scores are not automatically filled by the native score method.",
        "ModernTCN": "mode=thre has no padding and omits a tail remainder shorter than seq_len; label threshold uses step-1 loaders, while final prediction uses non-overlap windows.",
        "iTransformer": "Same SegLoader rule: no padding, tail remainder omitted in mode=thre; step-1 train/test loaders are used only for label threshold estimation.",
        "DualTF": "Overlapping folds define coverage counts; incomplete/tail behavior is determined by the branch loaders and fold arrays, not zero-padded as a causal stream.",
        "Anomaly Transformer": "Provider/SegLoader determines cold-start and tail coverage; no separate native padding contract is declared in the adapter.",
        "DCdetector": "Provider/SegLoader determines cold-start and tail coverage; missing points are not given a new score by this wrapper.",
        "TimesNet": "No padding in full-window mode=thre; tail remainder is omitted, while label threshold loaders use step 1.",
        "PatchTST": "No padding in full-window mode=thre; tail remainder is omitted, while label threshold loaders use step 1.",
        "DLinear": "No padding in full-window mode=thre; tail remainder is omitted, while label threshold loaders use step 1.",
        "NLinear": "No padding in full-window mode=thre; tail remainder is omitted, while label threshold loaders use step 1.",
        "AutoEncoder": "Merlion adapter reindexes returned scores to input index and fills absent timestamps with 0; these filled values can enter downstream evaluation.",
        "OCSVM": "TODS adapter returns the backend array; padding/missing handling is not added by the wrapper and must be read from the backend strategy.",
        "Isolation Forest": "Legacy adapter reindexes/fills absent timestamps; common runner emits the fixed aligned contract instead.",
        "PCA": "Legacy TODS behavior is backend-defined; common runner explicitly evaluates the fixed aligned contract.",
        "HBOS": "TODS adapter has no extra padding layer; backend/provider determines missing points.",
        "TFAD": "Uncovered fold positions become NaN and detect_score applies nan_to_num(..., nan=0.0), so those zero scores can enter downstream metrics if not masked by the strategy.",
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
            "temporal_context_and_future_information": context_notes[method],
            "cold_start_tail_padding_missing_scores": cold_notes[method],
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

PAPER_VERSION = "ICLR 2025 conference paper, Table 2"
PAPER_URL = "https://proceedings.iclr.cc/paper_files/paper/2025/file/2b25c39788e5cf11d3541de433ebf4c0-Paper-Conference.pdf"
PAPER_METHOD_ORDER = METHODS
PAPER_RAW = {
    "CICIDS": {
        "aff": ["0.787", "0.654", "0.708", "0.692", "0.560", "0.664", "0.657", "0.660", "0.669", "0.669", "0.243", "0.693", "0.604", "0.619", "0.542", "0.579"],
        "auc": ["0.795", "0.697", "0.692", "0.603", "0.528", "0.638", "0.732", "0.716", "0.751", "0.691", "0.629", "0.537", "0.787", "0.601", "0.760", "0.504"],
    },
    "CalIt2": {
        "aff": ["0.835", "0.780", "0.812", "0.751", "0.729", "0.697", "0.794", "0.793", "0.793", "0.757", "0.587", "0.783", "0.402", "0.768", "0.756", "0.744"],
        "auc": ["0.838", "0.676", "0.791", "0.574", "0.533", "0.527", "0.771", "0.808", "0.752", "0.695", "0.767", "0.804", "0.775", "0.790", "0.798", "0.504"],
    },
    "Creditcard": {
        "aff": ["0.750", "0.744", "0.713", "0.663", "0.650", "0.632", "0.744", "0.746", "0.738", "0.742", "0.561", "0.714", "0.634", "0.710", "0.695", "0.600"],
        "auc": ["0.958", "0.957", "0.934", "0.703", "0.552", "0.504", "0.957", "0.957", "0.954", "0.948", "0.909", "0.953", "0.860", "0.871", "0.951", "0.500"],
    },
    "GECCO": {
        "aff": ["0.908", "0.893", "0.839", "0.701", "0.782", "0.687", "0.894", "0.906", "0.893", "0.882", "0.823", "0.666", "0.424", "0.785", "0.708", "0.627"],
        "auc": ["0.970", "0.952", "0.795", "0.714", "0.516", "0.555", "0.954", "0.949", "0.947", "0.936", "0.769", "0.804", "0.619", "0.711", "0.557", "0.499"],
    },
    "Genesis": {
        "aff": ["0.896", "0.833", "0.891", "0.810", "0.856", "0.776", "0.864", "0.856", "0.856", "0.829", "0.854", "0.677", "0.788", "0.814", "0.721", "0.535"],
        "auc": ["0.974", "0.676", "0.690", "0.937", "0.947", "0.659", "0.913", "0.685", "0.696", "0.755", "0.931", "0.733", "0.549", "0.815", "0.897", "0.497"],
    },
    "MSL": {
        "aff": ["0.740", "0.726", "0.710", "0.588", "0.692", "0.694", "0.734", "0.724", "0.725", "0.723", "0.625", "0.641", "0.584", "0.678", "0.680", "0.665"],
        "auc": ["0.664", "0.633", "0.611", "0.576", "0.508", "0.507", "0.613", "0.637", "0.624", "0.592", "0.562", "0.524", "0.524", "0.552", "0.574", "0.500"],
    },
    "NYC": {
        "aff": ["0.994", "0.769", "0.684", "0.708", "0.853", "0.862", "0.794", "0.776", "0.828", "0.819", "0.689", "0.667", "0.648", "0.680", "0.675", "0.689"],
        "auc": ["0.816", "0.466", "0.640", "0.633", "0.671", "0.549", "0.791", "0.709", "0.768", "0.671", "0.504", "0.456", "0.475", "0.666", "0.446", "0.502"],
    },
    "PSM": {
        "aff": ["0.859", "0.825", "0.854", "0.725", "0.710", "0.682", "0.842", "0.831", "0.831", "0.843", "0.707", "0.531", "0.620", "0.702", "0.658", "0.628"],
        "auc": ["0.652", "0.593", "0.592", "0.600", "0.514", "0.501", "0.592", "0.586", "0.580", "0.585", "0.650", "0.619", "0.542", "0.648", "0.620", "0.500"],
    },
}
PAPER_TABLE = {
    dataset: {
        method: (PAPER_RAW[dataset]["auc"][idx], PAPER_RAW[dataset]["aff"][idx])
        for idx, method in enumerate(PAPER_METHOD_ORDER)
    }
    for dataset in DATASETS
}


def _script_command(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if "run_benchmark.py" in line:
            return line.strip()
    return ""


def _command_hyperparams(command):
    """Decode the literal JSON passed by a task launcher, if present."""
    match = re.search(r"--model-hyper-params\s+'([^']*)'", command)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _effective_fields(method, entry_mode, command):
    """Concrete effective settings used by the launcher or native default."""
    params = _command_hyperparams(command)
    if method == "CATCH":
        window = params.get("seq_len", 192)
        patch = params.get("inference_patch_size", 32)
        weight = params.get("score_lambda", 0.05)
        aggregation = "mean variable reconstruction + score_lambda * frequency error"
    elif method in {"ModernTCN", "iTransformer", "TimesNet", "PatchTST", "DLinear", "NLinear"}:
        window = params.get("seq_len", 100)
        patch = params.get("inference_patch_size", params.get("patch_size", params.get("patch_len", 16)))
        weight = "1.0 (MSE)"
        aggregation = "mean squared reconstruction error over variables"
    elif method == "DualTF":
        window = params.get("seq_len", 100)
        patch = params.get("inference_patch_size", "not applicable")
        weight = "1.0 + 1.0 after branch normalization"
        aggregation = "normalized time_score + normalized fre_score"
    elif method in {"Anomaly Transformer", "DCdetector", "TFAD"}:
        window = params.get("win_size", params.get("seq_len", 100))
        patch = "not applicable (native window)"
        weight = "native objective weights; no extra score weight"
        aggregation = "native provider/window alignment"
    elif method == "AutoEncoder":
        window = params.get("sequence_len", 1)
        patch = "not applicable"
        weight = "1.0 reconstruction loss"
        aggregation = "Merlion timestamp score"
    elif method in {"OCSVM", "Isolation Forest", "PCA", "HBOS"}:
        window = params.get("window_size", params.get("shingle_size", 1))
        patch = "not applicable (tabular/TODS window)"
        weight = "1.0 native decision score"
        aggregation = ("PyOD weighted component-distance sum: "
                       "sum(cdist(X_scaled, selected_components)/selected_explained_variance_ratio)"
                       if method == "PCA" else "native row/window decision score")
    else:
        window = params.get("seq_len", "unknown")
        patch = "not applicable"
        weight = "native"
        aggregation = "native"
    if entry_mode == "detect_score":
        threshold = "not applied by score entry (continuous score)"
    elif method in {"CATCH", "ModernTCN", "iTransformer", "DualTF", "Anomaly Transformer",
                    "DCdetector", "TimesNet", "PatchTST", "DLinear", "NLinear", "TFAD"}:
        ratios = params.get("anomaly_ratio", "class default [0.1,0.5,1,2,3,5,10,15,20,25]")
        threshold = "percentile(combined train+test score, 100-ratio); strict score > threshold; ratio={}".format(ratios)
    else:
        threshold = "native backend/detector label threshold; adapter command provides no anomaly_ratio override"
    return {
        "effective_score_weight": str(weight),
        "effective_window": str(window),
        "effective_inference_patch": str(patch),
        "effective_aggregation": aggregation,
        "effective_threshold": threshold,
    }


def build_baseline_settings(root: Path):
    common_dir = root / "result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun"
    common = read_csv(common_dir / "common_protocol_metrics.csv")
    common_map = {(r.get("dataset"), r.get("method")): r for r in common}
    catch_summary = read_csv(root.parent / "CATCH-master/result/CATCH_reproduction_summary/physical_task_metrics.csv")
    catch_map = {r.get("physical_task"): r for r in catch_summary}
    inherited_defaults = {
        "CATCH": "CATCH.py/config defaults apply only to omitted fields; label mode exposes anomaly_ratio candidates unless the task script supplies one. score and label configs are separate.",
        "ModernTCN": "ModernTCN/task-config defaults for omitted model fields; adapter criterion is MSE and provider defaults determine loader behavior.",
        "iTransformer": "TransformerAdapter/config defaults for omitted fields; criterion is MSE and omitted transformer options remain class/config defaults.",
        "DualTF": "DualTF class/config defaults for omitted time/frequency options; label anomaly_ratio and score fre_anormly_ratio are independent command fields.",
        "Anomaly Transformer": "AnomalyTransformer class defaults for omitted k/attention options; label ratio is a command/config field when present.",
        "DCdetector": "DCdetector class defaults for omitted attention/discrepancy options; label ratio is a command/config field when present.",
        "TimesNet": "TransformerAdapter/config defaults for omitted fields; criterion is MSE and provider loader defaults remain active.",
        "PatchTST": "TransformerAdapter/config defaults for omitted fields; criterion is MSE and provider loader defaults remain active.",
        "DLinear": "TransformerAdapter/config defaults for omitted fields; criterion is MSE and provider loader defaults remain active.",
        "NLinear": "TransformerAdapter/config defaults for omitted fields; criterion is MSE and provider loader defaults remain active.",
        "AutoEncoder": "Merlion AutoEncoderConfig defaults include hidden_size=5, layer_sizes=(25,10,5), sequence_len=1, lr=1e-3, batch_size=512, num_epochs=50; command overrides are separate.",
        "Isolation Forest": "Merlion IsolationForestConfig defaults include max_n_samples=None (all samples), n_estimators=100, random_state=0, DifferenceTransform plus Shingle(size=2,stride=1).",
        "OCSVM": "Project-local TODS/PyOD defaults include RBF kernel, nu=0.5, gamma=auto and contamination=0.1 unless adapter/config overrides them.",
        "PCA": "Project-local TODS PCA defaults include contamination=0.1, n_components=None, n_selected_components=None, standardization=True and window step defaults; command overrides are separate.",
        "HBOS": "Project-local TODS/PyOD defaults include n_bins=10, alpha=0.1, tol=0.1 and contamination=0.1 unless command/config overrides them.",
        "TFAD": "TFAD class/data-module defaults include the configured synthetic anomaly injection and BCE window-label target; omitted architecture and ratio fields remain task-config defaults.",
    }
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            common_method = {"Isolation Forest": "IsolationForest", "PCA": "PCA"}.get(method)
            common_row = common_map.get((dataset, common_method), {}) if common_method else {}
            paper_auc, paper_aff = PAPER_TABLE[dataset][method]
            archive = catch_map.get(dataset, {}) if method == "CATCH" else {}
            for entry_mode in ("detect_score", "detect_label"):
                script = root / "scripts/multivariate_detection" / entry_mode / (dataset + "_script") / METHOD_SCRIPT[method]
                command = _script_command(script) if script.exists() else ""
                model_params = ""
                if "--model-hyper-params" in command:
                    model_params = command.split("--model-hyper-params", 1)[1].split(" --gpus", 1)[0].strip()
                config_name = "unfixed_{}_multi_config.json".format(entry_mode)
                if method == "CATCH":
                    if dataset == "PSM" and entry_mode == "detect_score":
                        effective = "Explicit score config: cf_dim=16, n_heads=4; do not infer label settings from this row."
                    elif dataset == "PSM" and entry_mode == "detect_label":
                        effective = "Explicit label config: cf_dim=32, n_heads=16, anomaly_ratio=3.0; this is one fixed ratio, not an unresolved candidate list."
                    else:
                        effective = "CATCH dimensions and ratio are the literal command values; config defaults are not substituted for omitted fields."
                    adapter = "native catch.CATCH; score and label are separate entry modes"
                elif method in {"AutoEncoder", "Isolation Forest"}:
                    effective = "Model parameters are command overrides; remaining backend defaults come from the installed Merlion adapter and are not treated as explicit command parameters."
                    adapter = "native merlion adapter"
                elif method in {"OCSVM", "PCA", "HBOS"}:
                    effective = "Model parameters are command overrides; remaining backend defaults come from the project-local TODS/PyOD adapter."
                    adapter = "project-local tods adapter (not top-level tods import)"
                elif "--adapter" in command:
                    effective = "Transformer adapter parameters are command overrides; omitted model defaults are inherited from the adapter/config."
                    adapter = "native model through transformer_adapter"
                else:
                    effective = "Native self_impl command parameters are explicit overrides; omitted values are inherited from the task config/class defaults."
                    adapter = "native self_impl entry"
                if entry_mode == "detect_score":
                    difference = "Returns continuous scores; thresholding is external to this score entry."
                else:
                    difference = "Returns label-mode predictions/threshold outputs; this entry may use its explicit anomaly_ratio or label strategy and is not interchangeable with detect_score."
                if method == "CATCH" and dataset in catch_map:
                    archive_score = archive.get("reproduced_score_auc_roc", "")
                    archive_aff = archive.get("reproduced_label_affiliation_f", "")
                else:
                    archive_score = archive_aff = ""
                effective_fields = _effective_fields(method, entry_mode, command)
                rows.append({
                    "method": method,
                    "dataset": dataset,
                    "entry_mode": entry_mode,
                    "entry_script": str(script),
                    "script_exists": str(script.exists()).lower(),
                    "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest() if script.exists() else "",
                    "server_command": command,
                    "implementation_or_adapter": adapter,
                    "explicit_parameters": model_params,
                    "inherited_parameters_or_defaults": "{}; score_weight={}; window={}; inference_patch={}; aggregation={}; threshold={}".format(
                        inherited_defaults[method], effective_fields["effective_score_weight"],
                        effective_fields["effective_window"], effective_fields["effective_inference_patch"],
                        effective_fields["effective_aggregation"], effective_fields["effective_threshold"]),
                    "effective_configuration_notes": effective,
                    **effective_fields,
                    "score_label_difference": difference,
                    "data_file": dataset + ".csv",
                    "special_handling": "Paper dataset name is 'Credit' for this row." if dataset == "Creditcard" else "",
                    "paper_dataset_name": "Credit" if dataset == "Creditcard" else dataset,
                    "paper_reference_version": PAPER_VERSION,
                    "paper_reference_url": PAPER_URL,
                    "paper_table": "Table 2",
                    "paper_auc_roc": paper_auc,
                    "paper_affiliation_f": paper_aff,
                    "original_archive_score_run": ARCHIVE_SCORE.get(dataset, "") if method == "CATCH" else "",
                    "original_archive_label_run": ARCHIVE_LABEL.get(dataset, "") if method == "CATCH" else "",
                    "original_archive_auc_roc": archive_score,
                    "original_archive_affiliation_f": archive_aff,
                    "common_protocol_evidence": str(common_dir / "common_protocol_metrics.csv") if common_row else "",
                    "common_protocol_source": common_row.get("source_full_run_id", ""),
                    "local_run_required_for_this_investigation": "no",
                    "evidence_status": ("direct-code/settings; paper Table 2 value available; archive/common fields are separate"
                                        if script.exists() else
                                        "direct-code/settings; paper Table 2 value available; this entry script is missing on the server"),
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

The primary numerical baseline reference is the fixed [ICLR 2025 conference-paper Table 2]({PAPER_URL}) for all eight scoped datasets and all 16 methods. The PDF reports `A-R (AUC-ROC)` and `Aff-F`; its dataset label `Credit` is mapped to the repository task `Creditcard`. Paper values are stored only in the numeric `paper_auc_roc` and `paper_affiliation_f` columns. The server scripts and configurations are treated as the authoritative implementation settings; a method does not need a local checkpoint or successful run to be included in the settings table.

The report keeps three protocol classes separate: `paper reference` (the Table 2 PDF), `original-protocol local archive` (CATCH-master score and label runs), and `common-protocol diagnostic` (the APD-CATCH frozen-source comparison). Local archive values never populate the paper columns. `baseline_settings.csv` has one `detect_score` and one `detect_label` row for every method x dataset pair, so an explicit command override is distinguishable from config/class defaults.

The existing eight-dataset APD-CATCH common contract (MNRF Full, CATCH-common, IsolationForest, and PCA) is reported separately as a diagnostic comparison. Its rows use the same frozen Full source, endpoint coordinates, independent calibration convention, and common metric entry point. This is a protocol comparison, not a claim that the historical implementations are identical.

Historical CATCH is not one jointly evaluated row. `CATCH-master/result/score/CATCH` and `.../label/CATCH` are separate runs; the archive summary records score AUC from the score run and Affiliation-F from the label run. Those values are original-protocol records, not paper values, and cannot be joined with a common-protocol AP or point metric. The CATCH paper table is a third category and is not a rerun.

Most methods have no local eight-dataset result in the scoped APD-CATCH result roots. This is an evidence-availability fact, not a failed experiment. Their server scripts and code paths are still recorded for all eight datasets in `baseline_settings.csv`; the paper table remains the numerical reference. Historical TODS artifacts in CATCH-master are sparse and do not establish an eight-dataset matrix. `tods` is not importable in the current `catch_env`, while current Merlion is 1.3.1 and PyOD is 2.1.0; no historical environment recovery is attempted.

The available common-protocol snapshot is:

{table}

These rows support ranking and alarm comparisons only within the common contract. They do not establish a 16-method comparison, and they do not prove a representation or causal mechanism. A threshold is one alarm-metric difference; changing a threshold on the same continuous score cannot change ROC-AUC or AP. Training configuration, score formula, model context, window alignment and threshold source also differ here, and no causal ordering of those differences is claimed without a controlled intervention.

## Time and score alignment

The previous generic “past/current, no future” wording is not used. The native CATCH `mode=thre` path uses non-overlapping full windows. For sequence length `L=4`, the window `[0,1,2,3]` produces flattened scores for positions 0, 1, 2 and 3; the score reported for position 0 can therefore read through input index 3. MNRF's frozen endpoint score for endpoint 3 reads the causal window ending at 3 and emits only that endpoint. The CATCH path is offline full-window scoring, not endpoint-causal scoring or a claim of leakage. `SegLoader` omits a remainder shorter than `L`, and the native score path does not invent scores for omitted positions. Each method's actual context, stride, overlap, cold-start, tail and missing-score behavior is recorded in `method_mechanisms.csv`; adapter-dependent cases remain explicitly unknown.

The project-local TODS adapter is under `ts_benchmark/baselines/tods/third_party`; a failed top-level `import tods` in `catch_env` does not mean the repository entry is unavailable. Merlion and PyOD behavior is described from the currently installed packages (Merlion 1.3.1 and PyOD 2.1.0); package-relative `merlion/...` references in `method_mechanisms.csv` point to that environment's site-packages source. Historical package versions are not inferred. The one missing scoped entry (`CICIDS` `detect_label` `Isolation Forest`) is recorded as missing, not as a failed experiment.

## Dataset-specific qualifications

- **CalIt2 (direct-record):** the old local input is a modulo-10 conversion of UCI counts. The RawCount rerun report records MNRF AUC/AP changing from 0.713040/0.053971 to 0.853538/0.170287, CATCH from 0.835812/0.106986 to 0.881578/0.281496, IF from 0.800190/0.101147 to 0.890166/0.253774, and PCA from 0.732786/0.080117 to 0.822361/0.146648. The RawCount report's sentence saying “PCA AP decreases slightly” contradicts its own numbers; the numbers imply an increase. The PCA comparison must remain three-way: old full-dimensional reconstruction, old data-repair residual PCA, and RawCount residual PCA. The historical file is left unchanged.
- **MSL (direct-record/unknown):** 27 telemetry sequences have 55 input fields each. `col_0` is a candidate telemetry target and `col_1..54` are not proven exogenous conditions. The existing col_0 aggregation is a post-hoc diagnostic on concatenated data, not a segmented retraining result.
- **NYC (direct-record):** three fields are passenger count, time-of-day, and day-of-week. The field-role diagnostic excluded evidenced calendar fields from target aggregation and did not improve results; this does not prove that removing context is generally correct.
- **Creditcard and CICIDS (direct-record):** the frozen inputs do not retain card/customer or flow/session identifiers sufficient to reconstruct entity continuity. Row adjacency must not be presented as a verified per-entity time series.

## What is confirmed versus unknown

The code directly confirms reconstruction versus discrepancy heads, losses, score direction, provider window behavior, and label-mode percentile thresholding for the 16 methods; see `method_mechanisms.csv`. The server entry script, exact command, hyperparameters, special handling, and paper-reference availability for every method x dataset pair are in `baseline_settings.csv`. Existing common-run source/metric bindings and the separate CATCH archive score/label runs are diagnostic evidence only.

Unknowns include the final selection rule when a method emits several anomaly ratios and results for methods without a local run. Historical package archaeology is outside this round; the current `catch_env` version is recorded only to explain current server behavior. No test-label selection was inferred merely from an `anomaly_ratio` list; the code does show train+test score distributions in label-mode thresholding, which is a protocol difference from the independent q=.995 common calibration.

## Minimum next experiment

Use the eight registered Full sources and one fixed common contract. Preserve raw file hashes, original row order and known contamination; use the saved train/validation/calibration/test endpoints and valid masks; prohibit crossing verified MSL sequence boundaries in the evaluation diagnostic, while clearly separating that from a segmented retrain. Fit each method with its native head and loss on training data only, use validation only for early stopping, derive one independent q=.995 threshold from the calibration segment, and predict with strict `score > threshold`. Do not force GMM, PCA, or a common head onto methods whose native scores differ. This is a minimum diagnostic, not a default eight-dataset x 16-method rerun: first freeze an auxiliary decoder/score-head diagnostic and the corrected CalIt2 RawCount input, then compare the resulting score and alarm changes under the same coordinates.

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

Each method keeps its native detector, head, loss, and score definition. Reconstruction methods score reconstruction error; discrepancy methods retain their association/discrepancy term; density and classic detectors retain their native decision score. A common causal endpoint rule would change native full-window methods, so native and causal-adapted variants must be named separately. No test label or hidden label-derived cleaning is allowed. Overlap, cold start, tail, and missing-score behavior is recorded per adapter and then mapped to the common valid point set; no implicit zero fill may create an evaluated point. The native CATCH mode=thre example is a full non-overlap window flattened to all positions, whereas MNRF is an endpoint score; this is an alignment difference, not a generic no-future assertion.

## Fitting and thresholding

Fit on the prescribed training range only. Validation may control early stopping. Use the independent calibration range for one model-specific q=0.995 threshold and strict `score > threshold`; never choose a threshold, anomaly ratio, hyperparameter, or report row using test labels. If an original method's historical protocol uses train+test percentile thresholds or an anomaly-ratio sweep, report it only in the original-protocol table and do not call it common-protocol evidence.

## Metrics and reporting

Compute ROC-AUC and average precision on the identical valid test points. Compute point precision/recall/F1/FPR and TP/FP/FN/TN at each method's own independent threshold. Compute Affiliation precision/recall/F with the same contiguous valid intervals, raw time/entity boundaries, and no point adjustment; name segment macro explicitly when used. Preserve no-score anomaly segments as `valid_points=0` rather than deleting them. Report coverage and all boundary exclusions.

## What can and cannot be unified

The scored coordinates, labels, threshold source, comparison symbol, metric definitions, and provenance can be unified. Native model heads, reconstruction/discrepancy formulas, PCA/TODS versus sklearn implementations, Merlion versions, context-field handling, and historical train+test thresholding cannot be made equivalent without changing the method. Therefore the final report must retain three labeled tables: paper reference, original-protocol reproduction, and common-protocol result. Missing artifacts remain unknown; they are not imputed from another method or dataset.

## Required evidence before execution

The final selection rule for some historical anomaly-ratio candidate lists remains unknown and is recorded as such; recovering it is not a prerequisite for an independent frozen auxiliary-decoder/score-head diagnostic. The smallest useful execution is that diagnostic plus the corrected CalIt2 RawCount comparison. Preserve the paper table as a separate reference; this is not a default eight-dataset x 16-method rerun.
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
        "method", "dataset", "entry_mode", "entry_script", "script_exists", "script_sha256",
        "server_command", "implementation_or_adapter", "explicit_parameters",
        "inherited_parameters_or_defaults", "effective_configuration_notes",
        "effective_score_weight", "effective_window", "effective_inference_patch",
        "effective_aggregation", "effective_threshold",
        "score_label_difference", "data_file", "special_handling", "paper_dataset_name",
        "paper_reference_version", "paper_reference_url", "paper_table", "paper_auc_roc",
        "paper_affiliation_f", "original_archive_score_run", "original_archive_label_run",
        "original_archive_auc_roc", "original_archive_affiliation_f", "common_protocol_evidence",
        "common_protocol_source", "local_run_required_for_this_investigation", "evidence_status",
    ])
    (out / "README.md").write_text(build_readme(root, out), encoding="utf-8")
    (out / "comparison_protocol.md").write_text(build_comparison_protocol(), encoding="utf-8")
    old_protocol = out / "run_protocols.csv"
    if old_protocol.exists():
        old_protocol.unlink()
    print("output_dir={}".format(out.resolve()))
    print("method_rows=16")
    print("dataset_count=8")
    print("settings_rows=256 (16 methods x 8 datasets x detect_score/detect_label)")
    print("common_protocol_rows=8 x MNRF/CATCH/IF/PCA where present")
    print("environment=read-only; no benchmark execution")


if __name__ == "__main__":
    main()
