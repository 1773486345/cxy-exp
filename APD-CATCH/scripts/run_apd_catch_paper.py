#!/usr/bin/env python3
"""Run APD-CATCH once per task on the real datasets used by CATCH."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PAPER_REFERENCES = {
    "CICIDS": {"auc_roc": 0.795, "affiliation_f": 0.787},
    "CalIt2": {"auc_roc": 0.838, "affiliation_f": 0.835},
    "SWAT": {"auc_roc": 0.345, "affiliation_f": 0.755},
    "Creditcard": {"auc_roc": 0.958, "affiliation_f": 0.750},
    "GECCO": {"auc_roc": 0.970, "affiliation_f": 0.908},
    "Genesis": {"auc_roc": 0.974, "affiliation_f": 0.896},
    "MSL": {"auc_roc": 0.664, "affiliation_f": 0.740},
    "NYC": {"auc_roc": 0.816, "affiliation_f": 0.994},
    "PSM": {"auc_roc": 0.652, "affiliation_f": 0.859},
    "SMD": {"auc_roc": 0.811, "affiliation_f": 0.847},
    "SMAP": {"auc_roc": 0.504, "affiliation_f": 0.699},
    "ASD": {"auc_roc": 0.824, "affiliation_f": 0.804},
}

SINGLE_FILE_DATASETS = {
    name: f"{name}.csv"
    for name in PAPER_REFERENCES
    if name != "ASD"
}
ASD_FILES = [f"ASD_dataset_{index}.csv" for index in range(1, 13)]
VARIANTS = ("causal_catch", "state", "state_scale")


def paper_group(file_name: str) -> str:
    return "ASD" if file_name.startswith("ASD_dataset_") else Path(file_name).stem


def canonical_dataset_name(value: str) -> str:
    lookup = {name.casefold(): name for name in PAPER_REFERENCES}
    lookup.update({"all": "all"})
    try:
        return lookup[value.casefold()]
    except KeyError as error:
        choices = ", ".join([*PAPER_REFERENCES, "all"])
        raise ValueError(f"Unknown dataset {value!r}. Choose from: {choices}") from error


def expand_datasets(values: Iterable[str]) -> list[str]:
    names = [canonical_dataset_name(value) for value in values]
    if "all" in names:
        names = list(PAPER_REFERENCES)
    files: list[str] = []
    for name in names:
        candidates = ASD_FILES if name == "ASD" else [SINGLE_FILE_DATASETS[name]]
        for candidate in candidates:
            if candidate not in files:
                files.append(candidate)
    return files


def source_script(file_name: str) -> Path:
    stem = Path(file_name).stem
    return (
        REPO_ROOT
        / "scripts"
        / "multivariate_detection"
        / "detect_score"
        / f"{stem}_script"
        / "CATCH.sh"
    )


def original_catch_params(file_name: str) -> dict:
    script = source_script(file_name)
    if not script.is_file():
        raise FileNotFoundError(f"Missing original CATCH script: {script}")
    tokens = shlex.split(script.read_text(encoding="utf-8"))
    option = "--model-hyper-params"
    if option not in tokens:
        raise ValueError(f"{script} does not define {option}")
    return json.loads(tokens[tokens.index(option) + 1])


def apd_params(file_name: str, variant: str, seed: int) -> tuple[dict, dict]:
    from ts_benchmark.baselines.apd_catch.APDCATCH import DEFAULT_HYPER_PARAMS

    original = original_catch_params(file_name)
    supported = {
        key: value for key, value in original.items() if key in DEFAULT_HYPER_PARAMS
    }
    supported.update({"variant": variant, "seed": seed})
    return supported, original


def anomaly_dataset_root(value: Path) -> Path:
    value = value.resolve()
    candidates = [value / "anomaly_detect", value]
    for candidate in candidates:
        if (candidate / "DETECT_META.csv").is_file():
            return candidate
    return candidates[0]


def missing_data_files(dataset_root: Path, file_names: Iterable[str]) -> list[Path]:
    root = anomaly_dataset_root(dataset_root)
    missing = []
    if not (root / "DETECT_META.csv").is_file():
        missing.append(root / "DETECT_META.csv")
    for file_name in file_names:
        path = root / "data" / file_name
        if not path.is_file():
            missing.append(path)
    return missing


def load_dataset(dataset_root: Path, file_name: str):
    from ts_benchmark.data.utils import read_data

    root = anomaly_dataset_root(dataset_root)
    metadata = pd.read_csv(root / "DETECT_META.csv")
    rows = metadata.loc[metadata["file_name"] == file_name]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one metadata row for {file_name}, found {len(rows)}"
        )
    train_length = int(rows.iloc[0]["train_lens"])
    data = read_data(str(root / "data" / file_name))
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame for {file_name}, got {type(data).__name__}")
    data = data.reset_index(drop=True)
    if "label" not in data.columns:
        raise ValueError(f"{file_name} has no label column")
    if not 0 < train_length < len(data):
        raise ValueError(
            f"Invalid train_lens={train_length} for {file_name} with {len(data)} rows"
        )

    features = data.drop(columns="label").apply(pd.to_numeric, errors="raise")
    labels = pd.to_numeric(data["label"], errors="raise").astype(np.int64)
    values = features.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{file_name} contains missing or non-finite features")
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{file_name} contains non-binary labels")
    train = features.iloc[:train_length].reset_index(drop=True)
    train_labels = labels.iloc[:train_length].to_numpy(dtype=np.int64, copy=True)
    test = features.iloc[train_length:].reset_index(drop=True)
    test_labels = labels.iloc[train_length:].to_numpy(dtype=np.int64, copy=True)
    return train, train_labels, test, test_labels


def safe_metric(
    name: str,
    function: Callable,
    actual: np.ndarray,
    predicted: np.ndarray,
    errors: dict[str, str],
):
    try:
        value = float(function(actual, predicted))
    except Exception as error:  # Preserve scores even if one optional metric fails.
        errors[name] = f"{type(error).__name__}: {error}"
        return None
    if not np.isfinite(value):
        errors[name] = f"non-finite value: {value}"
        return None
    return value


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    predicted_labels: np.ndarray,
    full_metrics: bool,
) -> tuple[dict, dict]:
    from ts_benchmark.evaluation.metrics.classification_metrics_label import (
        affiliation_f,
        affiliation_precision,
        affiliation_recall,
        f_score,
        precision,
        recall,
    )
    from ts_benchmark.evaluation.metrics.classification_metrics_score import (
        R_AUC_PR,
        R_AUC_ROC,
        VUS_PR,
        VUS_ROC,
        auc_pr,
        auc_roc,
    )

    score_metrics = {"auc_roc": auc_roc, "auc_pr": auc_pr}
    if full_metrics:
        score_metrics.update(
            {
                "r_auc_roc": R_AUC_ROC,
                "r_auc_pr": R_AUC_PR,
                "vus_roc": VUS_ROC,
                "vus_pr": VUS_PR,
            }
        )
    label_metrics = {
        "affiliation_f": affiliation_f,
        "affiliation_precision": affiliation_precision,
        "affiliation_recall": affiliation_recall,
        "point_f1": f_score,
        "point_precision": precision,
        "point_recall": recall,
    }
    errors: dict[str, str] = {}
    metrics = {
        name: safe_metric(name, function, labels, scores, errors)
        for name, function in score_metrics.items()
    }
    metrics.update(
        {
            name: safe_metric(name, function, labels, predicted_labels, errors)
            for name, function in label_metrics.items()
        }
    )
    return metrics, errors


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def result_path(output_root: Path, file_name: str, variant: str, seed: int) -> Path:
    return output_root / Path(file_name).stem / variant / f"seed_{seed}.json"


def collect_results(output_root: Path) -> list[dict]:
    records = []
    # Workers may use isolated subdirectories while sharing one experiment root.
    for path in sorted(output_root.rglob("seed_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        record = {
            "dataset_file": payload["dataset_file"],
            "paper_dataset": payload["paper_dataset"],
            "variant": payload["variant"],
            "seed": payload["seed"],
            "fit_seconds": payload["timing_seconds"]["fit"],
            "inference_seconds": payload["timing_seconds"]["inference"],
            "parameters": payload["fit_summary"]["trainable_parameters"],
            "epochs": payload["fit_summary"]["epochs"],
        }
        record.update(payload["metrics"])
        records.append(record)
    return records


def write_summaries(output_root: Path) -> None:
    records = collect_results(output_root)
    if not records:
        return
    frame = pd.DataFrame(records)
    frame.to_csv(output_root / "summary_runs.csv", index=False)

    metric_columns = [
        column
        for column in (
            "auc_roc",
            "auc_pr",
            "r_auc_roc",
            "r_auc_pr",
            "vus_roc",
            "vus_pr",
            "affiliation_f",
            "affiliation_precision",
            "affiliation_recall",
            "point_f1",
            "point_precision",
            "point_recall",
        )
        if column in frame
    ]
    grouped = (
        frame.groupby(["paper_dataset", "variant", "seed"], as_index=False)[
            metric_columns
        ]
        .mean(numeric_only=True)
        .sort_values(["paper_dataset", "variant", "seed"])
    )
    grouped["paper_catch_auc_roc"] = grouped["paper_dataset"].map(
        lambda name: PAPER_REFERENCES[name]["auc_roc"]
    )
    grouped["paper_catch_affiliation_f"] = grouped["paper_dataset"].map(
        lambda name: PAPER_REFERENCES[name]["affiliation_f"]
    )
    grouped["delta_auc_roc_vs_paper_catch"] = (
        grouped["auc_roc"] - grouped["paper_catch_auc_roc"]
    )
    grouped["delta_affiliation_f_vs_paper_catch"] = (
        grouped["affiliation_f"] - grouped["paper_catch_affiliation_f"]
    )
    grouped.to_csv(output_root / "summary_paper_comparison.csv", index=False)


def run_task(args, file_name: str, variant: str) -> Path:
    from ts_benchmark.baselines.apd_catch import APDCATCH

    output_root = args.output_dir.resolve()
    json_path = result_path(output_root, file_name, variant, args.seed)
    if json_path.is_file() and not args.force:
        print(f"skip existing result {json_path}", flush=True)
        return json_path

    params, original = apd_params(file_name, variant, args.seed)
    print(
        f"start dataset={file_name} variant={variant} seed={args.seed}",
        flush=True,
    )
    print(f"APD parameters: {json.dumps(params, sort_keys=True)}", flush=True)
    train, train_labels, test, labels = load_dataset(args.dataset_root, file_name)
    train_anomaly_rate = float(train_labels.mean())
    if train_anomaly_rate > 0:
        print(
            f"warning: official training split anomaly_rate={train_anomaly_rate:.6f}; "
            "labels are recorded but not passed to the model",
            flush=True,
        )
    model = APDCATCH(**params)

    fit_start = time.time()
    model.detect_fit(train)
    fit_seconds = time.time() - fit_start
    inference_start = time.time()
    diagnostics = None
    if args.save_diagnostics:
        aligned_scores, diagnostics = model.score_with_diagnostics(test)
    else:
        aligned_scores, _ = model.detect_score(test)
    inference_seconds = time.time() - inference_start

    evaluation_start = model.config.seq_len
    if len(test) <= evaluation_start:
        raise ValueError(
            f"Test length {len(test)} does not exceed seq_len={evaluation_start}"
        )
    evaluation_labels = labels[evaluation_start:]
    evaluation_scores = aligned_scores[evaluation_start:]
    predicted_labels = (
        evaluation_scores > model.calibration_threshold
    ).astype(np.int64)
    metrics, metric_errors = evaluate_scores(
        evaluation_labels,
        evaluation_scores,
        predicted_labels,
        full_metrics=args.metrics == "full",
    )

    score_path = json_path.with_suffix(".npz")
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_payload = {
        "labels": evaluation_labels.astype(np.uint8),
        "scores": evaluation_scores.astype(np.float32),
        "predicted_labels": predicted_labels.astype(np.uint8),
    }
    if diagnostics is not None:
        score_payload.update(diagnostics)
        score_payload["reference_location"] = (
            model.model.reference_location.detach().cpu().numpy()
        )
        score_payload["reference_scale"] = (
            model.model.reference_scale.detach().cpu().numpy()
        )
    np.savez_compressed(score_path, **score_payload)
    if args.save_checkpoint:
        import torch

        checkpoint_path = json_path.with_suffix(".pt")
        torch.save(
            {
                "model_state_dict": model.model.state_dict(),
                "apd_params": params,
                "effective_apd_params": model.config.effective_hyper_params(),
                "calibration_threshold": model.calibration_threshold,
                "fit_summary": dataclasses.asdict(model.fit_summary),
            },
            checkpoint_path,
        )
    payload = {
        "schema_version": 1,
        "model_version": "Causal-State-CATCH-v2.0",
        "dataset_file": file_name,
        "paper_dataset": paper_group(file_name),
        "variant": variant,
        "seed": args.seed,
        "protocol": {
            "input": "past_to_next_point",
            "training": "official_train_split_labels_not_used",
            "validation": "chronological_holdout_labels_not_used",
            "threshold": f"normal_validation_{model.config.calibration_fpr:g}_fpr",
            "unscored_test_prefix_excluded": evaluation_start,
        },
        "apd_params": params,
        "effective_apd_params": model.config.effective_hyper_params(),
        "original_catch_script": str(source_script(file_name).relative_to(REPO_ROOT)),
        "original_catch_params": original,
        "fit_summary": dataclasses.asdict(model.fit_summary),
        "timing_seconds": {"fit": fit_seconds, "inference": inference_seconds},
        "test_points_evaluated": int(len(evaluation_labels)),
        "train_anomaly_rate_labels_not_used": train_anomaly_rate,
        "test_anomaly_rate": float(evaluation_labels.mean()),
        "predicted_anomaly_rate": float(predicted_labels.mean()),
        "metrics": metrics,
        "metric_errors": metric_errors,
        "paper_catch_reference": PAPER_REFERENCES[paper_group(file_name)],
        "score_file": str(score_path.relative_to(output_root)),
        "diagnostics_saved": diagnostics is not None,
    }
    atomic_json(json_path, payload)
    write_summaries(output_root)
    print(f"complete {json_path}", flush=True)
    print(f"metrics: {json.dumps(metrics, sort_keys=True)}", flush=True)
    return json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Genesis"],
        help="Paper dataset names, ASD, or all (default: Genesis).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[*VARIANTS, "all"],
        default=["all"],
        help="APD-CATCH variants (default: all).",
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--gpu", default="0", help="CUDA device id, or cpu.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "dataset",
        help="Path to dataset or dataset/anomaly_detect.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "result" / "causal_state_catch_v2",
    )
    parser.add_argument(
        "--metrics",
        choices=("paper", "full"),
        default="full",
        help="paper computes AUC-ROC/AUC-PR and label metrics; full adds range/VUS.",
    )
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Save per-variable NLL, conditional mean/scale, causal state, and innovation scale.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the task and parameter matrix without loading data or training.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check that selected official data files exist, then exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        files = expand_datasets(args.datasets)
    except ValueError as error:
        parser.error(str(error))
    variants = list(VARIANTS) if "all" in args.variants else args.variants

    if args.gpu.casefold() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    missing = missing_data_files(args.dataset_root, files)
    if args.check_data:
        if missing:
            print("Missing data files:")
            for path in missing:
                print(f"  {path}")
            raise SystemExit(1)
        print(f"Data check passed for {len(files)} file(s).")
        return

    plan = []
    for file_name in files:
        for variant in variants:
            params, _ = apd_params(file_name, variant, args.seed)
            plan.append(
                {
                    "dataset_file": file_name,
                    "paper_dataset": paper_group(file_name),
                    "variant": variant,
                    "params": params,
                }
            )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if missing:
        print("Required data is missing. Run:", file=sys.stderr)
        print("  bash scripts/download_tab_datasets.sh", file=sys.stderr)
        for path in missing[:10]:
            print(f"  missing: {path}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)
        raise SystemExit(1)

    print(f"task_count={len(plan)} gpu={args.gpu} metrics={args.metrics}", flush=True)
    for task in plan:
        run_task(args, task["dataset_file"], task["variant"])
    write_summaries(args.output_dir.resolve())
    print(f"all requested tasks complete: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
