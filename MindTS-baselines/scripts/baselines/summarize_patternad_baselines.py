#!/usr/bin/env python
import csv
import io
import math
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "result" / "label"
PATTERNAD_RESULT_ROOT = PROJECT_ROOT.parent / "PatternAD-main" / "result" / "label"
SUMMARY_OUT_PATH = RESULT_ROOT / "patternad_baseline_summary.csv"
THREE_METRIC_OUT_PATH = RESULT_ROOT / "patternad_baseline_three_metrics.csv"

DATASETS = ["MetroPT3", "HAI21", "SMD"]

MODELS = [
    "PatternAD",
    "PatternAD_raw",
    "PCA",
    "IsolationForest",
    "LOF",
    "OCSVM",
    "USAD",
    "OmniAnomaly",
    "DAGMM",
    "TranAD",
    "AnomalyTransformer",
    "GDN",
    "MTAD-GAT",
    "InterFusion",
    "DADA",
    "UniTS",
    "Timer",
    "LMixer",
    "DLinear",
    "PatchTST",
    "iTransformer",
    "TimesNet",
]

TSLIB_RESULT_DIRS = {
    "DLinear": "{dataset}_DLinear_baseline",
    "PatchTST": "{dataset}_PatchTST_baseline",
    "iTransformer": "{dataset}_iTransformer_baseline",
    "TimesNet": "{dataset}_TimesNet_baseline_h0",
}


def result_dir_for(model: str, dataset: str) -> Path:
    if model in {"PatternAD", "PatternAD_raw"}:
        return PATTERNAD_RESULT_ROOT / f"{dataset}_{model}"
    if model in TSLIB_RESULT_DIRS:
        return RESULT_ROOT / TSLIB_RESULT_DIRS[model].format(dataset=dataset)
    return RESULT_ROOT / f"baselines_{dataset}_{model}"


def normalize_legacy_tslib_result_dirs():
    for model in TSLIB_RESULT_DIRS:
        for dataset in DATASETS:
            legacy_dir = RESULT_ROOT / f"baselines_{dataset}_{model}"
            canonical_dir = result_dir_for(model, dataset)
            if legacy_dir.is_dir() and not canonical_dir.exists():
                legacy_dir.rename(canonical_dir)


def latest_report(result_dir: Path):
    reports = sorted(result_dir.glob("test_report.*.csv"))
    return reports[-1] if reports else None


def latest_archive(result_dir: Path):
    archives = sorted(result_dir.glob("*.csv.tar.gz"))
    return archives[-1] if archives else None


def format_score(value, empty_value=""):
    if value is None or value == "":
        return empty_value
    try:
        score = float(value)
    except ValueError:
        return value
    return "nan" if math.isnan(score) else f"{score:.12g}"


def read_report_metrics(report_path: Path):
    with report_path.open(newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return {}
    return {
        row[1]: format_score(row[-1], empty_value="nan")
        for row in rows[1:]
        if len(row) >= 3
    }


def read_archive_vus_metrics(archive_path: Path):
    with tarfile.open(archive_path, "r:gz") as archive:
        csv_members = [
            member for member in archive.getmembers() if member.name.endswith(".csv")
        ]
        if not csv_members:
            return {}
        extracted = archive.extractfile(csv_members[0])
        if extracted is None:
            return {}
        reader = csv.DictReader(io.TextIOWrapper(extracted, encoding="utf-8"))
        for row in reader:
            return {
                "VUS_ROC": format_score(row.get("VUS_ROC")),
                "VUS_PR": format_score(row.get("VUS_PR")),
            }
    return {}


def read_three_metrics(result_dir: Path):
    report = latest_report(result_dir)
    report_metrics = read_report_metrics(report) if report else {}
    archive_metrics = {}
    if "VUS_ROC" not in report_metrics or "VUS_PR" not in report_metrics:
        archive = latest_archive(result_dir)
        archive_metrics = read_archive_vus_metrics(archive) if archive else {}
    return {
        "Aff-F": report_metrics.get("affiliation_f", ""),
        "V-PR": report_metrics.get("VUS_PR", archive_metrics.get("VUS_PR", "")),
        "V-ROC": report_metrics.get("VUS_ROC", archive_metrics.get("VUS_ROC", "")),
    }


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    normalize_legacy_tslib_result_dirs()
    summary_rows = []
    three_metric_rows = []

    for model in MODELS:
        summary_row = {"model": model}
        for dataset in DATASETS:
            result_dir = result_dir_for(model, dataset)
            metrics = read_three_metrics(result_dir)
            summary_row[dataset] = (
                f"{metrics['Aff-F']} / {metrics['V-PR']} / {metrics['V-ROC']}"
                if any(metrics.values())
                else ""
            )
            three_metric_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    **metrics,
                    "result_dir": str(result_dir),
                }
            )
        summary_rows.append(summary_row)

    with SUMMARY_OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + DATASETS)
        writer.writeheader()
        writer.writerows(summary_rows)

    with THREE_METRIC_OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "dataset",
                "Aff-F",
                "V-PR",
                "V-ROC",
                "result_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(three_metric_rows)

    print(SUMMARY_OUT_PATH)
    print(THREE_METRIC_OUT_PATH)


if __name__ == "__main__":
    main()
