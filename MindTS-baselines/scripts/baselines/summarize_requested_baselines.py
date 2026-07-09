#!/usr/bin/env python
import csv
import io
import math
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "result" / "label"
OUT_PATH = RESULT_ROOT / "_baseline_logs" / "requested_baseline_summary.csv"
THREE_METRIC_OUT_PATH = (
    RESULT_ROOT / "_baseline_logs" / "requested_baseline_three_metrics.csv"
)

DATASETS = [
    "Genesis",
    "Weather",
    "Energy",
    "SKAB",
    "MSDS",
    "Daphnet",
    "GECCO",
    "ExathlonSmall",
    "Metro",
]

BASELINES = [
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
]

TSLIB_RESULT_DIRS = {
    "DLinear": "{dataset}_DLinear_baseline",
    "PatchTST": "{dataset}_PatchTST_baseline",
    "iTransformer": "{dataset}_iTransformer_baseline",
    "TimesNet": "{dataset}_TimesNet_baseline_h0",
}


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
        csv_members = [member for member in archive.getmembers() if member.name.endswith(".csv")]
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
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    three_metric_rows = []
    for baseline in BASELINES:
        row = {"baseline": baseline}
        for dataset in DATASETS:
            result_dir = RESULT_ROOT / f"baselines_{dataset}_{baseline}"
            metrics = read_three_metrics(result_dir)
            row[dataset] = metrics["Aff-F"]
            three_metric_rows.append(
                {"baseline": baseline, "dataset": dataset, **metrics}
            )
        rows.append(row)

    for baseline, result_dir_template in TSLIB_RESULT_DIRS.items():
        for dataset in DATASETS:
            result_dir = RESULT_ROOT / result_dir_template.format(dataset=dataset)
            three_metric_rows.append(
                {
                    "baseline": baseline,
                    "dataset": dataset,
                    **read_three_metrics(result_dir),
                }
            )

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["baseline"] + DATASETS)
        writer.writeheader()
        writer.writerows(rows)

    with THREE_METRIC_OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["baseline", "dataset", "Aff-F", "V-PR", "V-ROC"]
        )
        writer.writeheader()
        writer.writerows(three_metric_rows)

    print(OUT_PATH)
    with OUT_PATH.open() as f:
        print(f.read())
    print(THREE_METRIC_OUT_PATH)


if __name__ == "__main__":
    main()
