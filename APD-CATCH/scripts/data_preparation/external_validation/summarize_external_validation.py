"""Read completed external CATCH/MSD archives and summarize the fixed validation.

This module has no dependency on benchmark runners. It never trains, scores, or
reconstructs a sequence; it only reads completed CSV archives and frozen
descriptors.
"""

from __future__ import annotations

import csv
import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import PAPER_DATASET, PROJECT_ROOT, REGISTRY_PATH, RESULT_ROOT, TASK_ORDER
from external_baseline_assets import BASELINE_SPECS


CANDIDATES = {
    "mean_drift": 1,
    "low_frequency_energy_ratio_mean": 1,
    "periodicity_top3_ratio": 1,
    "correlation_drift": -1,
}


ALL_METHODS = ("CATCH", "MSDCATCH", *(spec["paper_name"] for spec in BASELINE_SPECS))
METHOD_RESULT_DIR = {
    "CATCH": "CATCH",
    "MSDCATCH": "MSDCATCH",
    **{spec["paper_name"]: spec["result_name"] for spec in BASELINE_SPECS},
}


def read_archive(path: Path) -> dict | None:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.name.endswith(".csv")]
            if not members:
                return None
            with archive.extractfile(members[0]) as handle:
                if handle is None:
                    return None
                row = next(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))
    except (OSError, tarfile.TarError, StopIteration):
        return None
    try:
        auc_pr, auc_roc = float(row["auc_pr"]), float(row["auc_roc"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(auc_pr) or not math.isfinite(auc_roc):
        return None
    if "traceback" in str(row.get("log_info", "")).lower():
        return None
    return {"auc_pr": auc_pr, "auc_roc": auc_roc, "model_params": row.get("model_params", ""), "strategy_args": row.get("strategy_args", "")}


def single_valid_archive(task: str, model: str) -> tuple[Path, dict]:
    root = PROJECT_ROOT / "result" / "score" / "external_validation" / task / model
    candidates = []
    for path in sorted(root.rglob("*.tar.gz")) if root.exists() else []:
        row = read_archive(path)
        if row is not None:
            candidates.append((path, row))
    if len(candidates) != 1:
        raise RuntimeError(f"{task}/{model}: expected exactly one valid archive, found {len(candidates)}")
    return candidates[0]


def archive_status(root: Path) -> tuple[str, Path | None, dict | None]:
    """Collect a method archive without pretending incomplete coverage is valid."""
    archives = sorted(root.rglob("*.tar.gz")) if root.exists() else []
    valid = [(path, row) for path in archives if (row := read_archive(path)) is not None]
    if len(valid) == 1:
        return "valid", valid[0][0], valid[0][1]
    if len(valid) > 1:
        return "multiple_valid", None, None
    return ("invalid" if archives else "missing"), None, None


def collect_all_method_task_results(
    result_root: Path = PROJECT_ROOT / "result" / "score" / "external_validation",
    tasks: tuple[str, ...] | list[str] = TASK_ORDER,
) -> pd.DataFrame:
    """Return the fixed 20 by 17 result grid without reading descriptors."""
    rows = []
    for task in tasks:
        for method in ALL_METHODS:
            status, path, values = archive_status(result_root / task / METHOD_RESULT_DIR[method])
            if path is None:
                archive = ""
            else:
                try:
                    archive = str(path.relative_to(PROJECT_ROOT))
                except ValueError:
                    archive = str(path)
            rows.append(
                {
                    "task": task,
                    "paper_dataset": PAPER_DATASET[task],
                    "method": method,
                    "auc_pr": values["auc_pr"] if values else np.nan,
                    "auc_roc": values["auc_roc"] if values else np.nan,
                    "status": status,
                    "archive": archive,
                }
            )
    return pd.DataFrame(rows)


def aggregate_all_method_results(task_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Macro-average by paper dataset and rank only complete formal observations."""
    expected = pd.Series(PAPER_DATASET).value_counts().rename("expected_task_count")
    valid = task_results.loc[task_results["status"] == "valid"].copy()
    metrics = valid.groupby(["paper_dataset", "method"], as_index=False)[["auc_pr", "auc_roc"]].mean()
    counts = valid.groupby(["paper_dataset", "method"], as_index=False).size().rename(columns={"size": "valid_task_count"})
    grid = pd.MultiIndex.from_product([expected.index.tolist(), list(ALL_METHODS)], names=["paper_dataset", "method"]).to_frame(index=False)
    dataset = grid.merge(metrics, on=["paper_dataset", "method"], how="left").merge(counts, on=["paper_dataset", "method"], how="left")
    dataset.rename(columns={"auc_pr": "provisional_auc_pr", "auc_roc": "provisional_auc_roc"}, inplace=True)
    dataset["expected_task_count"] = dataset["paper_dataset"].map(expected).astype(int)
    dataset["valid_task_count"] = dataset["valid_task_count"].fillna(0).astype(int)
    dataset["complete"] = (
        (dataset["valid_task_count"] == dataset["expected_task_count"])
        & np.isfinite(dataset["provisional_auc_pr"])
        & np.isfinite(dataset["provisional_auc_roc"])
    )
    dataset["auc_pr"] = dataset["provisional_auc_pr"].where(dataset["complete"])
    dataset["auc_roc"] = dataset["provisional_auc_roc"].where(dataset["complete"])
    dataset["auc_pr_rank"] = dataset.groupby("paper_dataset")["auc_pr"].rank(ascending=False, method="average")
    dataset["auc_roc_rank"] = dataset.groupby("paper_dataset")["auc_roc"].rank(ascending=False, method="average")
    msd = dataset.loc[dataset["method"] == "MSDCATCH", ["paper_dataset", "auc_pr", "auc_roc"]].rename(
        columns={"auc_pr": "msd_auc_pr", "auc_roc": "msd_auc_roc"}
    )
    dataset = dataset.merge(msd, on="paper_dataset", how="left", validate="many_to_one")
    dataset["msd_minus_method_auc_pr"] = dataset["msd_auc_pr"] - dataset["auc_pr"]
    dataset["msd_minus_method_auc_roc"] = dataset["msd_auc_roc"] - dataset["auc_roc"]

    formal = dataset.loc[dataset["complete"]].copy()
    summary = formal.groupby("method", as_index=False).agg(
        auc_pr_mean=("auc_pr", "mean"),
        auc_pr_median=("auc_pr", "median"),
        auc_roc_mean=("auc_roc", "mean"),
        auc_roc_median=("auc_roc", "median"),
        average_auc_pr_rank=("auc_pr_rank", "mean"),
        average_auc_roc_rank=("auc_roc_rank", "mean"),
        valid_dataset_count=("auc_pr", "count"),
        complete_dataset_count=("complete", "sum"),
    )
    summary = pd.DataFrame({"method": list(ALL_METHODS)}).merge(summary, on="method", how="left")
    coverage = dataset.groupby("method", as_index=False).agg(
        provisional_dataset_count=("provisional_auc_pr", "count"),
        complete_dataset_count=("complete", "sum"),
    )
    summary.drop(columns=["complete_dataset_count"], errors="ignore", inplace=True)
    summary = summary.merge(coverage, on="method", how="left")
    summary["expected_dataset_count"] = len(expected)
    summary["complete"] = summary["complete_dataset_count"] == summary["expected_dataset_count"]
    return dataset, summary


def write_all_method_outputs() -> None:
    """Write the competitive comparison only; no scores are recomputed here."""
    task = collect_all_method_task_results()
    dataset, summary = aggregate_all_method_results(task)
    task.to_csv(RESULT_ROOT / "external_all_methods_task_results.csv", index=False)
    dataset.to_csv(RESULT_ROOT / "external_all_methods_dataset_results.csv", index=False)
    summary.to_csv(RESULT_ROOT / "external_all_methods_average_ranks.csv", index=False)
    lines = ["# External All-Methods Summary", "", "Results are read-only archive metadata. Missing or invalid tasks are never ranked.", ""]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.method}: PR mean={row.auc_pr_mean}, ROC mean={row.auc_roc_mean}, "
            f"provisional dataset coverage={row.provisional_dataset_count}/{row.expected_dataset_count}, "
            f"complete datasets={row.complete_dataset_count}/{row.expected_dataset_count}, complete={row.complete}"
        )
    (RESULT_ROOT / "external_all_methods_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rho(frame: pd.DataFrame, x: str, y: str) -> float | None:
    values = frame[[x, y]].dropna()
    if len(values) < 3 or values[x].nunique() < 2 or values[y].nunique() < 2:
        return None
    return float(values[x].corr(values[y], method="spearman"))


def roc_group(value: float) -> str:
    if value > 0.01:
        return "gain"
    if value < -0.01:
        return "loss"
    return "neutral"


def candidate_validation(task: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for descriptor, expected in CANDIDATES.items():
        task_rho = rho(task, descriptor, "delta_auc_roc")
        dataset_rho = rho(dataset, descriptor, "delta_auc_roc")
        leave_out = []
        for paper_dataset in dataset["paper_dataset"]:
            value = rho(task[task["paper_dataset"] != paper_dataset], descriptor, "delta_auc_roc")
            leave_out.append(value)
        sign_stable = bool(
            task_rho is not None
            and all(value is not None and value * expected > 0 for value in leave_out)
        )
        grouped = dataset.assign(roc_group=dataset["delta_auc_roc"].map(roc_group))
        gain = grouped.loc[grouped["roc_group"] == "gain", descriptor].dropna()
        other = grouped.loc[grouped["roc_group"] != "gain", descriptor].dropna()
        if len(gain) < 2 or len(other) < 2:
            group_status = "inconclusive_due_to_group_size"
            median_direction_ok = None
        else:
            group_status = "computed"
            median_direction_ok = bool((gain.median() - other.median()) * expected > 0)
        direction_ok = bool(task_rho is not None and dataset_rho is not None and task_rho * expected > 0 and dataset_rho * expected > 0)
        magnitude_ok = bool(task_rho is not None and dataset_rho is not None and abs(task_rho) >= 0.35 and abs(dataset_rho) >= 0.35)
        externally_supported = bool(direction_ok and magnitude_ok and sign_stable and median_direction_ok is True)
        rows.append(
            {
                "descriptor": descriptor,
                "expected_direction": "positive" if expected > 0 else "negative",
                "task_level_rho": task_rho,
                "dataset_level_rho": dataset_rho,
                "grouped_leave_one_dataset_out_rhos": json.dumps(leave_out),
                "leave_one_dataset_out_sign_stable": sign_stable,
                "gain_median": float(gain.median()) if len(gain) else None,
                "loss_or_neutral_median": float(other.median()) if len(other) else None,
                "group_status": group_status,
                "median_direction_consistent": median_direction_ok,
                "externally_supported": externally_supported,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    freeze_path = RESULT_ROOT / "external_descriptor_freeze.json"
    descriptor_path = RESULT_ROOT / "external_task_descriptors.csv"
    if not freeze_path.exists() or not descriptor_path.exists():
        raise FileNotFoundError("frozen external descriptors are required before result summarization")
    with freeze_path.open(encoding="utf-8") as handle:
        freeze = json.load(handle)
    if set(freeze["candidate_expected_delta_auc_roc_directions"]) != set(CANDIDATES):
        raise RuntimeError("frozen descriptor candidate set differs from the fixed external protocol")
    registry = pd.read_csv(REGISTRY_PATH).set_index("task")
    status = registry.get("status", pd.Series("valid", index=registry.index)).fillna("valid")
    valid_tasks = [task for task in TASK_ORDER if task in registry.index and status.loc[task] == "valid"]
    excluded = registry.loc[[task for task in TASK_ORDER if task in registry.index and status.loc[task] != "valid"]].reset_index()
    source_rows, metric_rows = [], []
    for task in valid_tasks:
        catch_path, catch = single_valid_archive(task, "CATCH")
        msd_path, msd = single_valid_archive(task, "MSDCATCH")
        if catch["model_params"] != msd["model_params"] or catch["strategy_args"] != msd["strategy_args"]:
            raise RuntimeError(f"{task}: CATCH/MSD archive configuration mismatch")
        metric_rows.append(
            {
                "task": task,
                "paper_dataset": PAPER_DATASET[task],
                "catch_auc_pr": catch["auc_pr"],
                "msd_auc_pr": msd["auc_pr"],
                "delta_auc_pr": msd["auc_pr"] - catch["auc_pr"],
                "catch_auc_roc": catch["auc_roc"],
                "msd_auc_roc": msd["auc_roc"],
                "delta_auc_roc": msd["auc_roc"] - catch["auc_roc"],
            }
        )
        source_rows.extend(
            [
                {"task": task, "model": "CATCH", "archive": str(catch_path.relative_to(PROJECT_ROOT))},
                {"task": task, "model": "MSDCATCH", "archive": str(msd_path.relative_to(PROJECT_ROOT))},
            ]
        )
    task_metrics = pd.DataFrame(metric_rows)
    descriptors = pd.read_csv(descriptor_path)
    task = task_metrics.merge(descriptors, on=["task", "paper_dataset"], validate="one_to_one")
    dataset_metrics = task_metrics.groupby("paper_dataset", sort=False)[
        ["catch_auc_pr", "msd_auc_pr", "delta_auc_pr", "catch_auc_roc", "msd_auc_roc", "delta_auc_roc"]
    ].mean().reset_index()
    dataset_descriptors = pd.read_csv(RESULT_ROOT / "external_dataset_descriptors.csv")
    dataset = dataset_metrics.merge(dataset_descriptors, on="paper_dataset", validate="one_to_one")
    validation = candidate_validation(task, dataset)
    task.to_csv(RESULT_ROOT / "external_task_results.csv", index=False)
    dataset.to_csv(RESULT_ROOT / "external_dataset_results.csv", index=False)
    validation.to_csv(RESULT_ROOT / "external_descriptor_roc_validation.csv", index=False)
    pd.DataFrame(source_rows).to_csv(RESULT_ROOT / "external_result_sources.csv", index=False)
    excluded.to_csv(RESULT_ROOT / "external_excluded_tasks.csv", index=False)
    # The baseline comparison is deliberately separate from the frozen CATCH/MSD
    # descriptor path above; it neither changes candidate validation nor adds data.
    write_all_method_outputs()
    print("read-only external result summary complete")


if __name__ == "__main__":
    main()
