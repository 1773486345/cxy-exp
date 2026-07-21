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

from common import PAPER_DATASET, PROJECT_ROOT, RESULT_ROOT, TASK_ORDER


CANDIDATES = {
    "mean_drift": 1,
    "low_frequency_energy_ratio_mean": 1,
    "periodicity_top3_ratio": 1,
    "correlation_drift": -1,
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
    source_rows, metric_rows = [], []
    for task in TASK_ORDER:
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
    print("read-only external result summary complete")


if __name__ == "__main__":
    main()
