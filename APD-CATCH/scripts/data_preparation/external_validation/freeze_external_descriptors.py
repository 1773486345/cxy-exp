"""Freeze the four predeclared external-validation descriptors before scoring.

This is a read-only data analysis step. It imports the frozen descriptor function
from ``analyze_decomposition_applicability.py`` and never constructs a benchmark
model, runs inference, or reads result archives.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

from common import (
    PAPER_DATASET,
    PROJECT_ROOT,
    REGISTRY_PATH,
    RESULT_ROOT,
    TASK_ORDER,
    load_prepared_task,
    utc_now,
)


SEQ_LEN = 192
PATCH_SIZE = 16
CANDIDATES = {
    "mean_drift": "positive",
    "low_frequency_energy_ratio_mean": "positive",
    "periodicity_top3_ratio": "positive",
    "correlation_drift": "negative",
}


def analysis_module():
    path = PROJECT_ROOT / "scripts" / "analyze_decomposition_applicability.py"
    spec = importlib.util.spec_from_file_location("external_descriptor_formula", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen descriptor script {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT_ROOT.parent), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError("prepare all external tasks before freezing descriptors")
    result_score_root = PROJECT_ROOT / "result" / "score" / "external_validation"
    if result_score_root.exists() and any(result_score_root.rglob("*.tar.gz")):
        raise RuntimeError("external model results already exist; descriptor freeze must precede scoring")
    registry = pd.read_csv(REGISTRY_PATH).set_index("task")
    missing = [task for task in TASK_ORDER if task not in registry.index]
    if missing:
        raise RuntimeError(f"cannot freeze descriptors; missing prepared tasks: {missing}")
    formula, formula_path = analysis_module()
    rows = []
    for task in TASK_ORDER:
        train, test, train_labels, test_labels, _ = load_prepared_task(task)
        base, _ = formula.describe_training_data(
            train,
            test_labels,
            len(test),
            SEQ_LEN,
            PATCH_SIZE,
            None,
        )
        rows.append(
            {
                "task": task,
                "paper_dataset": PAPER_DATASET[task],
                "prepared_sha256": registry.loc[task, "prepared_sha256"],
                "seq_len": SEQ_LEN,
                "patch_size": PATCH_SIZE,
                "descriptor_mode": "frozen analyze_decomposition_applicability.describe_training_data; train only; deterministic equal-weight decomposition unavailable to score models",
                **{name: base[name] for name in CANDIDATES},
            }
        )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    task_output = pd.DataFrame(rows)
    task_output.to_csv(RESULT_ROOT / "external_task_descriptors.csv", index=False)
    macro = task_output.groupby("paper_dataset", sort=False).mean(numeric_only=True).reset_index()
    macro["task_count"] = macro["paper_dataset"].map(task_output.groupby("paper_dataset").size())
    paper_order = ["HAI20_07", "BATADAL", "MetroPT3", "OPPORTUNITY", "Occupancy", "Metro", "SWAN-SF"]
    macro["paper_dataset"] = pd.Categorical(macro["paper_dataset"], paper_order, ordered=True)
    macro.sort_values("paper_dataset", inplace=True)
    macro.to_csv(RESULT_ROOT / "external_dataset_descriptors.csv", index=False)
    source_digest = hashlib.sha256(formula_path.read_bytes()).hexdigest()
    freeze = {
        "analysis_code_commit": git_head(),
        "descriptor_formula_file": str(formula_path.relative_to(PROJECT_ROOT)),
        "descriptor_formula_sha256": source_digest,
        "descriptor_function": "describe_training_data",
        "formula_version": "frozen applicability formulas at formal seq_len windows",
        "seq_len": SEQ_LEN,
        "patch_size": PATCH_SIZE,
        "candidate_expected_delta_auc_roc_directions": CANDIDATES,
        "prepared_data_sha256": {task: registry.loc[task, "prepared_sha256"] for task in TASK_ORDER},
        "computed_at": utc_now(),
        "model_results_present_at_freeze": False,
    }
    with (RESULT_ROOT / "external_descriptor_freeze.json").open("w", encoding="utf-8") as handle:
        json.dump(freeze, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"frozen {len(task_output)} task descriptors and {len(macro)} dataset descriptors")


if __name__ == "__main__":
    main()
