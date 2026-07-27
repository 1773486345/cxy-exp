#!/usr/bin/env python3
"""Select valid TypeFusion-CATCH runs and produce read-only real-task summaries.

Archives, not leaderboard reports, provide the registered AUC-PR/AUC-ROC and
supplementary metrics.  Every discovered run is audited newest-first.  A broken
newer run therefore cannot hide an older valid run.  This program never trains,
reruns a baseline, or derives any metric from labels or score vectors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from typefusion_catch_result_utils import (
    RESULT_ROOT,
    failure_marker,
    parse_leaderboard_report,
    parse_metric_archive,
    parse_run_timestamp,
    read_registry,
    write_registry,
)


ROOT = Path(__file__).resolve().parents[1]
SCORE_ROOT = ROOT / "result/score/TypeFusion-CATCH"
ASD_TASKS = tuple(f"ASD_dataset_{index}" for index in range(1, 13))
PAPER_DATASETS = (
    "ASD",
    "CICIDS",
    "CalIt2",
    "Creditcard",
    "GECCO",
    "Genesis",
    "MSL",
    "NYC",
    "PSM",
    "SMAP",
    "SMD",
    "SWAT",
)
COMPARISON_CONFIG_STATUSES = {
    "exact_shared_config",
    "architecture_compatibility_override",
    "fairness_rerun_required",
    "batch_paired_rerun_required",
    "historical_code_mismatch",
    "source_metric_conflict",
    "missing_valid_baseline",
    "missing_valid_typefusion_result",
}
FIXED_TYPEFUSION_KEYS = {
    "seed": 2021,
    "fit_mode": "three_stage",
    "training_budget_mode": "equal_total_steps",
    "joint_finetune_lr_scale": 0.1,
    "lambda_freq": 0.1,
    "lambda_mask": 0.1,
    "temporal_hidden_dim": 128,
    "temporal_layers": 3,
    "memory_size": 32,
    "memory_topk": 4,
    "branch_dim": 128,
    "fusion_layers": 2,
    "fusion_heads": 4,
    "relation_mask_groups": 4,
    "pattern_mask_ratio": 0.25,
}


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def value_or_na(value: Any) -> float | str:
    return float(value) if finite(value) else "N/A"


def json_contains_forbidden_label_use(value: Any) -> bool:
    text = json.dumps(value, sort_keys=True).lower()
    return any(term in text for term in ("detect_label", "test_label", "test-label", "label_based"))


def expected_params(source: Mapping[str, str]) -> Dict[str, Any]:
    return json.loads(source["typefusion_hyper_params_json"])


def _single(run_dir: Path, pattern: str, reason: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"{reason}:{len(matches)}")
    return matches[0]


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    candidates = (run_dir / "typefusion_run_config.json", run_dir / "run_config.json")
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("run_config_json_invalid") from exc
    raise ValueError("run_config_missing")


def _training_summary(run_dir: Path, run_config: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = run_config.get("training_budget_summary")
    if isinstance(summary, Mapping):
        return summary
    path = run_dir / "typefusion_training_summary.json"
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("training_summary_json_invalid") from exc
        if isinstance(value, Mapping):
            return value
    raise ValueError("training_budget_summary_missing")


def _validate_budget(summary: Mapping[str, Any]) -> None:
    if summary.get("mode") != "equal_total_steps":
        raise ValueError("training_budget_mode_not_equal_total_steps")
    try:
        reference = int(summary["reference_total_steps"])
        planned_steps = [
            int(summary[f"{stage}_steps"])
            for stage in ("branch_pretrain", "fusion_train", "joint_finetune")
        ]
        planned = sum(planned_steps)
        actual = [
            int(summary[f"actual_{stage}_steps"])
            for stage in ("branch_pretrain", "fusion_train", "joint_finetune")
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("training_budget_summary_incomplete") from exc
    if reference < 3 or planned != reference or any(step < 1 for step in planned_steps):
        raise ValueError("training_budget_plan_invalid")
    if any(step < 1 for step in actual):
        raise ValueError("training_stage_not_completed")
    if sum(actual) > reference:
        raise ValueError("optimizer_steps_exceed_formal_budget")


def _check_params(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key, expected_value in expected.items():
        if key == "batch_size":
            continue
        if actual.get(key) != expected_value:
            raise ValueError(f"model_param_mismatch:{key}")
    for key, expected_value in FIXED_TYPEFUSION_KEYS.items():
        if actual.get(key) != expected_value:
            raise ValueError(f"fixed_typefusion_param_mismatch:{key}")


def validate_typefusion_run(run_dir: Path, source: Mapping[str, str]) -> Dict[str, Any]:
    """Validate a completed run and return archive-backed data, or raise reason."""

    if not run_dir.is_dir():
        raise ValueError("run_directory_missing")
    run_config = _load_run_config(run_dir)
    report_path = _single(run_dir, "test_report*.csv", "expected_one_test_report")
    archive_path = _single(run_dir, "TypeFusion-CATCH.*.csv.tar.gz", "expected_one_typefusion_archive")
    marker = failure_marker([*run_dir.glob("*.log"), *run_dir.glob("*.txt")])
    if marker:
        raise ValueError(marker)
    archive = parse_metric_archive(archive_path, source["data_name"], "TypeFusion-CATCH")
    report = parse_leaderboard_report(report_path)
    if report["model_name"] != "TypeFusion-CATCH":
        raise ValueError("test_report_model_name_mismatch")
    if report["strategy"].get("strategy_name") != "unfixed_detect_score":
        raise ValueError("test_report_strategy_mismatch")
    if int(report["strategy"].get("seed")) != 2021:
        raise ValueError("test_report_seed_mismatch")
    if run_config.get("data_name") != source["data_name"]:
        raise ValueError("run_config_data_name_mismatch")
    if run_config.get("model_name") not in {"TypeFusion-CATCH", "typefusion_catch.TypeFusionCATCH"}:
        raise ValueError("run_config_model_name_mismatch")
    if int(run_config.get("seed", -1)) != 2021:
        raise ValueError("run_config_seed_mismatch")
    run_params = run_config.get("model_hyper_params")
    if not isinstance(run_params, Mapping):
        raise ValueError("run_config_model_hyper_params_missing")
    expected = expected_params(source)
    _check_params(run_params, expected)
    _check_params(archive.model_params, expected)
    _check_params(report["model_params"], expected)
    if archive.model_params.get("batch_size") != run_params.get("batch_size"):
        raise ValueError("archive_batch_size_mismatch")
    if json_contains_forbidden_label_use(run_config) or json_contains_forbidden_label_use(archive.model_params):
        raise ValueError("test_label_parameter_detected")
    _validate_budget(_training_summary(run_dir, run_config))
    if run_config.get("completed_stages") != [
        "branch_pretrain",
        "fusion_train",
        "joint_finetune",
    ]:
        raise ValueError("training_stage_completion_record_missing")
    return {
        "run_path": str(run_dir),
        "run_timestamp": parse_run_timestamp(run_dir).isoformat(),
        "archive_path": str(archive_path),
        "archive_sha256": archive.archive_sha256,
        "test_report": str(report_path),
        "seed": archive.strategy["seed"],
        "seq_len": archive.model_params["seq_len"],
        "patch_size": archive.model_params["patch_size"],
        "patch_stride": archive.model_params["patch_stride"],
        "batch_size": archive.model_params["batch_size"],
        **{f"typefusion_{key}": metric for key, metric in archive.metrics.items()},
    }


def candidate_run_dirs(task: str, score_root: Path = SCORE_ROOT) -> List[Path]:
    parent = score_root / task
    if not parent.is_dir():
        return []
    return sorted(
        (path for path in parent.glob("run-*") if path.is_dir()),
        key=parse_run_timestamp,
        reverse=True,
    )


def select_latest_valid_run(task: str, source: Mapping[str, str], score_root: Path = SCORE_ROOT) -> tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    """Check all candidates newest-first and retain a row for every rejection."""

    audit_rows: List[Dict[str, Any]] = []
    selected: Dict[str, Any] | None = None
    candidates = candidate_run_dirs(task, score_root)
    if not candidates:
        return None, [
            {
                "task": task,
                "run_path": "",
                "run_timestamp": "",
                "candidate_order": 0,
                "valid": False,
                "rejection_reason": "missing_valid_typefusion_result",
                "selected": False,
                "archive_path": "",
                "archive_sha256": "",
                "seed": "",
                "seq_len": "",
                "batch_size": "",
                "auc_roc": "N/A",
                "auc_pr": "N/A",
            }
        ]
    for index, run_dir in enumerate(candidates, start=1):
        row: Dict[str, Any] = {
            "task": task,
            "run_path": str(run_dir),
            "run_timestamp": parse_run_timestamp(run_dir).isoformat(),
            "candidate_order": index,
            "valid": False,
            "rejection_reason": "",
            "selected": False,
            "archive_path": "",
            "archive_sha256": "",
            "seed": "",
            "seq_len": "",
            "batch_size": "",
            "auc_roc": "N/A",
            "auc_pr": "N/A",
        }
        try:
            parsed = validate_typefusion_run(run_dir, source)
            row.update(
                {
                    "valid": True,
                    "archive_path": parsed["archive_path"],
                    "archive_sha256": parsed["archive_sha256"],
                    "seed": parsed["seed"],
                    "seq_len": parsed["seq_len"],
                    "batch_size": parsed["batch_size"],
                    "auc_roc": parsed["typefusion_auc_roc"],
                    "auc_pr": parsed["typefusion_auc_pr"],
                }
            )
            if selected is None:
                selected = parsed
                row["selected"] = True
        except Exception as exc:
            row["rejection_reason"] = str(exc)
        audit_rows.append(row)
    return selected, audit_rows


def _baseline(source: Mapping[str, str], gecco_fair: Mapping[str, str] | None) -> Dict[str, Any]:
    if source["task"] == "GECCO" and gecco_fair and as_bool(gecco_fair.get("fair_baseline_available")):
        return {
            "valid": True,
            "archive_commit": gecco_fair.get("fair_source_code_version", ""),
            "code_group": "fair_baseline_pending_code_audit",
            "auc_roc": value_or_na(gecco_fair.get("fair_auc_roc")),
            "auc_pr": value_or_na(gecco_fair.get("fair_auc_pr")),
            "r_auc_roc": value_or_na(gecco_fair.get("fair_r_auc_roc")),
            "r_auc_pr": value_or_na(gecco_fair.get("fair_r_auc_pr")),
            "vus_roc": value_or_na(gecco_fair.get("fair_vus_roc")),
            "vus_pr": value_or_na(gecco_fair.get("fair_vus_pr")),
            "fit_time": value_or_na(gecco_fair.get("fair_fit_time")),
            "inference_time": value_or_na(gecco_fair.get("fair_inference_time")),
            "seq_len": gecco_fair.get("fair_seq_len", ""),
            "patch_size": gecco_fair.get("fair_patch_size", ""),
            "patch_stride": gecco_fair.get("fair_patch_stride", ""),
            "batch_size": gecco_fair.get("fair_batch_size", ""),
            "historical_code_comparable": True,
            "report_conflict": False,
            "is_fair_baseline": True,
        }
    return {
        "valid": as_bool(source.get("baseline_archive_valid")),
        "archive_commit": source.get("source_catch_master_commit", ""),
        "code_group": source.get("historical_code_group", ""),
        "auc_roc": value_or_na(source.get("baseline_auc_roc")),
        "auc_pr": value_or_na(source.get("baseline_auc_pr")),
        "r_auc_roc": value_or_na(source.get("baseline_r_auc_roc")),
        "r_auc_pr": value_or_na(source.get("baseline_r_auc_pr")),
        "vus_roc": value_or_na(source.get("baseline_vus_roc")),
        "vus_pr": value_or_na(source.get("baseline_vus_pr")),
        "fit_time": value_or_na(source.get("baseline_fit_time")),
        "inference_time": value_or_na(source.get("baseline_inference_time")),
        "seq_len": source.get("baseline_seq_len", source.get("seq_len", "")),
        "patch_size": source.get("patch_size", ""),
        "patch_stride": source.get("patch_stride", ""),
        "batch_size": source.get("baseline_batch_size", source.get("batch_size", "")),
        "historical_code_comparable": as_bool(source.get("historical_code_comparable")),
        "report_conflict": as_bool(source.get("report_archive_metric_conflict")),
        "is_fair_baseline": False,
    }


def comparison_status(source: Mapping[str, str], baseline: Mapping[str, Any], selected: Mapping[str, Any] | None) -> str:
    if not baseline["valid"]:
        return "missing_valid_baseline"
    # With no selected TypeFusion archive, this is the actionable current task
    # state. Baseline fairness/code audit details remain registered separately
    # and will govern comparability once a TypeFusion result exists.
    if selected is None:
        return "missing_valid_typefusion_result"
    if (
        source["task"] == "GECCO"
        and as_bool(source.get("baseline_requires_fair_rerun"))
        and not baseline["is_fair_baseline"]
    ):
        return "fairness_rerun_required"
    if baseline["report_conflict"]:
        return "source_metric_conflict"
    if not baseline["historical_code_comparable"]:
        return "historical_code_mismatch"
    if str(selected["batch_size"]) != str(baseline["batch_size"]):
        return "batch_paired_rerun_required"
    if source["task"] == "ASD_dataset_1":
        return "architecture_compatibility_override"
    return "exact_shared_config"


def _delta(left: Any, right: Any) -> float | str:
    return float(right) - float(left) if finite(left) and finite(right) else "N/A"


def task_metric_row(source: Mapping[str, str], selected: Mapping[str, Any] | None, gecco_fair: Mapping[str, str] | None) -> Dict[str, Any]:
    baseline = _baseline(source, gecco_fair)
    status = comparison_status(source, baseline, selected)
    if status not in COMPARISON_CONFIG_STATUSES:
        raise AssertionError(f"unregistered comparison status: {status}")
    typefusion = selected or {}
    geometry_match = selected is not None and all(
        str(typefusion.get(key, "")) == str(baseline[key])
        for key in ("seq_len", "patch_size", "patch_stride")
    )
    primary_present = all(
        finite(value)
        for value in (
            baseline["auc_pr"],
            baseline["auc_roc"],
            typefusion.get("typefusion_auc_pr"),
            typefusion.get("typefusion_auc_roc"),
        )
    )
    comparable = bool(
        status in {"exact_shared_config", "architecture_compatibility_override"}
        and primary_present
        and geometry_match
        and str(typefusion.get("seed", "")) == "2021"
    )
    row = {
        "task": source["task"],
        "paper_dataset": source["paper_dataset"],
        "comparison_config_status": status,
        "comparison_config_reason": (
            "CATCH source uses cf_dim=4 and n_heads=16, while TypeFusion MultiheadAttention requires num_heads not exceeding and dividing the embedding dimension; the preregistered compatibility mapping uses n_heads=4."
            if status == "architecture_compatibility_override"
            else source.get("historical_code_difference", "") if status == "historical_code_mismatch" else ""
        ),
        "catch_archive_commit": baseline["archive_commit"],
        "catch_code_group": baseline["code_group"],
        "catch_auc_pr": baseline["auc_pr"],
        "catch_auc_roc": baseline["auc_roc"],
        "typefusion_auc_pr": typefusion.get("typefusion_auc_pr", "N/A"),
        "typefusion_auc_roc": typefusion.get("typefusion_auc_roc", "N/A"),
        "delta_auc_pr": _delta(baseline["auc_pr"], typefusion.get("typefusion_auc_pr")),
        "delta_auc_roc": _delta(baseline["auc_roc"], typefusion.get("typefusion_auc_roc")),
        "catch_r_auc_pr": baseline["r_auc_pr"],
        "catch_r_auc_roc": baseline["r_auc_roc"],
        "typefusion_r_auc_pr": typefusion.get("typefusion_r_auc_pr", "N/A"),
        "typefusion_r_auc_roc": typefusion.get("typefusion_r_auc_roc", "N/A"),
        "catch_vus_pr": baseline["vus_pr"],
        "catch_vus_roc": baseline["vus_roc"],
        "typefusion_vus_pr": typefusion.get("typefusion_vus_pr", "N/A"),
        "typefusion_vus_roc": typefusion.get("typefusion_vus_roc", "N/A"),
        "catch_fit_time": baseline["fit_time"],
        "catch_inference_time": baseline["inference_time"],
        "typefusion_fit_time": typefusion.get("typefusion_fit_time", "N/A"),
        "typefusion_inference_time": typefusion.get("typefusion_inference_time", "N/A"),
        "catch_seq_len": baseline["seq_len"],
        "catch_patch_size": baseline["patch_size"],
        "catch_patch_stride": baseline["patch_stride"],
        "typefusion_seq_len": typefusion.get("seq_len", ""),
        "typefusion_patch_size": typefusion.get("patch_size", ""),
        "typefusion_patch_stride": typefusion.get("patch_stride", ""),
        "catch_batch_size": baseline["batch_size"],
        "typefusion_batch_size": typefusion.get("batch_size", ""),
        "typefusion_run_path": typefusion.get("run_path", ""),
        "typefusion_archive_path": typefusion.get("archive_path", ""),
        "comparable": comparable,
        "exclusion_reason": "" if comparable else status,
    }
    return row


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [float(row[field]) for row in rows if finite(row.get(field))]
    return sum(values) / len(values) if values else "N/A"


def paper_metric_rows(task_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    paper_rows: List[Dict[str, Any]] = []
    for paper in PAPER_DATASETS:
        members = [row for row in task_rows if row["paper_dataset"] == paper]
        comparable_members = [row for row in members if as_bool(row["comparable"])]
        expected_count = 12 if paper == "ASD" else 1
        complete = len(members) == expected_count and len(comparable_members) == expected_count
        paper_rows.append(
            {
                "paper_dataset": paper,
                "physical_task_count": len(members),
                "comparable_task_count": len(comparable_members),
                "complete_for_overall": complete,
                "catch_auc_pr": _mean(comparable_members, "catch_auc_pr"),
                "catch_auc_roc": _mean(comparable_members, "catch_auc_roc"),
                "typefusion_auc_pr": _mean(comparable_members, "typefusion_auc_pr"),
                "typefusion_auc_roc": _mean(comparable_members, "typefusion_auc_roc"),
                "delta_auc_pr": _mean(comparable_members, "delta_auc_pr"),
                "delta_auc_roc": _mean(comparable_members, "delta_auc_roc"),
                "available_task_catch_auc_pr": _mean(comparable_members, "catch_auc_pr"),
                "available_task_catch_auc_roc": _mean(comparable_members, "catch_auc_roc"),
                "available_task_typefusion_auc_pr": _mean(comparable_members, "typefusion_auc_pr"),
                "available_task_typefusion_auc_roc": _mean(comparable_members, "typefusion_auc_roc"),
                "incomplete_reason": "" if complete else f"{len(comparable_members)}/{expected_count}_comparable_tasks",
            }
        )
    return paper_rows


def _macro_values(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "catch_auc_pr": _mean(rows, "catch_auc_pr"),
        "catch_auc_roc": _mean(rows, "catch_auc_roc"),
        "typefusion_auc_pr": _mean(rows, "typefusion_auc_pr"),
        "typefusion_auc_roc": _mean(rows, "typefusion_auc_roc"),
    }


def macro_summary(
    paper_rows: Sequence[Mapping[str, Any]], task_rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    complete = [row for row in paper_rows if as_bool(row["complete_for_overall"])]
    exact_task_rows = [
        row
        for row in task_rows
        if as_bool(row["comparable"]) and row["comparison_config_status"] == "exact_shared_config"
    ]
    exact_paper_rows = paper_metric_rows(exact_task_rows)
    exact_complete = [row for row in exact_paper_rows if as_bool(row["complete_for_overall"])]
    return {
        "formal_overall": {
            "status": "complete" if len(complete) == 12 else "incomplete",
            "complete_paper_dataset_count": len(complete),
            "expected_paper_dataset_count": 12,
            **_macro_values(complete),
        },
        "available_paper_macro": {
            "status": "diagnostic_not_complete_12_dataset_overall",
            "paper_dataset_count": len(complete),
            **_macro_values(complete),
        },
        "including_architecture_compatibility_override": {
            "status": "complete" if len(complete) == 12 else "incomplete",
            "paper_dataset_count": len(complete),
            **_macro_values(complete),
        },
        "exact_shared_config_only": {
            "status": "complete" if len(exact_complete) == 12 else "incomplete",
            "paper_dataset_count": len(exact_complete),
            **_macro_values(exact_complete),
        },
    }


def _load_gecco_fair() -> Mapping[str, str] | None:
    path = RESULT_ROOT / "gecco_fair_baseline_registry.csv"
    if not path.is_file():
        return None
    rows = read_registry(path)
    return rows[0] if len(rows) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="audit and write incomplete outputs without training")
    parser.add_argument("--source-registry", type=Path, default=RESULT_ROOT / "typefusion_catch_source_registry.csv")
    parser.add_argument("--score-root", type=Path, default=SCORE_ROOT)
    args = parser.parse_args()
    source_rows = read_registry(args.source_registry)
    if len(source_rows) != 23:
        raise ValueError(f"source registry must contain exactly 23 tasks, found {len(source_rows)}")
    gecco_fair = _load_gecco_fair()
    task_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for source in source_rows:
        selected, selection = select_latest_valid_run(source["task"], source, args.score_root)
        task_rows.append(task_metric_row(source, selected, gecco_fair))
        audit_rows.extend(selection)
    paper_rows = paper_metric_rows(task_rows)
    summary = {
        "task_count": len(task_rows),
        "paper_dataset_count": len(paper_rows),
        "selected_typefusion_run_count": sum(as_bool(row["selected"]) for row in audit_rows),
        "missing_valid_typefusion_result_count": sum(
            row["comparison_config_status"] == "missing_valid_typefusion_result" for row in task_rows
        ),
        "macro": macro_summary(paper_rows, task_rows),
    }
    write_registry(RESULT_ROOT / "typefusion_run_selection_audit.csv", audit_rows)
    write_registry(RESULT_ROOT / "typefusion_task_metrics.csv", task_rows)
    write_registry(RESULT_ROOT / "typefusion_paper_metrics.csv", paper_rows)
    write_registry(RESULT_ROOT / "typefusion_comparison.csv", task_rows)
    (RESULT_ROOT / "typefusion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"TypeFusion run selection: {summary['selected_typefusion_run_count']}/23 selected; "
        f"formal overall: {summary['macro']['formal_overall']['status']}"
    )


if __name__ == "__main__":
    main()
