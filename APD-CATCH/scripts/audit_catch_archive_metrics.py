#!/usr/bin/env python3
"""Audit complete CATCH score archives and enrich the baseline source registry.

This is a read-only audit of CATCH-master.  It never reruns CATCH and never
uses detect_label output.  Leaderboard test reports are checked against the
complete `CATCH.*.csv.tar.gz` record, whose AUC-PR/AUC-ROC values are the
registered baseline metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from typefusion_catch_result_utils import (
    RESULT_ROOT,
    failure_marker,
    parse_leaderboard_report,
    parse_metric_archive,
    read_registry,
    sha256_file,
    write_registry,
)


def _find_archive(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("CATCH.*.csv.tar.gz"))
    if len(candidates) != 1:
        raise ValueError(f"expected_one_catch_archive:{len(candidates)}")
    return candidates[0]


def _bool(value: bool) -> bool:
    return bool(value)


def audit_row(row: Dict[str, str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return a registry update and one standalone, append-only audit row."""

    task = row["task"]
    report_path = Path(row["baseline_report"])
    run_dir = report_path.parent
    update: Dict[str, Any] = {
        "baseline_archive_path": "",
        "baseline_archive_sha256": "",
        "baseline_archive_valid": False,
        "baseline_archive_validation_reason": "",
        "baseline_auc_roc": "N/A",
        "baseline_auc_pr": "N/A",
        "baseline_r_auc_roc": "N/A",
        "baseline_r_auc_pr": "N/A",
        "baseline_vus_roc": "N/A",
        "baseline_vus_pr": "N/A",
        "baseline_fit_time": "N/A",
        "baseline_inference_time": "N/A",
        "baseline_strategy_name": "",
        "baseline_seed": "",
        "baseline_seq_len": row.get("seq_len", ""),
        "baseline_batch_size": row.get("batch_size", ""),
        "baseline_config_match": False,
        "report_archive_metric_conflict": False,
        "baseline_report_auc_roc": "N/A",
        "baseline_archive_auc_roc": "N/A",
    }
    reason = "ok"
    try:
        if not report_path.is_file() or report_path.stat().st_size == 0:
            raise ValueError("test_report_missing_or_empty")
        archive_path = _find_archive(run_dir)
        update["baseline_archive_path"] = str(archive_path)
        archive = parse_metric_archive(archive_path, row["data_name"], "CATCH")
        update["baseline_archive_sha256"] = archive.archive_sha256
        if archive_path.parent != run_dir:
            raise ValueError("archive_and_test_report_run_mismatch")
        marker = failure_marker(
            [
                *run_dir.glob("*.log"),
                run_dir / "metadata.txt",
                run_dir / "command.sh",
                run_dir / "official_CATCH.sh",
            ]
        )
        if marker:
            raise ValueError(marker)
        official = Path(row["official_source_script"])
        if not official.is_file():
            raise ValueError("official_script_missing")
        if sha256_file(official) != row["source_script_sha256"]:
            raise ValueError("official_script_sha256_mismatch")
        report = parse_leaderboard_report(report_path)
        if report["model_name"] != "CATCH":
            raise ValueError("test_report_model_name_mismatch")
        if report["strategy"].get("strategy_name") != "unfixed_detect_score":
            raise ValueError("test_report_strategy_mismatch")
        if int(report["strategy"].get("seed")) != 2021:
            raise ValueError("test_report_seed_mismatch")
        expected_params = json.loads(row["catch_model_hyper_params_json"])
        config_match = archive.model_params == expected_params and report["model_params"] == expected_params
        update["baseline_config_match"] = _bool(config_match)
        if not config_match:
            raise ValueError("archive_or_report_config_mismatch")
        for target, source in (
            ("baseline_auc_roc", "auc_roc"),
            ("baseline_auc_pr", "auc_pr"),
            ("baseline_r_auc_roc", "r_auc_roc"),
            ("baseline_r_auc_pr", "r_auc_pr"),
            ("baseline_vus_roc", "vus_roc"),
            ("baseline_vus_pr", "vus_pr"),
            ("baseline_fit_time", "fit_time"),
            ("baseline_inference_time", "inference_time"),
        ):
            update[target] = archive.metrics[source]
        update["baseline_strategy_name"] = archive.strategy["strategy_name"]
        update["baseline_seed"] = archive.strategy["seed"]
        update["baseline_report_auc_roc"] = report["metric_value"] if report["metric_name"] == "auc_roc" else "N/A"
        update["baseline_archive_auc_roc"] = archive.metrics["auc_roc"]
        conflict = (
            report["metric_name"] == "auc_roc"
            and not math.isclose(
                float(report["metric_value"]), float(archive.metrics["auc_roc"]), rel_tol=1e-12, abs_tol=1e-12
            )
        )
        update["report_archive_metric_conflict"] = _bool(conflict)
        update["baseline_archive_valid"] = True
        reason = "report_archive_auc_roc_conflict" if conflict else "ok"
    except Exception as exc:  # retain an auditable reason for every fixed task
        reason = str(exc)
    update["baseline_archive_validation_reason"] = reason
    audit = {
        "task": task,
        "successful_run_directory": str(run_dir),
        "test_report": str(report_path),
        "official_script": row["official_source_script"],
        "archive_commit": row["source_catch_master_commit"],
        **update,
    }
    return update, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=RESULT_ROOT / "typefusion_catch_source_registry.csv"
    )
    parser.add_argument(
        "--audit-output", type=Path, default=RESULT_ROOT / "catch_archive_metric_audit.csv"
    )
    args = parser.parse_args()

    rows = read_registry(args.registry)
    if len(rows) != 23:
        raise ValueError(f"expected 23 source-registry tasks, found {len(rows)}")
    audited: List[Dict[str, Any]] = []
    for row in rows:
        update, audit = audit_row(row)
        row.update(update)
        audited.append(audit)
    write_registry(args.registry, rows)
    write_registry(args.audit_output, audited)
    valid = sum(str(row["baseline_archive_valid"]).lower() == "true" for row in audited)
    conflicts = sum(str(row["report_archive_metric_conflict"]).lower() == "true" for row in audited)
    print(f"CATCH archive audit: {valid}/{len(audited)} valid; report/archive AUC-ROC conflicts: {conflicts}")


if __name__ == "__main__":
    main()
