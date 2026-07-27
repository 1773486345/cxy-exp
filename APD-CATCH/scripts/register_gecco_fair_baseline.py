#!/usr/bin/env python3
"""Register a future GECCO seq_len=192 CATCH fairness baseline, read-only.

The prepared CATCH_GECCO_FAIR.sh is never executed here.  Until a manually
completed, fully validated result exists, the registry contains one explicit
unavailable row rather than reusing the historical seq_len=96 result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from typefusion_catch_result_utils import (
    RESULT_ROOT,
    failure_marker,
    parse_leaderboard_report,
    parse_metric_archive,
    read_registry,
    write_registry,
)


ROOT = Path(__file__).resolve().parents[1]
CATCH_MASTER_ROOT = ROOT.parent / "CATCH-master"
FAIR_SCRIPT = ROOT / "scripts/multivariate_detection/detect_score/GECCO_script/CATCH_GECCO_FAIR.sh"


def fair_script_contract(script_path: Path = FAIR_SCRIPT) -> Dict[str, Any]:
    text = script_path.read_text(encoding="utf-8")
    params_match = re.search(r"MODEL_HYPER_PARAMS='([^']+)'", text)
    save_match = re.search(r"CATCH_SAVE_PATH:-([^}]+)", text)
    batch_match = re.search(r'BATCH_SIZE="\$\{BATCH_SIZE:-([0-9]+)\}"', text)
    if not params_match or not save_match or not batch_match:
        raise ValueError("fair_script_contract_parse_failed")
    params = json.loads(params_match.group(1).replace("__BATCH_SIZE__", batch_match.group(1)))
    if (params.get("seq_len"), params.get("patch_size"), params.get("patch_stride")) != (192, 16, 8):
        raise ValueError("fair_script_required_geometry_mismatch")
    return {
        "params": params,
        "batch_size": int(batch_match.group(1)),
        "save_path_prefix": save_match.group(1).split("/run-", 1)[0],
    }


def candidate_run_dirs(contract: Dict[str, Any]) -> List[Path]:
    relative = str(contract["save_path_prefix"])
    roots = (ROOT / "result", CATCH_MASTER_ROOT / "result")
    candidates: List[Path] = []
    for result_root in roots:
        root = result_root / relative
        if root.is_dir():
            candidates.extend(path for path in root.glob("run-*") if path.is_dir())
    return sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)


def _single(path: Path, pattern: str, reason: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"{reason}:{len(matches)}")
    return matches[0]


def validate_candidate(run_dir: Path, contract: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one candidate. Raise ValueError with the audit rejection reason."""

    report = _single(run_dir, "test_report*.csv", "expected_one_test_report")
    archive = _single(run_dir, "CATCH.*.csv.tar.gz", "expected_one_catch_archive")
    marker = failure_marker([*run_dir.glob("*.log"), run_dir / "metadata.txt", run_dir / "command.sh"])
    if marker:
        raise ValueError(marker)
    parsed_archive = parse_metric_archive(archive, "GECCO.csv", "CATCH")
    parsed_report = parse_leaderboard_report(report)
    if parsed_report["model_name"] != "CATCH":
        raise ValueError("test_report_model_name_mismatch")
    if parsed_report["strategy"].get("strategy_name") != "unfixed_detect_score":
        raise ValueError("test_report_strategy_mismatch")
    if int(parsed_report["strategy"].get("seed")) != 2021:
        raise ValueError("test_report_seed_mismatch")
    if parsed_archive.model_params != contract["params"] or parsed_report["model_params"] != contract["params"]:
        raise ValueError("script_archive_report_config_mismatch")
    metadata = run_dir / "metadata.txt"
    if not metadata.is_file() or "catch_master_commit=" not in metadata.read_text(encoding="utf-8", errors="replace"):
        raise ValueError("source_code_version_not_recorded")
    return {
        "task": "GECCO",
        "fair_baseline_available": True,
        "fair_baseline_reason": "ok",
        "fair_run_directory": str(run_dir),
        "fair_test_report": str(report),
        "fair_archive_path": str(archive),
        "fair_archive_sha256": parsed_archive.archive_sha256,
        "fair_auc_roc": parsed_archive.metrics["auc_roc"],
        "fair_auc_pr": parsed_archive.metrics["auc_pr"],
        "fair_r_auc_roc": parsed_archive.metrics["r_auc_roc"],
        "fair_r_auc_pr": parsed_archive.metrics["r_auc_pr"],
        "fair_vus_roc": parsed_archive.metrics["vus_roc"],
        "fair_vus_pr": parsed_archive.metrics["vus_pr"],
        "fair_fit_time": parsed_archive.metrics["fit_time"],
        "fair_inference_time": parsed_archive.metrics["inference_time"],
        "fair_seed": parsed_archive.strategy["seed"],
        "fair_seq_len": parsed_archive.model_params["seq_len"],
        "fair_patch_size": parsed_archive.model_params["patch_size"],
        "fair_patch_stride": parsed_archive.model_params["patch_stride"],
        "fair_batch_size": parsed_archive.model_params["batch_size"],
        "fair_strategy_name": parsed_archive.strategy["strategy_name"],
        "fair_source_code_version": next(
            line.split("=", 1)[1].strip()
            for line in metadata.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.startswith("catch_master_commit=")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=RESULT_ROOT / "gecco_fair_baseline_registry.csv"
    )
    args = parser.parse_args()
    contract = fair_script_contract()
    attempts: List[str] = []
    selected: Dict[str, Any] | None = None
    for run_dir in candidate_run_dirs(contract):
        try:
            selected = validate_candidate(run_dir, contract)
            break
        except Exception as exc:
            attempts.append(f"{run_dir}:{exc}")
    if selected is None:
        selected = {
            "task": "GECCO",
            "fair_baseline_available": False,
            "fair_baseline_reason": "not_run_or_no_valid_result",
            "fair_run_directory": "",
            "fair_test_report": "",
            "fair_archive_path": "",
            "fair_archive_sha256": "",
            "fair_auc_roc": "N/A",
            "fair_auc_pr": "N/A",
            "fair_rejected_candidates": "|".join(attempts),
        }
    else:
        selected["fair_rejected_candidates"] = "|".join(attempts)
    write_registry(args.output, [selected])
    print(f"GECCO fair baseline available: {str(selected['fair_baseline_available']).lower()}")


if __name__ == "__main__":
    main()
