#!/usr/bin/env python3
"""Hash CATCH source trees at every archive commit recorded in the registry.

The script reads historical content through `git show`; CATCH-master remains
read-only.  Differences include every tracked source file under the CATCH
directory, including comments and package files, so this audit never assumes a
code change is operationally irrelevant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from typefusion_catch_result_utils import RESULT_ROOT, read_registry, sha256_file, write_registry


ROOT = Path(__file__).resolve().parents[1]
GIT_ROOT = ROOT.parent
HISTORICAL_PATHS = (
    "CATCH-master/ts_benchmark/baselines/catch",
    "ts_benchmark/baselines/catch",
)
CURRENT_PATHS = (
    ROOT.parent / "CATCH-master/ts_benchmark/baselines/catch",
    ROOT.parent / "ts_benchmark/baselines/catch",
)


def _run_git(args: List[str]) -> bytes:
    return subprocess.check_output(["git", "-C", str(GIT_ROOT), *args], stderr=subprocess.PIPE)


def _source_name(name: str) -> bool:
    parts = Path(name).parts
    return not (
        "__pycache__" in parts
        or name.endswith(".pyc")
        or "/result/" in f"/{name}"
        or "/log/" in f"/{name}"
    )


def aggregate_hash(files: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, file_hash in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_files_from_commit(commit: str) -> tuple[str, Dict[str, str]]:
    """Return the detected historical directory and every retained file hash."""

    for candidate in HISTORICAL_PATHS:
        try:
            names = _run_git(["ls-tree", "-r", "--name-only", commit, "--", candidate]).decode().splitlines()
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"commit_unreadable:{commit}") from exc
        names = [name for name in names if name.startswith(candidate + "/") and _source_name(name)]
        if not names:
            continue
        files: Dict[str, str] = {}
        for name in sorted(names):
            content = _run_git(["show", f"{commit}:{name}"])
            relative_name = name[len(candidate) + 1 :]
            files[relative_name] = hashlib.sha256(content).hexdigest()
        required = {"CATCH.py", "models/CATCH_model.py", "__init__.py"}
        if not required.issubset(files):
            raise ValueError("historical_catch_required_sources_missing")
        if not any(name.startswith("layers/") for name in files):
            raise ValueError("historical_catch_layers_missing")
        if not any(name.startswith("utils/") for name in files):
            raise ValueError("historical_catch_utils_missing")
        return candidate, files
    raise ValueError("historical_catch_path_not_found")


def source_files_from_current() -> tuple[Path, Dict[str, str]]:
    for candidate in CURRENT_PATHS:
        if not candidate.is_dir():
            continue
        files = {
            str(path.relative_to(candidate)): sha256_file(path)
            for path in sorted(candidate.rglob("*"))
            if path.is_file() and _source_name(str(path.relative_to(candidate)))
        }
        required = {"CATCH.py", "models/CATCH_model.py", "__init__.py"}
        if required.issubset(files) and any(name.startswith("layers/") for name in files) and any(
            name.startswith("utils/") for name in files
        ):
            return candidate, files
    raise ValueError("current_catch_path_not_found_or_incomplete")


def changed_files(left: Mapping[str, str], right: Mapping[str, str]) -> List[str]:
    return [name for name in sorted(set(left).union(right)) if left.get(name) != right.get(name)]


def write_baseline_reference(rows: Iterable[Mapping[str, Any]], current_commit: str) -> None:
    """Render the small human-readable reference from audited registry fields."""

    lines = [
        "# TypeFusion-CATCH CATCH Baseline Reference",
        "",
        f"CATCH-master current commit: `{current_commit}`. The source archives and CATCH-master are read-only.",
        "A leaderboard `test_report` is used only to verify run status and its selected metric. Complete",
        "AUC-PR, AUC-ROC, R-AUC, VUS, fit-time, and inference-time values are read from the matching",
        "`CATCH.*.csv.tar.gz` archive. No detect_label metric is used here.",
        "",
        "GECCO remains excluded until a paired `seq_len=192, patch_size=16, patch_stride=8` CATCH baseline",
        "is registered. ASD_dataset_1 is an explicitly preregistered architecture compatibility override, not",
        "an exact shared configuration: CATCH source uses `cf_dim=4, n_heads=16`, while TypeFusion",
        "MultiheadAttention requires num_heads not exceeding and dividing the embedding dimension; the",
        "preregistered compatibility mapping uses `n_heads=4`.",
        "",
        "| Task | Archive | SHA-256 | AUC-PR | AUC-ROC | R-AUC-PR | R-AUC-ROC | VUS-PR | VUS-ROC | Fit time | Inference time | Code group | Historical code comparable | Comparison status | Directly comparable | Exclusion reason |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if str(row.get("baseline_requires_fair_rerun", "")).lower() == "true":
            status = "fairness_rerun_required"
        elif str(row.get("report_archive_metric_conflict", "")).lower() == "true":
            status = "source_metric_conflict"
        elif str(row.get("historical_code_comparable", "")).lower() != "true":
            status = "historical_code_mismatch"
        elif row["task"] == "ASD_dataset_1":
            status = "architecture_compatibility_override"
        else:
            status = "exact_shared_config"
        comparable = status in {"exact_shared_config", "architecture_compatibility_override"} and str(
            row.get("baseline_archive_valid", "")
        ).lower() == "true"
        exclusion = "" if comparable else status
        lines.append(
            "| {task} | `{archive}` | `{sha}` | {pr} | {roc} | {rapr} | {raroc} | {vuspr} | {vusroc} | {fit} | {inference} | {group} | {code} | {status} | {direct} | {exclusion} |".format(
                task=row["task"],
                archive=row.get("baseline_archive_path", ""),
                sha=row.get("baseline_archive_sha256", ""),
                pr=row.get("baseline_auc_pr", "N/A"),
                roc=row.get("baseline_auc_roc", "N/A"),
                rapr=row.get("baseline_r_auc_pr", "N/A"),
                raroc=row.get("baseline_r_auc_roc", "N/A"),
                vuspr=row.get("baseline_vus_pr", "N/A"),
                vusroc=row.get("baseline_vus_roc", "N/A"),
                fit=row.get("baseline_fit_time", "N/A"),
                inference=row.get("baseline_inference_time", "N/A"),
                group=row.get("historical_code_group", ""),
                code=row.get("historical_code_comparable", ""),
                status=status,
                direct="yes" if comparable else "no",
                exclusion=exclusion,
            )
        )
    (ROOT / "TYPEFUSION_CATCH_BASELINE_REFERENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=RESULT_ROOT / "typefusion_catch_source_registry.csv"
    )
    parser.add_argument(
        "--audit-output", type=Path, default=RESULT_ROOT / "catch_historical_code_audit.csv"
    )
    args = parser.parse_args()

    rows = read_registry(args.registry)
    commits = sorted({row["source_catch_master_commit"] for row in rows if row["source_catch_master_commit"]})
    if not commits:
        raise ValueError("no archive commits in source registry")
    current_path, current_files = source_files_from_current()
    current_aggregate = aggregate_hash(current_files)
    audits: List[Dict[str, Any]] = []
    by_commit: Dict[str, Dict[str, Any]] = {}
    for commit in commits:
        audit: Dict[str, Any] = {
            "archive_commit": commit,
            "resolved_catch_path": "",
            "source_file_count": 0,
            "aggregate_sha256": "",
            "source_file_sha256_json": "{}",
            "current_catch_path": str(current_path),
            "current_aggregate_sha256": current_aggregate,
            "matches_current_catch": False,
            "changed_file_count": 0,
            "changed_files": "",
            "audit_status": "failed",
            "audit_reason": "",
        }
        try:
            path, files = source_files_from_commit(commit)
            differences = changed_files(files, current_files)
            audit.update(
                {
                    "resolved_catch_path": path,
                    "source_file_count": len(files),
                    "aggregate_sha256": aggregate_hash(files),
                    "source_file_sha256_json": json.dumps(files, sort_keys=True, separators=(",", ":")),
                    "matches_current_catch": aggregate_hash(files) == current_aggregate,
                    "changed_file_count": len(differences),
                    "changed_files": ";".join(differences),
                    "audit_status": "ok",
                    "audit_reason": "",
                }
            )
        except Exception as exc:
            audit["audit_reason"] = str(exc)
        audits.append(audit)
        by_commit[commit] = audit
    for row in rows:
        audit = by_commit[row["source_catch_master_commit"]]
        if audit["audit_status"] == "ok":
            row["historical_code_group"] = "catch_code_" + str(audit["aggregate_sha256"])[:12]
            row["historical_code_comparable"] = bool(audit["matches_current_catch"])
            row["historical_code_difference"] = audit["changed_files"] or ""
        else:
            row["historical_code_group"] = "unavailable_" + row["source_catch_master_commit"][:12]
            row["historical_code_comparable"] = False
            row["historical_code_difference"] = "audit_unavailable:" + str(audit["audit_reason"])
    write_registry(args.registry, rows)
    write_registry(args.audit_output, audits)
    write_baseline_reference(rows, subprocess.check_output(["git", "-C", str(GIT_ROOT), "rev-parse", "HEAD"], text=True).strip())
    print(
        "historical CATCH audit: "
        f"{len(audits)} commits, {len(set(row['aggregate_sha256'] for row in audits if row['aggregate_sha256']))} code groups, "
        f"{sum(bool(row['matches_current_catch']) for row in audits)}/{len(audits)} match current CATCH, "
        f"{sum(row['audit_status'] != 'ok' for row in audits)} unavailable"
    )


if __name__ == "__main__":
    main()
