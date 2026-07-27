"""Read-only helpers for CATCH and TypeFusion-CATCH result archives."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result/typefusion_catch_main"
METRIC_COLUMNS = {
    "auc_roc": "auc_roc",
    "auc_pr": "auc_pr",
    "r_auc_roc": "R_AUC_ROC",
    "r_auc_pr": "R_AUC_PR",
    "vus_roc": "VUS_ROC",
    "vus_pr": "VUS_PR",
    "fit_time": "fit_time",
    "inference_time": "inference_time",
}
FAILURE_MARKERS = ("traceback", "cuda out of memory", "cuda oom", "interrupted")


@dataclass(frozen=True)
class ArchiveMetrics:
    archive_path: Path
    archive_sha256: str
    member_name: str
    model_name: str
    strategy: Dict[str, Any]
    model_params: Dict[str, Any]
    file_name: str
    metrics: Dict[str, float | str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def read_registry(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_registry(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty registry: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_or_na(value: Any) -> float | str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return numeric if math.isfinite(numeric) else "N/A"


def failure_marker(paths: Iterable[Path]) -> str:
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for marker in FAILURE_MARKERS:
            if marker in text:
                return f"failure_marker:{marker}:{path.name}"
    return ""


def _read_archive_frame(archive_path: Path) -> tuple[str, pd.DataFrame]:
    if not archive_path.is_file():
        raise ValueError("archive_missing")
    if archive_path.stat().st_size == 0:
        raise ValueError("archive_empty")
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.lower().endswith(".csv") and member.size > 0
            ]
            if not members:
                raise ValueError("archive_has_no_nonempty_csv")
            member = members[0]
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("archive_csv_unreadable")
            frame = pd.read_csv(io.BytesIO(source.read()))
    except tarfile.TarError as exc:
        raise ValueError(f"archive_tar_error:{type(exc).__name__}") from exc
    if frame.empty:
        raise ValueError("archive_csv_empty")
    return member.name, frame


def parse_metric_archive(
    archive_path: Path,
    expected_task_file: str,
    expected_model_name: str,
) -> ArchiveMetrics:
    """Parse an archive row and enforce the common score-protocol contract."""

    member_name, frame = _read_archive_frame(archive_path)
    if len(frame) != 1:
        raise ValueError(f"archive_expected_one_result_row:{len(frame)}")
    required = {"model_name", "strategy_args", "model_params", "file_name", "auc_roc", "auc_pr"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("archive_missing_columns:" + ",".join(sorted(missing)))
    row = frame.iloc[0]
    model_name = str(row["model_name"])
    if model_name != expected_model_name:
        raise ValueError(f"archive_model_name_mismatch:{model_name}")
    file_name = str(row["file_name"])
    if file_name != expected_task_file:
        raise ValueError(f"archive_task_file_mismatch:{file_name}")
    try:
        strategy = json.loads(str(row["strategy_args"]))
        model_params = json.loads(str(row["model_params"]))
    except json.JSONDecodeError as exc:
        raise ValueError("archive_json_parse_error") from exc
    if strategy.get("strategy_name") != "unfixed_detect_score":
        raise ValueError("archive_strategy_mismatch")
    if int(strategy.get("seed")) != 2021:
        raise ValueError("archive_seed_mismatch")
    failure_text = " ".join(str(value) for value in row.astype(str)).lower()
    for marker in FAILURE_MARKERS:
        if marker in failure_text:
            raise ValueError(f"archive_failure_marker:{marker}")
    metrics = {key: finite_or_na(row.get(column)) for key, column in METRIC_COLUMNS.items()}
    if metrics["auc_roc"] == "N/A" or metrics["auc_pr"] == "N/A":
        raise ValueError("archive_primary_metric_not_finite")
    return ArchiveMetrics(
        archive_path=archive_path,
        archive_sha256=sha256_file(archive_path),
        member_name=member_name,
        model_name=model_name,
        strategy=strategy,
        model_params=model_params,
        file_name=file_name,
        metrics=metrics,
    )


def parse_leaderboard_report(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("test_report_missing_or_empty")
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"test_report_expected_one_row:{len(frame)}")
    row = frame.iloc[0]
    metric_columns = [
        column for column in frame.columns if column not in {"strategy_args", "metric_name"}
    ]
    if len(metric_columns) != 1:
        raise ValueError("test_report_expected_one_metric_column")
    metric_column = metric_columns[0]
    model_name, params_text = metric_column.split(";", 1)
    try:
        params = json.loads(params_text)
        strategy = json.loads(str(row["strategy_args"]))
    except json.JSONDecodeError as exc:
        raise ValueError("test_report_json_parse_error") from exc
    value = finite_or_na(row[metric_column])
    if value == "N/A":
        raise ValueError("test_report_metric_not_finite")
    return {
        "model_name": model_name,
        "model_params": params,
        "strategy": strategy,
        "metric_name": str(row["metric_name"]),
        "metric_value": value,
    }


def parse_run_timestamp(run_path: Path) -> datetime:
    """Prefer the explicit UTC run name, then fall back to directory mtime."""

    match = re.match(r"run-(\d{8}T\d{6}Z)(?:-|$)", run_path.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(run_path.stat().st_mtime, tz=timezone.utc)
