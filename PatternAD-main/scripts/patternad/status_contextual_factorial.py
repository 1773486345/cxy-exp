#!/usr/bin/env python3
"""Report live progress for a PatternAD contextual factorial run."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


DEFAULT_INPUT = Path(
    "result/patternad_synthetic/p1_contextual_calibrated_v2_holdout/development"
)
TERMINAL_STATUSES = {"completed", "failed", "interrupted"}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _tail_nonempty(path: Path, line_count: int) -> List[str]:
    if line_count <= 0 or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = [line.rstrip() for line in handle if line.strip()]
    return lines[-line_count:]


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def collect_status(
    input_root: Path, tail_lines: int = 3, now: Optional[datetime] = None
) -> Dict[str, Any]:
    input_root = input_root.resolve()
    plan_path = input_root / "run_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing run plan: {plan_path}")
    plan = _load_json(plan_path)
    identities = plan.get("expected_identities")
    if not isinstance(identities, list) or not identities:
        raise ValueError("run_plan.json expected_identities must be a non-empty list.")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    rows: List[Dict[str, Any]] = []
    counts: Counter = Counter()
    durations_by_variant: Dict[str, List[float]] = defaultdict(list)
    for index, identity in enumerate(identities, start=1):
        if not isinstance(identity, dict):
            raise ValueError(f"Invalid expected identity at index {index - 1}.")
        relative_dir = identity.get("result_dir")
        if not isinstance(relative_dir, str) or not relative_dir:
            raise ValueError(f"Identity {index} has no result_dir.")
        result_dir = input_root / relative_dir
        metadata_path = result_dir / "identity_metadata.json"
        metadata: Mapping[str, Any] = {}
        status = "missing"
        error = None
        if metadata_path.is_file():
            try:
                metadata = _load_json(metadata_path)
                status = str(metadata.get("status", "invalid"))
            except (OSError, ValueError, json.JSONDecodeError) as exception:
                status = "invalid"
                error = f"{type(exception).__name__}: {exception}"
        counts[status] += 1

        started_at = _parse_timestamp(metadata.get("started_at"))
        completed_at = _parse_timestamp(metadata.get("completed_at"))
        wall_seconds = metadata.get("runner_wall_seconds")
        if isinstance(wall_seconds, (int, float)) and wall_seconds >= 0:
            duration = float(wall_seconds)
        elif started_at is not None:
            end = completed_at or now
            duration = max(0.0, (end - started_at).total_seconds())
        else:
            duration = None
        variant = str(identity.get("variant", "?"))
        if status == "completed" and duration is not None:
            durations_by_variant[variant].append(duration)

        log_path = result_dir / "run.log"
        log_age = None
        if log_path.is_file():
            log_age = max(0.0, now.timestamp() - log_path.stat().st_mtime)
        rows.append(
            {
                "index": index,
                "variant": variant,
                "generator_seed": identity.get("generator_seed", "?"),
                "model_seed": identity.get("model_seed", "?"),
                "status": status,
                "duration_seconds": duration,
                "log_age_seconds": log_age,
                "log_tail": _tail_nonempty(log_path, tail_lines),
                "error": metadata.get("error", error),
                "result_dir": relative_dir,
            }
        )

    completed = counts.get("completed", 0)
    total = len(rows)
    remaining_rows = [row for row in rows if row["status"] != "completed"]
    eta_seconds: Optional[float] = 0.0
    for row in remaining_rows:
        durations = durations_by_variant.get(row["variant"], [])
        if not durations:
            eta_seconds = None
            break
        eta_seconds += statistics.median(durations)

    return {
        "input_root": str(input_root),
        "run_name": plan.get("run_name", input_root.parent.name),
        "phase": plan.get("phase", "unknown"),
        "plan_hash": plan.get("plan_hash", "unknown"),
        "total": total,
        "completed": completed,
        "progress_fraction": completed / total,
        "counts": dict(counts),
        "rows": rows,
        "running": [row for row in rows if row["status"] == "running"],
        "problems": [
            row
            for row in rows
            if row["status"] in {"failed", "interrupted", "invalid"}
        ],
        "eta_seconds": eta_seconds,
        "observed_variant_median_seconds": {
            variant: statistics.median(durations)
            for variant, durations in sorted(durations_by_variant.items())
        },
        "observed_at": now.astimezone().isoformat(timespec="seconds"),
        "finished": completed == total,
    }


def render_status(snapshot: Mapping[str, Any]) -> str:
    total = int(snapshot["total"])
    completed = int(snapshot["completed"])
    percent = 100.0 * float(snapshot["progress_fraction"])
    counts = snapshot["counts"]
    lines = [
        f"[{snapshot['observed_at']}] {snapshot['run_name']} / {snapshot['phase']}",
        (
            f"Progress: {completed}/{total} ({percent:.1f}%) | "
            f"running={counts.get('running', 0)} failed={counts.get('failed', 0)} "
            f"interrupted={counts.get('interrupted', 0)} "
            f"missing={counts.get('missing', 0)}"
        ),
    ]
    if snapshot["eta_seconds"] is None:
        observed = snapshot["observed_variant_median_seconds"]
        known = ", ".join(
            f"{variant}~{_format_duration(seconds)}"
            for variant, seconds in observed.items()
        )
        lines.append(
            "ETA: unavailable until every remaining variant has a completed sample"
            + (f" (observed {known})" if known else "")
        )
    else:
        lines.append(f"Rough ETA from per-variant medians: {_format_duration(snapshot['eta_seconds'])}")

    for row in snapshot["running"]:
        lines.append(
            "Current [{}/{}]: {} generator={} model={} elapsed={} log_age={}".format(
                row["index"],
                total,
                row["variant"],
                row["generator_seed"],
                row["model_seed"],
                _format_duration(row["duration_seconds"]),
                _format_duration(row["log_age_seconds"]),
            )
        )
        for log_line in row["log_tail"]:
            lines.append(f"  log> {log_line}")
    if not snapshot["running"] and not snapshot["finished"]:
        lines.append("Current: no identity is marked running; the runner may be stopped.")
    for row in snapshot["problems"]:
        lines.append(
            "Problem [{}]: {} generator={} model={} status={} error={}".format(
                row["index"],
                row["variant"],
                row["generator_seed"],
                row["model_seed"],
                row["status"],
                row["error"] or "none",
            )
        )
    return "\n".join(lines)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--watch",
        type=float,
        help="Refresh interval in seconds. Omit for a one-shot report.",
    )
    parser.add_argument("--tail-lines", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.watch is not None and args.watch <= 0:
        raise ValueError("--watch must be positive.")
    if args.tail_lines < 0:
        raise ValueError("--tail-lines cannot be negative.")
    try:
        while True:
            snapshot = collect_status(args.input, args.tail_lines)
            if args.watch is not None and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(render_status(snapshot), flush=True)
            if args.watch is None or snapshot["finished"]:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nStatus monitor stopped; the experiment runner was not touched.")
        return 130


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
