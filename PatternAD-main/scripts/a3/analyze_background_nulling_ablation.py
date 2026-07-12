#!/usr/bin/env python3
"""Decide whether frozen A3-N1 development depends on its event-pre trigger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping


def _load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an A3-N1 summary object in {path}.")
    return value


def analyze_ablation(
    development: Mapping[str, Any], control: Mapping[str, Any]
) -> Dict[str, Any]:
    """Require the past-free control to fail N1's frozen primary relation gate."""
    violations = []
    if str(development.get("experiment_id")) != "a3_n1_background_nulling_route_graph_development_v1":
        violations.append("development summary has the wrong A3-N1 experiment ID")
    if str(control.get("experiment_id")) != "a3_n1_background_nulling_route_graph_past_free_control_v1":
        violations.append("control summary has the wrong A3-N1 experiment ID")
    for field in ("contract_config_hash", "background_protocol_hash", "preflight_config_hash"):
        if development.get(field) != control.get(field):
            violations.append(f"development/control {field} mismatch")
    if development.get("experiment_config", {}).get("model", {}).get("condition_on_event_pre") is not True:
        violations.append("development did not condition on event-pre state")
    if control.get("experiment_config", {}).get("model", {}).get("condition_on_event_pre") is not False:
        violations.append("control did not remove event-pre state")
    if not bool(development.get("all_gates_passed")):
        violations.append("development did not pass its frozen gates")
    primary = control.get("gates", {}).get("primary_misrouted_null_route")
    if not isinstance(primary, dict):
        violations.append("control lacks the primary null-route gate")
        primary = {}
    control_failed_primary = not bool(primary.get("passed", False))
    if not control_failed_primary:
        violations.append("past-free control unexpectedly passed the primary relation gate")
    return {
        "passed": not violations,
        "violations": violations,
        "metrics": {
            "development_primary": development.get("gates", {}).get("primary_misrouted_null_route"),
            "control_primary": primary,
            "control_failed_primary": control_failed_primary,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = analyze_ablation(
        _load_summary(arguments.development_summary), _load_summary(arguments.control_summary)
    )
    if arguments.output.exists():
        raise FileExistsError(f"Refusing to overwrite {arguments.output}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
