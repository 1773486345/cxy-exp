#!/usr/bin/env python3
"""Verify A2-M2's frozen event-pre ablation against matched result artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping


def _load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an A2 result object in {path}.")
    return value


def _experiment_except_condition(summary: Mapping[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(dict(summary["experiment_config"]))
    config.pop("experiment_id", None)
    model = dict(config["model"])
    model.pop("condition_on_event_pre", None)
    config["model"] = model
    return config


def analyze_ablation(
    conditioned: Mapping[str, Any], unconditional: Mapping[str, Any]
) -> Dict[str, Any]:
    """Apply the M2 ablation gate frozen before observing the control result."""
    for label, summary, expected_condition in (
        ("conditioned", conditioned, True),
        ("unconditional", unconditional, False),
    ):
        if summary.get("experiment_id", "").find("contrastive_energy") == -1:
            raise ValueError(f"{label} result is not an A2-M2 contrastive-energy run.")
        if bool(summary.get("condition_on_event_pre")) is not expected_condition:
            raise ValueError(f"{label} result has the wrong event-pre condition setting.")
        if summary.get("raw_score_name") != "event_pre_future_contrastive_energy":
            raise ValueError(f"{label} result has the wrong A2-M2 score family.")
    if conditioned.get("contract_config_hash") != unconditional.get("contract_config_hash"):
        raise ValueError("A2-M2 ablation requires the same generated contract configuration.")
    if _experiment_except_condition(conditioned) != _experiment_except_condition(unconditional):
        raise ValueError(
            "A2-M2 ablation configurations may differ only in experiment_id and "
            "model.condition_on_event_pre."
        )
    conditioned_primary = dict(conditioned["gates"]["primary_ordering"])
    unconditional_primary = dict(unconditional["gates"]["primary_ordering"])
    conditioned_all = bool(conditioned["all_gates_passed"])
    unconditional_primary_failed = not bool(unconditional_primary["passed"])
    return {
        "ablation_id": "a2_m2_event_pre_control_v1",
        "contract_config_hash": str(conditioned["contract_config_hash"]),
        "conditioned": {
            "all_gates_passed": conditioned_all,
            "primary_positive_pairs": int(conditioned_primary["positive_pairs"]),
            "primary_pair_count": int(conditioned_primary["pair_count"]),
            "primary_median_tail_margin": float(conditioned_primary["median_tail_margin"]),
        },
        "unconditional": {
            "all_gates_passed": bool(unconditional["all_gates_passed"]),
            "primary_gate_passed": bool(unconditional_primary["passed"]),
            "primary_positive_pairs": int(unconditional_primary["positive_pairs"]),
            "primary_pair_count": int(unconditional_primary["pair_count"]),
            "primary_median_tail_margin": float(unconditional_primary["median_tail_margin"]),
        },
        "gate": {
            "conditioned_all_gates_passed": conditioned_all,
            "unconditional_primary_gate_failed": unconditional_primary_failed,
            "passed": conditioned_all and unconditional_primary_failed,
        },
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioned-summary", type=Path, required=True)
    parser.add_argument("--unconditional-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing ablation result: {arguments.output}")
    result = analyze_ablation(
        _load_summary(arguments.conditioned_summary), _load_summary(arguments.unconditional_summary)
    )
    _write_json_atomic(arguments.output, result)
    print(
        f"A2-M2 event-pre ablation: passed={result['gate']['passed']} "
        f"output={arguments.output}"
    )
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
