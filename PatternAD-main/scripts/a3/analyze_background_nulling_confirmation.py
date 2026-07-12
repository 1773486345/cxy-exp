#!/usr/bin/env python3
"""Verify that every frozen A3-N1 confirmation pair is a complete CUDA pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a3.n1_confirmation import (
    DEFAULT_CONFIRMATION_CONFIG,
    canonical_hash,
    load_confirmation_plan,
    prepared_confirmation_pair,
)


def _load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an A3-N1 summary object in {path}.")
    return value


def _gate_snapshot(summary: Mapping[str, Any]) -> Dict[str, Any]:
    gates = summary.get("gates", {})
    return {
        "primary_misrouted_null_route": gates.get("primary_misrouted_null_route"),
        "secondary_partial_propagation_null_route": gates.get("secondary_partial_propagation_null_route"),
        "untriggered_null_route": gates.get("untriggered_null_route"),
        "normal_transition_controls": gates.get("normal_transition_controls"),
        "independent_background": gates.get("independent_background"),
    }


def analyze_confirmation(
    summaries: List[Mapping[str, Any]], config_path: Path = DEFAULT_CONFIRMATION_CONFIG
) -> Dict[str, Any]:
    plan, _, background, preflight, _ = load_confirmation_plan(config_path)
    if len(summaries) != len(plan["pairs"]):
        raise ValueError("A3-N1 confirmation summaries do not cover every frozen pair.")
    violations: List[str] = []
    runs: List[Dict[str, Any]] = []
    for pair_index, summary in enumerate(summaries):
        _, contract, _, _, experiment, output_dir = prepared_confirmation_pair(pair_index, config_path)
        pair = plan["pairs"][pair_index]
        pair_violations: List[str] = []
        expected_hashes = {
            "contract_config_hash": canonical_hash(contract),
            "background_protocol_hash": canonical_hash(background),
            "preflight_config_hash": canonical_hash(preflight),
        }
        if str(summary.get("experiment_id", "")) != "a3_n1_background_nulling_route_graph_development_v1":
            pair_violations.append("wrong experiment ID")
        if int(summary.get("seed", -1)) != int(pair["model_seed"]):
            pair_violations.append("model seed mismatch")
        if str(summary.get("device", "")) != "cuda":
            pair_violations.append("result was not produced on CUDA")
        if summary.get("experiment_config") != experiment:
            pair_violations.append("experiment configuration mismatch")
        for name, expected in expected_hashes.items():
            if summary.get(name) != expected:
                pair_violations.append(f"{name} mismatch")
        if not bool(summary.get("all_gates_passed")):
            pair_violations.append("frozen gates did not all pass")
        if pair_violations:
            violations.extend(
                f"pair {pair_index} (contract={pair['contract_seed']}, model={pair['model_seed']}): {item}"
                for item in pair_violations
            )
        runs.append(
            {
                "pair_index": pair_index,
                "contract_seed": int(pair["contract_seed"]),
                "model_seed": int(pair["model_seed"]),
                "summary": str(output_dir / "summary.json"),
                "passed": not pair_violations,
                "violations": pair_violations,
                "gates": _gate_snapshot(summary),
            }
        )
    complete_passes = sum(int(run["passed"]) for run in runs)
    if complete_passes != int(plan["required_complete_passes"]):
        violations.append(
            f"confirmation has {complete_passes}/{plan['required_complete_passes']} complete frozen passes"
        )
    return {
        "confirmation_id": plan["confirmation_id"],
        "required_complete_passes": int(plan["required_complete_passes"]),
        "complete_passes": complete_passes,
        "passed": not violations,
        "violations": violations,
        "runs": runs,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-config", type=Path, default=DEFAULT_CONFIRMATION_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    plan, _, _, _, _ = load_confirmation_plan(arguments.confirmation_config)
    summaries = [
        _load_summary(prepared_confirmation_pair(pair_index, arguments.confirmation_config)[-1] / "summary.json")
        for pair_index in range(len(plan["pairs"]))
    ]
    result = analyze_confirmation(summaries, arguments.confirmation_config)
    if arguments.output.exists():
        raise FileExistsError(f"Refusing to overwrite {arguments.output}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
