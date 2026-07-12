#!/usr/bin/env python3
"""Audit A3-G3's fixed observable inputs before fitting its grammar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a3.audit_observable_graph_grammar import audit_graph_grammar_inputs
from scripts.a3.generate_trigger_response_contract import (
    DEFAULT_CONFIG as DEFAULT_CONTRACT_CONFIG,
    _load_json,
    generate_suite,
)

DEFAULT_EXPERIMENT_CONFIG = (
    REPO_ROOT / "config" / "a3" / "counterfactual_effect_graph_g3_v1.json"
)


def _validate_experiment(
    contract_config: Mapping[str, Any], experiment_config: Mapping[str, Any]
) -> List[str]:
    violations: List[str] = []
    if int(experiment_config.get("schema_version", 0)) != 1:
        violations.append("A3-G3 experiment config must use schema_version=1")
    if not str(experiment_config.get("experiment_id", "")).startswith(
        "a3_g3_counterfactual_effect_graph_"
    ):
        violations.append("A3-G3 experiment_id does not identify the counterfactual effect graph")
    required = {"trigger_extractor", "counterfactual", "effect_extractor", "model", "calibration"}
    missing = sorted(required.difference(experiment_config))
    if missing:
        violations.append(f"A3-G3 experiment config is missing {missing}")
        return violations
    if float(experiment_config["counterfactual"].get("ridge_penalty", 0.0)) <= 0.0:
        violations.append("A3-G3 ridge_penalty must be positive")
    effect_threshold = float(experiment_config["effect_extractor"].get("token_energy_threshold", 0.0))
    if effect_threshold <= 0.0:
        violations.append("A3-G3 effect token threshold must be positive")
    contract_threshold = float(contract_config["episodes"]["token_energy_threshold"])
    if abs(effect_threshold - contract_threshold) > 1e-12:
        violations.append("A3-G3 effect token threshold must equal the frozen contract threshold")
    if not bool(experiment_config["model"].get("condition_on_event_pre", False)):
        violations.append("A3-G3 development model must condition on its event-pre state")
    if not 0.0 < float(experiment_config["calibration"].get("outer_alpha", 0.0)) < 1.0:
        violations.append("A3-G3 outer_alpha must be in (0, 1)")
    return violations


def audit_counterfactual_effect_graph_inputs(
    contract_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> Dict[str, Any]:
    """Verify raw trigger/response invariants and G3's frozen configuration."""
    violations = _validate_experiment(contract_config, experiment_config)
    raw_graph_audit = audit_graph_grammar_inputs(contract_config, experiment_config, suite)
    violations.extend(raw_graph_audit["violations"])
    return {
        "passed": not violations,
        "violations": violations,
        "metrics": {
            **raw_graph_audit["metrics"],
            "effect_token_energy_threshold": float(
                experiment_config["effect_extractor"]["token_energy_threshold"]
            ),
            "ridge_penalty": float(experiment_config["counterfactual"]["ridge_penalty"]),
        },
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-config", type=Path, default=DEFAULT_CONTRACT_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--contract-seed", type=int, default=None)
    arguments = parser.parse_args(argv)
    contract = _load_json(arguments.contract_config)
    experiment = _load_json(arguments.experiment_config)
    if arguments.contract_seed is not None:
        contract["seed"] = int(arguments.contract_seed)
    result = audit_counterfactual_effect_graph_inputs(contract, experiment, generate_suite(contract))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
