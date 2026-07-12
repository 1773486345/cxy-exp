#!/usr/bin/env python3
"""Audit A3's route-identifiable successor contract without fitting a detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a3.audit_independent_background_contract import audit_independent_background_contract
from scripts.a3.audit_trigger_response_contract import audit_suite
from scripts.a3.generate_trigger_response_contract import _load_json, generate_suite, validate_config


DEFAULT_CONTRACT_CONFIG = (
    REPO_ROOT / "config" / "a3" / "trigger_response_route_identifiability_v2.json"
)
DEFAULT_BACKGROUND_PROTOCOL = (
    REPO_ROOT / "config" / "a3" / "independent_background_route_v2.json"
)


def response_background_alignment(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Return fixed-route alignment with the normal latent loading direction."""
    response_channels = np.asarray(contract["episodes"]["response_channels"], dtype=np.int64)
    loading = np.asarray(contract["normal_process"]["loadings"], dtype=np.float64)[response_channels]
    patterns = np.asarray(contract["episodes"]["response_patterns"], dtype=np.float64)
    denominator = float(np.linalg.norm(loading) * np.linalg.norm(patterns[0]))
    if denominator <= 0.0:
        raise ValueError("A3 route alignment has a zero loading or response vector.")
    cosine = float(abs(np.dot(loading, patterns[0]) / denominator))
    return {
        "response_channels": response_channels.tolist(),
        "background_loading": loading.tolist(),
        "response_mode": patterns[0].tolist(),
        "absolute_cosine": cosine,
        "dot_product": float(np.dot(loading, patterns[0])),
    }


def audit_route_identifiability_contract(
    contract: Mapping[str, Any], background_protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate route/background separation plus inherited raw A3 invariants."""
    violations: list[str] = []
    try:
        validate_config(contract)
    except (KeyError, TypeError, ValueError) as error:
        return {"passed": False, "violations": [str(error)], "metrics": {}}
    if str(contract.get("suite_id")) != "a3_trigger_response_route_identifiability_v2":
        violations.append("Route-identifiability audit requires the A3-v2 successor suite")
    route = contract.get("route_identifiability", {})
    try:
        alignment = response_background_alignment(contract)
        patterns = np.asarray(contract["episodes"]["response_patterns"], dtype=np.float64)
        if alignment["absolute_cosine"] > float(route["max_background_loading_cosine"]):
            violations.append("Response route is too aligned with the normal latent loading")
        if np.min(np.abs(patterns[0])) < float(route["minimum_abs_response_component"]):
            violations.append("A3-v2 route has a response component too weak for the declared graph")
        if not np.allclose(patterns[0], -patterns[1], atol=1e-12, rtol=0.0):
            violations.append("A3-v2 route modes are not exactly opposed")
    except (KeyError, TypeError, ValueError) as error:
        return {"passed": False, "violations": [str(error)], "metrics": {}}
    raw_audit = audit_suite(contract, generate_suite(contract))
    if not raw_audit["passed"]:
        violations.extend(raw_audit["violations"])
    background_audit = audit_independent_background_contract(contract, background_protocol)
    if not background_audit["passed"]:
        violations.extend(background_audit["violations"])
    return {
        "passed": not violations,
        "violations": violations,
        "metrics": {
            "route_alignment": alignment,
            "route_cosine_limit": float(route["max_background_loading_cosine"]),
            "minimum_abs_response_component": float(route["minimum_abs_response_component"]),
            "raw_contract": raw_audit["metrics"],
            "independent_background": background_audit["metrics"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-config", type=Path, default=DEFAULT_CONTRACT_CONFIG)
    parser.add_argument("--background-protocol", type=Path, default=DEFAULT_BACKGROUND_PROTOCOL)
    arguments = parser.parse_args(argv)
    result = audit_route_identifiability_contract(
        _load_json(arguments.contract_config), _load_json(arguments.background_protocol)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
