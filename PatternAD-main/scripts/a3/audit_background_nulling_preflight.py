#!/usr/bin/env python3
"""Run A3-N1's normal-only background-subspace preflight without a detector."""

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

from scripts.a3.audit_route_identifiability_contract import (
    DEFAULT_BACKGROUND_PROTOCOL,
    DEFAULT_CONTRACT_CONFIG,
    audit_route_identifiability_contract,
)
from scripts.a3.generate_trigger_response_contract import _load_json, generate_suite


DEFAULT_PREFLIGHT_CONFIG = REPO_ROOT / "config" / "a3" / "background_nulling_n1_v1.json"


def _validate_preflight_config(
    contract: Mapping[str, Any], background_protocol: Mapping[str, Any], preflight: Mapping[str, Any]
) -> None:
    if int(preflight.get("schema_version", 0)) != 1:
        raise ValueError("A3-N1 preflight config must use schema_version=1.")
    if str(preflight.get("experiment_id", "")) != "a3_n1_background_nulling_route_graph_preflight_v1":
        raise ValueError("Unexpected A3-N1 preflight experiment_id.")
    if str(preflight.get("base_contract_id", "")) != str(contract.get("suite_id", "")):
        raise ValueError("A3-N1 preflight is paired with the wrong route contract.")
    if str(preflight.get("background_protocol_id", "")) != str(background_protocol.get("protocol_id", "")):
        raise ValueError("A3-N1 preflight is paired with the wrong background protocol.")
    subspace = preflight["background_subspace"]
    if str(subspace.get("channels", "")) != "all_raw":
        raise ValueError("A3-N1 must fit its background factor on all raw channels.")
    if int(subspace["components"]) != 1 or str(subspace["fit_split"]) != "optimization":
        raise ValueError("A3-N1 only permits one factor fitted on normal optimization increments.")
    if not bool(subspace["center_increments"]):
        raise ValueError("A3-N1 requires centered normal increments for PCA.")
    if not 0.0 < float(subspace["minimum_factor_alignment"]) <= 1.0:
        raise ValueError("A3-N1 minimum factor alignment is invalid.")
    if not 0.0 < float(subspace["minimum_route_retention"]) <= 1.0:
        raise ValueError("A3-N1 minimum route retention is invalid.")
    if float(preflight["token_extractor"]["token_energy_threshold"]) != float(
        contract["episodes"]["token_energy_threshold"]
    ):
        raise ValueError("A3-N1 token threshold must equal the frozen route contract threshold.")


def _optimization_increments(suite: Mapping[str, Any], contract: Mapping[str, Any]) -> np.ndarray:
    start, end = suite["normal_split_ranges"]["optimization"]
    values = np.asarray(suite["train_values"], dtype=np.float64)[start:end]
    if len(values) < 3:
        raise ValueError("A3-N1 optimization split is too short for a PCA factor.")
    return np.diff(values, axis=0)


def fit_normal_background_factor(increments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the deterministic first centered PCA direction from normal increments."""
    values = np.asarray(increments, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("A3-N1 increments must be finite [samples, response_channels].")
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    direction = right_vectors[0]
    if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 0.0:
        raise ValueError("A3-N1 PCA did not produce a valid background direction.")
    return direction / np.linalg.norm(direction), singular_values


def audit_background_nulling_preflight(
    contract: Mapping[str, Any], background_protocol: Mapping[str, Any], preflight: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate N1's normal-only factor and preserve the declared response route."""
    violations: list[str] = []
    try:
        _validate_preflight_config(contract, background_protocol, preflight)
        route_audit = audit_route_identifiability_contract(contract, background_protocol)
        if not route_audit["passed"]:
            violations.extend(route_audit["violations"])
        suite = generate_suite(contract)
        increments = _optimization_increments(suite, contract)
        factor, singular_values = fit_normal_background_factor(increments)
        loading = np.asarray(contract["normal_process"]["loadings"], dtype=np.float64)
        loading = loading / np.linalg.norm(loading)
        route = np.zeros(int(contract["dimensions"]), dtype=np.float64)
        route[np.asarray(contract["episodes"]["response_channels"], dtype=np.int64)] = np.asarray(
            contract["episodes"]["response_patterns"], dtype=np.float64
        )[0]
        route = route / np.linalg.norm(route)
        alignment = float(abs(np.dot(factor, loading)))
        projected_route = route - factor * float(np.dot(factor, route))
        retention = float(np.linalg.norm(projected_route))
        subspace = preflight["background_subspace"]
        if alignment < float(subspace["minimum_factor_alignment"]):
            violations.append("A3-N1 normal PCA factor does not recover the background direction")
        if retention < float(subspace["minimum_route_retention"]):
            violations.append("A3-N1 background projection removes too much declared response route")
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        return {"passed": False, "violations": [str(error)], "metrics": {}}
    return {
        "passed": not violations,
        "violations": violations,
        "metrics": {
            "fit_source": "normal_optimization_increments_only",
            "increment_count": int(len(increments)),
            "factor": factor.astype(float).tolist(),
            "singular_values": singular_values.astype(float).tolist(),
            "factor_background_alignment": alignment,
            "minimum_factor_alignment": float(subspace["minimum_factor_alignment"]),
            "route_retention_after_projection": retention,
            "minimum_route_retention": float(subspace["minimum_route_retention"]),
            "route_contract": route_audit["metrics"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-config", type=Path, default=DEFAULT_CONTRACT_CONFIG)
    parser.add_argument("--background-protocol", type=Path, default=DEFAULT_BACKGROUND_PROTOCOL)
    parser.add_argument("--preflight-config", type=Path, default=DEFAULT_PREFLIGHT_CONFIG)
    arguments = parser.parse_args(argv)
    result = audit_background_nulling_preflight(
        _load_json(arguments.contract_config),
        _load_json(arguments.background_protocol),
        _load_json(arguments.preflight_config),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
