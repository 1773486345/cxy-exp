#!/usr/bin/env python3
"""Audit A3-v2 independent-background FPR evaluation without fitting a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a3.generate_independent_background_contract import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_PROTOCOL_CONFIG,
    generate_independent_background_suite,
    validate_protocol,
)
from scripts.a3.generate_trigger_response_contract import _load_json
from ts_benchmark.baselines.A3TriggerResponse import extract_trigger_states


def one_sided_wilson_upper(exceedances: int, count: int, confidence_level: float) -> float:
    """Upper Wilson bound used by later frozen detector decisions."""
    if count < 1 or not 0 <= exceedances <= count:
        raise ValueError("Wilson inputs are invalid.")
    proportion = float(exceedances) / float(count)
    z = NormalDist().inv_cdf(float(confidence_level))
    z_squared = z * z
    denominator = 1.0 + z_squared / count
    center = (proportion + z_squared / (2.0 * count)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / count + z_squared / (4.0 * count * count)
    ) / denominator
    return float(min(1.0, center + radius))


def maximum_accepted_exceedances(
    count: int, operating_fpr: float, confidence_level: float
) -> int:
    """Return the largest observed count whose frozen upper bound meets the target."""
    accepted = -1
    for exceedances in range(count + 1):
        if one_sided_wilson_upper(exceedances, count, confidence_level) <= operating_fpr:
            accepted = exceedances
        else:
            break
    return accepted


def _window_hashes(windows: np.ndarray) -> list[str]:
    return [hashlib.sha256(np.ascontiguousarray(window).tobytes()).hexdigest() for window in windows]


def audit_independent_background_contract(
    base_contract: Mapping[str, Any], protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    """Check independence provenance and the frozen future FPR decision rule."""
    violations: list[str] = []
    try:
        validate_protocol(base_contract, protocol)
        suite = generate_independent_background_suite(base_contract, protocol)
    except (KeyError, TypeError, ValueError) as error:
        return {"passed": False, "violations": [str(error)], "metrics": {}}

    windows = np.asarray(suite["windows"], dtype=np.float32)
    provenance = suite["provenance"]
    history = int(base_contract["history_length"])
    horizon = int(base_contract["horizon_length"])
    dimensions = int(base_contract["dimensions"])
    scales = np.asarray(base_contract["normal_process"]["regime_noise_scales"], dtype=np.float64)
    blocks_per_regime = int(protocol["background"]["blocks_per_regime"])
    expected_count = len(scales) * blocks_per_regime
    if windows.shape != (expected_count, history + horizon, dimensions):
        violations.append("A3-v2 independent background window shape is invalid")
    if not np.isfinite(windows).all():
        violations.append("A3-v2 independent background contains non-finite values")
    source_ids = [str(row["source_id"]) for row in provenance]
    spawn_keys = [tuple(int(value) for value in row["spawn_key"]) for row in provenance]
    if len(provenance) != expected_count or len(set(source_ids)) != expected_count:
        violations.append("A3-v2 independent background source IDs are not one-to-one")
    if len(set(spawn_keys)) != expected_count:
        violations.append("A3-v2 independent background seed streams are not one-to-one")
    hashes = _window_hashes(windows)
    if len(set(hashes)) != expected_count:
        violations.append("A3-v2 independent background contains duplicate raw windows")
    regime_counts = [sum(int(row["regime_index"]) == index for row in provenance) for index in range(len(scales))]
    if regime_counts != [blocks_per_regime] * len(scales):
        violations.append("A3-v2 independent background is not balanced across regimes")
    scale_errors = [
        abs(float(row["regime_scale"]) - float(scales[int(row["regime_index"])])) for row in provenance
    ]
    if max(scale_errors, default=0.0) > 1e-12:
        violations.append("A3-v2 background provenance does not match its declared regime scale")
    extractor = protocol["trigger_extractor"]
    states = extract_trigger_states(
        windows[:, :history],
        cue_length=int(extractor["cue_length"]),
        minimum_amplitude=float(extractor["minimum_amplitude"]),
        linear_tolerance=float(extractor["linear_tolerance"]),
    )
    false_triggers = int(np.sum(np.asarray(states[:, 0], dtype=np.int64) != 0))
    if false_triggers:
        violations.append("A3-v2 ordinary background contains a fixed-trigger acceptance")
    evaluation = protocol["evaluation"]
    total = int(len(windows))
    if total != int(evaluation["minimum_total_blocks"]):
        violations.append("A3-v2 generated count does not match the frozen FPR sample size")
    max_accepted = maximum_accepted_exceedances(
        total,
        float(evaluation["operating_fpr"]),
        float(evaluation["confidence_level"]),
    )
    if max_accepted < 0:
        violations.append(
            "A3-v2 background sample size is too small for any observation to meet its FPR confidence gate"
        )
    per_regime_maximum = maximum_accepted_exceedances(
        blocks_per_regime,
        float(evaluation["operating_fpr"]),
        float(evaluation["confidence_level"]),
    )
    if per_regime_maximum < 0:
        violations.append(
            "A3-v2 per-regime sample size is too small for any observation to meet its FPR confidence gate"
        )
    return {
        "passed": not violations,
        "violations": violations,
        "metrics": {
            "independent_block_count": total,
            "window_length": int(history + horizon),
            "dimensions": dimensions,
            "regime_counts": regime_counts,
            "unique_source_count": len(set(source_ids)),
            "unique_spawn_key_count": len(set(spawn_keys)),
            "unique_raw_window_count": len(set(hashes)),
            "fixed_trigger_false_acceptances": false_triggers,
            "operating_fpr": float(evaluation["operating_fpr"]),
            "confidence_level": float(evaluation["confidence_level"]),
            "maximum_accepted_exceedances": max_accepted,
            "maximum_accepted_empirical_fpr": (
                float(max_accepted / total) if max_accepted >= 0 else None
            ),
            "upper_bound_at_maximum_accepted": (
                one_sided_wilson_upper(max_accepted, total, float(evaluation["confidence_level"]))
                if max_accepted >= 0
                else None
            ),
            "per_regime_maximum_accepted_exceedances": per_regime_maximum,
            "per_regime_maximum_accepted_empirical_fpr": (
                float(per_regime_maximum / blocks_per_regime)
                if per_regime_maximum >= 0
                else None
            ),
            "per_regime_upper_bound_at_maximum_accepted": (
                one_sided_wilson_upper(
                    per_regime_maximum,
                    blocks_per_regime,
                    float(evaluation["confidence_level"]),
                )
                if per_regime_maximum >= 0
                else None
            ),
            "base_contract_hash": suite["base_contract_hash"],
            "protocol_hash": suite["protocol_hash"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-contract-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL_CONFIG)
    arguments = parser.parse_args(argv)
    result = audit_independent_background_contract(
        _load_json(arguments.base_contract_config), _load_json(arguments.protocol_config)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
