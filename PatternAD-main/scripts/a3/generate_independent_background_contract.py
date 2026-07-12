#!/usr/bin/env python3
"""Generate A3-v2's independent ordinary-background evaluation blocks.

This is a contract generator, not a detector or a calibration sweep.  Every
returned window is produced from a distinct seeded normal-process trajectory
that remains in one declared noise regime through burn-in, event-pre, and
future.  It removes the overlapping-window ambiguity in A3-v1's background
FPR measurement while preserving the same normal-process equations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from scripts.a3.generate_trigger_response_contract import _load_json, validate_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = REPO_ROOT / "config" / "a3" / "trigger_response_contract_v1.json"
DEFAULT_PROTOCOL_CONFIG = REPO_ROOT / "config" / "a3" / "independent_background_calibration_v2.json"
SUPPORTED_PROTOCOL_BASES = {
    "a3_independent_background_calibration_v2": "a3_trigger_response_contract_v1",
    "a3_independent_background_route_v2": "a3_trigger_response_route_identifiability_v2",
}


def canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_protocol(
    base_contract: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    """Reject a background protocol that is not fixed before detector fitting."""
    validate_config(base_contract)
    if int(protocol.get("schema_version", 0)) != 1:
        raise ValueError("A3-v2 background protocol must use schema_version=1.")
    protocol_id = str(protocol.get("protocol_id", ""))
    if protocol_id not in SUPPORTED_PROTOCOL_BASES:
        raise ValueError("Unexpected A3 independent-background protocol_id.")
    if str(base_contract["suite_id"]) != SUPPORTED_PROTOCOL_BASES[protocol_id]:
        raise ValueError("A3 independent-background protocol has an unexpected base suite.")
    if str(protocol.get("base_contract_id", "")) != str(base_contract["suite_id"]):
        raise ValueError("A3-v2 background protocol is paired with the wrong base contract.")
    background = protocol["background"]
    if int(background["seed"]) < 0 or int(background["burn_in"]) < int(base_contract["history_length"]):
        raise ValueError("A3-v2 background seed or burn-in is invalid.")
    if int(background["blocks_per_regime"]) < 2 or not bool(background["fixed_regime_per_block"]):
        raise ValueError("A3-v2 requires at least two fixed-regime independent blocks.")
    extractor = protocol["trigger_extractor"]
    if int(extractor["cue_length"]) > int(base_contract["history_length"]):
        raise ValueError("A3-v2 trigger cue length exceeds the history window.")
    if float(extractor["minimum_amplitude"]) <= 0.0 or float(extractor["linear_tolerance"]) <= 0.0:
        raise ValueError("A3-v2 trigger extractor parameters must be positive.")
    evaluation = protocol["evaluation"]
    if not 0.0 < float(evaluation["operating_fpr"]) < 1.0:
        raise ValueError("A3-v2 operating FPR must be in (0, 1).")
    if not 0.5 < float(evaluation["confidence_level"]) < 1.0:
        raise ValueError("A3-v2 confidence level must be in (0.5, 1).")
    if str(evaluation["interval"]) != "wilson_one_sided":
        raise ValueError("A3-v2 requires the frozen one-sided Wilson interval.")
    if not bool(evaluation.get("require_per_regime_bound", False)):
        raise ValueError("A3-v2 requires a per-regime FPR confidence bound.")
    total = int(background["blocks_per_regime"]) * len(base_contract["normal_process"]["regime_noise_scales"])
    if int(evaluation["minimum_total_blocks"]) != total:
        raise ValueError("A3-v2 minimum_total_blocks must equal the generated block count.")


def _simulate_fixed_regime_window(
    base_contract: Mapping[str, Any],
    scale: float,
    burn_in: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one independent stationary-regime event-pre/future block."""
    process = base_contract["normal_process"]
    dimensions = int(base_contract["dimensions"])
    length = int(base_contract["history_length"]) + int(base_contract["horizon_length"])
    total = int(burn_in) + length
    values = np.zeros((total, dimensions), dtype=np.float64)
    latent = 0.0
    loadings = np.asarray(process["loadings"], dtype=np.float64)
    channel_ar = np.asarray(process["channel_ar"], dtype=np.float64)
    channel_noise = np.asarray(process["channel_noise"], dtype=np.float64)
    for index in range(total):
        latent = (
            float(process["latent_ar"]) * latent
            + float(process["latent_std"]) * float(scale) * rng.normal()
        )
        previous = values[index - 1] if index else np.zeros(dimensions, dtype=np.float64)
        values[index] = channel_ar * previous + loadings * latent + channel_noise * float(scale) * rng.normal(
            size=dimensions
        )
    return values[burn_in:].astype(np.float32)


def generate_independent_background_suite(
    base_contract: Mapping[str, Any], protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return balanced, independently seeded normal blocks plus provenance."""
    validate_protocol(base_contract, protocol)
    process = base_contract["normal_process"]
    scales = np.asarray(process["regime_noise_scales"], dtype=np.float64)
    background = protocol["background"]
    blocks_per_regime = int(background["blocks_per_regime"])
    count = len(scales) * blocks_per_regime
    seed_sequence = np.random.SeedSequence(int(background["seed"]))
    children = seed_sequence.spawn(count)
    windows: List[np.ndarray] = []
    provenance: List[Dict[str, Any]] = []
    child_index = 0
    for regime_index, scale in enumerate(scales.tolist()):
        for block_index in range(blocks_per_regime):
            child = children[child_index]
            window = _simulate_fixed_regime_window(
                base_contract,
                scale=float(scale),
                burn_in=int(background["burn_in"]),
                rng=np.random.default_rng(child),
            )
            windows.append(window)
            provenance.append(
                {
                    "source_id": f"independent_r{regime_index}_b{block_index:04d}",
                    "regime_index": int(regime_index),
                    "regime_scale": float(scale),
                    "block_index": int(block_index),
                    "spawn_key": list(child.spawn_key),
                    "window_start": 0,
                }
            )
            child_index += 1
    return {
        "base_contract_hash": canonical_hash(base_contract),
        "protocol_hash": canonical_hash(protocol),
        "windows": np.stack(windows).astype(np.float32, copy=False),
        "provenance": provenance,
    }


if __name__ == "__main__":
    raise SystemExit("Use audit_independent_background_contract.py to inspect this contract.")
