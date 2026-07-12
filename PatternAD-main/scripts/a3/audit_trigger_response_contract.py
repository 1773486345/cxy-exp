#!/usr/bin/env python3
"""Audit A3 trigger-response construction invariants without fitting a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a3.generate_trigger_response_contract import (
    DEFAULT_CONFIG,
    ROLE_ORDER,
    SPLIT_NAMES,
    _load_json,
    extract_response_tokens,
    generate_suite,
    validate_config,
)


def _majority_accuracy(rows: List[tuple[int, int]]) -> float:
    counts: Dict[int, List[int]] = {}
    for feature, label in rows:
        counts.setdefault(feature, [0, 0])[label] += 1
    return float(sum(max(values) for values in counts.values()) / len(rows))


def _grouped(suite: Mapping[str, Any]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    result: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for episode in suite["episodes"]:
        result.setdefault(str(episode["pair_id"]), {})[str(episode["role"])] = episode
    return result


def _mode_from_raw_cue(event_pre: np.ndarray, config: Mapping[str, Any]) -> tuple[int, float]:
    episodes = config["episodes"]
    source = int(episodes["source_channel"])
    cue_length = int(episodes["cue_length"])
    amplitudes = np.asarray(episodes["cue_amplitudes"], dtype=np.float64)
    delta = float(event_pre[-1, source] - event_pre[-cue_length, source])
    return int(np.argmin(np.abs(amplitudes - delta))), delta


def _mode_from_target_token(tokens: Mapping[str, np.ndarray], target: int) -> int | None:
    if int(tokens["active"][target]) != 1:
        return None
    return 0 if int(tokens["direction"][target]) == 1 else 1


def audit_suite(
    config: Mapping[str, Any], suite: Mapping[str, Any], tolerance: float = 1e-7
) -> Dict[str, Any]:
    """Validate raw A3-v1 facts required before model training."""
    validate_config(config)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    history = int(config["history_length"])
    episodes_config = config["episodes"]
    response_channels = np.asarray(episodes_config["response_channels"], dtype=np.int64)
    target = int(response_channels[0])
    threshold = float(episodes_config["token_energy_threshold"])
    cue_amplitudes = np.asarray(episodes_config["cue_amplitudes"], dtype=np.float64)
    cue_tolerance = float(episodes_config["cue_amplitude_tolerance"])
    grouped = _grouped(suite)
    violations: List[str] = []
    primary_rows_cue: List[tuple[int, int]] = []
    primary_rows_response: List[tuple[int, int]] = []
    cue_errors: List[float] = []
    primary_relation_count = 0
    secondary_target_ties = 0
    deterministic_count = 0
    normal_token_count = 0

    expected_ranges: Dict[str, tuple[int, int]] = {}
    start = 0
    guard = int(config["normal_splits"]["guard_length"])
    for index, split in enumerate(SPLIT_NAMES):
        end = start + int(config["normal_splits"][f"{split}_length"])
        expected_ranges[split] = (start, end)
        start = end + (guard if index < len(SPLIT_NAMES) - 1 else 0)
    if dict(suite["normal_split_ranges"]) != expected_ranges:
        violations.append("normal split ranges differ from frozen config")

    for split in SPLIT_NAMES:
        bank = suite["normal_event_banks"].get(split, [])
        required = 3 * int(episodes_config["normal_transition_sources_per_regime"][split])
        if len(bank) != required:
            violations.append(f"{split}: normal event bank count is not frozen count")
        for entry in bank:
            values = np.asarray(entry["values"], dtype=np.float32)
            first = extract_response_tokens(values[history:], threshold)
            second = extract_response_tokens(values[history:], threshold)
            if any(not np.array_equal(first[name], second[name]) for name in first):
                violations.append(f"{split}:{entry['bank_id']}: response tokens are not deterministic")
            deterministic_count += 1
            normal_token_count += int(np.isfinite(values).all())

    for pair_id, roles in sorted(grouped.items()):
        role_set = set(roles)
        if pair_id.startswith("triggered_"):
            expected = {"normal_routed_response", "misrouted_response", "partial_propagation_response"}
            if role_set != expected:
                violations.append(f"{pair_id}: triggered role set is incomplete")
                continue
            normal = roles["normal_routed_response"]
            misrouted = roles["misrouted_response"]
            partial = roles["partial_propagation_response"]
            normal_values = np.asarray(normal["values"], dtype=np.float32)
            misrouted_values = np.asarray(misrouted["values"], dtype=np.float32)
            partial_values = np.asarray(partial["values"], dtype=np.float32)
            if not np.allclose(normal_values[:history], misrouted_values[:history], atol=tolerance, rtol=0.0):
                violations.append(f"{pair_id}: primary event-pre values differ")
            if not np.allclose(normal_values[:history], partial_values[:history], atol=tolerance, rtol=0.0):
                violations.append(f"{pair_id}: secondary event-pre values differ")
            if not np.allclose(normal_values[history:, target], partial_values[history:, target], atol=tolerance, rtol=0.0):
                violations.append(f"{pair_id}: partial propagation changes target trajectory")
            else:
                secondary_target_ties += 1
            normal_tokens = extract_response_tokens(normal_values[history:], threshold)
            misrouted_tokens = extract_response_tokens(misrouted_values[history:], threshold)
            if any(not np.array_equal(normal_tokens[name], extract_response_tokens(normal_values[history:], threshold)[name]) for name in normal_tokens):
                violations.append(f"{pair_id}: primary tokens are non-deterministic")
            cue_mode, cue_delta = _mode_from_raw_cue(normal_values[:history], config)
            cue_errors.append(abs(cue_delta - float(cue_amplitudes[cue_mode])))
            normal_mode = _mode_from_target_token(normal_tokens, target)
            misrouted_mode = _mode_from_target_token(misrouted_tokens, target)
            if normal_mode is None or misrouted_mode is None:
                violations.append(f"{pair_id}: target response token is inactive")
                continue
            if normal_mode != cue_mode or misrouted_mode == cue_mode:
                violations.append(f"{pair_id}: raw trigger-response relation is not routed/misrouted")
            else:
                primary_relation_count += 1
            primary_rows_cue.extend(((cue_mode, 0), (cue_mode, 1)))
            primary_rows_response.extend(((normal_mode, 0), (misrouted_mode, 1)))
        elif pair_id.startswith("untriggered_"):
            expected = {"normal_no_trigger", "untriggered_response"}
            if role_set != expected:
                violations.append(f"{pair_id}: no-trigger role set is incomplete")
                continue
            normal = np.asarray(roles["normal_no_trigger"]["values"], dtype=np.float32)
            untriggered = np.asarray(roles["untriggered_response"]["values"], dtype=np.float32)
            if not np.allclose(normal[:history], untriggered[:history], atol=tolerance, rtol=0.0):
                violations.append(f"{pair_id}: untriggered event-pre values differ")
            untriggered_tokens = extract_response_tokens(untriggered[history:], threshold)
            if int(untriggered_tokens["active"][target]) != 1:
                violations.append(f"{pair_id}: untriggered target is not observable")
        else:
            violations.append(f"{pair_id}: unknown pair family")

    if cue_errors and max(cue_errors) > cue_tolerance:
        violations.append("observable cue amplitude exceeds its declared tolerance")
    cue_only_accuracy = _majority_accuracy(primary_rows_cue) if primary_rows_cue else 0.0
    response_only_accuracy = _majority_accuracy(primary_rows_response) if primary_rows_response else 0.0
    if abs(cue_only_accuracy - 0.5) > tolerance:
        violations.append("cue alone predicts primary normal/misrouted label above chance")
    if abs(response_only_accuracy - 0.5) > tolerance:
        violations.append("response mode alone predicts primary normal/misrouted label above chance")
    expected_primary = 2 * int(episodes_config["pairs_per_mode"])
    if primary_relation_count != expected_primary:
        violations.append("not every primary pair has its exact raw trigger-response relation")
    if secondary_target_ties != expected_primary:
        violations.append("not every secondary pair ties its target trajectory")
    if deterministic_count != normal_token_count:
        violations.append("normal event bank contains non-finite values")

    return {
        "passed": not violations,
        "violations": violations,
        "metrics": {
            "cue_only_primary_accuracy": cue_only_accuracy,
            "response_mode_only_primary_accuracy": response_only_accuracy,
            "primary_raw_relation_count": primary_relation_count,
            "secondary_target_tie_count": secondary_target_ties,
            "max_cue_amplitude_error": max(cue_errors) if cue_errors else None,
            "deterministic_normal_token_count": deterministic_count,
            "normal_bank_finite_count": normal_token_count,
        },
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args(argv)
    config = _load_json(arguments.config)
    result = audit_suite(config, generate_suite(config))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
