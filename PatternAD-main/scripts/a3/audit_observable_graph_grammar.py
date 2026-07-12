#!/usr/bin/env python3
"""Audit A3-G2's raw trigger and response-graph representation."""

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

from scripts.a3.audit_trigger_response_contract import audit_suite as audit_a3_suite
from scripts.a3.generate_trigger_response_contract import (
    DEFAULT_CONFIG as DEFAULT_CONTRACT_CONFIG,
    _load_json,
    generate_suite,
)
from ts_benchmark.baselines.A3TriggerResponse.A3ObservableGraphGrammar import (
    extract_trigger_states,
    response_graph_tokens,
)


DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "config" / "a3" / "observable_graph_grammar_g2_v1.json"


def _majority_accuracy(rows: List[tuple[int, int]]) -> float:
    counts: Dict[int, List[int]] = {}
    for feature, label in rows:
        counts.setdefault(feature, [0, 0])[label] += 1
    return float(sum(max(value) for value in counts.values()) / len(rows))


def _grouped(suite: Mapping[str, Any]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    grouped: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for episode in suite["episodes"]:
        grouped.setdefault(str(episode["pair_id"]), {})[str(episode["role"])] = episode
    return grouped


def _check_extractor_config(
    contract_config: Mapping[str, Any], experiment_config: Mapping[str, Any]
) -> None:
    trigger = experiment_config.get("trigger_extractor", {})
    if int(trigger.get("cue_length", -1)) != int(contract_config["episodes"]["cue_length"]):
        raise ValueError("A3-G2 cue_length must equal the frozen contract cue_length.")
    if not 0.0 < float(trigger.get("minimum_amplitude", 0.0)) < min(
        abs(float(value)) for value in contract_config["episodes"]["cue_amplitudes"]
    ):
        raise ValueError("A3-G2 trigger minimum amplitude must lie below the frozen cue amplitude.")
    if not 0.0 < float(trigger.get("linear_tolerance", 0.0)) < 1e-3:
        raise ValueError("A3-G2 trigger linear_tolerance must be in (0, 1e-3).")


def audit_graph_grammar_inputs(
    contract_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    suite: Mapping[str, Any],
    tolerance: float = 1e-7,
) -> Dict[str, Any]:
    """Certify fixed observable G2 states before any grammar model is fitted."""
    _check_extractor_config(contract_config, experiment_config)
    contract_audit = audit_a3_suite(contract_config, suite, tolerance=tolerance)
    violations = list(contract_audit["violations"])
    if not contract_audit["passed"]:
        return {"passed": False, "violations": violations, "metrics": {}}
    history = int(contract_config["history_length"])
    threshold = float(contract_config["episodes"]["token_energy_threshold"])
    target = int(contract_config["episodes"]["response_channels"][0])
    extractor = experiment_config["trigger_extractor"]
    grouped = _grouped(suite)
    primary_trigger_rows: List[tuple[int, int]] = []
    primary_response_rows: List[tuple[int, int]] = []
    routed_exact = 0
    no_trigger_exact = 0
    routed_count = 0
    no_trigger_count = 0
    trigger_errors: List[float] = []
    ordinary_trigger_count = 0
    ordinary_window_count = 0

    def states(values: np.ndarray) -> np.ndarray:
        return extract_trigger_states(
            values,
            cue_length=int(extractor["cue_length"]),
            minimum_amplitude=float(extractor["minimum_amplitude"]),
            linear_tolerance=float(extractor["linear_tolerance"]),
        )

    def target_mode(values: np.ndarray) -> int:
        tokens = response_graph_tokens(values[None, ...], threshold)[0]
        code = int(tokens[target])
        if code == 0:
            return -1
        # The fixed direction bit is one for a positive target response;
        # A3-v1 mode zero is positive and mode one is negative.
        return 0 if int((code - 1) % 2) == 1 else 1

    for pair_id, roles in sorted(grouped.items()):
        if pair_id.startswith("triggered_"):
            normal = np.asarray(roles["normal_routed_response"]["values"], dtype=np.float32)
            misrouted = np.asarray(roles["misrouted_response"]["values"], dtype=np.float32)
            expected_mode = int(roles["normal_routed_response"]["cue_mode"])
            expected_state = 1 + expected_mode
            normal_state, normal_error = states(normal[None, :history])[0]
            misrouted_state, misrouted_error = states(misrouted[None, :history])[0]
            routed_count += 2
            routed_exact += int(normal_state == expected_state) + int(misrouted_state == expected_state)
            trigger_errors.extend((normal_error, misrouted_error))
            normal_mode = target_mode(normal[history:])
            misrouted_mode = target_mode(misrouted[history:])
            if normal_mode != expected_mode or misrouted_mode == expected_mode:
                violations.append(f"{pair_id}: response graph target mode is not balanced/routed")
            primary_trigger_rows.extend(((int(normal_state), 0), (int(misrouted_state), 1)))
            primary_response_rows.extend(((normal_mode, 0), (misrouted_mode, 1)))
        elif pair_id.startswith("untriggered_"):
            for role in ("normal_no_trigger", "untriggered_response"):
                values = np.asarray(roles[role]["values"], dtype=np.float32)
                state, error = states(values[None, :history])[0]
                no_trigger_count += 1
                no_trigger_exact += int(state == 0)
                trigger_errors.append(error)
        else:
            violations.append(f"{pair_id}: unknown A3 pair family")

    for split, (start, end) in suite["normal_split_ranges"].items():
        values = np.asarray(suite["train_values"], dtype=np.float32)
        starts = np.arange(start, end - history + 1, dtype=np.int64)
        windows = np.stack([values[index : index + history] for index in starts])
        observed = states(windows)
        ordinary_trigger_count += int(np.sum(observed[:, 0] != 0))
        ordinary_window_count += len(observed)

    trigger_only_accuracy = _majority_accuracy(primary_trigger_rows) if primary_trigger_rows else 0.0
    response_only_accuracy = _majority_accuracy(primary_response_rows) if primary_response_rows else 0.0
    if abs(trigger_only_accuracy - 0.5) > tolerance:
        violations.append("fixed trigger state alone predicts primary label above chance")
    if abs(response_only_accuracy - 0.5) > tolerance:
        violations.append("fixed response target mode alone predicts primary label above chance")
    if routed_exact != routed_count:
        violations.append("fixed trigger extractor does not recover every routed trigger state")
    if no_trigger_exact != no_trigger_count:
        violations.append("fixed trigger extractor falsely accepts a paired no-trigger state")
    if ordinary_trigger_count:
        violations.append("fixed trigger extractor falsely accepts ordinary normal continuation")

    return {
        "passed": not violations,
        "violations": violations,
        "metrics": {
            "routed_trigger_exact": routed_exact,
            "routed_trigger_count": routed_count,
            "no_trigger_exact": no_trigger_exact,
            "no_trigger_count": no_trigger_count,
            "ordinary_trigger_count": ordinary_trigger_count,
            "ordinary_window_count": ordinary_window_count,
            "trigger_only_primary_accuracy": trigger_only_accuracy,
            "response_mode_only_primary_accuracy": response_only_accuracy,
            "max_trigger_linear_error": max(trigger_errors) if trigger_errors else None,
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
    result = audit_graph_grammar_inputs(contract, experiment, generate_suite(contract))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
