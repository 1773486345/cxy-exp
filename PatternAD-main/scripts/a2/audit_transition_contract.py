#!/usr/bin/env python3
"""Audit A2 construction invariants without fitting or scoring a model."""

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

from scripts.a2.generate_transition_contract import (
    DEFAULT_CONFIG,
    ROLE_ORDER,
    _load_json,
    generate_suite,
    validate_config,
)


def _episodes_by_source(suite: Mapping[str, Any]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    grouped: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for episode in suite["episodes"]:
        grouped.setdefault(str(episode["source_id"]), {})[str(episode["role"])] = episode
    return grouped


def _increment_statistics(values: np.ndarray) -> tuple[float, float, float]:
    increments = np.diff(np.asarray(values, dtype=np.float64), axis=0)
    return (
        float(np.max(np.abs(increments))),
        float(np.sum(np.abs(increments))),
        float(np.sum(np.square(increments))),
    )


def _majority_accuracy(rows: List[tuple[int, int, int]], feature: int) -> float:
    """Best label accuracy from one generator-audit feature alone."""
    counts: Dict[int, List[int]] = {}
    for row in rows:
        feature_value = row[feature]
        label = row[2]
        counts.setdefault(feature_value, [0, 0])[label] += 1
    return sum(max(label_counts) for label_counts in counts.values()) / len(rows)


def audit_suite(
    config: Mapping[str, Any], suite: Mapping[str, Any], tolerance: float = 1e-7
) -> Dict[str, Any]:
    """Check that the suite cannot be passed by endpoint or global-onset shortcuts."""
    validate_config(config)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    history = int(config["history_length"])
    horizon = int(config["horizon_length"])
    target = int(config["episodes"]["target_channel"])
    normal_onsets = np.asarray(config["episodes"]["normal_transition_onsets"], dtype=np.int64)
    incompatible_onsets = np.asarray(
        config["episodes"]["incompatible_transition_onsets"], dtype=np.int64
    )
    grouped = _episodes_by_source(suite)
    violations: List[str] = []
    normal_observed_onsets: List[int] = []
    incompatible_observed_onsets: List[int] = []
    primary_rows: List[tuple[int, int, int]] = []
    cue_observability_rows: List[tuple[int, float]] = []
    cue_channel = int(config["episodes"]["cue_channel"])
    cue_length = int(config["episodes"]["cue_length"])
    cue_encoding = str(config["episodes"].get("cue_encoding", "additive_v1"))
    cue_amplitudes = np.asarray(config["episodes"]["cue_amplitudes"], dtype=np.float64)
    cue_amplitude_tolerance = float(config["episodes"].get("cue_amplitude_tolerance", 0.0))
    cue_amplitude_errors: List[float] = []
    expected_split_names = ("optimization", "validation", "reference", "outer_calibration")
    expected_split_ranges: Dict[str, tuple[int, int]] = {}
    split_start = 0
    guard_length = int(config["normal_splits"]["guard_length"])
    for index, name in enumerate(expected_split_names):
        split_end = split_start + int(config["normal_splits"][f"{name}_length"])
        expected_split_ranges[name] = (split_start, split_end)
        split_start = split_end + (guard_length if index < len(expected_split_names) - 1 else 0)
    if dict(suite["normal_split_ranges"]) != expected_split_ranges:
        violations.append("normal split ranges differ from the frozen config")

    for source_id, roles in sorted(grouped.items()):
        if set(roles) != set(ROLE_ORDER):
            violations.append(f"{source_id}: incomplete role set")
            continue
        scheduled = roles["normal_scheduled_transition"]
        incompatible = roles["incompatible_timing_transition"]
        coordinated = roles["normal_coordinated_transition"]
        unsupported = roles["unsupported_transition"]
        scheduled_values = np.asarray(scheduled["values"], dtype=np.float64)
        incompatible_values = np.asarray(incompatible["values"], dtype=np.float64)
        coordinated_values = np.asarray(coordinated["values"], dtype=np.float64)
        unsupported_values = np.asarray(unsupported["values"], dtype=np.float64)

        if not np.allclose(
            scheduled_values[:history], incompatible_values[:history], atol=tolerance, rtol=0.0
        ):
            violations.append(f"{source_id}: primary event-pre state differs")
        if not np.allclose(
            scheduled_values[-1], incompatible_values[-1], atol=tolerance, rtol=0.0
        ):
            violations.append(f"{source_id}: primary endpoint differs")
        if np.max(np.abs(scheduled_values[history:] - incompatible_values[history:])) <= tolerance:
            violations.append(f"{source_id}: primary trajectories do not differ")
        scheduled_statistics = _increment_statistics(scheduled_values[history:])
        incompatible_statistics = _increment_statistics(incompatible_values[history:])
        if any(
            abs(first - second) > tolerance
            for first, second in zip(scheduled_statistics, incompatible_statistics)
        ):
            violations.append(f"{source_id}: primary observed trajectory summaries differ")

        cue_mode = int(scheduled["cue_mode"])
        cue_values = scheduled_values[:history, cue_channel]
        cue_delta = float(cue_values[-1] - cue_values[-cue_length])
        cue_observability_rows.append((cue_mode, cue_delta))
        if cue_encoding == "anchored_overwrite_v1":
            cue_amplitude_errors.append(abs(cue_delta - float(cue_amplitudes[cue_mode])))
        expected_onset = int(scheduled["expected_onset"])
        scheduled_onset = int(scheduled["observed_onset"])
        incompatible_onset = int(incompatible["observed_onset"])
        if expected_onset != int(normal_onsets[cue_mode]) or scheduled_onset != expected_onset:
            violations.append(f"{source_id}: normal timing does not follow its cue")
        if int(incompatible["expected_onset"]) != expected_onset:
            violations.append(f"{source_id}: primary expected onset is not tied")
        if incompatible_onset != int(incompatible_onsets[cue_mode]):
            violations.append(f"{source_id}: incompatible timing does not follow its mapping")
        if incompatible_onset == expected_onset:
            violations.append(f"{source_id}: incompatible timing is cue-compatible")
        normal_observed_onsets.append(scheduled_onset)
        incompatible_observed_onsets.append(incompatible_onset)
        primary_rows.extend(
            (
                (cue_mode, scheduled_onset, 0),
                (cue_mode, incompatible_onset, 1),
            )
        )

        if not np.allclose(
            coordinated_values[:history], unsupported_values[:history], atol=tolerance, rtol=0.0
        ):
            violations.append(f"{source_id}: coordination event-pre state differs")
        if not np.allclose(
            coordinated_values[:, target], unsupported_values[:, target], atol=tolerance, rtol=0.0
        ):
            violations.append(f"{source_id}: coordination target trajectory differs")
        drivers = [index for index in range(coordinated_values.shape[1]) if index != target]
        if np.max(np.abs(coordinated_values[history:, drivers] - unsupported_values[history:, drivers])) <= tolerance:
            violations.append(f"{source_id}: coordination drivers do not differ")

    if sorted(normal_observed_onsets) != sorted(incompatible_observed_onsets):
        violations.append("primary onset marginals differ, enabling a global time shortcut")
    cue_only_accuracy = _majority_accuracy(primary_rows, feature=0)
    onset_only_accuracy = _majority_accuracy(primary_rows, feature=1)
    conditional_accuracy = _majority_accuracy(
        [(cue_mode * 1000 + onset, onset, label) for cue_mode, onset, label in primary_rows],
        feature=0,
    )
    if cue_only_accuracy > 0.5 + tolerance:
        violations.append("primary cue alone predicts the label above chance")
    if onset_only_accuracy > 0.5 + tolerance:
        violations.append("primary onset alone predicts the label above chance")
    if conditional_accuracy < 1.0 - tolerance:
        violations.append("primary cue-onset mapping does not determine the intended label")
    cue_labels = np.asarray([cue_mode for cue_mode, _ in cue_observability_rows], dtype=np.int64)
    cue_deltas = np.asarray([cue_delta for _, cue_delta in cue_observability_rows])
    cue_observability_accuracy = float(np.mean((cue_deltas > 0.0).astype(np.int64) == cue_labels))
    cue_observability_margin = float(
        min(
            np.min(cue_deltas[cue_labels == 1]),
            -np.max(cue_deltas[cue_labels == 0]),
        )
    )
    if cue_observability_accuracy < 1.0 - tolerance or cue_observability_margin <= tolerance:
        violations.append("event-pre cue is not observable with its predeclared raw-state rule")
    maximum_cue_amplitude_error = float(max(cue_amplitude_errors, default=0.0))
    if (
        cue_encoding == "anchored_overwrite_v1"
        and maximum_cue_amplitude_error > cue_amplitude_tolerance
    ):
        violations.append("anchored event-pre cue does not equal its predeclared raw amplitude")

    for contract in suite["contracts"]:
        source_id = str(contract["source_id"])
        for key in (
            "event_pre_max_abs_difference",
            "primary_endpoint_max_abs_difference",
            "primary_increment_max_abs_difference",
            "primary_increment_l1_abs_difference",
            "primary_increment_l2_abs_difference",
            "coordination_target_trajectory_max_abs_difference",
        ):
            if float(contract[key]) > tolerance:
                violations.append(f"{source_id}: {key} exceeds tolerance")
        if float(contract["primary_trajectory_max_abs_difference"]) <= tolerance:
            violations.append(f"{source_id}: recorded primary trajectory difference is zero")
        if float(contract["coordination_driver_trajectory_max_abs_difference"]) <= tolerance:
            violations.append(f"{source_id}: recorded coordination driver difference is zero")
        if float(contract["coordination_error_target_std"]) < float(
            contract["minimum_coordination_error_target_std"]
        ):
            violations.append(f"{source_id}: coordination support gap is too small")

    banks = dict(suite["normal_transition_banks"])
    expected_bank_counts = config["episodes"]["normal_transition_sources_per_regime"]
    for split_name, expected_range in expected_split_ranges.items():
        bank = list(banks.get(split_name, []))
        expected_count = 6 * int(expected_bank_counts[split_name])
        if len(bank) != expected_count:
            violations.append(f"normal {split_name} transition bank has an unexpected size")
        roles = {str(episode["role"]) for episode in bank}
        if roles != {
            "normal_scheduled_transition",
            "normal_coordinated_transition",
            "no_event_normal_control",
        }:
            violations.append("normal transition bank does not contain every normal trajectory role")
        for episode in bank:
            if str(episode["split"]) != split_name:
                violations.append("normal transition bank split label is inconsistent")
            if str(episode["role"]) != "no_event_normal_control":
                cue_mode = int(episode["cue_mode"])
                if int(episode["expected_onset"]) != int(normal_onsets[cue_mode]):
                    violations.append("normal transition timing does not follow its cue")
            elif episode["cue_mode"] is not None or episode["expected_onset"] is not None:
                violations.append("no-event normal bank episode contains a timing label")
            source_start = int(episode["source_start"])
            split_start, split_end = expected_range
            if source_start - history < split_start or source_start + horizon > split_end:
                violations.append("normal transition window crosses its time-disjoint split")
    references = list(banks.get("reference", []))
    if list(suite["normal_transition_references"]) != references:
        violations.append("normal transition reference alias differs from the reference bank")

    return {
        "passed": not violations,
        "source_count": len(grouped),
        "episode_count": len(suite["episodes"]),
        "normal_transition_reference_count": len(references),
        "normal_transition_bank_counts": {
            split_name: len(list(banks.get(split_name, [])))
            for split_name in expected_split_ranges
        },
        "normal_split_ranges": {
            name: list(value) for name, value in expected_split_ranges.items()
        },
        "primary_normal_onset_marginal": sorted(normal_observed_onsets),
        "primary_incompatible_onset_marginal": sorted(incompatible_observed_onsets),
        "primary_cue_only_majority_accuracy": cue_only_accuracy,
        "primary_onset_only_majority_accuracy": onset_only_accuracy,
        "primary_cue_onset_majority_accuracy": conditional_accuracy,
        "event_pre_cue_observability_accuracy": cue_observability_accuracy,
        "event_pre_cue_observability_margin": cue_observability_margin,
        "event_pre_cue_encoding": cue_encoding,
        "event_pre_cue_maximum_amplitude_error": maximum_cue_amplitude_error,
        "event_pre_cue_amplitude_tolerance": cue_amplitude_tolerance,
        "violations": violations,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    arguments = parser.parse_args(argv)
    config = _load_json(arguments.config)
    result = audit_suite(config, generate_suite(config), tolerance=arguments.tolerance)
    print(json.dumps(result, sort_keys=True, ensure_ascii=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
