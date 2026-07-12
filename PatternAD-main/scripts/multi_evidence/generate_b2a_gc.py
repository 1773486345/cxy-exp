#!/usr/bin/env python3
"""Generate B2a-GC's terminally valid drift-rotated counterfactual suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "multi_evidence" / "b2a_gc_drift_rotation.json"

from scripts.multi_evidence.generate_b2a_holdout import (  # noqa: E402
    DRIFT_CONTROL_ROLE,
    PAIR_ROLE_ORDER,
    _canonical_hash,
    _file_sha256,
    _load_json,
    _phase_candidates,
    _select_drift_controls,
    _simulate_drift_var,
    validate_config as _validate_b2a_base,
)


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate B2a-GC plus the unchanged B2a process and evaluation fields."""
    _validate_b2a_base(config)
    if not str(config.get("suite_id", "")).startswith("multi_evidence_b2a_gc_"):
        raise ValueError("B2a-GC suite_id must start with 'multi_evidence_b2a_gc_'.")
    contract = config.get("counterfactual_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("B2a-GC requires counterfactual_contract.")
    for name in (
        "maximum_relation_value_delta",
        "minimum_structural_cross_gap_target_std",
        "minimum_terminal_target_gap_target_std",
    ):
        if float(contract.get(name, 0.0)) <= 0.0:
            raise ValueError(f"B2a-GC counterfactual_contract.{name} must be positive.")
    if float(contract["maximum_relation_value_delta"]) > 2.0:
        raise ValueError("B2a-GC relation matching tolerance is implausibly broad.")
    if int(contract.get("source_candidates_per_chronological_block", 0)) < 1:
        raise ValueError(
            "B2a-GC source_candidates_per_chronological_block must be positive."
        )


def _structural_cross_support(
    values: np.ndarray,
    factors: np.ndarray,
    relation_value: np.ndarray,
    process: Mapping[str, Any],
    target_index: int,
    terminal_index: int,
) -> float:
    """Generator-side target support excluding the target's self-lag term."""
    return float(
        _structural_cross_support_many(
            values,
            factors,
            relation_value,
            process,
            target_index,
            np.asarray([terminal_index], dtype=np.int64),
        )[0]
    )


def _structural_cross_support_many(
    values: np.ndarray,
    factors: np.ndarray,
    relation_value: np.ndarray,
    process: Mapping[str, Any],
    target_index: int,
    terminal_indices: np.ndarray,
) -> np.ndarray:
    """Vectorized generator-side support excluding the target self-lag."""
    indices = np.asarray(terminal_indices, dtype=np.int64).reshape(-1)
    if not len(indices) or np.any(indices < 1):
        raise ValueError("B2a-GC structural support needs non-empty prior indices.")
    if np.any(indices >= len(values)):
        raise ValueError("B2a-GC structural support index is outside the stream.")
    relation = np.asarray(relation_value, dtype=np.float64)[indices]
    lag_base = np.asarray(process["lag_base"], dtype=np.float64)[target_index]
    lag_drift = np.asarray(process["lag_drift"], dtype=np.float64)[target_index]
    loading_base = np.asarray(process["loading_base"], dtype=np.float64)[target_index]
    loading_drift = np.asarray(process["loading_drift"], dtype=np.float64)[target_index]
    drivers = np.ones(values.shape[1], dtype=bool)
    drivers[target_index] = False
    lag = lag_base[drivers][None, :] + relation[:, None] * lag_drift[drivers][None, :]
    loading = loading_base[None, :] + relation[:, None] * loading_drift[None, :]
    return (
        np.sum(lag * values[indices - 1][:, drivers], axis=1)
        + np.sum(loading * factors[indices], axis=1)
    ).astype(np.float64, copy=False)


def _legacy_structural_cross_support(
    values: np.ndarray,
    factors: np.ndarray,
    relation_value: np.ndarray,
    process: Mapping[str, Any],
    target_index: int,
    terminal_index: int,
) -> float:
    """Reference scalar formula retained for direct unit-test comparison."""
    if terminal_index < 1:
        raise ValueError("B2a-GC structural support needs a prior point.")
    relation = float(relation_value[terminal_index])
    lag = np.asarray(process["lag_base"], dtype=np.float64) + relation * np.asarray(
        process["lag_drift"], dtype=np.float64
    )
    loading = np.asarray(process["loading_base"], dtype=np.float64) + relation * np.asarray(
        process["loading_drift"], dtype=np.float64
    )
    drivers = np.ones(values.shape[1], dtype=bool)
    drivers[target_index] = False
    return float(
        np.dot(lag[target_index, drivers], values[terminal_index - 1, drivers])
        + np.dot(loading[target_index], factors[terminal_index])
    )


def _select_contract_donor(
    values: np.ndarray,
    factors: np.ndarray,
    phase_bins: np.ndarray,
    relation_value: np.ndarray,
    target_std: float,
    process: Mapping[str, Any],
    history: int,
    target_index: int,
    source_terminal_index: int,
    contract: Mapping[str, Any],
    forbidden_terminal_indices: Sequence[int] = (),
) -> Tuple[int, Dict[str, float]]:
    """Choose a locally matched donor with certified terminal incompatibility."""
    candidates = _phase_candidates(
        phase_bins, history, int(phase_bins[source_terminal_index])
    )
    candidates = candidates[np.abs(candidates - source_terminal_index) > history]
    if forbidden_terminal_indices:
        candidates = candidates[
            ~np.isin(candidates, np.asarray(forbidden_terminal_indices, dtype=np.int64))
        ]
    relation_delta = np.abs(relation_value[candidates] - relation_value[source_terminal_index])
    candidates = candidates[
        relation_delta <= float(contract["maximum_relation_value_delta"])
    ]
    if not len(candidates):
        raise ValueError(
            f"B2a-GC found no locally relation-matched donor for target {target_index} "
            f"at source {source_terminal_index}."
        )
    source_support = _structural_cross_support(
        values, factors, relation_value, process, target_index, source_terminal_index
    )
    source_target = float(values[source_terminal_index, target_index])
    donor_support = _structural_cross_support_many(
        values, factors, relation_value, process, target_index, candidates
    )
    structural_gap = np.abs(source_support - donor_support) / target_std
    terminal_target_gap = np.abs(values[candidates, target_index] - source_target) / target_std
    local_relation_gap = np.abs(
        relation_value[candidates] - relation_value[source_terminal_index]
    )
    qualified = (
        structural_gap >= float(contract["minimum_structural_cross_gap_target_std"])
    ) & (
        terminal_target_gap >= float(contract["minimum_terminal_target_gap_target_std"])
    )
    if not np.any(qualified):
        raise ValueError(
            f"B2a-GC found no donor satisfying the terminal contract for target "
            f"{target_index} at source {source_terminal_index}."
        )
    candidate_indices = np.flatnonzero(qualified)
    # The local relation match has priority; structural strength breaks ties.
    selected_position = candidate_indices[
        np.lexsort(
            (
                candidates[candidate_indices],
                -terminal_target_gap[candidate_indices],
                -structural_gap[candidate_indices],
                -np.minimum(
                    structural_gap[candidate_indices],
                    terminal_target_gap[candidate_indices],
                ),
                local_relation_gap[candidate_indices],
            )
        )[0]
    ]
    donor = int(candidates[selected_position])
    diagnostics = {
        "source_structural_cross_support": float(source_support),
        "donor_structural_cross_support": float(donor_support[selected_position]),
        "structural_cross_gap_target_std": float(structural_gap[selected_position]),
        "terminal_target_gap_target_std": float(terminal_target_gap[selected_position]),
        "relation_value_abs_difference": float(local_relation_gap[selected_position]),
    }
    return donor, diagnostics


def _select_contract_pairs(
    values: np.ndarray,
    factors: np.ndarray,
    phase_bins: np.ndarray,
    relation_value: np.ndarray,
    target_std: float,
    process: Mapping[str, Any],
    history: int,
    target_index: int,
    phase_bin: int,
    count: int,
    contract: Mapping[str, Any],
) -> List[Tuple[int, int, Dict[str, float], int]]:
    """Select one valid pair in each chronological phase block.

    Fixed evenly spaced sources can land at a normal state with no donor that
    satisfies a strong terminal contract. Source selection is therefore still
    generator-only, but picks one valid pair from each chronological block so
    every phase retains temporal coverage.
    """
    source_candidates = _phase_candidates(phase_bins, history, phase_bin)
    blocks = np.array_split(source_candidates, count)
    candidates_per_block = int(contract["source_candidates_per_chronological_block"])
    selections: List[Tuple[int, int, Dict[str, float], int]] = []
    used_terminals: set[int] = set()
    for block_index, block in enumerate(blocks):
        candidates: List[
            Tuple[Tuple[float, float, int, int], int, int, Dict[str, float]]
        ] = []
        positions = np.linspace(
            0, len(block) - 1, min(len(block), candidates_per_block), dtype=np.int64
        )
        for source_terminal_index in block[np.unique(positions)]:
            source = int(source_terminal_index)
            if source in used_terminals:
                continue
            try:
                donor, diagnostics = _select_contract_donor(
                    values,
                    factors,
                    phase_bins,
                    relation_value,
                    target_std,
                    process,
                    history,
                    target_index,
                    source,
                    contract,
                    forbidden_terminal_indices=tuple(used_terminals),
                )
            except ValueError:
                continue
            rank = (
                -min(
                    diagnostics["structural_cross_gap_target_std"],
                    diagnostics["terminal_target_gap_target_std"],
                ),
                diagnostics["relation_value_abs_difference"],
                source,
                donor,
            )
            candidates.append((rank, source, donor, diagnostics))
        if not candidates:
            raise ValueError(
                f"B2a-GC found no valid source/donor pair for target {target_index}, "
                f"phase {phase_bin}, chronological block {block_index}."
            )
        _, source, donor, diagnostics = min(candidates, key=lambda item: item[0])
        selections.append((source, donor, diagnostics, block_index))
        used_terminals.update((source, donor))
    return selections


def generate_suite(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Generate normal streams and all target-specific valid counterfactuals."""
    validate_config(config)
    rng = np.random.default_rng(int(config["seed"]))
    train = _simulate_drift_var(int(config["train_length"]), config, rng)
    background = _simulate_drift_var(int(config["test_length"]), config, rng)
    history = int(config["history_length"])
    dimensions = int(config["dimensions"])
    targets = tuple(int(value) for value in config["target_indices"])
    episodes_config = config["episodes"]
    contract = config["counterfactual_contract"]
    phase_count = int(episodes_config["phase_bins"])
    pairs_per_phase = int(episodes_config["pairs_per_phase"])
    controls_per_phase = int(episodes_config["drift_controls_per_phase"])
    target_std = train["values"].std(axis=0, dtype=np.float64).astype(np.float32)
    episodes: List[Dict[str, Any]] = []
    contracts: List[Dict[str, Any]] = []
    source_values = background["values"]
    for target_index in targets:
        drivers = np.asarray(
            [index for index in range(dimensions) if index != target_index], dtype=np.int64
        )
        for phase_bin in range(phase_count):
            selections = _select_contract_pairs(
                source_values,
                background["factors"],
                background["relation_phase_bin"],
                background["relation_value"],
                float(target_std[target_index]),
                config["normal_process"],
                history,
                target_index,
                phase_bin,
                pairs_per_phase,
                contract,
            )
            for ordinal, (source_terminal, donor_terminal, diagnostics, block_index) in enumerate(selections):
                coherent = source_values[source_terminal - history : source_terminal + 1].copy()
                donor = source_values[donor_terminal - history : donor_terminal + 1].copy()
                unsupported = coherent.copy()
                unsupported[:, drivers] = donor[:, drivers]
                omission = coherent.copy()
                omission[:, target_index] = donor[:, target_index]
                spike = coherent.copy()
                spike_sign = 1.0 if coherent[-1, target_index] >= 0.0 else -1.0
                spike[-1, target_index] += (
                    spike_sign
                    * float(episodes_config["target_spike_multiplier"])
                    * float(target_std[target_index])
                )
                pair_id = f"target_{target_index}_phase_{phase_bin}_pair_{ordinal:02d}"
                role_values = {
                    "coherent_control": coherent,
                    "unsupported_target_break": unsupported,
                    "target_omission_break": omission,
                    "target_spike": spike,
                }
                for role in PAIR_ROLE_ORDER:
                    episodes.append(
                        {
                            "pair_id": pair_id,
                            "role": role,
                            "is_pair": True,
                            "target_index": target_index,
                            "phase_bin": phase_bin,
                            "source_terminal_index": source_terminal,
                            "donor_terminal_index": donor_terminal,
                            "source_selection_block": block_index,
                            "relation_value": float(background["relation_value"][source_terminal]),
                            "relation_velocity": float(background["relation_velocity"][source_terminal]),
                            "values": role_values[role].astype(np.float32),
                        }
                    )
                contracts.append(
                    {
                        "pair_id": pair_id,
                        "target_index": target_index,
                        "phase_bin": phase_bin,
                        "source_terminal_index": source_terminal,
                        "donor_terminal_index": donor_terminal,
                        "source_selection_block": block_index,
                        "source_donor_terminal_distance": int(
                            abs(source_terminal - donor_terminal)
                        ),
                        **diagnostics,
                        "coherent_unsupported_target_max_abs_difference": float(
                            np.max(
                                np.abs(
                                    coherent[:, target_index]
                                    - unsupported[:, target_index]
                                )
                            )
                        ),
                        "coherent_omission_driver_max_abs_difference": float(
                            np.max(np.abs(coherent[:, drivers] - omission[:, drivers]))
                        ),
                        "coherent_unsupported_driver_terminal_max_abs_difference": float(
                            np.max(
                                np.abs(
                                    coherent[-1, drivers] - unsupported[-1, drivers]
                                )
                            )
                        ),
                        "coherent_omission_target_terminal_abs_difference": float(
                            abs(coherent[-1, target_index] - omission[-1, target_index])
                        ),
                    }
                )
            control_indices = _select_drift_controls(
                background["relation_phase_bin"],
                background["relation_velocity"],
                history,
                phase_bin,
                controls_per_phase,
            )
            for ordinal, terminal_index in enumerate(control_indices):
                episodes.append(
                    {
                        "pair_id": f"target_{target_index}_phase_{phase_bin}_drift_{ordinal:02d}",
                        "role": DRIFT_CONTROL_ROLE,
                        "is_pair": False,
                        "target_index": target_index,
                        "phase_bin": phase_bin,
                        "source_terminal_index": int(terminal_index),
                        "donor_terminal_index": -1,
                        "relation_value": float(background["relation_value"][terminal_index]),
                        "relation_velocity": float(background["relation_velocity"][terminal_index]),
                        "values": source_values[terminal_index - history : terminal_index + 1].copy(),
                    }
                )
    return {
        "train": train,
        "background": background,
        "target_indices": targets,
        "episodes": episodes,
        "contracts": contracts,
        "target_std": target_std,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_suite(config: Mapping[str, Any], suite: Mapping[str, Any], output_dir: Path) -> Path:
    """Persist inputs and generator-only terminal-contract diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=False)
    episodes = list(suite["episodes"])
    np.savez_compressed(
        output_dir / "normal_streams.npz",
        train_values=np.asarray(suite["train"]["values"], dtype=np.float32),
        background_values=np.asarray(suite["background"]["values"], dtype=np.float32),
        background_phase_bin=np.asarray(suite["background"]["relation_phase_bin"], dtype=np.int64),
        background_relation_value=np.asarray(suite["background"]["relation_value"], dtype=np.float32),
        background_relation_velocity=np.asarray(suite["background"]["relation_velocity"], dtype=np.float32),
    )
    np.savez_compressed(
        output_dir / "episodes.npz",
        windows=np.stack([episode["values"] for episode in episodes], axis=0),
        pair_ids=np.asarray([episode["pair_id"] for episode in episodes]),
        roles=np.asarray([episode["role"] for episode in episodes]),
        is_pair=np.asarray([episode["is_pair"] for episode in episodes], dtype=np.uint8),
        target_indices=np.asarray([episode["target_index"] for episode in episodes], dtype=np.int64),
        phase_bins=np.asarray([episode["phase_bin"] for episode in episodes], dtype=np.int64),
        source_terminal_indices=np.asarray(
            [episode["source_terminal_index"] for episode in episodes], dtype=np.int64
        ),
        donor_terminal_indices=np.asarray(
            [episode["donor_terminal_index"] for episode in episodes], dtype=np.int64
        ),
    )
    _write_json(output_dir / "resolved_config.json", dict(config))
    manifest = {
        "suite_id": str(config["suite_id"]),
        "config_hash": _canonical_hash(config),
        "generator_sha256": _file_sha256(Path(__file__)),
        "target_indices": list(suite["target_indices"]),
        "episode_count": len(episodes),
        "pair_role_order": list(PAIR_ROLE_ORDER),
        "drift_control_role": DRIFT_CONTROL_ROLE,
        "counterfactual_contract": dict(config["counterfactual_contract"]),
        "contracts": list(suite["contracts"]),
        "target_std_from_normal_train": np.asarray(suite["target_std"], dtype=float).tolist(),
        "phase_metadata_used_by_model_or_calibration": False,
        "latent_factors_used_by_model_or_calibration": False,
        "structural_support_used_by_model_or_calibration": False,
    }
    _write_json(output_dir / "suite_metadata.json", manifest)
    return output_dir / "suite_metadata.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config = _load_json(arguments.config)
    suite = generate_suite(config)
    manifest = write_suite(config, suite, arguments.output_dir)
    print(f"Wrote B2a-GC held-out suite: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
