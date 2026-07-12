#!/usr/bin/env python3
"""Generate B2a's held-out drift-rotated multi-target mechanism suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "multi_evidence" / "b2a_drift_rotation.json"
PAIR_ROLE_ORDER = (
    "coherent_control",
    "unsupported_target_break",
    "target_omission_break",
    "target_spike",
)
DRIFT_CONTROL_ROLE = "normal_relation_drift_control"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return array


def validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("B2a config must use schema_version=1.")
    dimensions = int(config["dimensions"])
    history = int(config["history_length"])
    if dimensions < 3 or history < 2:
        raise ValueError("B2a requires at least three channels and history >= 2.")
    if len(config.get("channel_names", [])) != dimensions:
        raise ValueError("channel_names must match dimensions.")
    targets = tuple(int(value) for value in config.get("target_indices", []))
    if len(targets) != dimensions or tuple(sorted(targets)) != tuple(range(dimensions)):
        raise ValueError("B2a must rotate over every target channel exactly once.")
    if int(config["train_length"]) <= 10 * history or int(config["test_length"]) <= 8 * history:
        raise ValueError("B2a train/test lengths are too short.")
    process = config["normal_process"]
    _array(process["factor_transition"], (2, 2), "factor_transition")
    factor_radius = float(
        np.max(np.abs(np.linalg.eigvals(np.asarray(process["factor_transition"]))))
    )
    if factor_radius >= 0.98:
        raise ValueError("factor_transition is not safely stable.")
    cholesky = _array(
        process["factor_noise_cholesky"], (2, 2), "factor_noise_cholesky"
    )
    if np.any(np.diag(cholesky) <= 0.0):
        raise ValueError("factor_noise_cholesky must have positive diagonal.")
    for name in ("lag_base", "lag_drift"):
        matrix = _array(process[name], (dimensions, dimensions), name)
        if np.max(np.sum(np.abs(matrix), axis=1)) > 0.30:
            raise ValueError(f"{name} row absolute sum is too large for B2a.")
    _array(process["loading_base"], (dimensions, 2), "loading_base")
    _array(process["loading_drift"], (dimensions, 2), "loading_drift")
    noise = _array(process["observation_noise"], (dimensions,), "observation_noise")
    if np.any(noise <= 0.0):
        raise ValueError("observation_noise must be positive.")
    if int(process["relation_period"]) < 4 * history:
        raise ValueError("relation_period must exceed four history lengths.")
    episodes = config["episodes"]
    phase_bins = int(episodes["phase_bins"])
    pairs_per_phase = int(episodes["pairs_per_phase"])
    controls_per_phase = int(episodes["drift_controls_per_phase"])
    if phase_bins != 4 or pairs_per_phase < 1 or controls_per_phase < 1:
        raise ValueError("B2a requires four phase bins and positive episode counts.")
    if float(episodes["target_spike_multiplier"]) <= 0.0:
        raise ValueError("target_spike_multiplier must be positive.")
    fractions = [float(config["split"][key]) for key in (
        "outer_calibration_fraction", "validation_fraction", "reference_fraction"
    )]
    if any(value <= 0.0 for value in fractions) or sum(fractions) >= 0.8:
        raise ValueError("B2a split fractions leave too little optimization data.")
    model = config["model"]
    if int(model["d_model"]) < 1 or int(model["batch_size"]) < 1:
        raise ValueError("B2a model configuration is invalid.")
    if float(model["learning_rate"]) <= 0.0 or int(model["epochs"]) < 1:
        raise ValueError("B2a model configuration is invalid.")
    if float(model.get("dropout", 0.0)) != 0.0:
        raise ValueError("B2a forbids dropout.")
    calibration = config["calibration"]
    if calibration.get("mode") != "input_energy_stratified":
        raise ValueError("B2a requires B1 input-energy reliability calibration.")
    if int(calibration["reliability_strata"]) != 3:
        raise ValueError("B2a freezes three reliability strata.")
    if int(calibration["min_reference_per_stratum"]) < 50:
        raise ValueError("B2a requires at least 50 reference samples per stratum.")
    evaluation = config["evaluation"]
    total_pairs = phase_bins * pairs_per_phase
    if not 1 <= int(evaluation["paired_order_min"]) <= total_pairs:
        raise ValueError("paired_order_min is incompatible with B2a episode count.")
    if not 1 <= int(evaluation["target_spike_exceedance_min"]) <= total_pairs:
        raise ValueError("target_spike_exceedance_min is incompatible with B2a pairs.")
    for name in ("coherent_exceedance_max", "drift_control_exceedance_max"):
        if not 0 <= int(evaluation[name]) <= total_pairs:
            raise ValueError(f"{name} is incompatible with B2a episode count.")


def _simulate_drift_var(
    length: int, config: Mapping[str, Any], rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    process = config["normal_process"]
    dimensions = int(config["dimensions"])
    burn_in = int(config["burn_in"])
    total = length + burn_in
    period = int(process["relation_period"])
    indexes = np.arange(total, dtype=np.float64)
    relation_value = np.sin(2.0 * np.pi * indexes / float(period))
    relation_phase = np.mod(2.0 * np.pi * indexes / float(period), 2.0 * np.pi)
    relation_phase_bin = np.floor(4.0 * relation_phase / (2.0 * np.pi)).astype(np.int64)
    relation_velocity = np.abs(
        np.cos(2.0 * np.pi * indexes / float(period))
        * 2.0
        * np.pi
        / float(period)
    )
    factor_transition = np.asarray(process["factor_transition"], dtype=np.float64)
    factor_cholesky = np.asarray(process["factor_noise_cholesky"], dtype=np.float64)
    lag_base = np.asarray(process["lag_base"], dtype=np.float64)
    lag_drift = np.asarray(process["lag_drift"], dtype=np.float64)
    loading_base = np.asarray(process["loading_base"], dtype=np.float64)
    loading_drift = np.asarray(process["loading_drift"], dtype=np.float64)
    observation_noise = np.asarray(process["observation_noise"], dtype=np.float64)
    values = np.zeros((total, dimensions), dtype=np.float64)
    factors = np.zeros((total, 2), dtype=np.float64)
    for index in range(total):
        previous_factor = factors[index - 1] if index else np.zeros(2, dtype=np.float64)
        previous_values = values[index - 1] if index else np.zeros(dimensions, dtype=np.float64)
        factors[index] = (
            factor_transition @ previous_factor
            + factor_cholesky @ rng.normal(size=2)
        )
        relation = relation_value[index]
        lag = lag_base + relation * lag_drift
        loading = loading_base + relation * loading_drift
        values[index] = (
            lag @ previous_values
            + loading @ factors[index]
            + observation_noise * rng.normal(size=dimensions)
        )
    retained = slice(burn_in, None)
    return {
        "values": values[retained].astype(np.float32),
        "factors": factors[retained].astype(np.float32),
        "relation_value": relation_value[retained].astype(np.float32),
        "relation_phase_bin": relation_phase_bin[retained].astype(np.int64),
        "relation_velocity": relation_velocity[retained].astype(np.float32),
    }


def _phase_candidates(
    phase_bins: np.ndarray, history: int, phase_bin: int
) -> np.ndarray:
    indices = np.arange(len(phase_bins))
    return np.flatnonzero((indices >= history) & (phase_bins == phase_bin))


def _select_source_indices(
    phase_bins: np.ndarray,
    history: int,
    phase_bin: int,
    count: int,
    offset: int,
) -> np.ndarray:
    candidates = _phase_candidates(phase_bins, history, phase_bin)
    if len(candidates) < count:
        raise ValueError("B2a test stream cannot provide requested source episodes.")
    positions = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
    return candidates[(positions + offset) % len(candidates)]


def _select_donor(
    factors: np.ndarray,
    phase_bins: np.ndarray,
    relation_value: np.ndarray,
    history: int,
    source_terminal_index: int,
) -> int:
    candidates = _phase_candidates(
        phase_bins, history, int(phase_bins[source_terminal_index])
    )
    candidates = candidates[np.abs(candidates - source_terminal_index) > history]
    if len(candidates) == 0:
        raise ValueError("B2a cannot find a non-overlapping donor.")
    factor_distance = np.linalg.norm(
        factors[candidates] - factors[source_terminal_index], axis=1
    )
    relation_difference = np.abs(
        relation_value[candidates] - relation_value[source_terminal_index]
    )
    # Generator-only selection holds drift phase approximately matched while
    # selecting a different normal factor realization.
    score = factor_distance - 0.25 * relation_difference
    return int(candidates[int(np.argmax(score))])


def _select_drift_controls(
    phase_bins: np.ndarray,
    velocity: np.ndarray,
    history: int,
    phase_bin: int,
    count: int,
) -> np.ndarray:
    candidates = _phase_candidates(phase_bins, history, phase_bin)
    if len(candidates) < count:
        raise ValueError("B2a cannot select normal drift controls.")
    order = np.argsort(velocity[candidates])[::-1]
    return candidates[order[:count]]


def generate_suite(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Generate B2a normal streams and per-target paired counterfactuals."""
    validate_config(config)
    rng = np.random.default_rng(int(config["seed"]))
    train = _simulate_drift_var(int(config["train_length"]), config, rng)
    background = _simulate_drift_var(int(config["test_length"]), config, rng)
    history = int(config["history_length"])
    dimensions = int(config["dimensions"])
    targets = tuple(int(value) for value in config["target_indices"])
    episodes_config = config["episodes"]
    phase_count = int(episodes_config["phase_bins"])
    pairs_per_phase = int(episodes_config["pairs_per_phase"])
    controls_per_phase = int(episodes_config["drift_controls_per_phase"])
    spike_std = train["values"].std(axis=0, dtype=np.float64).astype(np.float32)
    episodes: List[Dict[str, Any]] = []
    contracts: List[Dict[str, Any]] = []
    source_values = background["values"]
    for target_index in targets:
        drivers = np.asarray(
            [index for index in range(dimensions) if index != target_index], dtype=np.int64
        )
        for phase_bin in range(phase_count):
            source_indices = _select_source_indices(
                background["relation_phase_bin"],
                history,
                phase_bin,
                pairs_per_phase,
                offset=target_index * pairs_per_phase,
            )
            for ordinal, terminal_index in enumerate(source_indices):
                coherent = source_values[
                    terminal_index - history : terminal_index + 1
                ].copy()
                donor_terminal_index = _select_donor(
                    background["factors"],
                    background["relation_phase_bin"],
                    background["relation_value"],
                    history,
                    int(terminal_index),
                )
                donor = source_values[
                    donor_terminal_index - history : donor_terminal_index + 1
                ].copy()
                unsupported = coherent.copy()
                unsupported[:, drivers] = donor[:, drivers]
                omission = coherent.copy()
                omission[:, target_index] = donor[:, target_index]
                spike = coherent.copy()
                spike_sign = 1.0 if coherent[-1, target_index] >= 0.0 else -1.0
                spike[-1, target_index] += (
                    spike_sign
                    * float(episodes_config["target_spike_multiplier"])
                    * float(spike_std[target_index])
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
                            "source_terminal_index": int(terminal_index),
                            "donor_terminal_index": int(donor_terminal_index),
                            "relation_value": float(background["relation_value"][terminal_index]),
                            "relation_velocity": float(background["relation_velocity"][terminal_index]),
                            "values": role_values[role].astype(np.float32),
                        }
                    )
                contracts.append(
                    {
                        "pair_id": pair_id,
                        "target_index": target_index,
                        "phase_bin": phase_bin,
                        "source_terminal_index": int(terminal_index),
                        "donor_terminal_index": int(donor_terminal_index),
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
                        "coherent_unsupported_driver_difference": float(
                            np.max(np.abs(coherent[:, drivers] - unsupported[:, drivers]))
                        ),
                        "coherent_omission_target_difference": float(
                            np.max(np.abs(coherent[:, target_index] - omission[:, target_index]))
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
                        "values": source_values[
                            terminal_index - history : terminal_index + 1
                        ].copy(),
                    }
                )
    return {
        "train": train,
        "background": background,
        "target_indices": targets,
        "episodes": episodes,
        "contracts": contracts,
        "spike_std": spike_std,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_suite(config: Mapping[str, Any], suite: Mapping[str, Any], output_dir: Path) -> Path:
    """Persist every input and diagnostic needed to reproduce B2a."""
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
        "contracts": list(suite["contracts"]),
        "spike_std_from_normal_train": np.asarray(suite["spike_std"], dtype=float).tolist(),
        "phase_metadata_used_by_model_or_calibration": False,
        "latent_factors_used_by_model_or_calibration": False,
    }
    _write_json(output_dir / "suite_metadata.json", manifest)
    return output_dir / "suite_metadata.json"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config = _load_json(arguments.config)
    suite = generate_suite(config)
    manifest = write_suite(config, suite, arguments.output_dir)
    print(f"Wrote B2a held-out suite: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
