#!/usr/bin/env python3
"""Generate A2's model-independent matched transition contract.

The suite is intentionally not a detector and contains no score, normalizer,
calibration, or model choice. It certifies the question an A2 model must solve:
whether a future trajectory is compatible with an event-pre observable state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "a2" / "transition_contract_v1.json"
ROLE_ORDER: Sequence[str] = (
    "normal_gradual_transition",
    "incompatible_abrupt_transition",
    "normal_coordinated_transition",
    "unsupported_transition",
    "no_event_normal_control",
)


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


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate only generator inputs; model/calibration keys are irrelevant."""
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("A2 config must use schema_version=1.")
    if not str(config.get("suite_id", "")).startswith("a2_transition_contract_"):
        raise ValueError("A2 suite_id must start with 'a2_transition_contract_'.")
    dimensions = int(config["dimensions"])
    history = int(config["history_length"])
    horizon = int(config["horizon_length"])
    if dimensions < 2 or history < 2 or horizon < 4:
        raise ValueError("A2 requires dimensions >= 2, history >= 2, and horizon >= 4.")
    if len(config.get("channel_names", [])) != dimensions:
        raise ValueError("channel_names must match dimensions.")
    if int(config["train_length"]) <= 4 * history:
        raise ValueError("train_length is too short for an A2 normal reference stream.")
    if int(config["background_length"]) <= 2 * (history + horizon):
        raise ValueError("background_length is too short for A2 source windows.")
    process = config["normal_process"]
    for name in ("loadings", "channel_ar", "channel_noise"):
        value = np.asarray(process[name], dtype=np.float64)
        if value.shape != (dimensions,):
            raise ValueError(f"normal_process.{name} must have {dimensions} entries.")
    if not np.all(np.asarray(process["channel_noise"], dtype=np.float64) > 0.0):
        raise ValueError("normal_process.channel_noise must be positive.")
    if not 0.0 <= float(process["latent_ar"]) < 0.99:
        raise ValueError("normal_process.latent_ar must be in [0, 0.99).")
    if float(process["latent_std"]) <= 0.0:
        raise ValueError("normal_process.latent_std must be positive.")
    if int(process["regime_segment_length"]) < history + horizon:
        raise ValueError("normal_process.regime_segment_length is too short.")
    noise_scales = np.asarray(process["regime_noise_scales"], dtype=np.float64)
    if noise_scales.shape != (2,) or np.any(noise_scales <= 0.0):
        raise ValueError("normal_process.regime_noise_scales must contain two positives.")
    maximum_increment = float(process["normal_transition_max_profile_increment"])
    if not 0.0 < maximum_increment < 1.0:
        raise ValueError("normal_transition_max_profile_increment must be in (0, 1).")
    episodes = config["episodes"]
    target = int(episodes["target_channel"])
    if not 0 <= target < dimensions:
        raise ValueError("episodes.target_channel is outside dimensions.")
    if int(episodes["pairs_per_regime"]) < 1:
        raise ValueError("episodes.pairs_per_regime must be positive.")
    if float(episodes["transition_amplitude"]) <= 0.0:
        raise ValueError("episodes.transition_amplitude must be positive.")
    if not 0.0 < float(episodes["coordination_amplitude_multiplier"]) <= 1.0:
        raise ValueError("coordination_amplitude_multiplier must be in (0, 1].")
    abrupt_step = int(episodes["abrupt_step_index"])
    if not 1 <= abrupt_step < horizon - 1:
        raise ValueError("abrupt_step_index must lie inside the A2 horizon.")
    if abs(float(episodes["unsupported_driver_multiplier"]) - 1.0) < 1e-9:
        raise ValueError("unsupported_driver_multiplier must differ from one.")
    if float(episodes["minimum_coordination_error_target_std"]) <= 0.0:
        raise ValueError("minimum_coordination_error_target_std must be positive.")


def _simulate_normal_stream(
    length: int, config: Mapping[str, Any], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a stable multivariate normal stream with two observable regimes."""
    process = config["normal_process"]
    burn_in = int(config["burn_in"])
    dimensions = int(config["dimensions"])
    total = length + burn_in
    regime_length = int(process["regime_segment_length"])
    regime = (np.arange(total, dtype=np.int64) // regime_length) % 2
    scales = np.asarray(process["regime_noise_scales"], dtype=np.float64)
    loadings = np.asarray(process["loadings"], dtype=np.float64)
    channel_ar = np.asarray(process["channel_ar"], dtype=np.float64)
    channel_noise = np.asarray(process["channel_noise"], dtype=np.float64)
    values = np.zeros((total, dimensions), dtype=np.float64)
    latent = 0.0
    for index in range(total):
        scale = scales[regime[index]]
        latent = (
            float(process["latent_ar"]) * latent
            + float(process["latent_std"]) * scale * rng.normal()
        )
        previous = values[index - 1] if index else np.zeros(dimensions, dtype=np.float64)
        values[index] = (
            channel_ar * previous
            + loadings * latent
            + channel_noise * scale * rng.normal(size=dimensions)
        )
    return values[burn_in:].astype(np.float32), regime[burn_in:].astype(np.int64)


def _simulate_baseline_continuation(
    event_pre: np.ndarray,
    horizon: int,
    config: Mapping[str, Any],
    regime: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one shared normal continuation for all counterfactual roles."""
    process = config["normal_process"]
    loadings = np.asarray(process["loadings"], dtype=np.float64)
    channel_ar = np.asarray(process["channel_ar"], dtype=np.float64)
    channel_noise = np.asarray(process["channel_noise"], dtype=np.float64)
    scale = float(np.asarray(process["regime_noise_scales"], dtype=np.float64)[regime])
    current = np.asarray(event_pre[-1], dtype=np.float64).copy()
    latent = float(np.dot(current, loadings) / max(np.dot(loadings, loadings), 1e-12))
    output = np.empty((horizon, len(current)), dtype=np.float64)
    for index in range(horizon):
        latent = (
            float(process["latent_ar"]) * latent
            + float(process["latent_std"]) * scale * rng.normal()
        )
        current = (
            channel_ar * current
            + loadings * latent
            + channel_noise * scale * rng.normal(size=len(current))
        )
        output[index] = current
    return output.astype(np.float32)


def _smooth_profile(horizon: int) -> np.ndarray:
    position = np.linspace(0.0, 1.0, horizon, dtype=np.float64)
    return (position * position * (3.0 - 2.0 * position)).astype(np.float32)


def _abrupt_profile(horizon: int, step_index: int) -> np.ndarray:
    profile = np.zeros(horizon, dtype=np.float32)
    profile[step_index:] = 1.0
    return profile


def _select_sources(
    regimes: np.ndarray, history: int, horizon: int, pairs_per_regime: int
) -> Dict[int, np.ndarray]:
    starts = np.arange(len(regimes), dtype=np.int64)
    selected: Dict[int, np.ndarray] = {}
    for regime in (0, 1):
        candidates = starts[
            (regimes == regime)
            & (starts >= history)
            & (starts + horizon <= len(regimes))
        ]
        if len(candidates) < pairs_per_regime:
            raise ValueError("A2 background stream cannot supply the requested sources.")
        positions = np.linspace(0, len(candidates) - 1, pairs_per_regime, dtype=np.int64)
        selected[regime] = candidates[np.unique(positions)]
        if len(selected[regime]) != pairs_per_regime:
            raise ValueError("A2 source selection did not produce distinct pairs.")
    return selected


def _episode(
    source_id: str,
    role: str,
    regime: int,
    source_start: int,
    event_pre: np.ndarray,
    future: np.ndarray,
    primary_pair_id: str | None,
    coordination_pair_id: str | None,
) -> Dict[str, Any]:
    return {
        "source_id": source_id,
        "role": role,
        "regime": int(regime),
        "source_start": int(source_start),
        "primary_pair_id": primary_pair_id,
        "coordination_pair_id": coordination_pair_id,
        "values": np.concatenate((event_pre, future), axis=0).astype(np.float32),
    }


def generate_suite(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Generate normal streams and matched endpoint/trajectory counterfactuals."""
    validate_config(config)
    rng = np.random.default_rng(int(config["seed"]))
    train_values, train_regime = _simulate_normal_stream(
        int(config["train_length"]), config, rng
    )
    background_values, background_regime = _simulate_normal_stream(
        int(config["background_length"]), config, rng
    )
    history = int(config["history_length"])
    horizon = int(config["horizon_length"])
    episodes_config = config["episodes"]
    process = config["normal_process"]
    target = int(episodes_config["target_channel"])
    drivers = np.asarray(
        [index for index in range(int(config["dimensions"])) if index != target],
        dtype=np.int64,
    )
    loadings = np.asarray(process["loadings"], dtype=np.float32)
    target_std = float(np.std(train_values[:, target], dtype=np.float64))
    target_std = max(target_std, 1e-6)
    gradual_profile = _smooth_profile(horizon)
    abrupt_profile = _abrupt_profile(horizon, int(episodes_config["abrupt_step_index"]))
    normal_max_increment = float(np.max(np.abs(np.diff(gradual_profile))))
    abrupt_max_increment = float(np.max(np.abs(np.diff(abrupt_profile))))
    allowed_increment = float(process["normal_transition_max_profile_increment"])
    if normal_max_increment > allowed_increment:
        raise ValueError("A2 normal gradual profile violates the configured support bound.")
    if abrupt_max_increment <= allowed_increment:
        raise ValueError("A2 abrupt profile is not outside normal transition support.")
    sources = _select_sources(
        background_regime, history, horizon, int(episodes_config["pairs_per_regime"])
    )
    episodes: List[Dict[str, Any]] = []
    contracts: List[Dict[str, Any]] = []
    for regime in (0, 1):
        for ordinal, source_start in enumerate(sources[regime]):
            source_id = f"regime_{regime}_source_{ordinal:02d}"
            event_pre = background_values[source_start - history : source_start].copy()
            base_future = _simulate_baseline_continuation(
                event_pre, horizon, config, regime, rng
            )
            sign = 1.0 if ordinal % 2 == 0 else -1.0
            primary_amplitude = sign * float(episodes_config["transition_amplitude"])
            coordinated_amplitude = (
                -sign
                * float(episodes_config["transition_amplitude"])
                * float(episodes_config["coordination_amplitude_multiplier"])
            )
            gradual_delta = primary_amplitude * gradual_profile[:, None] * loadings[None, :]
            abrupt_delta = primary_amplitude * abrupt_profile[:, None] * loadings[None, :]
            coordinated_delta = (
                coordinated_amplitude * gradual_profile[:, None] * loadings[None, :]
            )
            unsupported_delta = coordinated_delta.copy()
            unsupported_delta[:, drivers] *= float(
                episodes_config["unsupported_driver_multiplier"]
            )
            gradual = base_future + gradual_delta
            abrupt = base_future + abrupt_delta
            coordinated = base_future + coordinated_delta
            unsupported = base_future + unsupported_delta
            primary_pair_id = f"{source_id}_primary"
            coordination_pair_id = f"{source_id}_coordination"
            episodes.extend(
                (
                    _episode(
                        source_id,
                        "normal_gradual_transition",
                        regime,
                        int(source_start),
                        event_pre,
                        gradual,
                        primary_pair_id,
                        None,
                    ),
                    _episode(
                        source_id,
                        "incompatible_abrupt_transition",
                        regime,
                        int(source_start),
                        event_pre,
                        abrupt,
                        primary_pair_id,
                        None,
                    ),
                    _episode(
                        source_id,
                        "normal_coordinated_transition",
                        regime,
                        int(source_start),
                        event_pre,
                        coordinated,
                        None,
                        coordination_pair_id,
                    ),
                    _episode(
                        source_id,
                        "unsupported_transition",
                        regime,
                        int(source_start),
                        event_pre,
                        unsupported,
                        None,
                        coordination_pair_id,
                    ),
                    _episode(
                        source_id,
                        "no_event_normal_control",
                        regime,
                        int(source_start),
                        event_pre,
                        base_future,
                        None,
                        None,
                    ),
                )
            )
            expected_drivers = coordinated_delta[:, drivers]
            observed_drivers = unsupported_delta[:, drivers]
            coordination_error = float(
                np.max(np.abs(expected_drivers - observed_drivers)) / target_std
            )
            required_error = float(episodes_config["minimum_coordination_error_target_std"])
            if coordination_error < required_error:
                raise ValueError(
                    "A2 unsupported transition does not meet the coordination-error contract."
                )
            contracts.append(
                {
                    "source_id": source_id,
                    "regime": int(regime),
                    "source_start": int(source_start),
                    "primary_pair_id": primary_pair_id,
                    "coordination_pair_id": coordination_pair_id,
                    "event_pre_max_abs_difference": 0.0,
                    "primary_endpoint_max_abs_difference": float(
                        np.max(np.abs(gradual[-1] - abrupt[-1]))
                    ),
                    "primary_trajectory_max_abs_difference": float(
                        np.max(np.abs(gradual - abrupt))
                    ),
                    "coordination_target_trajectory_max_abs_difference": float(
                        np.max(
                            np.abs(
                                coordinated[:, target] - unsupported[:, target]
                            )
                        )
                    ),
                    "coordination_driver_trajectory_max_abs_difference": float(
                        np.max(
                            np.abs(
                                coordinated[:, drivers] - unsupported[:, drivers]
                            )
                        )
                    ),
                    "normal_profile_max_increment": normal_max_increment,
                    "abrupt_profile_max_increment": abrupt_max_increment,
                    "normal_transition_max_profile_increment": allowed_increment,
                    "coordination_error_target_std": coordination_error,
                    "minimum_coordination_error_target_std": required_error,
                }
            )
    return {
        "train_values": train_values,
        "train_regime": train_regime,
        "background_values": background_values,
        "background_regime": background_regime,
        "episodes": episodes,
        "contracts": contracts,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_suite(config: Mapping[str, Any], suite: Mapping[str, Any], output_dir: Path) -> Path:
    """Persist the contract inputs without producing a model-derived artifact."""
    output_dir.mkdir(parents=True, exist_ok=False)
    episodes = list(suite["episodes"])
    np.savez_compressed(
        output_dir / "normal_streams.npz",
        train_values=np.asarray(suite["train_values"], dtype=np.float32),
        train_regime=np.asarray(suite["train_regime"], dtype=np.int64),
        background_values=np.asarray(suite["background_values"], dtype=np.float32),
        background_regime=np.asarray(suite["background_regime"], dtype=np.int64),
    )
    np.savez_compressed(
        output_dir / "episodes.npz",
        windows=np.stack([episode["values"] for episode in episodes], axis=0),
        source_ids=np.asarray([episode["source_id"] for episode in episodes]),
        roles=np.asarray([episode["role"] for episode in episodes]),
        regimes=np.asarray([episode["regime"] for episode in episodes], dtype=np.int64),
        source_starts=np.asarray(
            [episode["source_start"] for episode in episodes], dtype=np.int64
        ),
        primary_pair_ids=np.asarray(
            [episode["primary_pair_id"] or "" for episode in episodes]
        ),
        coordination_pair_ids=np.asarray(
            [episode["coordination_pair_id"] or "" for episode in episodes]
        ),
    )
    _write_json(output_dir / "resolved_config.json", dict(config))
    metadata = {
        "suite_id": str(config["suite_id"]),
        "config_hash": _canonical_hash(config),
        "generator_sha256": _file_sha256(Path(__file__)),
        "episode_count": int(len(episodes)),
        "source_count": int(len(suite["contracts"])),
        "roles": list(ROLE_ORDER),
        "contracts": list(suite["contracts"]),
        "contains_model_scores": False,
        "contains_calibration_thresholds": False,
    }
    _write_json(output_dir / "suite_metadata.json", metadata)
    return output_dir / "suite_metadata.json"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config = _load_json(arguments.config)
    suite = generate_suite(config)
    metadata = write_suite(config, suite, arguments.output_dir)
    print(f"Wrote A2 transition contract: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
