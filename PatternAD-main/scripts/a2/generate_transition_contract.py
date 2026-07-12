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
    "normal_scheduled_transition",
    "incompatible_timing_transition",
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
    normal_splits = config["normal_splits"]
    split_names = (
        "optimization_length",
        "validation_length",
        "reference_length",
        "outer_calibration_length",
    )
    split_lengths = [int(normal_splits[name]) for name in split_names]
    guard_length = int(normal_splits["guard_length"])
    required_guard = history + horizon - 1
    if guard_length != required_guard:
        raise ValueError("normal_splits.guard_length must equal history_length + horizon_length - 1.")
    if any(length < history + horizon for length in split_lengths):
        raise ValueError("Each A2 normal split must contain at least one complete window.")
    if sum(split_lengths) + 3 * guard_length != int(config["train_length"]):
        raise ValueError("normal_splits and guards must partition train_length exactly.")
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
    episodes = config["episodes"]
    target = int(episodes["target_channel"])
    if not 0 <= target < dimensions:
        raise ValueError("episodes.target_channel is outside dimensions.")
    if int(episodes["pairs_per_regime"]) < 2:
        raise ValueError(
            "episodes.pairs_per_regime must be at least two to expose both cue modes."
        )
    bank_counts = episodes["normal_transition_sources_per_regime"]
    for split_name in ("optimization", "validation", "reference", "outer_calibration"):
        if int(bank_counts[split_name]) < 2:
            raise ValueError(
                "normal_transition_sources_per_regime must contain at least two "
                f"episodes for {split_name}."
            )
    if float(episodes["transition_amplitude"]) <= 0.0:
        raise ValueError("episodes.transition_amplitude must be positive.")
    if not 0.0 < float(episodes["coordination_amplitude_multiplier"]) <= 1.0:
        raise ValueError("coordination_amplitude_multiplier must be in (0, 1].")
    cue_channel = int(episodes["cue_channel"])
    if not 0 <= cue_channel < dimensions or cue_channel == target:
        raise ValueError("cue_channel must be a non-target channel.")
    cue_length = int(episodes["cue_length"])
    if not 1 <= cue_length <= history:
        raise ValueError("cue_length must be in [1, history_length].")
    cue_encoding = str(episodes.get("cue_encoding", "additive_v1"))
    if cue_encoding not in {"additive_v1", "anchored_overwrite_v1"}:
        raise ValueError(
            "episodes.cue_encoding must be 'additive_v1' or 'anchored_overwrite_v1'."
        )
    cue_amplitude_tolerance = float(episodes.get("cue_amplitude_tolerance", 0.0))
    if cue_encoding == "anchored_overwrite_v1" and not 0.0 < cue_amplitude_tolerance <= 1e-5:
        raise ValueError(
            "anchored_overwrite_v1 requires cue_amplitude_tolerance in (0, 1e-5]."
        )
    cue_amplitudes = np.asarray(episodes["cue_amplitudes"], dtype=np.float64)
    normal_onsets = np.asarray(episodes["normal_transition_onsets"], dtype=np.int64)
    incompatible_onsets = np.asarray(
        episodes["incompatible_transition_onsets"], dtype=np.int64
    )
    if (
        cue_amplitudes.shape != (2,)
        or normal_onsets.shape != (2,)
        or incompatible_onsets.shape != (2,)
        or not float(cue_amplitudes[0]) < 0.0 < float(cue_amplitudes[1])
    ):
        raise ValueError(
            "A2 must define negative/positive observable cues for its two timing modes."
        )
    ramp_length = int(episodes["transition_ramp_length"])
    if ramp_length < 2:
        raise ValueError("transition_ramp_length must be at least two.")
    for name, onsets in (
        ("normal_transition_onsets", normal_onsets),
        ("incompatible_transition_onsets", incompatible_onsets),
    ):
        if np.any(onsets < 1) or np.any(onsets + ramp_length > horizon):
            raise ValueError(f"{name} does not fit inside the A2 horizon.")
    if set(normal_onsets.tolist()) != set(incompatible_onsets.tolist()):
        raise ValueError(
            "Normal and incompatible onset distributions must match so a global timing rule cannot solve A2."
        )
    if np.array_equal(normal_onsets, incompatible_onsets):
        raise ValueError("Incompatible timing must disagree with the cue-conditioned normal timing.")
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


def _stationary_transition_carrier(event_pre: np.ndarray, horizon: int) -> np.ndarray:
    """Use a shared flat carrier so paired observed-window summaries can tie."""
    return np.repeat(np.asarray(event_pre[-1:], dtype=np.float32), horizon, axis=0)


def _timed_transition_profile(
    horizon: int, onset: int, ramp_length: int
) -> np.ndarray:
    """A fixed-rate rise whose onset, but not shape, varies by cue mode."""
    profile = np.zeros(horizon, dtype=np.float32)
    profile[onset : onset + ramp_length] = np.linspace(
        1.0 / ramp_length, 1.0, ramp_length, dtype=np.float32
    )
    profile[onset + ramp_length :] = 1.0
    return profile


def _trajectory_increment_statistics(values: np.ndarray) -> Dict[str, float]:
    increments = np.diff(np.asarray(values, dtype=np.float64), axis=0)
    return {
        "max_increment": float(np.max(np.abs(increments))),
        "increment_l1": float(np.sum(np.abs(increments))),
        "increment_l2": float(np.sum(np.square(increments))),
    }


def _with_event_pre_cue(
    event_pre: np.ndarray,
    cue_channel: int,
    cue_length: int,
    cue_amplitude: float,
    cue_encoding: str,
) -> np.ndarray:
    """Encode the expected transition timing in observable event-pre history."""
    output = np.asarray(event_pre, dtype=np.float32).copy()
    profile = np.linspace(
        0.0, cue_amplitude, cue_length, dtype=np.float32
    )
    if cue_encoding == "additive_v1":
        output[-cue_length:, cue_channel] += profile
    elif cue_encoding == "anchored_overwrite_v1":
        # The raw final-minus-initial cue rule is exactly the declared amplitude
        # for every generator seed, rather than a random-background perturbation.
        anchor = output[-cue_length, cue_channel]
        output[-cue_length:, cue_channel] = anchor + profile
    else:
        raise ValueError(f"Unsupported A2 cue encoding: {cue_encoding}")
    return output


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


def _normal_split_ranges(config: Mapping[str, Any]) -> Dict[str, tuple[int, int]]:
    """Return time-disjoint normal-data ranges, excluding their guard intervals."""
    normal_splits = config["normal_splits"]
    guard_length = int(normal_splits["guard_length"])
    ranges: Dict[str, tuple[int, int]] = {}
    start = 0
    for index, name in enumerate(
        ("optimization", "validation", "reference", "outer_calibration")
    ):
        end = start + int(normal_splits[f"{name}_length"])
        ranges[name] = (start, end)
        start = end + (guard_length if index < 3 else 0)
    if start != int(config["train_length"]):
        raise ValueError("A2 normal split ranges do not consume train_length exactly.")
    return ranges


def _episode(
    source_id: str,
    role: str,
    regime: int,
    source_start: int,
    event_pre: np.ndarray,
    future: np.ndarray,
    primary_pair_id: str | None,
    coordination_pair_id: str | None,
    cue_mode: int | None,
    expected_onset: int | None,
    observed_onset: int | None,
) -> Dict[str, Any]:
    return {
        "source_id": source_id,
        "role": role,
        "regime": int(regime),
        "source_start": int(source_start),
        "primary_pair_id": primary_pair_id,
        "coordination_pair_id": coordination_pair_id,
        "cue_mode": cue_mode,
        "expected_onset": expected_onset,
        "observed_onset": observed_onset,
        "values": np.concatenate((event_pre, future), axis=0).astype(np.float32),
    }


def _normal_transition_bank_episode(
    split_name: str,
    regime: int,
    ordinal: int,
    source_start: int,
    values: np.ndarray,
    history: int,
    horizon: int,
    cue_channel: int,
    cue_length: int,
    cue_amplitudes: np.ndarray,
    cue_encoding: str,
    normal_onsets: np.ndarray,
    ramp_length: int,
    transition_amplitude: float,
    coordination_amplitude_multiplier: float,
    loadings: np.ndarray,
) -> List[Dict[str, Any]]:
    """Build normal scheduled, coordinated, and no-event trajectories in one split."""
    cue_mode = ordinal % 2
    event_pre = _with_event_pre_cue(
        values[source_start - history : source_start],
        cue_channel,
        cue_length,
        float(cue_amplitudes[cue_mode]),
        cue_encoding,
    )
    profile = _timed_transition_profile(
        horizon, int(normal_onsets[cue_mode]), ramp_length
    )
    sign = 1.0 if ordinal % 2 == 0 else -1.0
    scheduled_future = _stationary_transition_carrier(event_pre, horizon) + (
        sign * transition_amplitude * profile[:, None] * loadings[None, :]
    )
    coordinated_future = _stationary_transition_carrier(event_pre, horizon) + (
        -sign
        * transition_amplitude
        * coordination_amplitude_multiplier
        * profile[:, None]
        * loadings[None, :]
    )
    no_event_pre = np.asarray(values[source_start - history : source_start], dtype=np.float32)
    no_event_future = np.asarray(values[source_start : source_start + horizon], dtype=np.float32)
    return [
        {
            "split": split_name,
            "role": "normal_scheduled_transition",
            "regime": int(regime),
            "source_start": int(source_start),
            "cue_mode": int(cue_mode),
            "expected_onset": int(normal_onsets[cue_mode]),
            "values": np.concatenate((event_pre, scheduled_future), axis=0).astype(np.float32),
        },
        {
            "split": split_name,
            "role": "normal_coordinated_transition",
            "regime": int(regime),
            "source_start": int(source_start),
            "cue_mode": int(cue_mode),
            "expected_onset": int(normal_onsets[cue_mode]),
            "values": np.concatenate((event_pre, coordinated_future), axis=0).astype(np.float32),
        },
        {
            "split": split_name,
            "role": "no_event_normal_control",
            "regime": int(regime),
            "source_start": int(source_start),
            "cue_mode": None,
            "expected_onset": None,
            "values": np.concatenate((no_event_pre, no_event_future), axis=0).astype(np.float32),
        },
    ]


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
    normal_split_ranges = _normal_split_ranges(config)
    episodes_config = config["episodes"]
    process = config["normal_process"]
    target = int(episodes_config["target_channel"])
    drivers = np.asarray(
        [index for index in range(int(config["dimensions"])) if index != target],
        dtype=np.int64,
    )
    loadings = np.asarray(process["loadings"], dtype=np.float32)
    optimization_start, optimization_end = normal_split_ranges["optimization"]
    target_std = float(
        np.std(train_values[optimization_start:optimization_end, target], dtype=np.float64)
    )
    target_std = max(target_std, 1e-6)
    sources = _select_sources(
        background_regime, history, horizon, int(episodes_config["pairs_per_regime"])
    )
    episodes: List[Dict[str, Any]] = []
    contracts: List[Dict[str, Any]] = []
    normal_transition_banks: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in normal_split_ranges
    }
    cue_channel = int(episodes_config["cue_channel"])
    cue_length = int(episodes_config["cue_length"])
    cue_amplitudes = np.asarray(episodes_config["cue_amplitudes"], dtype=np.float32)
    cue_encoding = str(episodes_config.get("cue_encoding", "additive_v1"))
    normal_onsets = np.asarray(episodes_config["normal_transition_onsets"], dtype=np.int64)
    incompatible_onsets = np.asarray(
        episodes_config["incompatible_transition_onsets"], dtype=np.int64
    )
    ramp_length = int(episodes_config["transition_ramp_length"])
    for regime in (0, 1):
        for ordinal, source_start in enumerate(sources[regime]):
            source_id = f"regime_{regime}_source_{ordinal:02d}"
            cue_mode = ordinal % 2
            event_pre = _with_event_pre_cue(
                background_values[source_start - history : source_start],
                cue_channel,
                cue_length,
                float(cue_amplitudes[cue_mode]),
                cue_encoding,
            )
            transition_carrier = _stationary_transition_carrier(event_pre, horizon)
            no_event_pre = background_values[source_start - history : source_start]
            # A no-event control must be a directly sampled normal continuation,
            # not a separately simulated forecast from an estimated latent state.
            no_event_future = background_values[source_start : source_start + horizon]
            sign = 1.0 if ordinal % 2 == 0 else -1.0
            primary_amplitude = sign * float(episodes_config["transition_amplitude"])
            coordinated_amplitude = (
                -sign
                * float(episodes_config["transition_amplitude"])
                * float(episodes_config["coordination_amplitude_multiplier"])
            )
            normal_profile = _timed_transition_profile(
                horizon, int(normal_onsets[cue_mode]), ramp_length
            )
            incompatible_profile = _timed_transition_profile(
                horizon, int(incompatible_onsets[cue_mode]), ramp_length
            )
            scheduled_delta = (
                primary_amplitude * normal_profile[:, None] * loadings[None, :]
            )
            incompatible_delta = (
                primary_amplitude * incompatible_profile[:, None] * loadings[None, :]
            )
            coordinated_delta = (
                coordinated_amplitude * normal_profile[:, None] * loadings[None, :]
            )
            unsupported_delta = coordinated_delta.copy()
            unsupported_delta[:, drivers] *= float(
                episodes_config["unsupported_driver_multiplier"]
            )
            scheduled = transition_carrier + scheduled_delta
            incompatible = transition_carrier + incompatible_delta
            coordinated = transition_carrier + coordinated_delta
            unsupported = transition_carrier + unsupported_delta
            primary_pair_id = f"{source_id}_primary"
            coordination_pair_id = f"{source_id}_coordination"
            episodes.extend(
                (
                    _episode(
                        source_id,
                        "normal_scheduled_transition",
                        regime,
                        int(source_start),
                        event_pre,
                        scheduled,
                        primary_pair_id,
                        None,
                        cue_mode,
                        int(normal_onsets[cue_mode]),
                        int(normal_onsets[cue_mode]),
                    ),
                    _episode(
                        source_id,
                        "incompatible_timing_transition",
                        regime,
                        int(source_start),
                        event_pre,
                        incompatible,
                        primary_pair_id,
                        None,
                        cue_mode,
                        int(normal_onsets[cue_mode]),
                        int(incompatible_onsets[cue_mode]),
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
                        cue_mode,
                        int(normal_onsets[cue_mode]),
                        int(normal_onsets[cue_mode]),
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
                        cue_mode,
                        int(normal_onsets[cue_mode]),
                        int(normal_onsets[cue_mode]),
                    ),
                    _episode(
                        source_id,
                        "no_event_normal_control",
                        regime,
                        int(source_start),
                        no_event_pre,
                        no_event_future,
                        None,
                        None,
                        None,
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
            scheduled_statistics = _trajectory_increment_statistics(scheduled)
            incompatible_statistics = _trajectory_increment_statistics(incompatible)
            contracts.append(
                {
                    "source_id": source_id,
                    "regime": int(regime),
                    "source_start": int(source_start),
                    "primary_pair_id": primary_pair_id,
                    "coordination_pair_id": coordination_pair_id,
                    "cue_mode": int(cue_mode),
                    "cue_channel": cue_channel,
                    "expected_onset": int(normal_onsets[cue_mode]),
                    "incompatible_onset": int(incompatible_onsets[cue_mode]),
                    "event_pre_max_abs_difference": 0.0,
                    "primary_endpoint_max_abs_difference": float(
                        np.max(np.abs(scheduled[-1] - incompatible[-1]))
                    ),
                    "primary_trajectory_max_abs_difference": float(
                        np.max(np.abs(scheduled - incompatible))
                    ),
                    "primary_increment_max_abs_difference": abs(
                        scheduled_statistics["max_increment"]
                        - incompatible_statistics["max_increment"]
                    ),
                    "primary_increment_l1_abs_difference": abs(
                        scheduled_statistics["increment_l1"]
                        - incompatible_statistics["increment_l1"]
                    ),
                    "primary_increment_l2_abs_difference": abs(
                        scheduled_statistics["increment_l2"]
                        - incompatible_statistics["increment_l2"]
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
                    "scheduled_trajectory_increment_statistics": scheduled_statistics,
                    "incompatible_trajectory_increment_statistics": incompatible_statistics,
                    "coordination_error_target_std": coordination_error,
                    "minimum_coordination_error_target_std": required_error,
                }
            )
    for split_name, (split_start, split_end) in normal_split_ranges.items():
        bank_sources = _select_sources(
            train_regime[split_start:split_end],
            history,
            horizon,
            int(episodes_config["normal_transition_sources_per_regime"][split_name]),
        )
        for regime in (0, 1):
            for ordinal, relative_source_start in enumerate(bank_sources[regime]):
                normal_transition_banks[split_name].extend(
                    _normal_transition_bank_episode(
                        split_name,
                        regime,
                        ordinal,
                        split_start + int(relative_source_start),
                        train_values,
                        history,
                        horizon,
                        cue_channel,
                        cue_length,
                        cue_amplitudes,
                        cue_encoding,
                        normal_onsets,
                        ramp_length,
                        float(episodes_config["transition_amplitude"]),
                        float(episodes_config["coordination_amplitude_multiplier"]),
                        loadings,
                    )
                )
    return {
        "train_values": train_values,
        "train_regime": train_regime,
        "background_values": background_values,
        "background_regime": background_regime,
        "normal_split_ranges": normal_split_ranges,
        "episodes": episodes,
        "contracts": contracts,
        "normal_transition_banks": normal_transition_banks,
        "normal_transition_references": normal_transition_banks["reference"],
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
        normal_split_names=np.asarray(list(suite["normal_split_ranges"].keys())),
        normal_split_starts=np.asarray(
            [item[0] for item in suite["normal_split_ranges"].values()], dtype=np.int64
        ),
        normal_split_ends=np.asarray(
            [item[1] for item in suite["normal_split_ranges"].values()], dtype=np.int64
        ),
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
        cue_modes=np.asarray(
            [episode["cue_mode"] if episode["cue_mode"] is not None else -1 for episode in episodes],
            dtype=np.int64,
        ),
        expected_onsets=np.asarray(
            [episode["expected_onset"] if episode["expected_onset"] is not None else -1 for episode in episodes],
            dtype=np.int64,
        ),
        observed_onsets=np.asarray(
            [episode["observed_onset"] if episode["observed_onset"] is not None else -1 for episode in episodes],
            dtype=np.int64,
        ),
    )
    banks = dict(suite["normal_transition_banks"])
    all_bank_episodes = [episode for bank in banks.values() for episode in bank]
    references = list(banks["reference"])
    np.savez_compressed(
        output_dir / "normal_transition_banks.npz",
        windows=np.stack([episode["values"] for episode in all_bank_episodes], axis=0),
        split_names=np.asarray([episode["split"] for episode in all_bank_episodes]),
        roles=np.asarray([episode["role"] for episode in all_bank_episodes]),
        regimes=np.asarray([episode["regime"] for episode in all_bank_episodes], dtype=np.int64),
        source_starts=np.asarray(
            [episode["source_start"] for episode in all_bank_episodes], dtype=np.int64
        ),
        cue_modes=np.asarray(
            [episode["cue_mode"] if episode["cue_mode"] is not None else -1 for episode in all_bank_episodes],
            dtype=np.int64,
        ),
        expected_onsets=np.asarray(
            [episode["expected_onset"] if episode["expected_onset"] is not None else -1 for episode in all_bank_episodes],
            dtype=np.int64,
        ),
    )
    np.savez_compressed(
        output_dir / "normal_transition_references.npz",
        windows=np.stack([reference["values"] for reference in references], axis=0),
        roles=np.asarray([reference["role"] for reference in references]),
        regimes=np.asarray([reference["regime"] for reference in references], dtype=np.int64),
        source_starts=np.asarray(
            [reference["source_start"] for reference in references], dtype=np.int64
        ),
        cue_modes=np.asarray(
            [reference["cue_mode"] if reference["cue_mode"] is not None else -1 for reference in references],
            dtype=np.int64,
        ),
        expected_onsets=np.asarray(
            [reference["expected_onset"] if reference["expected_onset"] is not None else -1 for reference in references],
            dtype=np.int64,
        ),
    )
    _write_json(output_dir / "resolved_config.json", dict(config))
    metadata = {
        "suite_id": str(config["suite_id"]),
        "config_hash": _canonical_hash(config),
        "generator_sha256": _file_sha256(Path(__file__)),
        "episode_count": int(len(episodes)),
        "source_count": int(len(suite["contracts"])),
        "normal_transition_reference_count": int(len(references)),
        "normal_transition_bank_counts": {
            split_name: int(len(bank)) for split_name, bank in banks.items()
        },
        "normal_split_ranges": {
            name: [int(start), int(end)]
            for name, (start, end) in suite["normal_split_ranges"].items()
        },
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
