#!/usr/bin/env python3
"""Generate A3's model-independent trigger-response mechanism suite.

The suite contains raw multivariate windows only. Generator metadata is
retained for construction audit and is never an input to the A3 model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "a3" / "trigger_response_contract_v1.json"
SPLIT_NAMES: Sequence[str] = (
    "optimization",
    "validation",
    "reference",
    "outer_calibration",
)
ROLE_ORDER: Sequence[str] = (
    "normal_routed_response",
    "misrouted_response",
    "partial_propagation_response",
    "normal_no_trigger",
    "untriggered_response",
)
SUPPORTED_SUITE_IDS: Sequence[str] = (
    "a3_trigger_response_contract_v1",
    "a3_trigger_response_route_identifiability_v2",
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate generator parameters before materializing any windows."""
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("A3 config must use schema_version=1.")
    if str(config.get("suite_id", "")) not in SUPPORTED_SUITE_IDS:
        raise ValueError(f"A3 suite_id must be one of {list(SUPPORTED_SUITE_IDS)}.")
    dimensions = int(config["dimensions"])
    history = int(config["history_length"])
    horizon = int(config["horizon_length"])
    if dimensions < 4 or history < 8 or horizon < 6:
        raise ValueError("A3 requires dimensions >= 4, history >= 8, horizon >= 6.")
    if len(config.get("channel_names", [])) != dimensions:
        raise ValueError("channel_names must match dimensions.")
    normal_splits = config["normal_splits"]
    required_guard = history + horizon - 1
    if int(normal_splits["guard_length"]) != required_guard:
        raise ValueError("normal_splits.guard_length must equal history_length + horizon_length - 1.")
    lengths = [int(normal_splits[f"{name}_length"]) for name in SPLIT_NAMES]
    if any(length < history for length in lengths):
        raise ValueError("Every A3 normal split must contain one history window.")
    if sum(lengths) + 3 * required_guard != int(config["train_length"]):
        raise ValueError("normal_splits and guards must partition train_length exactly.")
    if int(config["background_length"]) < 3 * history:
        raise ValueError("background_length is too short for A3 paired sources.")
    process = config["normal_process"]
    for key in ("loadings", "channel_ar", "channel_noise"):
        value = np.asarray(process[key], dtype=np.float64)
        if value.shape != (dimensions,):
            raise ValueError(f"normal_process.{key} must have {dimensions} entries.")
    if np.any(np.asarray(process["channel_noise"], dtype=np.float64) <= 0.0):
        raise ValueError("normal_process.channel_noise must be positive.")
    if not 0.0 <= float(process["latent_ar"]) < 0.99:
        raise ValueError("normal_process.latent_ar must be in [0, 0.99).")
    if int(process["regime_segment_length"]) < history:
        raise ValueError("normal_process.regime_segment_length is too short.")
    if np.any(np.asarray(process["regime_noise_scales"], dtype=np.float64) <= 0.0):
        raise ValueError("normal_process.regime_noise_scales must be positive.")

    episodes = config["episodes"]
    source = int(episodes["source_channel"])
    response_channels = np.asarray(episodes["response_channels"], dtype=np.int64)
    if not 0 <= source < dimensions:
        raise ValueError("episodes.source_channel is outside dimensions.")
    if response_channels.ndim != 1 or len(response_channels) < 2:
        raise ValueError("episodes.response_channels must contain at least two channels.")
    if np.any(response_channels < 0) or np.any(response_channels >= dimensions):
        raise ValueError("episodes.response_channels contains an invalid channel.")
    if source in response_channels or len(set(response_channels.tolist())) != len(response_channels):
        raise ValueError("source_channel must be distinct from unique response_channels.")
    cue_length = int(episodes["cue_length"])
    if not 2 <= cue_length <= history:
        raise ValueError("episodes.cue_length must be in [2, history_length].")
    cue_amplitudes = np.asarray(episodes["cue_amplitudes"], dtype=np.float64)
    onsets = np.asarray(episodes["response_onsets"], dtype=np.int64)
    patterns = np.asarray(episodes["response_patterns"], dtype=np.float64)
    if cue_amplitudes.shape != (2,) or not cue_amplitudes[0] < 0.0 < cue_amplitudes[1]:
        raise ValueError("A3 requires two signed cue amplitudes.")
    if onsets.shape != (2,) or np.any(onsets < 0):
        raise ValueError("A3 requires two non-negative response onsets.")
    if patterns.shape != (2, len(response_channels)):
        raise ValueError("response_patterns must be [two modes, response channels].")
    if not np.allclose(patterns[0], -patterns[1], atol=1e-12, rtol=0.0):
        raise ValueError("A3 response modes must be sign-opposed to balance response-mode marginals.")
    ramp_length = int(episodes["response_ramp_length"])
    if ramp_length < 2 or np.any(onsets + ramp_length > horizon):
        raise ValueError("response onsets and ramp length must fit inside the horizon.")
    if int(episodes["pairs_per_mode"]) < 2:
        raise ValueError("pairs_per_mode must be at least two.")
    if float(episodes["response_amplitude"]) <= 0.0:
        raise ValueError("response_amplitude must be positive.")
    if float(episodes["token_energy_threshold"]) <= 0.0:
        raise ValueError("token_energy_threshold must be positive.")
    if float(episodes["minimum_response_margin"]) <= 0.0:
        raise ValueError("minimum_response_margin must be positive.")
    partial = np.asarray(episodes["partial_propagation_channels"], dtype=np.int64)
    if partial.ndim != 1 or not len(partial) or np.any(~np.isin(partial, response_channels)):
        raise ValueError("partial_propagation_channels must be a non-empty response-channel subset.")
    if int(response_channels[0]) in partial:
        raise ValueError("A3-v1 partial propagation must leave its first target channel unchanged.")
    bank_counts = episodes["normal_transition_sources_per_regime"]
    for split in SPLIT_NAMES:
        if int(bank_counts[split]) < 2:
            raise ValueError(f"normal_transition_sources_per_regime.{split} must be at least two.")


def _split_ranges(config: Mapping[str, Any]) -> Dict[str, tuple[int, int]]:
    ranges: Dict[str, tuple[int, int]] = {}
    start = 0
    guard = int(config["normal_splits"]["guard_length"])
    for index, name in enumerate(SPLIT_NAMES):
        end = start + int(config["normal_splits"][f"{name}_length"])
        ranges[name] = (start, end)
        start = end + (guard if index < len(SPLIT_NAMES) - 1 else 0)
    return ranges


def _simulate_normal_stream(
    length: int, config: Mapping[str, Any], rng: np.random.Generator
) -> np.ndarray:
    process = config["normal_process"]
    dimensions = int(config["dimensions"])
    burn_in = int(config["burn_in"])
    total = length + burn_in
    values = np.zeros((total, dimensions), dtype=np.float64)
    latent = 0.0
    loadings = np.asarray(process["loadings"], dtype=np.float64)
    channel_ar = np.asarray(process["channel_ar"], dtype=np.float64)
    channel_noise = np.asarray(process["channel_noise"], dtype=np.float64)
    scales = np.asarray(process["regime_noise_scales"], dtype=np.float64)
    segment = int(process["regime_segment_length"])
    for index in range(total):
        scale = float(scales[(index // segment) % len(scales)])
        latent = float(process["latent_ar"]) * latent + float(process["latent_std"]) * scale * rng.normal()
        previous = values[index - 1] if index else np.zeros(dimensions, dtype=np.float64)
        values[index] = channel_ar * previous + loadings * latent + channel_noise * scale * rng.normal(size=dimensions)
    return values[burn_in:].astype(np.float32)


def _simulate_continuation(
    event_pre: np.ndarray,
    horizon: int,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a raw normal continuation from a supplied event-pre state."""
    process = config["normal_process"]
    dimensions = int(config["dimensions"])
    loadings = np.asarray(process["loadings"], dtype=np.float64)
    channel_ar = np.asarray(process["channel_ar"], dtype=np.float64)
    channel_noise = np.asarray(process["channel_noise"], dtype=np.float64)
    scale = float(np.mean(np.asarray(process["regime_noise_scales"], dtype=np.float64)))
    current = np.asarray(event_pre[-1], dtype=np.float64).copy()
    latent = float(np.dot(current, loadings) / max(np.dot(loadings, loadings), 1e-12))
    future = np.empty((horizon, dimensions), dtype=np.float64)
    for index in range(horizon):
        latent = float(process["latent_ar"]) * latent + float(process["latent_std"]) * scale * rng.normal()
        current = channel_ar * current + loadings * latent + channel_noise * scale * rng.normal(size=dimensions)
        future[index] = current
    return future.astype(np.float32)


def _with_cue(event_pre: np.ndarray, mode: int, config: Mapping[str, Any]) -> np.ndarray:
    episodes = config["episodes"]
    source = int(episodes["source_channel"])
    cue_length = int(episodes["cue_length"])
    cue_amplitude = float(np.asarray(episodes["cue_amplitudes"], dtype=np.float64)[mode])
    output = np.asarray(event_pre, dtype=np.float32).copy()
    anchor = float(output[-cue_length, source])
    output[-cue_length:, source] = np.linspace(
        anchor, anchor + cue_amplitude, cue_length, dtype=np.float32
    )
    return output


def _response_future(
    baseline: np.ndarray,
    mode: int,
    config: Mapping[str, Any],
    pattern: np.ndarray | None = None,
) -> np.ndarray:
    episodes = config["episodes"]
    response_channels = np.asarray(episodes["response_channels"], dtype=np.int64)
    patterns = np.asarray(episodes["response_patterns"], dtype=np.float64)
    selected = patterns[mode] if pattern is None else np.asarray(pattern, dtype=np.float64)
    onset = int(np.asarray(episodes["response_onsets"], dtype=np.int64)[mode])
    ramp = int(episodes["response_ramp_length"])
    progress = np.clip((np.arange(len(baseline), dtype=np.float64) - onset + 1.0) / ramp, 0.0, 1.0)
    output = np.asarray(baseline, dtype=np.float64).copy()
    # Specify the observable response as a net first-to-last displacement.
    # This removes arbitrary continuation drift from the mechanism token while
    # retaining the ordinary trajectory variation between both endpoints.
    desired_displacement = float(episodes["response_amplitude"]) * selected
    baseline_displacement = baseline[-1, response_channels] - baseline[0, response_channels]
    output[:, response_channels] += (
        progress[:, None] * (desired_displacement - baseline_displacement)[None, :]
    )
    return output.astype(np.float32)


def extract_response_tokens(
    future: np.ndarray, token_energy_threshold: float
) -> Dict[str, np.ndarray]:
    """Fixed observable future representation used by A3-G1.

    A channel is active when its end-to-start displacement clears the frozen
    threshold. Active onset is the first time its displacement reaches half of
    its terminal magnitude. Inactive channels use onset/direction `-1`.
    """
    values = np.asarray(future, dtype=np.float64)
    was_single = values.ndim == 2
    if was_single:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1] < 2:
        raise ValueError("future must have shape [samples, horizon, dimensions] with horizon >= 2.")
    terminal = values[:, -1, :] - values[:, 0, :]
    active = np.abs(terminal) >= float(token_energy_threshold)
    progress = np.abs(values - values[:, :1, :])
    half_terminal = 0.5 * np.abs(terminal)[:, None, :]
    reached = progress >= half_terminal
    onset = np.argmax(reached, axis=1).astype(np.int64)
    onset[~active] = -1
    direction = np.where(terminal >= 0.0, 1, 0).astype(np.int64)
    direction[~active] = -1
    output = {"active": active.astype(np.int64), "onset": onset, "direction": direction}
    if was_single:
        return {name: value[0] for name, value in output.items()}
    return output


def _window(event_pre: np.ndarray, future: np.ndarray) -> np.ndarray:
    return np.concatenate((event_pre, future), axis=0).astype(np.float32, copy=False)


def _source_pre(
    values: np.ndarray, start: int, history: int
) -> np.ndarray:
    return np.asarray(values[start : start + history], dtype=np.float32).copy()


def _normal_event_banks(
    train_values: np.ndarray,
    ranges: Mapping[str, tuple[int, int]],
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> Dict[str, List[Dict[str, Any]]]:
    history = int(config["history_length"])
    horizon = int(config["horizon_length"])
    banks: Dict[str, List[Dict[str, Any]]] = {}
    counts = config["episodes"]["normal_transition_sources_per_regime"]
    for split, (start, end) in ranges.items():
        available = np.arange(start, end - history + 1, dtype=np.int64)
        selected = rng.choice(available, size=int(counts[split]), replace=len(available) < int(counts[split]))
        bank: List[Dict[str, Any]] = []
        for index, source_start in enumerate(selected.tolist()):
            raw_pre = _source_pre(train_values, int(source_start), history)
            for mode in (0, 1):
                event_pre = _with_cue(raw_pre, mode, config)
                future = _response_future(
                    _simulate_continuation(event_pre, horizon, config, rng), mode, config
                )
                bank.append({
                    "bank_id": f"{split}_routed_{index:03d}_m{mode}",
                    "values": _window(event_pre, future),
                    "normal_kind": "routed_response",
                    "mode": mode,
                })
            future = _simulate_continuation(raw_pre, horizon, config, rng)
            bank.append({
                "bank_id": f"{split}_no_trigger_{index:03d}",
                "values": _window(raw_pre, future),
                "normal_kind": "no_trigger",
                "mode": None,
            })
        banks[split] = bank
    return banks


def generate_suite(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the complete A3-v1 suite without fitting a model."""
    validate_config(config)
    rng = np.random.default_rng(int(config["seed"]))
    history = int(config["history_length"])
    horizon = int(config["horizon_length"])
    episodes_config = config["episodes"]
    response_channels = np.asarray(episodes_config["response_channels"], dtype=np.int64)
    partial_channels = set(np.asarray(episodes_config["partial_propagation_channels"], dtype=np.int64).tolist())
    train_values = _simulate_normal_stream(int(config["train_length"]), config, rng)
    background_values = _simulate_normal_stream(int(config["background_length"]), config, rng)
    ranges = _split_ranges(config)
    normal_banks = _normal_event_banks(train_values, ranges, config, rng)
    episodes: List[Dict[str, Any]] = []
    source_starts = np.arange(0, len(background_values) - history + 1, dtype=np.int64)
    pair_index = 0
    for cue_mode in (0, 1):
        for within_mode in range(int(episodes_config["pairs_per_mode"])):
            raw_pre = _source_pre(background_values, int(rng.choice(source_starts)), history)
            event_pre = _with_cue(raw_pre, cue_mode, config)
            baseline = _simulate_continuation(event_pre, horizon, config, rng)
            normal_future = _response_future(baseline, cue_mode, config)
            other_mode = 1 - cue_mode
            misrouted_future = _response_future(baseline, other_mode, config)
            normal_pattern = np.asarray(episodes_config["response_patterns"], dtype=np.float64)[cue_mode]
            other_pattern = np.asarray(episodes_config["response_patterns"], dtype=np.float64)[other_mode]
            partial_pattern = normal_pattern.copy()
            for position, channel in enumerate(response_channels.tolist()):
                if channel in partial_channels:
                    partial_pattern[position] = other_pattern[position]
            partial_future = _response_future(baseline, cue_mode, config, pattern=partial_pattern)
            pair_id = f"triggered_m{cue_mode}_{within_mode:02d}"
            for role, future, future_mode in (
                ("normal_routed_response", normal_future, cue_mode),
                ("misrouted_response", misrouted_future, other_mode),
                ("partial_propagation_response", partial_future, cue_mode),
            ):
                episodes.append({
                    "pair_id": pair_id,
                    "role": role,
                    "values": _window(event_pre, future),
                    "cue_mode": cue_mode,
                    "future_mode": future_mode,
                    "pair_index": pair_index,
                })
            no_trigger_pre = _source_pre(background_values, int(rng.choice(source_starts)), history)
            no_trigger_base = _simulate_continuation(no_trigger_pre, horizon, config, rng)
            control_id = f"untriggered_m{cue_mode}_{within_mode:02d}"
            episodes.extend((
                {
                    "pair_id": control_id,
                    "role": "normal_no_trigger",
                    "values": _window(no_trigger_pre, no_trigger_base),
                    "cue_mode": None,
                    "future_mode": None,
                    "pair_index": pair_index,
                },
                {
                    "pair_id": control_id,
                    "role": "untriggered_response",
                    "values": _window(no_trigger_pre, _response_future(no_trigger_base, cue_mode, config)),
                    "cue_mode": None,
                    "future_mode": cue_mode,
                    "pair_index": pair_index,
                },
            ))
            pair_index += 1
    return {
        "suite_id": str(config["suite_id"]),
        "train_values": train_values,
        "background_values": background_values,
        "normal_split_ranges": ranges,
        "normal_event_banks": normal_banks,
        "episodes": episodes,
        "token_extractor": {
            "energy_threshold": float(episodes_config["token_energy_threshold"]),
            "definition": "terminal_displacement_with_half_displacement_onset",
        },
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args(argv)
    config = _load_json(arguments.config)
    suite = generate_suite(config)
    counts = {split: len(bank) for split, bank in suite["normal_event_banks"].items()}
    print(json.dumps({"suite_id": suite["suite_id"], "bank_counts": counts, "episode_count": len(suite["episodes"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
