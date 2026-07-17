"""Fixed synthetic data and anomaly injections for decomposition gate v0."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


TRAIN_LENGTH = 7680
TRAIN_CORE_LENGTH = 6144
VALIDATION_LENGTH = 1536
TEST_LENGTH = 3072
N_VARIABLES = 4
FREQUENCY = "s"
START_TIME = pd.Timestamp("2024-01-01T00:00:00")
TRAIN_SEEDS = (20260717, 20260718, 20260719)
SCORING_SEED = 20260717
ANOMALY_TYPES = (
    "level_shift",
    "slope_change",
    "spike",
    "variance_increase",
    "periodic_amplitude",
    "periodic_phase",
)

Z1_WEIGHTS = np.array([1.0, 0.8, 0.4, -0.3], dtype=np.float32)
Z2_WEIGHTS = np.array([0.2, -0.3, 1.0, 0.8], dtype=np.float32)
SLOW_AMPLITUDES = np.array([0.16, 0.12, 0.10, 0.14], dtype=np.float32)
SLOW_PERIODS = np.array([1536, 1920, 1280, 2560], dtype=np.float32)
SLOW_PHASES = np.array([0.1, 0.7, 1.1, 1.8], dtype=np.float32)
SLOW_TRENDS = np.array([0.08, -0.06, 0.04, -0.05], dtype=np.float32)
SHARED_NOISE_STD = 0.05
INDEPENDENT_NOISE_STD = 0.03

ANOMALY_SPECS = {
    "level_shift": {"length": 96, "variables": (0, 1), "multiplier": 2.5},
    "slope_change": {"length": 192, "variables": (0, 1), "multiplier": 3.0},
    "spike": {"count": 12, "minimum_gap": 48, "multiplier": 5.0},
    "variance_increase": {"length": 192, "variables": (2, 3), "multiplier": 3.0},
    "periodic_amplitude": {"length": 384, "variables": (0, 1), "multiplier": 2.0},
    "periodic_phase": {"length": 384, "variables": (2, 3), "phase_shift": np.pi / 2},
}


@dataclass(frozen=True)
class NormalSeries:
    frame: pd.DataFrame
    time_seconds: np.ndarray
    z1: np.ndarray
    z2: np.ndarray
    baseline_hash: str


def fixed_generator_parameters() -> Dict[str, object]:
    """Return the complete immutable normal and anomaly parameter set."""
    return {
        "z1_weights": Z1_WEIGHTS.tolist(),
        "z2_weights": Z2_WEIGHTS.tolist(),
        "slow_amplitudes": SLOW_AMPLITUDES.tolist(),
        "slow_periods": SLOW_PERIODS.tolist(),
        "slow_phases": SLOW_PHASES.tolist(),
        "slow_trends": SLOW_TRENDS.tolist(),
        "shared_noise_std": SHARED_NOISE_STD,
        "independent_noise_std": INDEPENDENT_NOISE_STD,
        "anomaly_specs": {
            name: {key: (list(value) if isinstance(value, tuple) else value) for key, value in spec.items()}
            for name, spec in ANOMALY_SPECS.items()
        },
    }


def generate_training_series(seed: int) -> NormalSeries:
    return _generate_normal_series(seed, TRAIN_LENGTH, 0, stream_offset=101)


def generate_test_baseline(seed: int) -> NormalSeries:
    return _generate_normal_series(seed, TEST_LENGTH, TRAIN_LENGTH, stream_offset=202)


def split_training_validation(train: NormalSeries) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(train.frame) != TRAIN_LENGTH:
        raise ValueError("training series must have the pre-registered length")
    return train.frame.iloc[:TRAIN_CORE_LENGTH].copy(), train.frame.iloc[TRAIN_CORE_LENGTH:].copy()


def precompute_anomaly_events(seed: int) -> Dict[str, object]:
    """Generate deterministic, in-bounds event locations independent of model windows."""
    rng = np.random.default_rng(seed + 303)

    def segment_start(length: int) -> int:
        return int(rng.integers(128, TEST_LENGTH - length - 128 + 1))

    spike_bins = np.sort(rng.choice(np.arange(2, TEST_LENGTH // 48 - 2), size=12, replace=False))
    return {
        "level_shift": {"start": segment_start(96), "length": 96},
        "slope_change": {"start": segment_start(192), "length": 192},
        "spike": {
            "positions": (spike_bins * 48).astype(int).tolist(),
            "variables": rng.integers(0, N_VARIABLES, size=12).astype(int).tolist(),
        },
        "variance_increase": {"start": segment_start(192), "length": 192},
        "periodic_amplitude": {"start": segment_start(384), "length": 384},
        "periodic_phase": {"start": segment_start(384), "length": 384},
    }


def inject_anomaly(
    baseline: NormalSeries,
    train_std: np.ndarray,
    anomaly_type: str,
    events: Dict[str, object],
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Inject exactly one pre-registered anomaly type into a copied normal baseline."""
    if anomaly_type not in ANOMALY_TYPES:
        raise ValueError(f"unknown anomaly type: {anomaly_type}")
    if len(baseline.frame) != TEST_LENGTH:
        raise ValueError("baseline must have the pre-registered test length")

    values = baseline.frame.to_numpy(copy=True)
    labels = np.zeros(TEST_LENGTH, dtype=np.int64)
    event = events[anomaly_type]

    if anomaly_type == "level_shift":
        indices = _segment_indices(event)
        values[np.ix_(indices, [0, 1])] += 2.5 * train_std[[0, 1]]
        labels[indices] = 1
    elif anomaly_type == "slope_change":
        indices = _segment_indices(event)
        ramp = np.linspace(0.0, 3.0, len(indices), dtype=np.float32)[:, None]
        values[np.ix_(indices, [0, 1])] += ramp * train_std[[0, 1]]
        labels[indices] = 1
    elif anomaly_type == "spike":
        positions = np.asarray(event["positions"], dtype=int)
        variables = np.asarray(event["variables"], dtype=int)
        for position, variable in zip(positions, variables):
            values[position, variable] += 5.0 * train_std[variable]
        labels[positions] = 1
    elif anomaly_type == "variance_increase":
        indices = _segment_indices(event)
        rng = np.random.default_rng(seed + 404)
        added_noise = rng.normal(
            0.0, 3.0 * train_std[[2, 3]], size=(len(indices), 2)
        ).astype(np.float32)
        values[np.ix_(indices, [2, 3])] += added_noise
        labels[indices] = 1
    elif anomaly_type == "periodic_amplitude":
        indices = _segment_indices(event)
        values[np.ix_(indices, [0, 1])] += baseline.z1[indices, None] * Z1_WEIGHTS[[0, 1]]
        labels[indices] = 1
    elif anomaly_type == "periodic_phase":
        indices = _segment_indices(event)
        phase_z2 = np.sin(2.0 * np.pi * baseline.time_seconds[indices] / 24.0 + np.pi / 2)
        delta = phase_z2 - baseline.z2[indices]
        values[np.ix_(indices, [2, 3])] += delta[:, None] * Z2_WEIGHTS[[2, 3]]
        labels[indices] = 1

    return pd.DataFrame(values, index=baseline.frame.index, columns=baseline.frame.columns), labels


def _generate_normal_series(seed: int, length: int, start_offset: int, stream_offset: int) -> NormalSeries:
    rng = np.random.default_rng(seed + stream_offset)
    time_seconds = np.arange(start_offset, start_offset + length, dtype=np.float32)
    z1 = np.sin(2.0 * np.pi * time_seconds / 48.0) + 0.5 * np.sin(
        2.0 * np.pi * time_seconds / 96.0 + 0.3
    )
    z2 = np.sin(2.0 * np.pi * time_seconds / 24.0)
    normalized_time = (time_seconds - time_seconds.mean()) / max(float(length), 1.0)
    slow = SLOW_AMPLITUDES * np.sin(
        2.0 * np.pi * time_seconds[:, None] / SLOW_PERIODS + SLOW_PHASES
    ) + normalized_time[:, None] * SLOW_TRENDS
    shared_noise = rng.normal(0.0, SHARED_NOISE_STD, size=(length, 1)).astype(np.float32)
    independent_noise = rng.normal(
        0.0, INDEPENDENT_NOISE_STD, size=(length, N_VARIABLES)
    ).astype(np.float32)
    values = (
        z1[:, None] * Z1_WEIGHTS
        + z2[:, None] * Z2_WEIGHTS
        + slow
        + shared_noise
        + independent_noise
    ).astype(np.float32)
    index = pd.date_range(START_TIME + pd.Timedelta(seconds=start_offset), periods=length, freq=FREQUENCY)
    frame = pd.DataFrame(values, index=index, columns=[f"x{index}" for index in range(N_VARIABLES)])
    return NormalSeries(
        frame=frame,
        time_seconds=time_seconds,
        z1=z1.astype(np.float32),
        z2=z2.astype(np.float32),
        baseline_hash=hashlib.sha256(values.tobytes()).hexdigest(),
    )


def _segment_indices(event: Dict[str, object]) -> np.ndarray:
    start = int(event["start"])
    length = int(event["length"])
    indices = np.arange(start, start + length, dtype=int)
    if start < 0 or indices[-1] >= TEST_LENGTH:
        raise ValueError("pre-registered event is out of bounds")
    return indices
