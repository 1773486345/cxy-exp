#!/usr/bin/env python3
"""Generate the deterministic PatternAD contextual-mechanism suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "patternad" / "synthetic_suite.json"
MECHANISM_ORDER = (
    "same_deviation_different_context",
    "slow_drift_vs_abrupt_shift",
    "dependency_break",
    "context_ood",
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _validated_regime(
    specification: Mapping[str, Any], dimensions: int
) -> Dict[str, Any]:
    transition = np.asarray(specification["transition"], dtype=np.float64)
    std = np.asarray(specification["innovation_std"], dtype=np.float64)
    correlation = np.asarray(
        specification["innovation_correlation"], dtype=np.float64
    )
    expected_matrix = (dimensions, dimensions)
    if transition.shape != expected_matrix or correlation.shape != expected_matrix:
        raise ValueError(
            f"Regime {specification.get('name')!r} matrices must be "
            f"{expected_matrix}."
        )
    if std.shape != (dimensions,) or np.any(std <= 0):
        raise ValueError(
            f"Regime {specification.get('name')!r} innovation_std is invalid."
        )
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if spectral_radius >= 0.98:
        raise ValueError(
            f"Regime {specification.get('name')!r} is not safely stable: "
            f"spectral radius={spectral_radius:.6f}."
        )
    covariance = correlation * np.outer(std, std)
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f"Regime {specification.get('name')!r} covariance is not positive definite."
        ) from error
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("Innovation covariance must have a positive determinant.")
    return {
        "name": str(specification["name"]),
        "transition": transition,
        "covariance": covariance,
        "cholesky": cholesky,
        "precision": np.linalg.inv(covariance),
        "logdet": float(logdet),
        "spectral_radius": spectral_radius,
    }


def validate_config(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    dimensions = int(config["dimensions"])
    if dimensions < 2:
        raise ValueError("The contextual suite must be multivariate.")
    if len(config.get("channel_names", [])) != dimensions:
        raise ValueError("channel_names must match dimensions.")
    replicate_seeds = [int(seed) for seed in config.get("replicate_seeds", [])]
    if replicate_seeds and len(replicate_seeds) != len(set(replicate_seeds)):
        raise ValueError("replicate_seeds must not contain duplicates.")
    seed_groups = config.get("seed_groups", {})
    if set(seed_groups) != {"development", "confirmation"}:
        raise ValueError("seed_groups must define development and confirmation.")
    development_seeds = [int(seed) for seed in seed_groups["development"]]
    confirmation_seeds = [int(seed) for seed in seed_groups["confirmation"]]
    if set(development_seeds) & set(confirmation_seeds):
        raise ValueError("Synthetic development and confirmation seeds must be disjoint.")
    if set(development_seeds + confirmation_seeds) != set(replicate_seeds):
        raise ValueError("seed_groups must partition replicate_seeds exactly.")
    normal_specs = config.get("normal_regimes", [])
    if len(normal_specs) != 2:
        raise ValueError("Exactly two observed normal regimes are required.")
    required_mechanisms = set(MECHANISM_ORDER)
    if set(config.get("mechanisms", {})) != required_mechanisms:
        raise ValueError(
            "The suite must define exactly the four contextual mechanisms."
        )
    segment_length = int(config["regime_segment_length"])
    train_length = int(config["train_length"])
    test_length = int(config["test_length"])
    if segment_length < 16 or train_length % segment_length != 0:
        raise ValueError("train_length must be a multiple of regime_segment_length >= 16.")
    if test_length < 8 * segment_length or test_length % segment_length != 0:
        raise ValueError("test_length must contain at least eight complete regime segments.")
    mechanisms = config["mechanisms"]
    starts = _test_segment_starts(train_length, test_length, segment_length)

    same = mechanisms["same_deviation_different_context"]
    same_count = int(same["pair_count"])
    same_offset = int(same["event_offset"])
    same_length = int(same["event_length"])
    if (
        same_count < 1
        or same_count > min(len(starts[0]), len(starts[1]))
        or same_offset < 0
        or same_length < 1
        or same_offset + same_length > segment_length
    ):
        raise ValueError("same-deviation events do not fit the available test segments.")

    drift = mechanisms["slow_drift_vs_abrupt_shift"]
    drift_count = int(drift["pair_count"])
    drift_offset = int(drift["event_offset"])
    gradual_length = int(drift["gradual_length"])
    abrupt_length = int(drift["abrupt_length"])
    if (
        drift_count < 1
        or 2 * drift_count > len(starts[0])
        or drift_offset < 0
        or drift_offset + max(gradual_length, abrupt_length) > segment_length
    ):
        raise ValueError("drift/shift events do not fit the available quiet segments.")
    _gradual_profile(gradual_length, abrupt_length)

    dependency = mechanisms["dependency_break"]
    dependency_count = int(dependency["event_count"])
    dependency_offset = int(dependency["event_offset"])
    dependency_length = int(dependency["event_length"])
    candidate_permutations = int(dependency.get("candidate_permutations", 0))
    if (
        dependency_count < 1
        or dependency_count > len(starts[0])
        or dependency_offset < 0
        or dependency_length < 2
        or candidate_permutations < 2
        or dependency_offset + dependency_length > segment_length
    ):
        raise ValueError("dependency-break events do not fit the test segments.")

    context_ood = mechanisms["context_ood"]
    ood_count = int(context_ood["event_count"])
    ood_offset = int(context_ood["event_offset"])
    ood_length = int(context_ood["event_length"])
    recovery_length = int(context_ood["recovery_length"])
    total_segments = len(starts[0]) + len(starts[1])
    if (
        ood_count < 1
        or ood_count > max(total_segments - 2, 0)
        or ood_offset < 0
        or ood_length < 1
        or recovery_length < 0
        or ood_offset + ood_length + recovery_length > segment_length
    ):
        raise ValueError("context-OOD events do not fit the interior test segments.")
    score_window_length = int(config["evaluation"].get("score_window_length", 1))
    calibration_gap = int(config["evaluation"]["calibration_gap"])
    if score_window_length < 1 or calibration_gap < score_window_length - 1:
        raise ValueError(
            "calibration_gap must cover at least score_window_length - 1 points."
        )
    ap_requirements = config["evaluation"].get(
        "minimum_ap_over_prevalence", {}
    )
    expected_ap_mechanisms = set(MECHANISM_ORDER) - {"context_ood"}
    if set(ap_requirements) != expected_ap_mechanisms or any(
        float(value) < 0 for value in ap_requirements.values()
    ):
        raise ValueError(
            "minimum_ap_over_prevalence must define non-negative requirements "
            "for the three positive-evidence mechanisms only."
        )
    if float(config["evaluation"].get("maximum_abs_raw_control_margin", -1)) < 0:
        raise ValueError("maximum_abs_raw_control_margin must be non-negative.")
    regimes = [_validated_regime(spec, dimensions) for spec in normal_specs]
    regimes.append(_validated_regime(config["ood_regime"], dimensions))
    return regimes


def _regime_schedule(length: int, segment_length: int) -> np.ndarray:
    return ((np.arange(length) // segment_length) % 2).astype(np.int8)


def _simulate_switching_var(
    length: int,
    burn_in: int,
    segment_length: int,
    regimes: Sequence[Mapping[str, Any]],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = burn_in + length
    # Burn-in must not phase-shift the declared train/test regime boundaries.
    schedule = np.concatenate(
        [
            _regime_schedule(burn_in, segment_length),
            _regime_schedule(length, segment_length),
        ]
    )
    values = np.zeros((total, regimes[0]["transition"].shape[0]), dtype=np.float64)
    for index in range(1, total):
        regime = regimes[int(schedule[index])]
        innovation = regime["cholesky"] @ rng.standard_normal(values.shape[1])
        values[index] = regime["transition"] @ values[index - 1] + innovation
    return values[burn_in:], schedule[burn_in:]


def _test_segment_starts(
    train_length: int, test_length: int, segment_length: int
) -> Dict[int, List[int]]:
    starts: Dict[int, List[int]] = {0: [], 1: []}
    for relative_start in range(0, test_length, segment_length):
        global_start = train_length + relative_start
        regime = (global_start // segment_length) % 2
        starts[regime].append(global_start)
    return starts


def _event(
    event_id: str,
    start: int,
    end: int,
    role: str,
    label: bool,
    pair_id: Optional[str] = None,
) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "event_id": event_id,
        "start": int(start),
        "end": int(end),
        "role": role,
        "label": bool(label),
    }
    if pair_id is not None:
        value["pair_id"] = pair_id
    return value


def _inject_same_deviation(
    clean: np.ndarray,
    labels: np.ndarray,
    fpr_eligible: np.ndarray,
    segments: Mapping[int, Sequence[int]],
    specification: Mapping[str, Any],
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, str]]]:
    values = clean.copy()
    length = int(specification["event_length"])
    offset = int(specification["event_offset"])
    deviation = np.asarray(specification["deviation"], dtype=np.float64)
    events: List[Dict[str, Any]] = []
    orderings: List[Dict[str, str]] = []
    for pair_index in range(int(specification["pair_count"])):
        pair_id = f"same_deviation_pair_{pair_index}"
        ids = {}
        for regime, role in ((0, "quiet_context"), (1, "volatile_context")):
            start = int(segments[regime][pair_index]) + offset
            end = start + length
            event_id = f"{pair_id}_{role}"
            is_anomaly = regime == 0
            values[start:end] += deviation
            if is_anomaly:
                labels[start:end] = 1
            fpr_eligible[start:end] = False
            events.append(
                _event(event_id, start, end, role, is_anomaly, pair_id)
            )
            ids[regime] = event_id
        orderings.append(
            {
                "name": pair_id,
                "higher_event": ids[0],
                "lower_event": ids[1],
                "hypothesis": "equal observed deviation is more surprising in quiet context",
            }
        )
    return values, events, orderings


def _gradual_profile(length: int, plateau_length: int) -> Tuple[np.ndarray, int]:
    if plateau_length < 1 or length < plateau_length + 4:
        raise ValueError(
            "A gradual drift needs a positive plateau and at least two ramp points "
            "on each side."
        )
    ramp_total = length - plateau_length
    up_length = ramp_total // 2
    down_length = ramp_total - up_length
    up = np.linspace(0.0, 1.0, up_length + 1, endpoint=True)[:-1]
    plateau = np.ones(plateau_length, dtype=np.float64)
    down = np.linspace(1.0, 0.0, down_length + 1, endpoint=True)[1:]
    return np.concatenate([up, plateau, down]), up_length


def _inject_drift_shift(
    clean: np.ndarray,
    labels: np.ndarray,
    fpr_eligible: np.ndarray,
    segments: Mapping[int, Sequence[int]],
    specification: Mapping[str, Any],
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, str]]]:
    values = clean.copy()
    pair_count = int(specification["pair_count"])
    quiet_segments = list(segments[0])
    if len(quiet_segments) < 2 * pair_count:
        raise ValueError("Not enough quiet-regime segments for drift/shift pairs.")
    gradual_length = int(specification["gradual_length"])
    abrupt_length = int(specification["abrupt_length"])
    offset = int(specification["event_offset"])
    shift = np.asarray(specification["endpoint_shift"], dtype=np.float64)
    profile, plateau_offset = _gradual_profile(gradual_length, abrupt_length)
    events: List[Dict[str, Any]] = []
    orderings: List[Dict[str, str]] = []
    for pair_index in range(pair_count):
        pair_id = f"drift_shift_pair_{pair_index}"
        gradual_start = quiet_segments[pair_index] + offset
        gradual_end = gradual_start + gradual_length
        gradual_id = f"{pair_id}_gradual_plateau_reference"
        values[gradual_start:gradual_end] += profile[:, None] * shift[None, :]
        fpr_eligible[gradual_start:gradual_end] = False
        plateau_start = gradual_start + plateau_offset
        plateau_end = plateau_start + abrupt_length
        gradual_event = _event(
            gradual_id,
            plateau_start,
            plateau_end,
            "gradual_plateau_reference",
            False,
            pair_id,
        )
        gradual_event["injection_start"] = gradual_start
        gradual_event["injection_end"] = gradual_end
        events.append(gradual_event)

        abrupt_start = quiet_segments[pair_index + pair_count] + offset
        abrupt_end = abrupt_start + abrupt_length
        abrupt_id = f"{pair_id}_abrupt_anomaly"
        values[abrupt_start:abrupt_end] += shift
        labels[abrupt_start:abrupt_end] = 1
        fpr_eligible[abrupt_start:abrupt_end] = False
        events.append(
            _event(
                abrupt_id,
                abrupt_start,
                abrupt_end,
                "abrupt_anomaly",
                True,
                pair_id,
            )
        )
        orderings.append(
            {
                "name": pair_id,
                "higher_event": abrupt_id,
                "lower_event": gradual_id,
                "hypothesis": "abrupt shift outranks a matched gradual excursion",
            }
        )
    return values, events, orderings


def _inject_dependency_break(
    clean: np.ndarray,
    labels: np.ndarray,
    fpr_eligible: np.ndarray,
    segments: Mapping[int, Sequence[int]],
    specification: Mapping[str, Any],
    regimes: np.ndarray,
    normal_regimes: Sequence[Mapping[str, Any]],
    seed: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, str]]]:
    values = clean.copy()
    # Use quiet segments so a preserved-marginal relation break cannot be
    # hidden by the intentionally broad volatile innovation distribution.
    starts = list(segments[0])
    event_count = int(specification["event_count"])
    chosen = starts[:event_count]
    length = int(specification["event_length"])
    offset = int(specification["event_offset"])
    channels = [int(value) for value in specification["shifted_channels"]]
    circular_offsets = [int(value) for value in specification["circular_offsets"]]
    candidate_count = int(specification["candidate_permutations"])
    if len(channels) != len(circular_offsets):
        raise ValueError("shifted_channels and circular_offsets must have equal length.")
    events = []
    for event_index, segment_start in enumerate(chosen):
        start = segment_start + offset
        end = start + length
        rng = np.random.default_rng(seed + event_index)
        for channel, circular_offset in zip(channels, circular_offsets):
            source = clean[start:end, channel].copy()
            candidates = [
                np.roll(source[::-1], circular_offset % length),
                *[source[rng.permutation(length)] for _ in range(candidate_count - 1)],
            ]
            best_score = -np.inf
            best_candidate = None
            original = values[start:end, channel].copy()
            for candidate in candidates:
                values[start:end, channel] = candidate
                score = 0.0
                for index in range(start, end):
                    score += _gaussian_nll(
                        values[index],
                        values[index - 1],
                        normal_regimes[int(regimes[index])],
                    )
                if score > best_score:
                    best_score = score
                    best_candidate = candidate.copy()
            values[start:end, channel] = (
                best_candidate if best_candidate is not None else original
            )
        labels[start:end] = 1
        fpr_eligible[start:end] = False
        dependency_event = _event(
            f"dependency_break_{event_index}",
            start,
            end,
            "dependency_break",
            True,
        )
        dependency_event["construction"] = (
            "per_channel_empirical_marginal_preserved; permutation selected from "
            f"{candidate_count} seeded candidates by ground-truth normal NLL"
        )
        events.append(dependency_event)
    return values, events, []


def _inject_context_ood(
    clean: np.ndarray,
    labels: np.ndarray,
    fpr_eligible: np.ndarray,
    regimes: np.ndarray,
    segments: Mapping[int, Sequence[int]],
    specification: Mapping[str, Any],
    ood_regime: Mapping[str, Any],
    seed: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, str]]]:
    values = clean.copy()
    starts = sorted(list(segments[0]) + list(segments[1]))
    event_count = int(specification["event_count"])
    chosen = np.linspace(1, len(starts) - 2, event_count, dtype=int)
    length = int(specification["event_length"])
    offset = int(specification["event_offset"])
    recovery = int(specification["recovery_length"])
    rng = np.random.default_rng(seed)
    events = []
    for event_index, chosen_index in enumerate(chosen):
        start = starts[int(chosen_index)] + offset
        injected_end = start + length
        end = min(injected_end + recovery, len(values))
        for index in range(start, injected_end):
            innovation = ood_regime["cholesky"] @ rng.standard_normal(values.shape[1])
            values[index] = (
                ood_regime["transition"] @ values[index - 1] + innovation
            )
        regimes[start:injected_end] = 2
        labels[start:end] = 1
        fpr_eligible[start:end] = False
        events.append(
            _event(
                f"context_ood_{event_index}",
                start,
                end,
                "unseen_coherent_regime_anomaly",
                True,
            )
        )
    return values, events, []


def _gaussian_nll(
    current: np.ndarray, previous: np.ndarray, regime: Mapping[str, Any]
) -> float:
    residual = current - regime["transition"] @ previous
    dimension = residual.size
    quadratic = float(residual @ regime["precision"] @ residual)
    return 0.5 * (quadratic + regime["logdet"] + dimension * np.log(2.0 * np.pi))


def _gaussian_quadratic_surprise(
    current: np.ndarray, previous: np.ndarray, regime: Mapping[str, Any]
) -> float:
    """Regime-invariant Gaussian tail ordering before the chi-square CDF."""
    residual = current - regime["transition"] @ previous
    return float(residual @ regime["precision"] @ residual)


def _oracle_context_score(
    values: np.ndarray,
    regimes: np.ndarray,
    normal_regimes: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    scores = np.zeros(len(values), dtype=np.float64)
    for index in range(1, len(values)):
        regime_id = int(regimes[index])
        if regime_id in (0, 1):
            scores[index] = _gaussian_quadratic_surprise(
                values[index], values[index - 1], normal_regimes[regime_id]
            )
        else:
            scores[index] = min(
                _gaussian_quadratic_surprise(
                    values[index], values[index - 1], regime
                )
                for regime in normal_regimes
            )
    scores[0] = float(np.median(scores[1:]))
    return scores


def generate_suite(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    validated = validate_config(config)
    normal_regimes = validated[:2]
    ood_regime = validated[2]
    train_length = int(config["train_length"])
    test_length = int(config["test_length"])
    total_length = train_length + test_length
    clean, base_regimes = _simulate_switching_var(
        total_length,
        int(config["burn_in"]),
        int(config["regime_segment_length"]),
        normal_regimes,
        int(config["seed"]),
    )
    segments = _test_segment_starts(
        train_length, test_length, int(config["regime_segment_length"])
    )
    artifacts: Dict[str, Dict[str, Any]] = {}
    for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
        labels = np.zeros(total_length, dtype=np.uint8)
        fpr_eligible = np.ones(total_length, dtype=bool)
        regimes = base_regimes.copy()
        specification = config["mechanisms"][mechanism]
        if mechanism == "same_deviation_different_context":
            values, events, orderings = _inject_same_deviation(
                clean, labels, fpr_eligible, segments, specification
            )
        elif mechanism == "slow_drift_vs_abrupt_shift":
            values, events, orderings = _inject_drift_shift(
                clean, labels, fpr_eligible, segments, specification
            )
        elif mechanism == "dependency_break":
            values, events, orderings = _inject_dependency_break(
                clean,
                labels,
                fpr_eligible,
                segments,
                specification,
                regimes,
                normal_regimes,
                int(config["seed"]) + 2000 + mechanism_index,
            )
        else:
            values, events, orderings = _inject_context_ood(
                clean,
                labels,
                fpr_eligible,
                regimes,
                segments,
                specification,
                ood_regime,
                int(config["seed"]) + 1000 + mechanism_index,
            )
        if np.any(labels[:train_length]):
            raise RuntimeError(f"{mechanism} contaminated the official train split.")
        ordered_events = sorted(events, key=lambda event: (event["start"], event["end"]))
        for event in ordered_events:
            injection_start = int(event.get("injection_start", event["start"]))
            injection_end = int(event.get("injection_end", event["end"]))
            if not (
                train_length <= injection_start < injection_end <= total_length
                and train_length <= int(event["start"]) < int(event["end"]) <= total_length
            ):
                raise ValueError(
                    f"{mechanism} event {event['event_id']!r} is outside the test split."
                )
        injection_spans = sorted(
            (
                int(event.get("injection_start", event["start"])),
                int(event.get("injection_end", event["end"])),
                event["event_id"],
            )
            for event in ordered_events
        )
        for previous, current in zip(injection_spans, injection_spans[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    f"{mechanism} events overlap: {previous[2]!r} and {current[2]!r}."
                )
        score_guard = int(config["evaluation"].get("score_window_length", 1)) - 1
        for start, end, _ in injection_spans:
            guarded_start = max(train_length, start - score_guard)
            guarded_end = min(total_length, end + score_guard)
            fpr_eligible[guarded_start:guarded_end] = False
        artifacts[mechanism] = {
            "values": values.astype(np.float32),
            "clean_values": clean.astype(np.float32),
            "labels": labels,
            "regime": regimes.astype(np.int8),
            "fpr_eligible": fpr_eligible.astype(np.uint8),
            "oracle_context_score": _oracle_context_score(
                values, regimes, normal_regimes
            ),
            "events": events,
            "orderings": orderings,
        }
    return artifacts


def _benchmark_stem(prefix: str, mechanism: str, seed: int) -> str:
    compact = "".join(part.title() for part in mechanism.split("_"))
    return f"{prefix}_Seed{seed}_{compact}"


def _write_benchmark_csv(
    path: Path,
    values: np.ndarray,
    labels: np.ndarray,
    channel_names: Sequence[str],
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=len(values), freq="min")
    blocks = []
    for channel_index, channel_name in enumerate(channel_names):
        blocks.append(
            pd.DataFrame(
                {
                    "date": timestamps,
                    "data": values[:, channel_index],
                    "cols": channel_name,
                }
            )
        )
    blocks.append(
        pd.DataFrame({"date": timestamps, "data": labels, "cols": "label"})
    )
    pd.concat(blocks, ignore_index=True).to_csv(path, index=False)


def _write_static_text(path: Path, suite_id: str, mechanism: str) -> None:
    pd.DataFrame(
        {
            "date": ["2024-01-01 00:00:00"],
            "data": [
                f"Synthetic switching-VAR fixture {suite_id}; mechanism={mechanism}."
            ],
            "cols": ["description"],
        }
    ).to_csv(path, index=False)


def _metadata_row(
    columns: Sequence[str], file_name: str, suite_id: str, train_length: int, total: int
) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {column: "" for column in columns}
    values = {
        "file_name": file_name,
        "trend": False,
        "seasonal": False,
        "stationary": False,
        "pattern": True,
        "shifting": True,
        "dataset_name": suite_id,
        "train_lens": train_length,
        "time_steps": total,
        "if_univariate": False,
        "size": "user",
        "type_value": "mult_new_new",
        "total_len": total,
        "train/total": train_length / total,
    }
    defaults.update({key: value for key, value in values.items() if key in defaults})
    return defaults


def _register_metadata(metadata_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    metadata = pd.read_csv(metadata_path)
    names = {str(row["file_name"]) for row in rows}
    metadata = metadata[~metadata["file_name"].astype(str).isin(names)]
    additions = pd.DataFrame(rows, columns=metadata.columns)
    combined = pd.concat([metadata, additions], ignore_index=True)
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    os.replace(temporary, metadata_path)


def write_suite(
    config: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    register_benchmark: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _canonical_hash(config)
    source_paths = {
        "generator": Path(__file__).resolve(),
        "evaluator": Path(__file__).resolve().with_name(
            "evaluate_contextual_mechanisms.py"
        ),
    }
    missing_sources = [path for path in source_paths.values() if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Missing synthetic-suite source: {missing_sources[0]}")
    source_hashes = {
        name: _file_sha256(path) for name, path in source_paths.items()
    }
    resolved_config_path = output_dir / "resolved_config.json"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    prefix = str(config["output"]["file_prefix"])
    train_length = int(config["train_length"])
    total_length = train_length + int(config["test_length"])
    manifest_entries = []
    registration_rows = []
    benchmark_target_dir = _resolve_repo_path(config["output"]["benchmark_data_dir"])
    if register_benchmark:
        benchmark_target_dir.mkdir(parents=True, exist_ok=True)

    for mechanism in MECHANISM_ORDER:
        artifact = artifacts[mechanism]
        stem = _benchmark_stem(prefix, mechanism, int(config["seed"]))
        npz_path = output_dir / f"{mechanism}.npz"
        np.savez_compressed(
            npz_path,
            values=artifact["values"],
            clean_values=artifact["clean_values"],
            labels=artifact["labels"],
            regime=artifact["regime"],
            fpr_eligible=artifact["fpr_eligible"],
            oracle_context_score=artifact["oracle_context_score"],
            split_index=np.asarray(train_length, dtype=np.int64),
        )
        sidecar_path = output_dir / f"{mechanism}.metadata.json"
        sidecar = {
            "schema_version": 1,
            "suite_id": config["suite_id"],
            "config_hash": config_hash,
            "source_hashes": source_hashes,
            "mechanism": mechanism,
            "seed": int(config["seed"]),
            "shape": list(artifact["values"].shape),
            "train_length": train_length,
            "test_length": int(config["test_length"]),
            "events": artifact["events"],
            "orderings": artifact["orderings"],
            "npz_file": npz_path.name,
            "score_contract": {
                "alignment": "one point score for every row, train then test",
                "external_key": config["evaluation"]["score_key"],
                "oracle_key": "oracle_context_score",
                "oracle_semantics": (
                    "regime-conditioned squared Mahalanobis surprise; monotone "
                    "with the conditional chi-square tail probability"
                ),
                "fpr_exclusion_guard": int(
                    config["evaluation"].get("score_window_length", 1)
                )
                - 1,
            },
        }
        with sidecar_path.open("w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")

        csv_name = f"{stem}.csv"
        text_name = f"{stem}_text.csv"
        csv_path = benchmark_dir / csv_name
        text_path = benchmark_dir / text_name
        _write_benchmark_csv(
            csv_path,
            artifact["values"],
            artifact["labels"],
            config["channel_names"],
        )
        _write_static_text(text_path, str(config["suite_id"]), mechanism)
        if register_benchmark:
            _write_benchmark_csv(
                benchmark_target_dir / csv_name,
                artifact["values"],
                artifact["labels"],
                config["channel_names"],
            )
            _write_static_text(
                benchmark_target_dir / text_name, str(config["suite_id"]), mechanism
            )

        row = _metadata_row(
            pd.read_csv(_resolve_repo_path(config["output"]["benchmark_metadata"]), nrows=0).columns,
            csv_name,
            str(config["suite_id"]),
            train_length,
            total_length,
        )
        registration_rows.append(row)
        manifest_entries.append(
            {
                "mechanism": mechanism,
                "npz": npz_path.name,
                "metadata": sidecar_path.name,
                "benchmark_data_name": csv_name,
                "benchmark_text_name": text_name,
                "npz_sha256": _file_sha256(npz_path),
                "benchmark_csv_sha256": _file_sha256(csv_path),
            }
        )

    if register_benchmark:
        _register_metadata(
            _resolve_repo_path(config["output"]["benchmark_metadata"]),
            registration_rows,
        )
    manifest = {
        "schema_version": 1,
        "suite_id": config["suite_id"],
        "config_hash": config_hash,
        "source_hashes": source_hashes,
        "config": config,
        "resolved_config": resolved_config_path.name,
        "registered_with_benchmark": bool(register_benchmark),
        "entries": manifest_entries,
    }
    manifest_path = output_dir / "suite_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the configured generator seed for an independent replicate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output.artifact_dir from the suite config.",
    )
    parser.add_argument(
        "--register-benchmark",
        action="store_true",
        help="Also install CSVs into dataset/anomaly_detect/data and update DETECT_META.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = _load_json(args.config.resolve())
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = _resolve_repo_path(config["output"]["artifact_dir"])
        if args.seed is not None:
            output_dir = output_dir / f"seed_{args.seed}"
    artifacts = generate_suite(config)
    manifest_path = write_suite(
        config, artifacts, output_dir, register_benchmark=args.register_benchmark
    )
    print(f"Generated {len(artifacts)} mechanisms: {manifest_path}")
    if not args.register_benchmark:
        print("Benchmark files were staged only; pass --register-benchmark to register them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
