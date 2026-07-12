#!/usr/bin/env python3
"""Generate the paired synthetic contract for Direction B0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "multi_evidence" / "b0_synthetic.json"
ROLE_ORDER = (
    "coherent_control",
    "unsupported_target_break",
    "target_omission_break",
    "target_spike",
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
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("B0 config must use schema_version=1.")
    dimensions = int(config["dimensions"])
    history = int(config["history_length"])
    if dimensions < 2 or history < 2:
        raise ValueError("B0 requires at least two channels and history_length >= 2.")
    if len(config.get("channel_names", [])) != dimensions:
        raise ValueError("channel_names must match dimensions.")
    if int(config["train_length"]) <= 6 * history or int(config["test_length"]) <= history:
        raise ValueError("train_length/test_length are too short for B0 splits and windows.")
    process = config["normal_process"]
    for key in ("loadings", "channel_ar", "channel_noise"):
        values = np.asarray(process[key], dtype=np.float64)
        if values.shape != (dimensions,):
            raise ValueError(f"normal_process.{key} must have {dimensions} entries.")
    if not np.all(np.asarray(process["channel_noise"], dtype=np.float64) > 0.0):
        raise ValueError("channel_noise must be positive.")
    if not 0.0 <= float(process["latent_ar"]) < 0.99:
        raise ValueError("latent_ar must be in [0, 0.99).")
    if not 0.0 < float(process["latent_std"]):
        raise ValueError("latent_std must be positive.")
    segment = int(process["regime_segment_length"])
    scales = np.asarray(process["regime_noise_scales"], dtype=np.float64)
    if segment < history + 2 or scales.shape != (2,) or np.any(scales <= 0.0):
        raise ValueError("normal regime configuration is invalid.")
    episodes = config["episodes"]
    target = int(episodes["target_channel"])
    if not 0 <= target < dimensions:
        raise ValueError("episodes.target_channel is outside dimensions.")
    if int(episodes["episodes_per_regime"]) < 1:
        raise ValueError("episodes_per_regime must be positive.")
    if float(episodes["shock_magnitude"]) <= 0.0:
        raise ValueError("shock_magnitude must be positive.")
    split = config["split"]
    fractions = [
        float(split["outer_calibration_fraction"]),
        float(split["validation_fraction"]),
        float(split["reference_fraction"]),
    ]
    if any(value <= 0.0 for value in fractions) or sum(fractions) >= 0.8:
        raise ValueError("B0 split fractions must be positive and leave optimization data.")
    model = config["model"]
    if int(model["d_model"]) < 1 or int(model["batch_size"]) < 1:
        raise ValueError("B0 model dimensions are invalid.")
    if float(model["learning_rate"]) <= 0.0 or int(model["epochs"]) < 1:
        raise ValueError("B0 model training configuration is invalid.")
    if float(model.get("dropout", 0.0)) != 0.0:
        raise ValueError("B0 deliberately forbids dropout.")
    evaluation = config["evaluation"]
    if not 0.0 < float(evaluation["target_fpr"]) < 1.0:
        raise ValueError("evaluation.target_fpr must be between zero and one.")
    pairs = 2 * int(episodes["episodes_per_regime"])
    if not 1 <= int(evaluation["paired_order_min"]) <= pairs:
        raise ValueError("paired_order_min is incompatible with episode count.")
    if not 0 <= int(evaluation["coherent_exceedance_max"]) <= pairs:
        raise ValueError("coherent_exceedance_max is incompatible with episode count.")
    if not 1 <= int(evaluation["target_spike_exceedance_min"]) <= pairs:
        raise ValueError("target_spike_exceedance_min is incompatible with episode count.")


def _simulate_factor_var(
    length: int, config: Mapping[str, Any], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    process = config["normal_process"]
    dimensions = int(config["dimensions"])
    burn_in = int(config["burn_in"])
    total = length + burn_in
    segment = int(process["regime_segment_length"])
    regime = (np.arange(total, dtype=np.int64) // segment) % 2
    scales = np.asarray(process["regime_noise_scales"], dtype=np.float64)
    loadings = np.asarray(process["loadings"], dtype=np.float64)
    channel_ar = np.asarray(process["channel_ar"], dtype=np.float64)
    channel_noise = np.asarray(process["channel_noise"], dtype=np.float64)
    values = np.zeros((total, dimensions), dtype=np.float64)
    latent_values = np.zeros(total, dtype=np.float64)
    latent = 0.0
    for index in range(total):
        scale = scales[regime[index]]
        latent = (
            float(process["latent_ar"]) * latent
            + float(process["latent_std"]) * scale * rng.normal()
        )
        latent_values[index] = latent
        previous = values[index - 1] if index else np.zeros(dimensions, dtype=np.float64)
        values[index] = (
            channel_ar * previous
            + loadings * latent
            + channel_noise * scale * rng.normal(size=dimensions)
        )
    return (
        values[burn_in:].astype(np.float32),
        regime[burn_in:].astype(np.int64),
        latent_values[burn_in:].astype(np.float32),
    )


def _select_sources(
    regimes: np.ndarray, history: int, per_regime: int
) -> Dict[int, np.ndarray]:
    selected: Dict[int, np.ndarray] = {}
    for regime in (0, 1):
        candidates = np.flatnonzero((regimes == regime) & (np.arange(len(regimes)) >= history))
        if len(candidates) < per_regime:
            raise ValueError("test stream cannot provide the requested paired episodes.")
        positions = np.linspace(0, len(candidates) - 1, per_regime, dtype=np.int64)
        selected[regime] = candidates[positions]
    return selected


def _select_counterfactual_donor(
    regimes: np.ndarray,
    latent: np.ndarray,
    history: int,
    source_terminal_index: int,
) -> int:
    candidates = np.flatnonzero(
        (regimes == regimes[source_terminal_index])
        & (np.arange(len(regimes)) >= history)
        & (np.abs(np.arange(len(regimes)) - source_terminal_index) > history)
    )
    if len(candidates) == 0:
        raise ValueError("Cannot find a non-overlapping normal counterfactual donor.")
    # This is generator-only construction: select an otherwise normal window
    # with a sufficiently different latent realization to make the relation
    # break observable. Latent state is never given to a model or calibrator.
    distance = np.abs(latent[candidates] - latent[source_terminal_index])
    return int(candidates[int(np.argmax(distance))])


def generate_suite(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Generate clean streams and exact paired counterfactual windows."""
    validate_config(config)
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    train_values, train_regime, _ = _simulate_factor_var(
        int(config["train_length"]), config, rng
    )
    background_values, background_regime, background_latent = _simulate_factor_var(
        int(config["test_length"]), config, rng
    )
    history = int(config["history_length"])
    episodes_config = config["episodes"]
    target = int(episodes_config["target_channel"])
    loadings = np.asarray(config["normal_process"]["loadings"], dtype=np.float32)
    drivers = np.asarray([index for index in range(int(config["dimensions"])) if index != target])
    sources = _select_sources(
        background_regime, history, int(episodes_config["episodes_per_regime"])
    )
    episodes: List[Dict[str, Any]] = []
    contracts: List[Dict[str, Any]] = []
    for regime in (0, 1):
        for ordinal, terminal_index in enumerate(sources[regime]):
            coherent = background_values[
                terminal_index - history : terminal_index + 1
            ].copy()
            donor_terminal_index = _select_counterfactual_donor(
                background_regime, background_latent, history, int(terminal_index)
            )
            donor = background_values[
                donor_terminal_index - history : donor_terminal_index + 1
            ].copy()
            sign = 1.0 if ordinal % 2 == 0 else -1.0
            unsupported = coherent.copy()
            unsupported[:, drivers] = donor[:, drivers]
            omission = coherent.copy()
            omission[:, target] = donor[:, target]
            spike = coherent.copy()
            spike[-1, target] += (
                sign
                * float(episodes_config["shock_magnitude"])
                * float(episodes_config["target_spike_multiplier"])
                * abs(float(loadings[target]))
            )
            pair_id = f"regime_{regime}_pair_{ordinal:02d}"
            role_windows = {
                "coherent_control": coherent,
                "unsupported_target_break": unsupported,
                "target_omission_break": omission,
                "target_spike": spike,
            }
            for role in ROLE_ORDER:
                episodes.append(
                    {
                        "pair_id": pair_id,
                        "role": role,
                        "regime": int(regime),
                        "source_terminal_index": int(terminal_index),
                        "donor_terminal_index": int(donor_terminal_index),
                        "values": role_windows[role].astype(np.float32),
                    }
                )
            contracts.append(
                {
                    "pair_id": pair_id,
                    "regime": int(regime),
                    "source_terminal_index": int(terminal_index),
                    "donor_terminal_index": int(donor_terminal_index),
                    "coherent_unsupported_target_max_abs_difference": float(
                        np.max(
                            np.abs(
                                coherent[:, target] - unsupported[:, target]
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
                        np.max(np.abs(coherent[:, target] - omission[:, target]))
                    ),
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
    """Persist all synthetic inputs required to reproduce a B0 result."""
    output_dir.mkdir(parents=True, exist_ok=False)
    episodes = list(suite["episodes"])
    windows = np.stack([episode["values"] for episode in episodes], axis=0)
    np.savez_compressed(
        output_dir / "normal_streams.npz",
        train_values=np.asarray(suite["train_values"], dtype=np.float32),
        train_regime=np.asarray(suite["train_regime"], dtype=np.int64),
        background_values=np.asarray(suite["background_values"], dtype=np.float32),
        background_regime=np.asarray(suite["background_regime"], dtype=np.int64),
    )
    np.savez_compressed(
        output_dir / "paired_episodes.npz",
        windows=windows,
        pair_ids=np.asarray([episode["pair_id"] for episode in episodes]),
        roles=np.asarray([episode["role"] for episode in episodes]),
        regimes=np.asarray([episode["regime"] for episode in episodes], dtype=np.int64),
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
        "episode_count": int(len(episodes)),
        "roles": list(ROLE_ORDER),
        "contracts": list(suite["contracts"]),
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
    print(f"Wrote B0 synthetic suite: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
