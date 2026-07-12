#!/usr/bin/env python3
"""Run B3a terminal-blind relation-history conditioned cross repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.multi_evidence.generate_b2a_gc import (  # noqa: E402
    DRIFT_CONTROL_ROLE,
    _canonical_hash,
    _file_sha256,
    _load_json,
)
from scripts.multi_evidence.multi_target_calibration import (  # noqa: E402
    MultiTargetEvidenceReliabilityCalibration,
)
from scripts.multi_evidence.run_b2a_transfer import (  # noqa: E402
    TAIL_COMPONENTS,
    _background_fpr,
    _episode_rows,
    _gate,
    _paired_order,
    _paired_records,
    _reliability_isolation_report,
    _select_device,
    _set_seed,
    _window_segment,
    _write_csv,
    _write_json_atomic,
    split_normal_train,
)
from ts_benchmark.baselines.MultiEvidenceRepair.MultiEvidenceRepair import (  # noqa: E402
    ChannelStandardizer,
    terminal_windows,
)
from ts_benchmark.baselines.MultiEvidenceRepair.MultiTargetRelationConditionedEvidenceRepair import (  # noqa: E402
    MultiTargetRelationConditionedEvidenceRepair,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "multi_evidence"
DEFAULT_CONFIG = (
    REPO_ROOT / "config" / "multi_evidence" / "b3_relation_conditioned_drift_rotation.json"
)
DEFAULT_FROZEN_CONTROL_DIR = (
    REPO_ROOT / "result" / "multi_evidence" / "b3a_baseline_seed4401_gpu"
)


def _validate_b3_config(config: Mapping[str, Any]) -> None:
    """Validate the B3a-only architecture and fixed capacity control."""
    if not str(config.get("suite_id", "")).startswith("multi_evidence_b2a_gc_"):
        raise ValueError("B3a must retain the B2a-GC terminal generator contract.")
    model = config["model"]
    for name in ("temporal_d_model", "cross_d_model", "cross_head_d_model"):
        if int(model.get(name, 0)) < 1:
            raise ValueError(f"B3a model.{name} must be positive.")
    if int(model["temporal_d_model"]) != int(model["d_model"]):
        raise ValueError("B3a must retain B2a-GC's temporal branch width.")
    if int(model.get("cross_parameter_count_max", 0)) < 1:
        raise ValueError("B3a must declare a cross-branch capacity ceiling.")


def _contract_gate(
    contracts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    history_length: int,
) -> Dict[str, Any]:
    contract = config["counterfactual_contract"]
    return _gate(
        all(
            float(record["coherent_unsupported_target_max_abs_difference"]) <= 1e-7
            and float(record["coherent_omission_driver_max_abs_difference"]) <= 1e-7
            and float(record["coherent_unsupported_driver_terminal_max_abs_difference"])
            > 1e-6
            and float(record["coherent_omission_target_terminal_abs_difference"]) > 1e-6
            and int(record["source_donor_terminal_distance"]) > history_length
            and float(record["relation_value_abs_difference"])
            <= float(contract["maximum_relation_value_delta"]) + 1e-12
            and float(record["structural_cross_gap_target_std"])
            >= float(contract["minimum_structural_cross_gap_target_std"])
            and float(record["terminal_target_gap_target_std"])
            >= float(contract["minimum_terminal_target_gap_target_std"])
            for record in contracts
        ),
        contracts=list(contracts),
        counterfactual_contract=dict(contract),
        tolerance=1e-7,
    )


def _load_frozen_control(
    config: Mapping[str, Any],
    control_dir: Path,
    target_indices: Sequence[int],
    history_length: int,
    seed: int,
) -> tuple[Dict[str, Any], ChannelStandardizer, Dict[str, Mapping[str, torch.Tensor]], Dict[str, Any]]:
    """Load the fixed B2a-GC control and reject any provenance drift.

    The B3a comparison must retain exactly the selected B2a-GC temporal
    checkpoints and the exact normal/episode inputs used by the control.  This
    deliberately does not regenerate a same-seed suite.
    """
    isolation = config.get("checkpoint_isolation")
    if not isinstance(isolation, Mapping):
        raise ValueError("B3a requires a frozen checkpoint_isolation contract.")
    evaluation_path = control_dir / "b2a_gc_evaluation.json"
    state_path = control_dir / "multi_target_model_state.pt"
    suite_dir = control_dir / "synthetic_suite"
    normal_path = suite_dir / "normal_streams.npz"
    episodes_path = suite_dir / "episodes.npz"
    manifest_path = suite_dir / "suite_metadata.json"
    for path in (evaluation_path, state_path, normal_path, episodes_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen B3a control artifact is missing: {path}")
    expected_hashes = {
        "model_state_sha256": state_path,
        "normal_streams_sha256": normal_path,
        "episodes_sha256": episodes_path,
        "suite_metadata_sha256": manifest_path,
    }
    observed_hashes = {name: _file_sha256(path) for name, path in expected_hashes.items()}
    for name, observed in observed_hashes.items():
        expected = str(isolation.get(f"expected_{name}", ""))
        if not expected or observed != expected:
            raise ValueError(
                f"Frozen B3a control {name} mismatch: expected={expected}; observed={observed}."
            )
    with evaluation_path.open("r", encoding="utf-8") as handle:
        evaluation = json.load(handle)
    expected_fields = {
        "phase": str(isolation.get("expected_phase", "")),
        "seed": int(isolation.get("expected_seed", -1)),
        "config_hash": str(isolation.get("expected_control_config_hash", "")),
    }
    for name, expected in expected_fields.items():
        if not expected or evaluation.get(name) != expected:
            raise ValueError(
                f"Frozen B3a control {name} mismatch: expected={expected}; "
                f"observed={evaluation.get(name)}."
            )
    if int(evaluation["seed"]) != int(seed):
        raise ValueError("B3a seed must equal the frozen control seed.")
    if tuple(int(value) for value in evaluation["target_indices"]) != tuple(target_indices):
        raise ValueError("Frozen B3a control target_indices do not match the B3a protocol.")
    if int(evaluation["history_length"]) != int(history_length):
        raise ValueError("Frozen B3a control history_length does not match the B3a protocol.")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("config_hash") != expected_fields["config_hash"]:
        raise ValueError("Frozen B3a control suite manifest config hash is inconsistent.")
    if tuple(int(value) for value in manifest["target_indices"]) != tuple(target_indices):
        raise ValueError("Frozen B3a control suite manifest target_indices are inconsistent.")
    state_payload = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state_payload, Mapping):
        raise ValueError("Frozen B3a control state must be a mapping.")
    source_states = state_payload.get("state_dict_by_target")
    if not isinstance(source_states, Mapping) or set(source_states) != {
        str(target) for target in target_indices
    }:
        raise ValueError("Frozen B3a control state has an incompatible target checkpoint set.")
    normalizer_metadata = state_payload.get("normalizer")
    if not isinstance(normalizer_metadata, Mapping):
        raise ValueError("Frozen B3a control state does not include its normalizer.")
    standardizer = ChannelStandardizer()
    standardizer.mean_ = np.asarray(normalizer_metadata.get("mean"), dtype=np.float32)
    standardizer.std_ = np.asarray(normalizer_metadata.get("std"), dtype=np.float32)
    if (
        standardizer.mean_.shape != (int(config["dimensions"]),)
        or standardizer.std_.shape != (int(config["dimensions"]),)
        or not np.isfinite(standardizer.mean_).all()
        or not np.isfinite(standardizer.std_).all()
        or np.any(standardizer.std_ <= 0.0)
    ):
        raise ValueError("Frozen B3a control normalizer is invalid for this protocol.")
    with np.load(normal_path, allow_pickle=False) as normal_data:
        train_values = np.asarray(normal_data["train_values"], dtype=np.float32)
        background_values = np.asarray(normal_data["background_values"], dtype=np.float32)
        background_phase = np.asarray(normal_data["background_phase_bin"], dtype=np.int64)
        background_relation_value = np.asarray(
            normal_data["background_relation_value"], dtype=np.float32
        )
        background_relation_velocity = np.asarray(
            normal_data["background_relation_velocity"], dtype=np.float32
        )
    with np.load(episodes_path, allow_pickle=False) as episode_data:
        episode_windows = np.asarray(episode_data["windows"], dtype=np.float32)
        pair_ids = episode_data["pair_ids"]
        roles = episode_data["roles"]
        is_pair = np.asarray(episode_data["is_pair"], dtype=np.uint8)
        episode_targets = np.asarray(episode_data["target_indices"], dtype=np.int64)
        phase_bins = np.asarray(episode_data["phase_bins"], dtype=np.int64)
        source_indices = np.asarray(episode_data["source_terminal_indices"], dtype=np.int64)
        donor_indices = np.asarray(episode_data["donor_terminal_indices"], dtype=np.int64)
    episode_count = len(episode_windows)
    episode_arrays = (
        pair_ids,
        roles,
        is_pair,
        episode_targets,
        phase_bins,
        source_indices,
        donor_indices,
    )
    if (
        episode_windows.ndim != 3
        or episode_windows.shape[1:] != (history_length + 1, int(config["dimensions"]))
        or any(len(array) != episode_count for array in episode_arrays)
        or int(manifest.get("episode_count", -1)) != episode_count
    ):
        raise ValueError("Frozen B3a control episode inputs have an invalid shape.")
    if (
        train_values.shape != (int(config["train_length"]), int(config["dimensions"]))
        or background_values.shape != (int(config["test_length"]), int(config["dimensions"]))
        or len(background_phase) != len(background_values)
        or len(background_relation_value) != len(background_values)
        or len(background_relation_velocity) != len(background_values)
    ):
        raise ValueError("Frozen B3a control normal streams have an invalid shape.")
    contracts = list(manifest.get("contracts", []))
    source_blocks = {
        str(record["pair_id"]): int(record["source_selection_block"])
        for record in contracts
    }
    episodes = []
    for index in range(episode_count):
        source_index = int(source_indices[index])
        if source_index < history_length or source_index >= len(background_values):
            raise ValueError("Frozen B3a control episode source index is outside background.")
        pair_id = str(pair_ids[index])
        episodes.append(
            {
                "pair_id": pair_id,
                "role": str(roles[index]),
                "is_pair": bool(is_pair[index]),
                "target_index": int(episode_targets[index]),
                "phase_bin": int(phase_bins[index]),
                "source_terminal_index": source_index,
                "donor_terminal_index": int(donor_indices[index]),
                "source_selection_block": source_blocks.get(pair_id, -1),
                "relation_value": float(background_relation_value[source_index]),
                "relation_velocity": float(background_relation_velocity[source_index]),
                "values": episode_windows[index],
            }
        )
    suite = {
        "train": {"values": train_values},
        "background": {
            "values": background_values,
            "relation_phase_bin": background_phase,
        },
        "target_indices": tuple(target_indices),
        "episodes": episodes,
        "contracts": contracts,
        "target_std": np.asarray(manifest["target_std_from_normal_train"], dtype=np.float32),
    }
    provenance = {
        "control_dir": str(control_dir),
        "control_evaluation": str(evaluation_path),
        "control_state": str(state_path),
        "control_suite_dir": str(suite_dir),
        "control_phase": evaluation["phase"],
        "control_seed": int(evaluation["seed"]),
        "control_device": str(evaluation["device"]),
        "control_config_hash": evaluation["config_hash"],
        **observed_hashes,
    }
    return suite, standardizer, dict(source_states), provenance


def run_experiment(
    config: Mapping[str, Any],
    output_dir: Path,
    device: str,
    seed: int,
    epoch_override: int | None = None,
    frozen_control_dir: Path = DEFAULT_FROZEN_CONTROL_DIR,
) -> Dict[str, Any]:
    """Train only B3a cross paths against a frozen B2a-GC control."""
    _validate_b3_config(config)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite B3a output directory: {output_dir}")
    history_length = int(config["history_length"])
    target_indices = tuple(int(value) for value in config["target_indices"])
    suite, standardizer, frozen_temporal_states, frozen_control = _load_frozen_control(
        config,
        Path(frozen_control_dir),
        target_indices,
        history_length,
        seed,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _set_seed(seed)
    values = np.asarray(suite["train"]["values"], dtype=np.float32)
    segments = split_normal_train(len(values), history_length, config["split"])
    windows = {
        name: _window_segment(values, segment, standardizer, history_length)
        for name, segment in segments.items()
    }
    model_config = dict(config["model"])
    if epoch_override is not None:
        if epoch_override < 1:
            raise ValueError("--epochs must be positive.")
        model_config["epochs"] = int(epoch_override)
        model_config["patience"] = min(int(model_config["patience"]), int(epoch_override))
    model = MultiTargetRelationConditionedEvidenceRepair(
        dimensions=int(config["dimensions"]),
        target_indices=target_indices,
        temporal_d_model=int(model_config["temporal_d_model"]),
        cross_d_model=int(model_config["cross_d_model"]),
        cross_head_d_model=int(model_config["cross_head_d_model"]),
        dropout=float(model_config["dropout"]),
        learning_rate=float(model_config["learning_rate"]),
        epochs=int(model_config["epochs"]),
        patience=int(model_config["patience"]),
        batch_size=int(model_config["batch_size"]),
        device=device,
    ).fit(
        windows["optimization"],
        windows["validation"],
        windows["reference"],
        seed,
        frozen_temporal_states=frozen_temporal_states,
    )
    frozen_temporal_hashes = {
        target: details["frozen_temporal_state_sha256"]
        for target, details in model.fit_metadata_["targets"].items()
    }
    final_temporal_hashes = model.temporal_state_sha256()
    if frozen_temporal_hashes != final_temporal_hashes:
        raise RuntimeError("B3a changed a frozen temporal checkpoint during cross training.")
    reference_raw = model.score_windows(windows["reference"], include_tails=False)
    outer_raw = model.score_windows(windows["outer_calibration"], include_tails=False)
    calibration_config = dict(config["calibration"])
    calibration = MultiTargetEvidenceReliabilityCalibration(
        dimensions=int(config["dimensions"]),
        target_indices=target_indices,
        target_fpr=float(config["evaluation"]["target_fpr"]),
        mode=str(calibration_config["mode"]),
        reliability_strata=int(calibration_config["reliability_strata"]),
        min_reference_per_stratum=int(calibration_config["min_reference_per_stratum"]),
    ).fit(
        windows["optimization"],
        windows["reference"],
        reference_raw,
        windows["outer_calibration"],
        outer_raw,
    )
    reference_scores = calibration.transform(windows["reference"], reference_raw)
    background_windows = terminal_windows(
        standardizer.transform(np.asarray(suite["background"]["values"])), history_length
    )
    background_raw = model.score_windows(background_windows, include_tails=False)
    with np.load(
        Path(frozen_control_dir) / "background_scores.npz", allow_pickle=False
    ) as baseline_background:
        expected_target = np.asarray(baseline_background["target"], dtype=np.float64)
        expected_mu_temporal = np.asarray(
            baseline_background["mu_temporal"], dtype=np.float64
        )
        expected_temporal_residual = np.asarray(
            baseline_background["temporal_residual"], dtype=np.float64
        )
    temporal_replay_deltas = {
        "target": float(np.max(np.abs(background_raw["target"] - expected_target))),
        "mu_temporal": float(
            np.max(np.abs(background_raw["mu_temporal"] - expected_mu_temporal))
        ),
        "temporal_residual": float(
            np.max(
                np.abs(
                    background_raw["temporal_residual"] - expected_temporal_residual
                )
            )
        ),
    }
    same_device_replay = str(device) == str(frozen_control["control_device"])
    temporal_replay_tolerance = 2e-6 if same_device_replay else 1e-2
    if any(value > temporal_replay_tolerance for value in temporal_replay_deltas.values()):
        raise RuntimeError(
            "B3a frozen temporal replay does not match the B2a-GC control: "
            f"{temporal_replay_deltas}."
        )
    background_scores = calibration.transform(background_windows, background_raw)
    background_phase = np.asarray(suite["background"]["relation_phase_bin"], dtype=np.int64)[
        history_length:
    ]
    fpr = _background_fpr(background_scores, calibration, background_phase)
    episode_rows, _ = _episode_rows(suite, standardizer, model, calibration)
    for row, episode in zip(episode_rows, suite["episodes"]):
        row["source_selection_block"] = int(episode.get("source_selection_block", -1))
    _write_csv(output_dir / "episode_scores.csv", episode_rows)
    pairs = _paired_records(episode_rows)
    evaluation = config["evaluation"]
    gates: Dict[str, Dict[str, Any]] = {}
    gates["frozen_temporal_checkpoint_and_replay"] = _gate(
        frozen_temporal_hashes == final_temporal_hashes
        and all(value <= temporal_replay_tolerance for value in temporal_replay_deltas.values()),
        frozen_temporal_state_sha256=frozen_temporal_hashes,
        final_temporal_state_sha256=final_temporal_hashes,
        background_replay_max_abs_difference=temporal_replay_deltas,
        replay_mode=(
            "same_device_exact_control" if same_device_replay else "cross_device_diagnostic"
        ),
        tolerance=temporal_replay_tolerance,
    )
    model_isolation = model.evidence_isolation_report(windows["reference"][:1])
    per_target_model_isolation = model_isolation["per_target"]
    gates["information_and_parameter_isolation"] = _gate(
        bool(model_isolation["all_branch_parameter_sets_disjoint"])
        and all(
            bool(report["parameter_sets_disjoint"])
            and float(report["temporal_driver_delta"]) <= 1e-7
            and float(report["temporal_terminal_target_delta"]) <= 1e-7
            and float(report["cross_terminal_target_delta"]) <= 1e-7
            and float(report["relation_history_terminal_target_input_delta"]) <= 1e-7
            and float(report["relation_history_target_history_input_delta"]) > 1e-7
            for report in per_target_model_isolation.values()
        ),
        **model_isolation,
        tolerance=1e-7,
    )
    cross_parameter_counts = {
        target: int(details["fit"]["parameter_counts"]["cross"])
        for target, details in model.fit_metadata_["targets"].items()
    }
    gates["cross_branch_capacity_control"] = _gate(
        all(
            count <= int(model_config["cross_parameter_count_max"])
            for count in cross_parameter_counts.values()
        ),
        per_target_cross_parameter_counts=cross_parameter_counts,
        maximum_allowed=int(model_config["cross_parameter_count_max"]),
    )
    reliability_isolation = _reliability_isolation_report(
        calibration, windows["reference"][:1]
    )
    gates["reliability_routing_isolation"] = _gate(
        bool(reliability_isolation["all_within_tolerance"]),
        **reliability_isolation,
        tolerance=1e-7,
    )
    contracts = list(suite["contracts"])
    gates["synthetic_terminal_contract"] = _contract_gate(
        contracts, config, history_length
    )
    temporal_ties = []
    cross_terminal_ties = []
    for (_, _), pair in pairs.items():
        temporal_ties.append(
            abs(
                float(pair["coherent_control"]["temporal_residual"])
                - float(pair["unsupported_target_break"]["temporal_residual"])
            )
        )
        cross_terminal_ties.append(
            abs(
                float(pair["coherent_control"]["mu_cross"])
                - float(pair["target_spike"]["mu_cross"])
            )
        )
    gates["counterfactual_input_ties"] = _gate(
        max(temporal_ties) <= 1e-7 and max(cross_terminal_ties) <= 1e-7,
        coherent_unsupported_temporal_residual_max_abs_difference=float(max(temporal_ties)),
        coherent_target_spike_cross_prediction_max_abs_difference=float(
            max(cross_terminal_ties)
        ),
        tolerance=1e-7,
    )
    required_order = int(evaluation["paired_order_min"])
    required_margin = float(evaluation["paired_tail_margin_min"])
    target_position = {target: position for position, target in enumerate(target_indices)}
    per_target_skill: Dict[str, Dict[str, float]] = {}
    optimization_target_means = np.mean(
        windows["optimization"][:, -1, list(target_indices)], axis=0, dtype=np.float64
    )
    for target_index in target_indices:
        position = target_position[target_index]
        for role in ("unsupported_target_break", "target_omission_break"):
            for component in ("cross_residual_tail", "disagreement_tail"):
                summary = _paired_order(pairs, target_index, role, component)
                gates[f"target_{target_index}_{role}_{component}"] = _gate(
                    int(summary["positive_count"]) >= required_order
                    and float(summary["median_delta"]) >= required_margin,
                    **summary,
                    required_order=required_order,
                    required_margin=required_margin,
                )
        target_pairs = [pair for (target, _), pair in pairs.items() if target == target_index]
        coherent_exceedance = np.asarray(
            [
                bool(pair["coherent_control"]["cross_residual_tail_exceeds"])
                or bool(pair["coherent_control"]["disagreement_tail_exceeds"])
                for pair in target_pairs
            ],
            dtype=bool,
        )
        gates[f"target_{target_index}_coherent_control"] = _gate(
            int(coherent_exceedance.sum()) <= int(evaluation["coherent_exceedance_max"]),
            count=int(coherent_exceedance.sum()),
            total=int(len(coherent_exceedance)),
            allowed=int(evaluation["coherent_exceedance_max"]),
        )
        spikes = [pair["target_spike"] for pair in target_pairs]
        spike_success = np.asarray(
            [
                bool(row["temporal_residual_tail_exceeds"])
                and bool(row["cross_residual_tail_exceeds"])
                for row in spikes
            ],
            dtype=bool,
        )
        gates[f"target_{target_index}_target_spike"] = _gate(
            int(spike_success.sum()) >= int(evaluation["target_spike_exceedance_min"]),
            success_count=int(spike_success.sum()),
            total=int(len(spike_success)),
            required=int(evaluation["target_spike_exceedance_min"]),
        )
        drift_rows = [
            row
            for row in episode_rows
            if int(row["designated_target"]) == target_index
            and row["role"] == DRIFT_CONTROL_ROLE
        ]
        drift_exceedance = np.asarray(
            [
                bool(row["cross_residual_tail_exceeds"])
                or bool(row["disagreement_tail_exceeds"])
                for row in drift_rows
            ],
            dtype=bool,
        )
        gates[f"target_{target_index}_normal_relation_drift_control"] = _gate(
            int(drift_exceedance.sum()) <= int(evaluation["drift_control_exceedance_max"]),
            count=int(drift_exceedance.sum()),
            total=int(len(drift_exceedance)),
            allowed=int(evaluation["drift_control_exceedance_max"]),
        )
        cross_mae = float(
            np.mean(
                np.abs(
                    reference_scores["target"][:, position]
                    - reference_scores["mu_cross"][:, position]
                )
            )
        )
        baseline_mae = float(
            np.mean(
                np.abs(
                    reference_scores["target"][:, position]
                    - optimization_target_means[position]
                )
            )
        )
        improvement = 1.0 - cross_mae / max(baseline_mae, 1e-12)
        per_target_skill[str(target_index)] = {
            "cross_mae": cross_mae,
            "target_mean_mae": baseline_mae,
            "relative_improvement": improvement,
        }
        gates[f"target_{target_index}_cross_normal_skill"] = _gate(
            improvement >= float(evaluation["cross_mean_mae_relative_improvement_min"]),
            **per_target_skill[str(target_index)],
            required_relative_improvement=float(
                evaluation["cross_mean_mae_relative_improvement_min"]
            ),
        )
    for target_index in target_indices:
        for tail_component, target_details in fpr.items():
            detail = target_details[str(target_index)]
            bin_rates = [
                float(value["fpr"])
                for value in detail["by_reliability_stratum"].values()
            ]
            phase_rates = [
                float(value["fpr"])
                for value in detail["by_hidden_relation_phase_diagnostic"].values()
            ]
            raw_component = tail_component[: -len("_tail")]
            bin_ok = all(
                value <= float(evaluation["background_component_fpr_max"])
                for value in bin_rates
            )
            phase_ok = all(
                value <= float(evaluation["drift_phase_fpr_max"])
                for value in phase_rates
            )
            if raw_component == "disagreement":
                bin_gap = max(bin_rates) - min(bin_rates)
                phase_gap = max(phase_rates) - min(phase_rates)
                bin_ok = bin_ok and bin_gap <= float(
                    evaluation["background_disagreement_fpr_gap_max"]
                )
                phase_ok = phase_ok and phase_gap <= float(
                    evaluation["drift_phase_disagreement_gap_max"]
                )
            else:
                bin_gap = None
                phase_gap = None
            gates[f"target_{target_index}_{tail_component}_background_fpr"] = _gate(
                bin_ok and phase_ok,
                target_index=target_index,
                component=tail_component,
                details=detail,
                reliability_bin_gap=bin_gap,
                phase_diagnostic_gap=phase_gap,
                maximum_reliability_bin_fpr=float(evaluation["background_component_fpr_max"]),
                maximum_phase_fpr=float(evaluation["drift_phase_fpr_max"]),
            )
    np.savez_compressed(
        output_dir / "background_scores.npz",
        terminal_indices=np.arange(history_length, history_length + len(background_windows)),
        hidden_relation_phase_bin=background_phase,
        target_indices=np.asarray(target_indices, dtype=np.int64),
        **{name: np.asarray(value) for name, value in background_scores.items()},
        **{
            f"{component}_exceeds": calibration.exceeds(background_scores, component)
            for component in TAIL_COMPONENTS
        },
    )
    torch.save(
        {
            "state_dict_by_target": model.state_dict(),
            "normalizer": standardizer.metadata(),
            "model_fit_metadata": model.fit_metadata_,
            "calibration_metadata": calibration.metadata(),
            "target_indices": target_indices,
        },
        output_dir / "multi_target_model_state.pt",
    )
    result = {
        "phase": "B3a-RelationConditioned-FrozenTemporal",
        "suite_id": str(config["suite_id"]),
        "status": "passed" if all(gate["pass"] for gate in gates.values()) else "failed_gates",
        "config_hash": _canonical_hash(config),
        "seed": int(seed),
        "device": device,
        "history_length": history_length,
        "target_indices": list(target_indices),
        "no_score_fusion": True,
        "no_cross_target_aggregation": True,
        "score_components": ["temporal_residual", "cross_residual", "disagreement"],
        "tail_components": list(TAIL_COMPONENTS),
        "split": {"segments": segments, "gaps": [history_length] * 3},
        "normalizer": standardizer.metadata(),
        "model": model.fit_metadata_,
        "calibration": calibration.metadata(),
        "counterfactual_contract": dict(config["counterfactual_contract"]),
        "cross_reference_skill_by_target": per_target_skill,
        "background_fpr": fpr,
        "gates": gates,
        "provenance": {
            "frozen_control": frozen_control,
            "generator_sha256": _file_sha256(
                REPO_ROOT / "scripts" / "multi_evidence" / "generate_b2a_gc.py"
            ),
            "runner_sha256": _file_sha256(Path(__file__)),
            "relation_conditioned_scalar_model_sha256": _file_sha256(
                REPO_ROOT
                / "ts_benchmark"
                / "baselines"
                / "MultiEvidenceRepair"
                / "RelationConditionedEvidenceRepair.py"
            ),
            "relation_conditioned_multi_target_model_sha256": _file_sha256(
                REPO_ROOT
                / "ts_benchmark"
                / "baselines"
                / "MultiEvidenceRepair"
                / "MultiTargetRelationConditionedEvidenceRepair.py"
            ),
            "scalar_calibration_sha256": _file_sha256(
                REPO_ROOT / "scripts" / "multi_evidence" / "reliability_calibration.py"
            ),
            "multi_target_calibration_sha256": _file_sha256(
                REPO_ROOT / "scripts" / "multi_evidence" / "multi_target_calibration.py"
            ),
            "test_scores_used_for_thresholds": False,
            "test_labels_used": False,
            "hidden_phase_used_by_model_or_calibration": False,
            "latent_factors_used_by_model_or_calibration": False,
            "structural_support_used_by_model_or_calibration": False,
        },
    }
    _write_json_atomic(output_dir / "b3a_evaluation.json", result)
    _write_json_atomic(
        output_dir / "run_metadata.json",
        {
            "status": result["status"],
            "phase": "B3a-RelationConditioned-FrozenTemporal",
            "config_hash": result["config_hash"],
            "outputs": {
                "evaluation": "b3a_evaluation.json",
                "episode_scores": "episode_scores.csv",
                "background_scores": "background_scores.npz",
                "model_state": "multi_target_model_state.pt",
                "frozen_control": frozen_control["control_dir"],
            },
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--frozen-control-dir",
        type=Path,
        default=DEFAULT_FROZEN_CONTROL_DIR,
        help="Retained B2a-GC control supplying immutable temporal checkpoints and inputs.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.torch_threads < 1:
        raise ValueError("--torch-threads must be positive.")
    torch.set_num_threads(arguments.torch_threads)
    config = _load_json(arguments.config)
    if arguments.seed is not None:
        config["seed"] = int(arguments.seed)
    output_dir = arguments.output_dir
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_ROOT / f"{config['suite_id']}_seed{config['seed']}"
    result = run_experiment(
        config=config,
        output_dir=output_dir,
        device=_select_device(str(arguments.device)),
        seed=int(config["seed"]),
        epoch_override=arguments.epochs,
        frozen_control_dir=arguments.frozen_control_dir,
    )
    print(f"B3a status: {result['status']}; results: {output_dir}")
    return 2 if arguments.strict and result["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
