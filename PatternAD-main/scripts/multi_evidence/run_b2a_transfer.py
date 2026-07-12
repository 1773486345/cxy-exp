#!/usr/bin/env python3
"""Run B2a held-out multi-target transfer without cross-target score fusion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.multi_evidence.generate_b2a_holdout import (  # noqa: E402
    DEFAULT_CONFIG,
    DRIFT_CONTROL_ROLE,
    PAIR_ROLE_ORDER,
    _canonical_hash,
    _file_sha256,
    _load_json,
    generate_suite,
    write_suite,
)
from scripts.multi_evidence.multi_target_calibration import (  # noqa: E402
    MultiTargetEvidenceReliabilityCalibration,
)
from ts_benchmark.baselines.MultiEvidenceRepair.MultiEvidenceRepair import (  # noqa: E402
    ChannelStandardizer,
    terminal_windows,
)
from ts_benchmark.baselines.MultiEvidenceRepair.MultiTargetEvidenceRepair import (  # noqa: E402
    MultiTargetEvidenceRepair,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "multi_evidence"
TAIL_COMPONENTS: Sequence[str] = (
    "temporal_residual_tail",
    "cross_residual_tail",
    "disagreement_tail",
)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _select_device(request: str) -> str:
    if request == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device requested CUDA but CUDA is unavailable.")
    return str(device)


def split_normal_train(
    train_length: int, history_length: int, split: Mapping[str, Any]
) -> Dict[str, Dict[str, int]]:
    """Allocate four disjoint normal intervals with H-point gaps."""
    outer_length = int(math.ceil(train_length * float(split["outer_calibration_fraction"])))
    validation_length = int(math.ceil(train_length * float(split["validation_fraction"])))
    reference_length = int(math.ceil(train_length * float(split["reference_fraction"])))
    gap = int(history_length)
    outer_start = train_length - outer_length
    reference_end = outer_start - gap
    reference_start = reference_end - reference_length
    validation_end = reference_start - gap
    validation_start = validation_end - validation_length
    optimization_end = validation_start - gap
    if optimization_end <= history_length + 1:
        raise ValueError("B2a split leaves too little optimization-normal data.")
    segments = {
        "optimization": {"start": 0, "end": optimization_end},
        "validation": {"start": validation_start, "end": validation_end},
        "reference": {"start": reference_start, "end": reference_end},
        "outer_calibration": {"start": outer_start, "end": train_length},
    }
    for name, segment in segments.items():
        length = segment["end"] - segment["start"]
        if length <= history_length:
            raise ValueError(f"B2a {name} segment is too short for terminal windows.")
        segment["length"] = length
    return segments


def _window_segment(
    values: np.ndarray,
    segment: Mapping[str, int],
    standardizer: ChannelStandardizer,
    history_length: int,
) -> np.ndarray:
    return terminal_windows(
        standardizer.transform(
            values[int(segment["start"]) : int(segment["end"])]
        ),
        history_length,
    )


def _serializable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot write an empty B2a CSV.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _serializable(value) for key, value in row.items()})


def _matrix_column(
    matrix_scores: Mapping[str, np.ndarray], position: int
) -> Dict[str, np.ndarray]:
    return {
        name: np.asarray(values, dtype=np.float64)[:, position]
        for name, values in matrix_scores.items()
    }


def _reliability_isolation_report(
    calibration: MultiTargetEvidenceReliabilityCalibration, windows: np.ndarray
) -> Dict[str, Any]:
    reports: Dict[str, Dict[str, float]] = {}
    base = np.asarray(windows[:1], dtype=np.float64).copy()
    for target_index in calibration.target_indices:
        scalar = calibration.calibrators[target_index]
        drivers = [index for index in range(scalar.dimensions) if index != target_index]
        driver_changed = base.copy()
        driver_changed[:, :, drivers] += np.linspace(
            0.0, 3.0, base.shape[1]
        )[None, :, None]
        terminal_changed = base.copy()
        terminal_changed[:, -1, target_index] += 3.0
        target_changed = base.copy()
        target_changed[:, :, target_index] += np.linspace(
            0.0, 3.0, base.shape[1]
        )[None, :]
        base_features = scalar.features(base)
        driver_features = scalar.features(driver_changed)
        terminal_features = scalar.features(terminal_changed)
        target_features = scalar.features(target_changed)
        reports[str(target_index)] = {
            "temporal_feature_driver_delta": float(
                np.max(
                    np.abs(
                        base_features["temporal_residual"]
                        - driver_features["temporal_residual"]
                    )
                )
            ),
            "temporal_feature_terminal_target_delta": float(
                np.max(
                    np.abs(
                        base_features["temporal_residual"]
                        - terminal_features["temporal_residual"]
                    )
                )
            ),
            "cross_feature_target_column_delta": float(
                np.max(
                    np.abs(
                        base_features["cross_residual"]
                        - target_features["cross_residual"]
                    )
                )
            ),
            "disagreement_feature_terminal_target_delta": float(
                np.max(
                    np.abs(
                        base_features["disagreement"]
                        - terminal_features["disagreement"]
                    )
                )
            ),
        }
    return {
        "per_target": reports,
        "all_within_tolerance": bool(
            all(value <= 1e-7 for report in reports.values() for value in report.values())
        ),
    }


def _gate(passed: bool, **details: Any) -> Dict[str, Any]:
    return {"pass": bool(passed), **details}


def _paired_records(
    rows: Sequence[Mapping[str, Any]]
) -> Dict[Tuple[int, str], Dict[str, Mapping[str, Any]]]:
    pairs: Dict[Tuple[int, str], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if bool(row["is_pair"]):
            key = (int(row["designated_target"]), str(row["pair_id"]))
            pairs[key][str(row["role"])] = row
    expected = set(PAIR_ROLE_ORDER)
    for key, roles in pairs.items():
        missing = expected - set(roles)
        if missing:
            raise ValueError(f"B2a pair {key} is missing roles {sorted(missing)}.")
    return dict(pairs)


def _paired_order(
    pairs: Mapping[Tuple[int, str], Mapping[str, Mapping[str, Any]]],
    target_index: int,
    role: str,
    component: str,
) -> Dict[str, Any]:
    selected = [pair for (target, _), pair in pairs.items() if target == target_index]
    deltas = np.asarray(
        [
            float(pair[role][component])
            - float(pair["coherent_control"][component])
            for pair in selected
        ],
        dtype=np.float64,
    )
    if not len(deltas):
        raise ValueError(f"No B2a pairs for target {target_index}.")
    return {
        "target_index": target_index,
        "role": role,
        "component": component,
        "count": int(len(deltas)),
        "positive_count": int(np.sum(deltas > 0.0)),
        "median_delta": float(np.median(deltas)),
        "minimum_delta": float(np.min(deltas)),
        "deltas": deltas.astype(float).tolist(),
    }


def _background_fpr(
    scores: Mapping[str, np.ndarray],
    calibration: MultiTargetEvidenceReliabilityCalibration,
    phase_bins: np.ndarray,
) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    for tail_component in TAIL_COMPONENTS:
        raw_component = tail_component[: -len("_tail")]
        exceedance = calibration.exceeds(scores, tail_component)
        strata = np.asarray(
            scores[f"{raw_component}_reliability_stratum"], dtype=np.int64
        )
        per_target: Dict[str, Any] = {}
        for position, target_index in enumerate(calibration.target_indices):
            by_stratum = {}
            for stratum in range(calibration.reliability_strata):
                mask = strata[:, position] == stratum
                if not mask.any():
                    raise ValueError(
                        f"B2a background has no {raw_component} target {target_index} "
                        f"reliability stratum {stratum} rows."
                    )
                by_stratum[str(stratum)] = {
                    "count": int(mask.sum()),
                    "fpr": float(exceedance[mask, position].mean()),
                }
            by_phase = {}
            for phase_bin in range(4):
                mask = phase_bins == phase_bin
                if not mask.any():
                    raise ValueError(f"B2a background has no phase {phase_bin} rows.")
                by_phase[str(phase_bin)] = {
                    "count": int(mask.sum()),
                    "fpr": float(exceedance[mask, position].mean()),
                }
            per_target[str(target_index)] = {
                "overall": float(exceedance[:, position].mean()),
                "by_reliability_stratum": by_stratum,
                "by_hidden_relation_phase_diagnostic": by_phase,
            }
        details[tail_component] = per_target
    return details


def _episode_rows(
    suite: Mapping[str, Any],
    standardizer: ChannelStandardizer,
    model: MultiTargetEvidenceRepair,
    calibration: MultiTargetEvidenceReliabilityCalibration,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    episodes = list(suite["episodes"])
    windows = standardizer.transform(
        np.stack([episode["values"] for episode in episodes], axis=0)
    )
    raw_scores = model.score_windows(windows, include_tails=False)
    scores = calibration.transform(windows, raw_scores)
    exceeds = {component: calibration.exceeds(scores, component) for component in TAIL_COMPONENTS}
    position_for_target = {
        target_index: position
        for position, target_index in enumerate(calibration.target_indices)
    }
    rows: List[Dict[str, Any]] = []
    for row_index, episode in enumerate(episodes):
        target_index = int(episode["target_index"])
        position = position_for_target[target_index]
        row: Dict[str, Any] = {
            "pair_id": str(episode["pair_id"]),
            "role": str(episode["role"]),
            "is_pair": bool(episode["is_pair"]),
            "designated_target": target_index,
            "scored_target": target_index,
            "phase_bin": int(episode["phase_bin"]),
            "source_terminal_index": int(episode["source_terminal_index"]),
            "donor_terminal_index": int(episode["donor_terminal_index"]),
            "relation_value": float(episode["relation_value"]),
            "relation_velocity": float(episode["relation_velocity"]),
        }
        for name, values in scores.items():
            array = np.asarray(values)
            if array.ndim == 2:
                row[name] = float(array[row_index, position])
        for component, matrix in exceeds.items():
            row[f"{component}_exceeds"] = bool(matrix[row_index, position])
        rows.append(row)
    return rows, scores


def run_experiment(
    config: Mapping[str, Any],
    output_dir: Path,
    device: str,
    seed: int,
    epoch_override: int | None = None,
) -> Dict[str, Any]:
    """Train six independent target models and evaluate B2a's frozen gates."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite B2a output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    _set_seed(seed)
    suite = generate_suite(config)
    suite_manifest = write_suite(config, suite, output_dir / "synthetic_suite")
    values = np.asarray(suite["train"]["values"], dtype=np.float32)
    history_length = int(config["history_length"])
    target_indices = tuple(int(value) for value in config["target_indices"])
    segments = split_normal_train(len(values), history_length, config["split"])
    standardizer = ChannelStandardizer().fit(
        values[segments["optimization"]["start"] : segments["optimization"]["end"]]
    )
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
    model = MultiTargetEvidenceRepair(
        dimensions=int(config["dimensions"]),
        target_indices=target_indices,
        d_model=int(model_config["d_model"]),
        dropout=float(model_config["dropout"]),
        learning_rate=float(model_config["learning_rate"]),
        epochs=int(model_config["epochs"]),
        patience=int(model_config["patience"]),
        batch_size=int(model_config["batch_size"]),
        device=device,
    ).fit(windows["optimization"], windows["validation"], windows["reference"], seed)
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
    background_scores = calibration.transform(background_windows, background_raw)
    background_phase = np.asarray(suite["background"]["relation_phase_bin"], dtype=np.int64)[
        history_length:
    ]
    fpr = _background_fpr(background_scores, calibration, background_phase)
    episode_rows, episode_score_matrices = _episode_rows(
        suite, standardizer, model, calibration
    )
    _write_csv(output_dir / "episode_scores.csv", episode_rows)
    pairs = _paired_records(episode_rows)
    evaluation = config["evaluation"]
    gates: Dict[str, Dict[str, Any]] = {}
    model_isolation = model.evidence_isolation_report(windows["reference"][:1])
    per_target_model_isolation = model_isolation["per_target"]
    gates["information_and_parameter_isolation"] = _gate(
        bool(model_isolation["all_branch_parameter_sets_disjoint"])
        and all(
            bool(report["parameter_sets_disjoint"])
            and float(report["temporal_driver_delta"]) <= 1e-7
            and float(report["temporal_terminal_target_delta"]) <= 1e-7
            and float(report["cross_target_column_delta"]) <= 1e-7
            for report in per_target_model_isolation.values()
        ),
        **model_isolation,
        tolerance=1e-7,
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
    gates["synthetic_pair_contract"] = _gate(
        all(
            float(contract["coherent_unsupported_target_max_abs_difference"]) <= 1e-7
            and float(contract["coherent_omission_driver_max_abs_difference"]) <= 1e-7
            and float(contract["coherent_unsupported_driver_difference"]) > 1e-6
            and float(contract["coherent_omission_target_difference"]) > 1e-6
            for contract in contracts
        ),
        contracts=contracts,
        tolerance=1e-7,
    )
    target_position = {target: position for position, target in enumerate(target_indices)}
    temporal_ties = []
    cross_ties = []
    for (target_index, _), pair in pairs.items():
        temporal_ties.append(
            abs(
                float(pair["coherent_control"]["temporal_residual"])
                - float(pair["unsupported_target_break"]["temporal_residual"])
            )
        )
        cross_ties.append(
            abs(
                float(pair["coherent_control"]["mu_cross"])
                - float(pair["target_omission_break"]["mu_cross"])
            )
        )
    gates["counterfactual_input_ties"] = _gate(
        max(temporal_ties) <= 1e-7 and max(cross_ties) <= 1e-7,
        coherent_unsupported_temporal_residual_max_abs_difference=float(max(temporal_ties)),
        coherent_omission_cross_prediction_max_abs_difference=float(max(cross_ties)),
        tolerance=1e-7,
    )
    required_order = int(evaluation["paired_order_min"])
    required_margin = float(evaluation["paired_tail_margin_min"])
    per_target_skill = {}
    optimization_target_means = np.mean(
        windows["optimization"][:, -1, list(target_indices)], axis=0, dtype=np.float64
    )
    for target_index in target_indices:
        position = target_position[target_index]
        for role in ("unsupported_target_break", "target_omission_break"):
            for component in ("cross_residual_tail", "disagreement_tail"):
                summary = _paired_order(pairs, target_index, role, component)
                gate_name = f"target_{target_index}_{role}_{component}"
                gates[gate_name] = _gate(
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
        spikes = [
            pair["target_spike"]
            for pair in target_pairs
        ]
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
        **{name: np.asarray(values) for name, values in background_scores.items()},
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
        "phase": "B2a",
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
        "cross_reference_skill_by_target": per_target_skill,
        "background_fpr": fpr,
        "gates": gates,
        "provenance": {
            "suite_manifest": str(suite_manifest.relative_to(output_dir)),
            "generator_sha256": _file_sha256(
                REPO_ROOT / "scripts" / "multi_evidence" / "generate_b2a_holdout.py"
            ),
            "runner_sha256": _file_sha256(Path(__file__)),
            "scalar_model_sha256": _file_sha256(
                REPO_ROOT
                / "ts_benchmark"
                / "baselines"
                / "MultiEvidenceRepair"
                / "MultiEvidenceRepair.py"
            ),
            "multi_target_model_sha256": _file_sha256(
                REPO_ROOT
                / "ts_benchmark"
                / "baselines"
                / "MultiEvidenceRepair"
                / "MultiTargetEvidenceRepair.py"
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
        },
    }
    _write_json_atomic(output_dir / "b2a_evaluation.json", result)
    _write_json_atomic(
        output_dir / "run_metadata.json",
        {
            "status": result["status"],
            "phase": "B2a",
            "config_hash": result["config_hash"],
            "outputs": {
                "evaluation": "b2a_evaluation.json",
                "episode_scores": "episode_scores.csv",
                "background_scores": "background_scores.npz",
                "model_state": "multi_target_model_state.pt",
                "synthetic_suite": "synthetic_suite",
            },
        },
    )
    return result


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
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
    )
    print(f"B2a status: {result['status']}; results: {output_dir}")
    return 2 if arguments.strict and result["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
