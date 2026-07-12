#!/usr/bin/env python3
"""Run B2c family-wise ECRC on the frozen B2a-GC counterfactual contract."""

from __future__ import annotations

import argparse
import json
import math
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
    generate_suite,
    write_suite,
)
from scripts.multi_evidence.multi_target_familywise_calibration import (  # noqa: E402
    MultiTargetFamilyWiseEvidenceReliabilityCalibration,
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
from ts_benchmark.baselines.MultiEvidenceRepair.MultiTargetEvidenceRepair import (  # noqa: E402
    MultiTargetEvidenceRepair,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "multi_evidence"
DEFAULT_CONFIG = REPO_ROOT / "config" / "multi_evidence" / "b2c_fw_ecrc_drift_rotation.json"


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


def run_experiment(
    config: Mapping[str, Any],
    output_dir: Path,
    device: str,
    seed: int,
    epoch_override: int | None = None,
) -> Dict[str, Any]:
    """Train B1-style repair heads and evaluate B2c FW-ECRC gates."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite B2c output directory: {output_dir}")
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
    calibration = MultiTargetFamilyWiseEvidenceReliabilityCalibration(
        dimensions=int(config["dimensions"]),
        target_indices=target_indices,
        component_alphas=calibration_config["component_alphas"],
        mode=str(calibration_config["mode"]),
        reliability_strata=int(calibration_config["reliability_strata"]),
        min_reference_per_stratum=int(calibration_config["min_reference_per_stratum"]),
        min_outer_per_stratum=int(calibration_config["min_outer_per_stratum"]),
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
    episode_rows, _ = _episode_rows(suite, standardizer, model, calibration)
    for row, episode in zip(episode_rows, suite["episodes"]):
        row["source_selection_block"] = int(episode.get("source_selection_block", -1))
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
    gates["synthetic_terminal_contract"] = _contract_gate(
        contracts, config, history_length
    )
    temporal_ties = []
    cross_ties = []
    for (_, _), pair in pairs.items():
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
        "phase": "B2c-FW-ECRC",
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
            "suite_manifest": str(suite_manifest.relative_to(output_dir)),
            "generator_sha256": _file_sha256(
                REPO_ROOT / "scripts" / "multi_evidence" / "generate_b2a_gc.py"
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
            "familywise_calibration_sha256": _file_sha256(
                REPO_ROOT / "scripts" / "multi_evidence" / "familywise_calibration.py"
            ),
            "multi_target_familywise_calibration_sha256": _file_sha256(
                REPO_ROOT
                / "scripts"
                / "multi_evidence"
                / "multi_target_familywise_calibration.py"
            ),
            "test_scores_used_for_thresholds": False,
            "test_labels_used": False,
            "hidden_phase_used_by_model_or_calibration": False,
            "latent_factors_used_by_model_or_calibration": False,
            "structural_support_used_by_model_or_calibration": False,
        },
    }
    _write_json_atomic(output_dir / "b2c_evaluation.json", result)
    _write_json_atomic(
        output_dir / "run_metadata.json",
        {
            "status": result["status"],
            "phase": "B2c-FW-ECRC",
            "config_hash": result["config_hash"],
            "outputs": {
                "evaluation": "b2c_evaluation.json",
                "episode_scores": "episode_scores.csv",
                "background_scores": "background_scores.npz",
                "model_state": "multi_target_model_state.pt",
                "synthetic_suite": "synthetic_suite",
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
    print(f"B2c-FW-ECRC status: {result['status']}; results: {output_dir}")
    return 2 if arguments.strict and result["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
