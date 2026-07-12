#!/usr/bin/env python3
"""Replay B2c seed 4301 with B1 global ECRC using the saved model weights.

This is a calibration attribution analysis, not a new experiment. It reads a
completed B2c run, reconstructs its saved normal streams and episodes, restores
the exact trained targetwise heads on CPU, and refits only B1's global outer
normal ECRC. No model is trained and no test scores or labels enter calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.multi_evidence.generate_b2a_gc import DRIFT_CONTROL_ROLE, _file_sha256
from scripts.multi_evidence.multi_target_calibration import (
    MultiTargetEvidenceReliabilityCalibration,
)
from scripts.multi_evidence.run_b2a_transfer import (
    TAIL_COMPONENTS,
    _background_fpr,
    _episode_rows,
    _gate,
    _paired_order,
    _paired_records,
    _window_segment,
    split_normal_train,
)
from ts_benchmark.baselines.MultiEvidenceRepair.MultiEvidenceRepair import (
    ChannelStandardizer,
    MultiEvidenceRepair,
    terminal_windows,
)
from ts_benchmark.baselines.MultiEvidenceRepair.MultiTargetEvidenceRepair import (
    MultiTargetEvidenceRepair,
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _restore_model(
    config: Mapping[str, Any], checkpoint: Mapping[str, Any], device: torch.device
) -> MultiTargetEvidenceRepair:
    targets = tuple(int(value) for value in config["target_indices"])
    model_config = config["model"]
    restored = MultiTargetEvidenceRepair(
        dimensions=int(config["dimensions"]),
        target_indices=targets,
        d_model=int(model_config["d_model"]),
        dropout=float(model_config["dropout"]),
        learning_rate=float(model_config["learning_rate"]),
        epochs=int(model_config["epochs"]),
        patience=int(model_config["patience"]),
        batch_size=int(model_config["batch_size"]),
        device=device,
    )
    states = checkpoint["state_dict_by_target"]
    for target_index in targets:
        scalar = MultiEvidenceRepair(
            dimensions=int(config["dimensions"]),
            target_index=target_index,
            d_model=int(model_config["d_model"]),
            dropout=float(model_config["dropout"]),
            learning_rate=float(model_config["learning_rate"]),
            epochs=int(model_config["epochs"]),
            patience=int(model_config["patience"]),
            batch_size=int(model_config["batch_size"]),
            device=device,
        )
        scalar.net.load_state_dict(states[str(target_index)])
        scalar.net.eval()
        restored.models[target_index] = scalar
    return restored


def _load_episodes(path: Path) -> list[Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as saved:
        windows = np.asarray(saved["windows"], dtype=np.float32)
        pair_ids = saved["pair_ids"]
        roles = saved["roles"]
        is_pair = saved["is_pair"]
        targets = saved["target_indices"]
        phase_bins = saved["phase_bins"]
        sources = saved["source_terminal_indices"]
        donors = saved["donor_terminal_indices"]
    return [
        {
            "pair_id": str(pair_ids[index]),
            "role": str(roles[index]),
            "is_pair": bool(is_pair[index]),
            "target_index": int(targets[index]),
            "phase_bin": int(phase_bins[index]),
            "source_terminal_index": int(sources[index]),
            "donor_terminal_index": int(donors[index]),
            # These fields are emitted by the shared row helper but never used
            # by calibration or gates in this replay.
            "relation_value": 0.0,
            "relation_velocity": 0.0,
            "values": windows[index],
        }
        for index in range(len(windows))
    ]


def _performance_gates(
    config: Mapping[str, Any],
    target_indices: Sequence[int],
    pairs: Mapping[tuple[int, str], Mapping[str, Mapping[str, Any]]],
    episode_rows: Sequence[Mapping[str, Any]],
    reference_scores: Mapping[str, np.ndarray],
    windows: Mapping[str, np.ndarray],
    fpr: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    evaluation = config["evaluation"]
    gates: Dict[str, Dict[str, Any]] = {}
    target_position = {target: position for position, target in enumerate(target_indices)}
    optimization_target_means = np.mean(
        windows["optimization"][:, -1, list(target_indices)], axis=0, dtype=np.float64
    )
    for target_index in target_indices:
        position = target_position[target_index]
        for role in ("unsupported_target_break", "target_omission_break"):
            for component in ("cross_residual_tail", "disagreement_tail"):
                summary = _paired_order(pairs, target_index, role, component)
                gates[f"target_{target_index}_{role}_{component}"] = _gate(
                    int(summary["positive_count"]) >= int(evaluation["paired_order_min"])
                    and float(summary["median_delta"])
                    >= float(evaluation["paired_tail_margin_min"]),
                    **summary,
                    required_order=int(evaluation["paired_order_min"]),
                    required_margin=float(evaluation["paired_tail_margin_min"]),
                )
        target_pairs = [
            pair for (target, _), pair in pairs.items() if target == target_index
        ]
        coherent_exceedance = np.asarray(
            [
                bool(pair["coherent_control"]["cross_residual_tail_exceeds"])
                or bool(pair["coherent_control"]["disagreement_tail_exceeds"])
                for pair in target_pairs
            ],
            dtype=bool,
        )
        gates[f"target_{target_index}_coherent_control"] = _gate(
            int(coherent_exceedance.sum())
            <= int(evaluation["coherent_exceedance_max"]),
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
            int(drift_exceedance.sum())
            <= int(evaluation["drift_control_exceedance_max"]),
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
        gates[f"target_{target_index}_cross_normal_skill"] = _gate(
            improvement
            >= float(evaluation["cross_mean_mae_relative_improvement_min"]),
            cross_mae=cross_mae,
            target_mean_mae=baseline_mae,
            relative_improvement=improvement,
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
                target_index=int(target_index),
                component=tail_component,
                details=detail,
                reliability_bin_gap=bin_gap,
                phase_diagnostic_gap=phase_gap,
                maximum_reliability_bin_fpr=float(
                    evaluation["background_component_fpr_max"]
                ),
                maximum_phase_fpr=float(evaluation["drift_phase_fpr_max"]),
            )
    return gates


def analyze(run_dir: Path, output_path: Path, device: torch.device) -> Dict[str, Any]:
    suite_dir = run_dir / "synthetic_suite"
    config = _load_json(suite_dir / "resolved_config.json")
    source_evaluation = _load_json(run_dir / "b2c_evaluation.json")
    checkpoint = torch.load(
        run_dir / "multi_target_model_state.pt", map_location=device, weights_only=False
    )
    with np.load(suite_dir / "normal_streams.npz", allow_pickle=False) as streams:
        train_values = np.asarray(streams["train_values"], dtype=np.float32)
        background_values = np.asarray(streams["background_values"], dtype=np.float32)
        background_phase = np.asarray(streams["background_phase_bin"], dtype=np.int64)
    history_length = int(config["history_length"])
    target_indices = tuple(int(value) for value in config["target_indices"])
    segments = split_normal_train(len(train_values), history_length, config["split"])
    standardizer = ChannelStandardizer().fit(
        train_values[segments["optimization"]["start"] : segments["optimization"]["end"]]
    )
    windows = {
        name: _window_segment(train_values, segment, standardizer, history_length)
        for name, segment in segments.items()
    }
    model = _restore_model(config, checkpoint, device)
    reference_raw = model.score_windows(windows["reference"], include_tails=False)
    outer_raw = model.score_windows(windows["outer_calibration"], include_tails=False)
    calibration_config = config["calibration"]
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
        standardizer.transform(background_values), history_length
    )
    background_raw = model.score_windows(background_windows, include_tails=False)
    background_scores = calibration.transform(background_windows, background_raw)
    source_background_path = run_dir / "background_scores.npz"
    with np.load(source_background_path, allow_pickle=False) as source_background:
        raw_deltas = {
            component: float(
                np.max(
                    np.abs(
                        background_raw[component]
                        - np.asarray(source_background[component], dtype=np.float64)
                    )
                )
            )
            for component in ("temporal_residual", "cross_residual", "disagreement")
        }
    fpr = _background_fpr(
        background_scores, calibration, background_phase[history_length:]
    )
    suite = {"episodes": _load_episodes(suite_dir / "episodes.npz")}
    episode_rows, _ = _episode_rows(suite, standardizer, model, calibration)
    pairs = _paired_records(episode_rows)
    gates = _performance_gates(
        config,
        target_indices,
        pairs,
        episode_rows,
        reference_scores,
        windows,
        fpr,
    )
    result = {
        "analysis": "B2c same-model global-ECRC replay",
        "source_run": str(run_dir),
        "source_phase": source_evaluation["phase"],
        "source_seed": int(source_evaluation["seed"]),
        "source_config_hash": source_evaluation["config_hash"],
        "analysis_device": str(device),
        "model_retrained": False,
        "test_scores_used_for_calibration": False,
        "test_labels_used_for_calibration": False,
        "saved_background_raw_score_max_abs_difference": raw_deltas,
        "calibration": calibration.metadata(),
        "background_fpr": fpr,
        "gates": gates,
        "passed_gate_count": int(sum(gate["pass"] for gate in gates.values())),
        "failed_gate_count": int(sum(not gate["pass"] for gate in gates.values())),
        "status": "passed" if all(gate["pass"] for gate in gates.values()) else "failed_gates",
        "provenance": {
            "analysis_runner_sha256": _file_sha256(Path(__file__)),
            "source_model_state": "multi_target_model_state.pt",
            "source_normal_streams": "synthetic_suite/normal_streams.npz",
            "source_episodes": "synthetic_suite/episodes.npz",
        },
    }
    _write_json_atomic(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.torch_threads < 1:
        raise ValueError("--torch-threads must be positive.")
    torch.set_num_threads(arguments.torch_threads)
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA replay was requested but CUDA is unavailable.")
    output = arguments.output
    if output is None:
        output = arguments.run_dir / "analysis" / "b2c_global_ecrc_same_model.json"
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"Refusing to overwrite analysis output: {output}")
    result = analyze(arguments.run_dir, output, device)
    print(
        "B2c same-model global-ECRC replay: "
        f"{result['status']}; {result['passed_gate_count']} passed, "
        f"{result['failed_gate_count']} failed; results: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
