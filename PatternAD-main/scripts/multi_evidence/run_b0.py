#!/usr/bin/env python3
"""Run the frozen Direction B0 dual-evidence synthetic experiment.

The runner has no score fusion.  It exports temporal residual, cross residual,
and branch disagreement independently, then tests their pre-registered roles.
"""

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
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.multi_evidence.generate_b0_synthetic import (  # noqa: E402
    DEFAULT_CONFIG,
    ROLE_ORDER,
    _canonical_hash,
    _file_sha256,
    _load_json,
    generate_suite,
    write_suite,
)
from scripts.multi_evidence.reliability_calibration import (  # noqa: E402
    EvidenceReliabilityCalibration,
)
from ts_benchmark.baselines.MultiEvidenceRepair.MultiEvidenceRepair import (  # noqa: E402
    ChannelStandardizer,
    MultiEvidenceRepair,
    terminal_windows,
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
    """Allocate disjoint normal intervals separated by H-point gaps."""
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
        raise ValueError("B0 split leaves too little optimization-normal data.")
    segments = {
        "optimization": {"start": 0, "end": optimization_end},
        "validation": {"start": validation_start, "end": validation_end},
        "reference": {"start": reference_start, "end": reference_end},
        "outer_calibration": {"start": outer_start, "end": train_length},
    }
    for name, segment in segments.items():
        if segment["end"] - segment["start"] <= history_length:
            raise ValueError(f"B0 {name} segment is too short for one terminal window.")
        segment["length"] = segment["end"] - segment["start"]
    return segments


def _window_segment(
    values: np.ndarray,
    segment: Mapping[str, int],
    standardizer: ChannelStandardizer,
    history_length: int,
) -> np.ndarray:
    raw = values[int(segment["start"]) : int(segment["end"])]
    return terminal_windows(standardizer.transform(raw), history_length)


def _serializable_float(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot write an empty B0 CSV.")
    fields = list(materialized[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _serializable_float(value) for key, value in row.items()})


def _episode_scores(
    suite: Mapping[str, Any],
    standardizer: ChannelStandardizer,
    model: MultiEvidenceRepair,
    calibration: EvidenceReliabilityCalibration,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for episode in suite["episodes"]:
        windows = standardizer.transform(np.asarray(episode["values"], dtype=np.float32))[None, :, :]
        raw_scores = model.score_windows(windows, include_tails=False)
        scores = calibration.transform(windows, raw_scores)
        row: Dict[str, Any] = {
            "pair_id": str(episode["pair_id"]),
            "role": str(episode["role"]),
            "regime": int(episode["regime"]),
            "source_terminal_index": int(episode["source_terminal_index"]),
            "donor_terminal_index": int(episode["donor_terminal_index"]),
        }
        for name, values in scores.items():
            row[name] = float(values[0])
        rows.append(row)
    return rows


def _paired_records(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    paired: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[str(row["pair_id"])][str(row["role"])] = row
    expected = set(ROLE_ORDER)
    for pair_id, pair in paired.items():
        missing = expected - set(pair)
        if missing:
            raise ValueError(f"Paired episode {pair_id} is missing roles: {sorted(missing)}")
    return dict(paired)


def _paired_order(
    pairs: Mapping[str, Mapping[str, Mapping[str, Any]]], role: str, metric: str
) -> Dict[str, Any]:
    deltas = np.asarray(
        [
            float(pair[role][metric]) - float(pair["coherent_control"][metric])
            for pair in pairs.values()
        ],
        dtype=np.float64,
    )
    return {
        "metric": metric,
        "role": role,
        "count": int(len(deltas)),
        "positive_count": int(np.sum(deltas > 0.0)),
        "median_delta": float(np.median(deltas)),
        "minimum_delta": float(np.min(deltas)),
        "deltas": deltas.astype(float).tolist(),
    }


def _gate(passed: bool, **details: Any) -> Dict[str, Any]:
    return {"pass": bool(passed), **details}


def _fpr_by_hidden_regime(
    scores: Mapping[str, np.ndarray],
    calibration: EvidenceReliabilityCalibration,
    regimes: np.ndarray,
) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    for component in TAIL_COMPONENTS:
        exceedance = calibration.exceeds(scores, component)
        by_regime = {
            str(regime): float(exceedance[regimes == regime].mean())
            for regime in (0, 1)
        }
        details[component] = {
            "overall": float(exceedance.mean()),
            "by_regime": by_regime,
        }
    return details


def _fpr_by_reliability_stratum(
    scores: Mapping[str, np.ndarray], calibration: EvidenceReliabilityCalibration
) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    for tail_component in TAIL_COMPONENTS:
        raw_component = tail_component[: -len("_tail")]
        exceedance = calibration.exceeds(scores, tail_component)
        strata = np.asarray(
            scores[f"{raw_component}_reliability_stratum"], dtype=np.int64
        )
        details[tail_component] = {
            "overall": float(exceedance.mean()),
            "by_reliability_stratum": {
                str(stratum): float(exceedance[strata == stratum].mean())
                for stratum in range(calibration.reliability_strata)
            },
        }
    return details


def _reliability_isolation_report(
    calibration: EvidenceReliabilityCalibration, windows: np.ndarray
) -> Dict[str, float]:
    """Verify B1 routing uses no forbidden terminal target or branch input."""
    base = np.asarray(windows[:1], dtype=np.float64).copy()
    target = calibration.target_index
    drivers = [index for index in range(calibration.dimensions) if index != target]
    driver_changed = base.copy()
    driver_changed[:, :, drivers] += np.linspace(0.0, 3.0, base.shape[1])[None, :, None]
    terminal_changed = base.copy()
    terminal_changed[:, -1, target] += 3.0
    target_changed = base.copy()
    target_changed[:, :, target] += np.linspace(0.0, 3.0, base.shape[1])[None, :]
    base_features = calibration.features(base)
    driver_features = calibration.features(driver_changed)
    terminal_features = calibration.features(terminal_changed)
    target_features = calibration.features(target_changed)
    return {
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


def run_experiment(
    config: Mapping[str, Any],
    output_dir: Path,
    device: str,
    seed: int,
    epoch_override: int | None = None,
) -> Dict[str, Any]:
    """Fit B0, write its full provenance, and evaluate its frozen gates."""
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing B0 output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    _set_seed(seed)
    suite = generate_suite(config)
    suite_dir = output_dir / "synthetic_suite"
    suite_manifest = write_suite(config, suite, suite_dir)
    history_length = int(config["history_length"])
    target_index = int(config["episodes"]["target_channel"])
    segments = split_normal_train(
        len(suite["train_values"]), history_length, config["split"]
    )
    standardizer = ChannelStandardizer().fit(
        np.asarray(suite["train_values"])[
            segments["optimization"]["start"] : segments["optimization"]["end"]
        ]
    )
    windows = {
        name: _window_segment(
            np.asarray(suite["train_values"]), segment, standardizer, history_length
        )
        for name, segment in segments.items()
    }
    model_config = dict(config["model"])
    if epoch_override is not None:
        if epoch_override < 1:
            raise ValueError("--epochs must be positive.")
        model_config["epochs"] = int(epoch_override)
        model_config["patience"] = min(int(model_config["patience"]), int(epoch_override))
    model = MultiEvidenceRepair(
        dimensions=int(config["dimensions"]),
        target_index=target_index,
        d_model=int(model_config["d_model"]),
        dropout=float(model_config["dropout"]),
        learning_rate=float(model_config["learning_rate"]),
        epochs=int(model_config["epochs"]),
        patience=int(model_config["patience"]),
        batch_size=int(model_config["batch_size"]),
        device=device,
    ).fit(
        windows["optimization"], windows["validation"], windows["reference"], seed
    )
    calibration_config = dict(config.get("calibration", {}))
    calibration_mode = str(calibration_config.get("mode", "global"))
    reliability_strata = int(calibration_config.get("reliability_strata", 1))
    min_reference_per_stratum = int(
        calibration_config.get("min_reference_per_stratum", 8)
    )
    reference_raw_scores = model.score_windows(windows["reference"], include_tails=False)
    outer_raw_scores = model.score_windows(
        windows["outer_calibration"], include_tails=False
    )
    calibration = EvidenceReliabilityCalibration(
        dimensions=int(config["dimensions"]),
        target_index=target_index,
        target_fpr=float(config["evaluation"]["target_fpr"]),
        mode=calibration_mode,
        reliability_strata=reliability_strata,
        min_reference_per_stratum=min_reference_per_stratum,
    ).fit(
        windows["optimization"],
        windows["reference"],
        reference_raw_scores,
        windows["outer_calibration"],
        outer_raw_scores,
    )
    calibration_metadata = calibration.metadata()
    thresholds = calibration_metadata["thresholds"]
    background_windows = terminal_windows(
        standardizer.transform(np.asarray(suite["background_values"])), history_length
    )
    background_raw_scores = model.score_windows(background_windows, include_tails=False)
    background_scores = calibration.transform(background_windows, background_raw_scores)
    background_regimes = np.asarray(suite["background_regime"], dtype=np.int64)[history_length:]
    if calibration_mode == "global":
        fpr = _fpr_by_hidden_regime(background_scores, calibration, background_regimes)
        fpr_group_key = "by_regime"
        fpr_group_name = "hidden_regime_diagnostic"
    else:
        fpr = _fpr_by_reliability_stratum(background_scores, calibration)
        fpr_group_key = "by_reliability_stratum"
        fpr_group_name = "observable_reliability_stratum"
    reference_scores = calibration.transform(windows["reference"], reference_raw_scores)
    optimization_target_mean = float(
        np.mean(windows["optimization"][:, -1, target_index], dtype=np.float64)
    )
    cross_mae = float(np.mean(np.abs(reference_scores["target"] - reference_scores["mu_cross"])))
    target_mean_mae = float(
        np.mean(np.abs(reference_scores["target"] - optimization_target_mean))
    )
    episode_rows = _episode_scores(suite, standardizer, model, calibration)
    _write_csv(output_dir / "episode_scores.csv", episode_rows)
    pairs = _paired_records(episode_rows)
    evaluation = config["evaluation"]
    paired_order: Dict[str, Dict[str, Any]] = {}
    gates: Dict[str, Dict[str, Any]] = {}
    isolation = model.evidence_isolation_report(windows["reference"][:1])
    gates["information_isolation"] = _gate(
        bool(isolation["parameter_sets_disjoint"])
        and float(isolation["temporal_driver_delta"]) <= 1e-7
        and float(isolation["temporal_terminal_target_delta"]) <= 1e-7
        and float(isolation["cross_target_column_delta"]) <= 1e-7,
        **isolation,
        tolerance=1e-7,
    )
    if calibration_mode == "input_energy_stratified":
        reliability_isolation = _reliability_isolation_report(
            calibration, windows["reference"][:1]
        )
        gates["reliability_routing_isolation"] = _gate(
            all(float(value) <= 1e-7 for value in reliability_isolation.values()),
            **reliability_isolation,
            tolerance=1e-7,
        )
    contracts = list(suite["contracts"])
    contract_pass = all(
        float(contract["coherent_unsupported_target_max_abs_difference"]) <= 1e-7
        and float(contract["coherent_omission_driver_max_abs_difference"]) <= 1e-7
        and float(contract["coherent_unsupported_driver_difference"]) > 1e-6
        and float(contract["coherent_omission_target_difference"]) > 1e-6
        for contract in contracts
    )
    gates["synthetic_pair_contract"] = _gate(
        contract_pass, contracts=contracts, tolerance=1e-7
    )
    temporal_tie = np.asarray(
        [
            abs(
                float(pair["coherent_control"]["temporal_residual"])
                - float(pair["unsupported_target_break"]["temporal_residual"])
            )
            for pair in pairs.values()
        ]
    )
    cross_tie = np.asarray(
        [
            abs(
                float(pair["coherent_control"]["mu_cross"])
                - float(pair["target_omission_break"]["mu_cross"])
            )
            for pair in pairs.values()
        ]
    )
    gates["counterfactual_input_ties"] = _gate(
        float(temporal_tie.max()) <= 1e-7 and float(cross_tie.max()) <= 1e-7,
        coherent_unsupported_temporal_residual_max_abs_difference=float(temporal_tie.max()),
        coherent_omission_cross_prediction_max_abs_difference=float(cross_tie.max()),
        tolerance=1e-7,
    )
    required_order = int(evaluation["paired_order_min"])
    required_margin = float(evaluation["paired_tail_margin_min"])
    for role in ("unsupported_target_break", "target_omission_break"):
        for component in ("cross_residual_tail", "disagreement_tail"):
            key = f"{role}_{component}"
            summary = _paired_order(pairs, role, component)
            paired_order[key] = summary
            gates[key] = _gate(
                int(summary["positive_count"]) >= required_order
                and float(summary["median_delta"]) >= required_margin,
                **summary,
                required_order=required_order,
                required_margin=required_margin,
            )
    coherent_exceedance = np.asarray(
        [
            calibration.row_exceeds(
                pair["coherent_control"], "cross_residual_tail"
            )
            or calibration.row_exceeds(
                pair["coherent_control"], "disagreement_tail"
            )
            for pair in pairs.values()
        ],
        dtype=bool,
    )
    max_coherent = int(evaluation["coherent_exceedance_max"])
    gates["coherent_control"] = _gate(
        int(coherent_exceedance.sum()) <= max_coherent,
        count=int(coherent_exceedance.sum()),
        total=int(len(coherent_exceedance)),
        allowed=max_coherent,
        thresholds=thresholds,
    )
    maximum_fpr = float(evaluation["background_component_fpr_max"])
    all_fpr_ok = all(
        float(value) <= maximum_fpr
        for component in fpr.values()
        for value in component[fpr_group_key].values()
    )
    disagreement_groups = fpr["disagreement_tail"][fpr_group_key]
    disagreement_gap = float(
        max(disagreement_groups.values()) - min(disagreement_groups.values())
    )
    gates["background_normal_fpr"] = _gate(
        all_fpr_ok
        and disagreement_gap <= float(evaluation["background_disagreement_fpr_gap_max"]),
        components=fpr,
        group=fpr_group_name,
        maximum_component_group_fpr=maximum_fpr,
        disagreement_fpr_gap=disagreement_gap,
        maximum_disagreement_fpr_gap=float(
            evaluation["background_disagreement_fpr_gap_max"]
        ),
    )
    improvement = 1.0 - cross_mae / max(target_mean_mae, 1e-12)
    gates["cross_normal_skill"] = _gate(
        improvement >= float(evaluation["cross_mean_mae_relative_improvement_min"]),
        cross_mae=cross_mae,
        target_mean_mae=target_mean_mae,
        relative_improvement=improvement,
        required_relative_improvement=float(
            evaluation["cross_mean_mae_relative_improvement_min"]
        ),
    )
    spike_success = np.asarray(
        [
            calibration.row_exceeds(
                pair["target_spike"], "temporal_residual_tail"
            )
            and calibration.row_exceeds(
                pair["target_spike"], "cross_residual_tail"
            )
            for pair in pairs.values()
        ],
        dtype=bool,
    )
    gates["target_spike_residuals"] = _gate(
        int(spike_success.sum()) >= int(evaluation["target_spike_exceedance_min"]),
        success_count=int(spike_success.sum()),
        total=int(len(spike_success)),
        required=int(evaluation["target_spike_exceedance_min"]),
    )
    np.savez_compressed(
        output_dir / "background_scores.npz",
        regimes=background_regimes,
        **{name: np.asarray(values, dtype=np.float64) for name, values in background_scores.items()},
    )
    state_dict = {
        name: tensor.detach().cpu() for name, tensor in model.net.state_dict().items()
    }
    torch.save(
        {
            "state_dict": state_dict,
            "normalizer": standardizer.metadata(),
            "fit_metadata": model.fit_metadata_,
            "target_index": target_index,
        },
        output_dir / "model_state.pt",
    )
    evaluation_filename = (
        "b1_evaluation.json"
        if calibration_mode == "input_energy_stratified"
        else "b0_evaluation.json"
    )
    result = {
        "phase": "B1" if calibration_mode == "input_energy_stratified" else "B0",
        "suite_id": str(config["suite_id"]),
        "status": "passed" if all(gate["pass"] for gate in gates.values()) else "failed_gates",
        "config_hash": _canonical_hash(config),
        "seed": int(seed),
        "device": device,
        "history_length": history_length,
        "target_index": target_index,
        "no_score_fusion": True,
        "score_components": [
            "temporal_residual",
            "cross_residual",
            "disagreement",
        ],
        "tail_components": list(TAIL_COMPONENTS),
        "thresholds": thresholds,
        "calibration": calibration_metadata,
        "split": {"segments": segments, "gaps": [history_length, history_length, history_length]},
        "normalizer": standardizer.metadata(),
        "model": model.fit_metadata_,
        "cross_reference_skill": {
            "cross_mae": cross_mae,
            "target_mean_mae": target_mean_mae,
            "relative_improvement": improvement,
        },
        "paired_order": paired_order,
        "gates": gates,
        "provenance": {
            "suite_manifest": str(suite_manifest.relative_to(output_dir)),
            "generator_sha256": _file_sha256(
                REPO_ROOT / "scripts" / "multi_evidence" / "generate_b0_synthetic.py"
            ),
            "runner_sha256": _file_sha256(Path(__file__)),
            "model_sha256": _file_sha256(
                REPO_ROOT
                / "ts_benchmark"
                / "baselines"
                / "MultiEvidenceRepair"
                / "MultiEvidenceRepair.py"
            ),
            "reliability_calibration_sha256": _file_sha256(
                REPO_ROOT
                / "scripts"
                / "multi_evidence"
                / "reliability_calibration.py"
            ),
            "outer_calibration_only_for_thresholds": True,
            "test_scores_used_for_thresholds": False,
        },
    }
    _write_json_atomic(output_dir / evaluation_filename, result)
    _write_json_atomic(
        output_dir / "run_metadata.json",
        {
            "status": result["status"],
            "config_hash": result["config_hash"],
            "outputs": {
                "evaluation": evaluation_filename,
                "episode_scores": "episode_scores.csv",
                "background_scores": "background_scores.npz",
                "model_state": "model_state.pt",
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
    parser.add_argument(
        "--strict", action="store_true", help="Return non-zero after writing a failed gate result."
    )
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
    print(f"{result['phase']} status: {result['status']}; results: {output_dir}")
    return 2 if arguments.strict and result["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
