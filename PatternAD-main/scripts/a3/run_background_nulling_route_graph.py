#!/usr/bin/env python3
"""Fit and evaluate the frozen A3-N1 background-nulling route graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a3.audit_background_nulling_preflight import (
    DEFAULT_PREFLIGHT_CONFIG,
    audit_background_nulling_preflight,
)
from scripts.a3.audit_independent_background_contract import (
    maximum_accepted_exceedances,
    one_sided_wilson_upper,
)
from scripts.a3.audit_route_identifiability_contract import (
    DEFAULT_BACKGROUND_PROTOCOL,
    DEFAULT_CONTRACT_CONFIG,
)
from scripts.a3.generate_independent_background_contract import generate_independent_background_suite
from scripts.a3.generate_trigger_response_contract import _load_json, generate_suite
from ts_benchmark.baselines.A3TriggerResponse import A3BackgroundNullingRouteGraph


DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "config" / "a3" / "background_nulling_n1_development_v1.json"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(requested: str) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("A3-N1 requested CUDA but CUDA is unavailable.")
        return "cuda"
    if requested != "cpu":
        raise ValueError("A3-N1 device must be either 'cpu' or 'cuda'.")
    return "cpu"


def _event_bank(suite: Mapping[str, Any], split: str) -> np.ndarray:
    bank = suite["normal_event_banks"][split]
    if not bank:
        raise ValueError(f"A3-N1 split {split} lacks normal event windows.")
    return np.stack([np.asarray(entry["values"], dtype=np.float32) for entry in bank])


def _ordinary_normal_windows(
    suite: Mapping[str, Any], split: str, history: int, horizon: int
) -> np.ndarray:
    start, end = suite["normal_split_ranges"][split]
    values = np.asarray(suite["train_values"], dtype=np.float32)
    starts = np.arange(start, end - history - horizon + 1, dtype=np.int64)
    if not len(starts):
        raise ValueError(f"A3-N1 split {split} lacks ordinary normal windows.")
    return np.stack([values[index : index + history + horizon] for index in starts])


def _normal_windows(
    suite: Mapping[str, Any], split: str, history: int, horizon: int
) -> np.ndarray:
    return np.concatenate(
        (_ordinary_normal_windows(suite, split, history, horizon), _event_bank(suite, split)), axis=0
    )


def _normal_optimization_values(suite: Mapping[str, Any]) -> np.ndarray:
    start, end = suite["normal_split_ranges"]["optimization"]
    return np.asarray(suite["train_values"], dtype=np.float32)[start:end]


def _episode_rows(suite: Mapping[str, Any], model: A3BackgroundNullingRouteGraph) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for episode in suite["episodes"]:
        scores = model.score_windows(np.asarray(episode["values"], dtype=np.float32)[None, ...])
        rows.append(
            {
                "pair_id": str(episode["pair_id"]),
                "role": str(episode["role"]),
                "cue_mode": episode["cue_mode"],
                "future_mode": episode["future_mode"],
                "null_route_surprisal": float(scores["null_route_surprisal"][0]),
                "null_route_tail": float(scores["null_route_tail"][0]),
                "null_route_threshold": float(scores["null_route_threshold"][0]),
                "null_route_exceedance": int(scores["null_route_exceedance"][0]),
                "trigger_state": int(scores["trigger_state"][0]),
                "node_surprisal": scores["node_surprisal"][0].astype(float).tolist(),
            }
        )
    return rows


def _paired_gate(
    rows: List[Mapping[str, Any]], normal_role: str, anomalous_role: str
) -> Dict[str, Any]:
    pairs: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pairs.setdefault(str(row["pair_id"]), {})[str(row["role"])] = row
    margins = []
    for _, roles in sorted(pairs.items()):
        if normal_role in roles and anomalous_role in roles:
            margins.append(
                float(roles[anomalous_role]["null_route_tail"])
                - float(roles[normal_role]["null_route_tail"])
            )
    if not margins:
        raise ValueError(f"A3-N1 lacks {normal_role}/{anomalous_role} pairs.")
    return {
        "pair_count": len(margins),
        "positive_pairs": int(np.sum(np.asarray(margins) > 0.0)),
        "median_tail_margin": float(np.median(margins)),
        "margins": [float(value) for value in margins],
    }


def _normal_role_report(rows: List[Mapping[str, Any]], role: str) -> Dict[str, int]:
    selected = [row for row in rows if row["role"] == role]
    if not selected:
        raise ValueError(f"A3-N1 lacks normal role {role}.")
    return {
        "count": len(selected),
        "below_threshold": int(sum(int(row["null_route_exceedance"]) == 0 for row in selected)),
    }


def _event_pre_isolation(
    model: A3BackgroundNullingRouteGraph, windows: np.ndarray, history: int
) -> float:
    changed = np.asarray(windows, dtype=np.float32).copy()
    changed[:, history:] = -3.0 * changed[:, history:] + 0.25
    return float(np.max(np.abs(model.event_pre_state(windows) - model.event_pre_state(changed))))


def _background_report(
    scores: Mapping[str, np.ndarray], provenance: List[Mapping[str, Any]], evaluation: Mapping[str, Any]
) -> Dict[str, Any]:
    exceedances = np.asarray(scores["null_route_exceedance"], dtype=np.int64)
    total = len(exceedances)
    target = float(evaluation["operating_fpr"])
    confidence = float(evaluation["confidence_level"])
    pooled_count = int(exceedances.sum())
    per_regime: Dict[str, Any] = {}
    for regime_index in sorted({int(row["regime_index"]) for row in provenance}):
        indices = np.asarray(
            [index for index, row in enumerate(provenance) if int(row["regime_index"]) == regime_index],
            dtype=np.int64,
        )
        count = len(indices)
        positive = int(exceedances[indices].sum())
        upper = one_sided_wilson_upper(positive, count, confidence)
        per_regime[str(regime_index)] = {
            "count": int(count),
            "exceedances": positive,
            "fpr": float(positive / count),
            "upper_bound": upper,
            "maximum_accepted_exceedances": maximum_accepted_exceedances(count, target, confidence),
            "passed": upper <= target,
        }
    pooled_upper = one_sided_wilson_upper(pooled_count, total, confidence)
    return {
        "count": int(total),
        "exceedances": pooled_count,
        "fpr": float(pooled_count / total),
        "upper_bound": pooled_upper,
        "maximum_accepted_exceedances": maximum_accepted_exceedances(total, target, confidence),
        "per_regime": per_regime,
        "passed": pooled_upper <= target and all(report["passed"] for report in per_regime.values()),
    }


def _validate_experiment_config(
    contract: Mapping[str, Any], preflight: Mapping[str, Any], experiment: Mapping[str, Any]
) -> None:
    if int(experiment.get("schema_version", 0)) != 1:
        raise ValueError("A3-N1 experiment config must use schema_version=1.")
    experiment_id = str(experiment.get("experiment_id", ""))
    valid_ids = {
        "a3_n1_background_nulling_route_graph_development_v1": True,
        "a3_n1_background_nulling_route_graph_past_free_control_v1": False,
    }
    if experiment_id not in valid_ids:
        raise ValueError("Unexpected A3-N1 experiment_id.")
    if bool(experiment["model"]["condition_on_event_pre"]) != valid_ids[experiment_id]:
        raise ValueError("A3-N1 experiment_id and event-pre conditioning do not agree.")
    if int(experiment["background_subspace"]["components"]) != 1:
        raise ValueError("A3-N1 development must use one background factor.")
    if str(experiment["background_subspace"]["channels"]) != "all_raw":
        raise ValueError("A3-N1 development must project all raw channels.")
    if str(experiment["background_subspace"]["fit_split"]) != "optimization":
        raise ValueError("A3-N1 development must fit its factor on optimization only.")
    if float(experiment["token_extractor"]["token_energy_threshold"]) != float(
        contract["episodes"]["token_energy_threshold"]
    ):
        raise ValueError("A3-N1 token threshold does not match its route contract.")
    if float(experiment["calibration"]["outer_alpha"]) != 0.05:
        raise ValueError("A3-N1 outer alpha is frozen at 0.05 for the independent FPR gate.")
    if str(preflight["base_contract_id"]) != str(contract["suite_id"]):
        raise ValueError("A3-N1 preflight/base contract mismatch.")


def run_experiment(
    contract: Mapping[str, Any],
    background_protocol: Mapping[str, Any],
    preflight: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> tuple[Dict[str, Any], Mapping[str, torch.Tensor], np.ndarray]:
    _validate_experiment_config(contract, preflight, experiment)
    preflight_result = audit_background_nulling_preflight(contract, background_protocol, preflight)
    if not preflight_result["passed"]:
        raise RuntimeError(f"A3-N1 raw preflight failed: {preflight_result['violations']}")
    suite = generate_suite(contract)
    seed = int(experiment["seed"])
    _set_seed(seed)
    history = int(contract["history_length"])
    horizon = int(contract["horizon_length"])
    model_config = experiment["model"]
    trigger = experiment["trigger_extractor"]
    model = A3BackgroundNullingRouteGraph(
        dimensions=int(contract["dimensions"]),
        history_length=history,
        horizon_length=horizon,
        token_energy_threshold=float(experiment["token_extractor"]["token_energy_threshold"]),
        cue_length=int(trigger["cue_length"]),
        minimum_trigger_amplitude=float(trigger["minimum_amplitude"]),
        trigger_linear_tolerance=float(trigger["linear_tolerance"]),
        hidden_size=int(model_config["hidden_size"]),
        condition_on_event_pre=bool(model_config["condition_on_event_pre"]),
        learning_rate=float(model_config["learning_rate"]),
        epochs=int(model_config["epochs"]),
        patience=int(model_config["patience"]),
        batch_size=int(model_config["batch_size"]),
        outer_alpha=float(experiment["calibration"]["outer_alpha"]),
        device=_select_device(str(experiment["device"])),
    ).fit(
        _normal_optimization_values(suite),
        _normal_windows(suite, "optimization", history, horizon),
        _normal_windows(suite, "validation", history, horizon),
        _normal_windows(suite, "reference", history, horizon),
        _normal_windows(suite, "outer_calibration", history, horizon),
        seed=seed,
    )
    rows = _episode_rows(suite, model)
    primary = _paired_gate(rows, "normal_routed_response", "misrouted_response")
    secondary = _paired_gate(rows, "normal_routed_response", "partial_propagation_response")
    untriggered = _paired_gate(rows, "normal_no_trigger", "untriggered_response")
    normal_reports = {
        role: _normal_role_report(rows, role)
        for role in ("normal_routed_response", "normal_no_trigger")
    }
    background_suite = generate_independent_background_suite(contract, background_protocol)
    background_scores = model.score_windows(background_suite["windows"])
    background = _background_report(
        background_scores, background_suite["provenance"], background_protocol["evaluation"]
    )
    primary_pass = primary["positive_pairs"] >= 14 and primary["median_tail_margin"] > 0.0
    secondary_pass = secondary["positive_pairs"] >= 14 and secondary["median_tail_margin"] > 0.0
    untriggered_pass = untriggered["positive_pairs"] >= 14 and untriggered["median_tail_margin"] > 0.0
    normal_pass = all(report["below_threshold"] >= 14 for report in normal_reports.values())
    isolation = _event_pre_isolation(
        model, _normal_windows(suite, "optimization", history, horizon)[:16], history
    )
    gates = {
        "normal_only_preflight": {"passed": True, "metrics": preflight_result["metrics"]},
        "event_pre_isolation": {"max_state_difference": isolation, "passed": isolation <= 1e-7},
        "primary_misrouted_null_route": {**primary, "passed": primary_pass},
        "secondary_partial_propagation_null_route": {**secondary, "passed": secondary_pass},
        "untriggered_null_route": {**untriggered, "passed": untriggered_pass},
        "normal_transition_controls": {"roles": normal_reports, "passed": normal_pass},
        "independent_background": background,
    }
    summary: Dict[str, Any] = {
        "experiment_id": str(experiment["experiment_id"]),
        "seed": seed,
        "device": str(model.device),
        "contract_config_hash": _canonical_hash(contract),
        "background_protocol_hash": _canonical_hash(background_protocol),
        "preflight_config_hash": _canonical_hash(preflight),
        "experiment_config": dict(experiment),
        "fit": model.fit_metadata_,
        "normal_split_window_counts": {
            name: {
                "ordinary_normal": int(len(_ordinary_normal_windows(suite, name, history, horizon))),
                "normal_event_bank": int(len(_event_bank(suite, name))),
                "combined": int(len(_normal_windows(suite, name, history, horizon))),
            }
            for name in suite["normal_event_banks"]
        },
        "gates": gates,
        "all_gates_passed": all(gate["passed"] for gate in gates.values()),
        "episode_scores": rows,
    }
    return summary, model.state_dict(), model.background_factor()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_result(
    output_dir: Path,
    summary: Mapping[str, Any],
    checkpoint: Mapping[str, torch.Tensor],
    background_factor: np.ndarray,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "summary.json", summary)
    torch.save(
        {
            "contract_config_hash": summary["contract_config_hash"],
            "background_protocol_hash": summary["background_protocol_hash"],
            "preflight_config_hash": summary["preflight_config_hash"],
            "experiment_config": summary["experiment_config"],
            "fit": summary["fit"],
            "background_factor": np.asarray(background_factor, dtype=np.float64),
            "model_state_dict": checkpoint,
        },
        output_dir / "model.pt",
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-config", type=Path, default=DEFAULT_CONTRACT_CONFIG)
    parser.add_argument("--background-protocol", type=Path, default=DEFAULT_BACKGROUND_PROTOCOL)
    parser.add_argument("--preflight-config", type=Path, default=DEFAULT_PREFLIGHT_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--contract-seed", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    arguments = parser.parse_args(argv)
    if arguments.torch_threads < 1:
        raise ValueError("--torch-threads must be positive.")
    torch.set_num_threads(arguments.torch_threads)
    contract = _load_json(arguments.contract_config)
    background_protocol = _load_json(arguments.background_protocol)
    preflight = _load_json(arguments.preflight_config)
    experiment = _load_json(arguments.experiment_config)
    if arguments.seed is not None:
        experiment["seed"] = int(arguments.seed)
    if arguments.contract_seed is not None:
        contract["seed"] = int(arguments.contract_seed)
    summary, checkpoint, factor = run_experiment(contract, background_protocol, preflight, experiment)
    write_result(arguments.output_dir, summary, checkpoint, factor)
    print(
        f"A3-N1 complete: gates_passed={summary['all_gates_passed']} "
        f"summary={arguments.output_dir / 'summary.json'}"
    )
    return 0 if summary["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
