#!/usr/bin/env python3
"""Fit and evaluate the frozen A3-G3 counterfactual effect-graph grammar."""

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

from scripts.a3.audit_counterfactual_effect_graph import (
    DEFAULT_EXPERIMENT_CONFIG,
    audit_counterfactual_effect_graph_inputs,
)
from scripts.a3.generate_trigger_response_contract import (
    DEFAULT_CONFIG as DEFAULT_CONTRACT_CONFIG,
    _load_json,
    generate_suite,
)
from ts_benchmark.baselines.A3TriggerResponse import A3CounterfactualEffectGraphGrammar


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
            raise RuntimeError("A3-G3 requested CUDA but CUDA is unavailable.")
        return "cuda"
    if requested != "cpu":
        raise ValueError("A3-G3 device must be either 'cpu' or 'cuda'.")
    return "cpu"


def _event_bank(suite: Mapping[str, Any], split: str) -> np.ndarray:
    bank = suite["normal_event_banks"][split]
    if not bank:
        raise ValueError(f"A3-G3 split {split} lacks normal event windows.")
    return np.stack([np.asarray(entry["values"], dtype=np.float32) for entry in bank])


def _ordinary_normal_windows(
    suite: Mapping[str, Any], split: str, history: int, horizon: int
) -> np.ndarray:
    start, end = suite["normal_split_ranges"][split]
    values = np.asarray(suite["train_values"], dtype=np.float32)
    starts = np.arange(start, end - history - horizon + 1, dtype=np.int64)
    if not len(starts):
        raise ValueError(f"A3-G3 split {split} lacks ordinary normal windows.")
    return np.stack([values[index : index + history + horizon] for index in starts])


def _normal_windows(
    suite: Mapping[str, Any], split: str, history: int, horizon: int
) -> np.ndarray:
    return np.concatenate(
        (_ordinary_normal_windows(suite, split, history, horizon), _event_bank(suite, split)), axis=0
    )


def _background_windows(suite: Mapping[str, Any], history: int, horizon: int) -> np.ndarray:
    values = np.asarray(suite["background_values"], dtype=np.float32)
    starts = np.arange(0, len(values) - history - horizon + 1, dtype=np.int64)
    return np.stack([values[index : index + history + horizon] for index in starts])


def _episode_rows(
    suite: Mapping[str, Any], model: A3CounterfactualEffectGraphGrammar
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for episode in suite["episodes"]:
        scores = model.score_windows(np.asarray(episode["values"], dtype=np.float32)[None, ...])
        rows.append(
            {
                "pair_id": str(episode["pair_id"]),
                "role": str(episode["role"]),
                "cue_mode": episode["cue_mode"],
                "future_mode": episode["future_mode"],
                "effect_graph_surprisal": float(scores["effect_graph_surprisal"][0]),
                "effect_graph_tail": float(scores["effect_graph_tail"][0]),
                "effect_graph_threshold": float(scores["effect_graph_threshold"][0]),
                "effect_graph_exceedance": int(scores["effect_graph_exceedance"][0]),
                "effect_terminal_l2": float(scores["effect_terminal_l2"][0]),
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
                float(roles[anomalous_role]["effect_graph_tail"])
                - float(roles[normal_role]["effect_graph_tail"])
            )
    if not margins:
        raise ValueError(f"A3-G3 lacks {normal_role}/{anomalous_role} pairs.")
    return {
        "pair_count": len(margins),
        "positive_pairs": int(np.sum(np.asarray(margins) > 0.0)),
        "median_tail_margin": float(np.median(margins)),
        "margins": [float(value) for value in margins],
    }


def _normal_role_report(rows: List[Mapping[str, Any]], role: str) -> Dict[str, int]:
    selected = [row for row in rows if row["role"] == role]
    if not selected:
        raise ValueError(f"A3-G3 lacks normal role {role}.")
    return {
        "count": len(selected),
        "below_threshold": int(sum(int(row["effect_graph_exceedance"]) == 0 for row in selected)),
    }


def _event_pre_isolation(
    model: A3CounterfactualEffectGraphGrammar, windows: np.ndarray, history: int
) -> Dict[str, float]:
    changed = np.asarray(windows, dtype=np.float32).copy()
    changed[:, history:] = -3.0 * changed[:, history:] + 0.25
    state_difference = np.max(np.abs(model.event_pre_state(windows) - model.event_pre_state(changed)))
    baseline_difference = np.max(
        np.abs(model.counterfactual_baseline(windows) - model.counterfactual_baseline(changed))
    )
    return {
        "max_state_difference": float(state_difference),
        "max_counterfactual_baseline_difference": float(baseline_difference),
    }


def run_experiment(
    contract_config: Mapping[str, Any], experiment_config: Mapping[str, Any]
) -> tuple[Dict[str, Any], Mapping[str, torch.Tensor]]:
    suite = generate_suite(contract_config)
    audit = audit_counterfactual_effect_graph_inputs(contract_config, experiment_config, suite)
    if not audit["passed"]:
        raise RuntimeError(f"A3-G3 raw effect-graph audit failed: {audit['violations']}")
    seed = int(experiment_config["seed"])
    _set_seed(seed)
    history = int(contract_config["history_length"])
    horizon = int(contract_config["horizon_length"])
    dimensions = int(contract_config["dimensions"])
    model_config = experiment_config["model"]
    trigger_config = experiment_config["trigger_extractor"]
    model = A3CounterfactualEffectGraphGrammar(
        dimensions=dimensions,
        history_length=history,
        horizon_length=horizon,
        effect_token_energy_threshold=float(
            experiment_config["effect_extractor"]["token_energy_threshold"]
        ),
        cue_length=int(trigger_config["cue_length"]),
        minimum_trigger_amplitude=float(trigger_config["minimum_amplitude"]),
        trigger_linear_tolerance=float(trigger_config["linear_tolerance"]),
        ridge_penalty=float(experiment_config["counterfactual"]["ridge_penalty"]),
        hidden_size=int(model_config["hidden_size"]),
        condition_on_event_pre=bool(model_config["condition_on_event_pre"]),
        learning_rate=float(model_config["learning_rate"]),
        epochs=int(model_config["epochs"]),
        patience=int(model_config["patience"]),
        batch_size=int(model_config["batch_size"]),
        outer_alpha=float(experiment_config["calibration"]["outer_alpha"]),
        device=_select_device(str(experiment_config["device"])),
    ).fit(
        _ordinary_normal_windows(suite, "optimization", history, horizon),
        _event_bank(suite, "optimization"),
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
    background = _background_windows(suite, history, horizon)
    background_scores = model.score_windows(background)
    background_fpr = float(np.mean(background_scores["effect_graph_exceedance"]))
    primary_pass = primary["positive_pairs"] >= 14 and primary["median_tail_margin"] > 0.0
    secondary_pass = secondary["positive_pairs"] >= 14 and secondary["median_tail_margin"] > 0.0
    untriggered_pass = untriggered["positive_pairs"] >= 14 and untriggered["median_tail_margin"] > 0.0
    normal_pass = all(report["below_threshold"] >= 14 for report in normal_reports.values())
    isolation = _event_pre_isolation(
        model, _normal_windows(suite, "optimization", history, horizon)[:16], history
    )
    isolation_pass = (
        isolation["max_state_difference"] <= 1e-7
        and isolation["max_counterfactual_baseline_difference"] <= 1e-7
    )
    gates = {
        "event_pre_isolation": {**isolation, "passed": isolation_pass},
        "primary_misrouted_effect_graph": {**primary, "passed": primary_pass},
        "secondary_partial_propagation_effect_graph": {**secondary, "passed": secondary_pass},
        "untriggered_effect_graph": {**untriggered, "passed": untriggered_pass},
        "normal_transition_controls": {"roles": normal_reports, "passed": normal_pass},
        "background_normal": {
            "count": int(len(background)),
            "exceedances": int(np.sum(background_scores["effect_graph_exceedance"])),
            "fpr": background_fpr,
            "passed": background_fpr <= 0.10,
        },
    }
    summary: Dict[str, Any] = {
        "experiment_id": str(experiment_config["experiment_id"]),
        "seed": seed,
        "device": str(model.device),
        "contract_config_hash": _canonical_hash(contract_config),
        "experiment_config": dict(experiment_config),
        "raw_effect_graph_audit": audit,
        "fit": model.fit_metadata_,
        "condition_on_event_pre": bool(model.condition_on_event_pre),
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
    return summary, model.state_dict()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_result(
    output_dir: Path, summary: Mapping[str, Any], checkpoint: Mapping[str, torch.Tensor]
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "summary.json", summary)
    torch.save(
        {
            "contract_config_hash": summary["contract_config_hash"],
            "experiment_config": summary["experiment_config"],
            "fit": summary["fit"],
            "model_state_dict": checkpoint,
        },
        output_dir / "model.pt",
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-config", type=Path, default=DEFAULT_CONTRACT_CONFIG)
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
    experiment = _load_json(arguments.experiment_config)
    if arguments.seed is not None:
        experiment["seed"] = int(arguments.seed)
    if arguments.contract_seed is not None:
        contract["seed"] = int(arguments.contract_seed)
    summary, checkpoint = run_experiment(contract, experiment)
    write_result(arguments.output_dir, summary, checkpoint)
    print(
        f"A3-G3 complete: gates_passed={summary['all_gates_passed']} "
        f"summary={arguments.output_dir / 'summary.json'}"
    )
    return 0 if summary["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
