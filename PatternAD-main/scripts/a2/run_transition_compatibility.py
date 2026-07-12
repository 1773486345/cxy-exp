#!/usr/bin/env python3
"""Run A2-M1's frozen conditional-mixture trajectory compatibility experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a2.audit_transition_contract import audit_suite  # noqa: E402
from scripts.a2.generate_transition_contract import (  # noqa: E402
    DEFAULT_CONFIG as DEFAULT_CONTRACT_CONFIG,
    _load_json,
    generate_suite,
)
from ts_benchmark.baselines.A2TransitionCompatibility import (  # noqa: E402
    A2TransitionCompatibility,
)


DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "config" / "a2" / "trajectory_gru_v1.json"
NORMAL_ROLES: Sequence[str] = (
    "normal_scheduled_transition",
    "normal_coordinated_transition",
    "no_event_normal_control",
)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot write an empty A2 score table.")
    fields = list(materialized[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


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


def _bank_windows(suite: Mapping[str, Any], split_name: str) -> np.ndarray:
    bank = list(suite["normal_transition_banks"][split_name])
    return np.stack([episode["values"] for episode in bank], axis=0).astype(np.float32)


def _normal_continuation_windows(
    suite: Mapping[str, Any], split_name: str, history: int, horizon: int
) -> np.ndarray:
    """Return every ordinary-normal window wholly contained in one frozen split."""
    split_start, split_end = suite["normal_split_ranges"][split_name]
    values = np.asarray(suite["train_values"], dtype=np.float32)
    starts = range(int(split_start) + history, int(split_end) - horizon + 1)
    windows = [values[start - history : start + horizon] for start in starts]
    if not windows:
        raise ValueError(f"A2 {split_name} split has no valid normal continuation window.")
    return np.stack(windows, axis=0)


def _normal_split_windows(
    suite: Mapping[str, Any], split_name: str, history: int, horizon: int
) -> np.ndarray:
    """Combine ordinary normal continuations with the split's normal transition bank."""
    return np.concatenate(
        (
            _normal_continuation_windows(suite, split_name, history, horizon),
            _bank_windows(suite, split_name),
        ),
        axis=0,
    )


def _background_windows(suite: Mapping[str, Any], history: int, horizon: int) -> np.ndarray:
    values = np.asarray(suite["background_values"], dtype=np.float32)
    return np.stack(
        [values[start - history : start + horizon] for start in range(history, len(values) - horizon + 1)],
        axis=0,
    )


def _event_pre_volatility(windows: np.ndarray, history: int) -> np.ndarray:
    event_pre = np.asarray(windows[:, :history], dtype=np.float64)
    return np.sqrt(np.mean(np.square(np.diff(event_pre, axis=1)), axis=(1, 2)) + 1e-12)


def _finite_sample_upper_threshold(scores: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    rank = int(np.ceil((len(ordered) + 1) * (1.0 - alpha))) - 1
    return float(ordered[min(max(rank, 0), len(ordered) - 1)])


def _episode_score_rows(
    suite: Mapping[str, Any], model: A2TransitionCompatibility
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    raw_score_key = str(getattr(model, "raw_score_key", "trajectory_nll"))
    raw_score_name = str(getattr(model, "raw_score_name", raw_score_key))
    for episode in suite["episodes"]:
        score = model.score_windows(np.asarray(episode["values"], dtype=np.float32)[None, :, :])
        rows.append(
            {
                "source_id": str(episode["source_id"]),
                "role": str(episode["role"]),
                "regime": int(episode["regime"]),
                "primary_pair_id": str(episode["primary_pair_id"] or ""),
                "coordination_pair_id": str(episode["coordination_pair_id"] or ""),
                "raw_score_name": raw_score_name,
                "raw_compatibility_score": float(score[raw_score_key][0]),
                "compatibility_tail": float(score["compatibility_tail"][0]),
                "reliability_bin": int(score["reliability_bin"][0]),
                "outer_threshold": float(score["outer_threshold"][0]),
                "outer_exceedance": int(score["outer_exceedance"][0]),
            }
        )
    return rows


def _role_report(rows: Sequence[Mapping[str, Any]], role: str) -> Dict[str, Any]:
    selected = [row for row in rows if row["role"] == role]
    exceedances = sum(int(row["outer_exceedance"]) for row in selected)
    return {
        "count": len(selected),
        "exceedances": exceedances,
        "below_threshold": len(selected) - exceedances,
        "fpr": exceedances / max(len(selected), 1),
    }


def _paired_gates(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_source: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_source[str(row["source_id"])][str(row["role"])] = row
    primary_deltas = []
    coordination_deltas = []
    for roles in by_source.values():
        primary_deltas.append(
            float(
                roles["incompatible_timing_transition"]["compatibility_tail"]
                - roles["normal_scheduled_transition"]["compatibility_tail"]
            )
        )
        coordination_deltas.append(
            float(
                roles["unsupported_transition"]["compatibility_tail"]
                - roles["normal_coordinated_transition"]["compatibility_tail"]
            )
        )
    return {
        "primary": {
            "positive_pairs": int(sum(delta > 0.0 for delta in primary_deltas)),
            "pair_count": len(primary_deltas),
            "median_tail_margin": float(np.median(primary_deltas)),
            "tail_margins": primary_deltas,
        },
        "coordination": {
            "positive_pairs": int(sum(delta > 0.0 for delta in coordination_deltas)),
            "pair_count": len(coordination_deltas),
            "median_tail_margin": float(np.median(coordination_deltas)),
            "tail_margins": coordination_deltas,
        },
    }


def _event_pre_isolation(
    model: A2TransitionCompatibility, windows: np.ndarray, history: int
) -> float:
    changed = np.asarray(windows, dtype=np.float32).copy()
    changed[:, history:] = -3.0 * changed[:, history:] + 0.25
    first = model.event_pre_state(windows)
    second = model.event_pre_state(changed)
    return float(np.max(np.abs(first - second)))


def run_experiment(
    contract_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    model_factory: Callable[..., Any] = A2TransitionCompatibility,
) -> tuple[Dict[str, Any], Mapping[str, torch.Tensor]]:
    if int(experiment_config.get("schema_version", 0)) != 1:
        raise ValueError("A2 experiment config must use schema_version=1.")
    audit = audit_suite(contract_config, generate_suite(contract_config))
    if not audit["passed"]:
        raise RuntimeError(f"A2 contract audit failed: {audit['violations']}")
    suite = generate_suite(contract_config)
    seed = int(experiment_config["seed"])
    _set_seed(seed)
    history = int(contract_config["history_length"])
    horizon = int(contract_config["horizon_length"])
    dimensions = int(contract_config["dimensions"])
    model_config = experiment_config["model"]
    calibration_config = experiment_config["calibration"]
    common_model_keys = {
        "hidden_size",
        "mixture_components",
        "condition_on_event_pre",
        "learning_rate",
        "epochs",
        "patience",
        "batch_size",
    }
    extra_model_kwargs = {
        key: value for key, value in model_config.items() if key not in common_model_keys
    }
    model = model_factory(
        dimensions=dimensions,
        history_length=history,
        horizon_length=horizon,
        hidden_size=int(model_config["hidden_size"]),
        condition_on_event_pre=bool(model_config.get("condition_on_event_pre", True)),
        learning_rate=float(model_config["learning_rate"]),
        epochs=int(model_config["epochs"]),
        patience=int(model_config["patience"]),
        batch_size=int(model_config["batch_size"]),
        outer_alpha=float(calibration_config["outer_alpha"]),
        reliability_bin_count=int(calibration_config["reliability_bin_count"]),
        device=_select_device(str(experiment_config["device"])),
        **(
            {"mixture_components": int(model_config["mixture_components"])}
            if "mixture_components" in model_config
            else {}
        ),
        **extra_model_kwargs,
    ).fit(
        _normal_split_windows(suite, "optimization", history, horizon),
        _normal_split_windows(suite, "validation", history, horizon),
        _normal_split_windows(suite, "reference", history, horizon),
        _normal_split_windows(suite, "outer_calibration", history, horizon),
        seed=seed,
    )
    rows = _episode_score_rows(suite, model)
    paired = _paired_gates(rows)
    normal_reports = {role: _role_report(rows, role) for role in NORMAL_ROLES}
    background = _background_windows(suite, history, horizon)
    background_scores = model.score_windows(background)
    background_reliability_bins = background_scores["reliability_bin"]
    stratum_reports = {}
    for bin_index in range(int(calibration_config["reliability_bin_count"])):
        mask = background_reliability_bins == bin_index
        values = background_scores["outer_exceedance"][mask]
        stratum_reports[f"reliability_bin_{bin_index}"] = {
            "count": int(len(values)),
            "exceedances": int(np.sum(values)),
            "fpr": float(np.mean(values)) if len(values) else None,
        }
    finite_fprs = [report["fpr"] for report in stratum_reports.values() if report["fpr"] is not None]
    background_fpr_gap = float(max(finite_fprs) - min(finite_fprs))
    predicted = model.predict_mean_trajectory(background)
    observed = background[:, history:]
    last_value_baseline = np.repeat(background[:, history - 1 : history], horizon, axis=1)
    model_mae = float(np.mean(np.abs(predicted - observed)))
    baseline_mae = float(np.mean(np.abs(last_value_baseline - observed)))
    normal_bank = _normal_split_windows(suite, "optimization", history, horizon)
    isolation_difference = _event_pre_isolation(model, normal_bank[: min(16, len(normal_bank))], history)
    primary_pass = (
        paired["primary"]["positive_pairs"] >= 14
        and paired["primary"]["median_tail_margin"] > 0.0
    )
    coordination_pass = (
        paired["coordination"]["positive_pairs"] >= 14
        and paired["coordination"]["median_tail_margin"] > 0.0
    )
    normal_pass = all(report["below_threshold"] >= 14 for report in normal_reports.values())
    background_pass = all(fpr <= 0.10 for fpr in finite_fprs) and background_fpr_gap <= 0.05
    normal_skill_pass = model_mae < baseline_mae
    gates = {
        "event_pre_isolation": {
            "max_state_difference": isolation_difference,
            "passed": isolation_difference <= 1e-7,
        },
        "primary_ordering": {**paired["primary"], "passed": primary_pass},
        "secondary_coordination_ordering": {**paired["coordination"], "passed": coordination_pass},
        "normal_transition_controls": {"roles": normal_reports, "passed": normal_pass},
        "background_normal": {
            "reliability_boundaries": model.fit_metadata_["reliability_boundaries"],
            "strata": stratum_reports,
            "fpr_gap": background_fpr_gap,
            "passed": background_pass,
        },
        "normal_skill": {
            "mixture_mean_mae": model_mae,
            "last_value_mae": baseline_mae,
            "relative_improvement": (baseline_mae - model_mae) / baseline_mae,
            "passed": normal_skill_pass,
        },
    }
    summary: Dict[str, Any] = {
        "experiment_id": str(experiment_config["experiment_id"]),
        "seed": seed,
        "device": str(model.device),
        "contract_config_hash": _canonical_hash(contract_config),
        "experiment_config": dict(experiment_config),
        "contract_audit": audit,
        "fit": model.fit_metadata_,
        "condition_on_event_pre": bool(model.condition_on_event_pre),
        "raw_score_name": str(getattr(model, "raw_score_name", "trajectory_nll")),
        "normal_split_window_counts": {
            split_name: int(
                len(_normal_split_windows(suite, split_name, history, horizon))
            )
            for split_name in ("optimization", "validation", "reference", "outer_calibration")
        },
        "gates": gates,
        "all_gates_passed": all(gate["passed"] for gate in gates.values()),
        "episode_scores": rows,
    }
    return summary, model.state_dict()


def write_result(
    output_dir: Path, summary: Mapping[str, Any], checkpoint: Mapping[str, torch.Tensor]
) -> None:
    """Persist one complete A2 run without overwriting another seed's result."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(output_dir / "summary.json", summary)
    _write_csv(output_dir / "episode_scores.csv", summary["episode_scores"])
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
    parser.add_argument("--outer-alpha", type=float, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    arguments = parser.parse_args(argv)
    if arguments.torch_threads < 1:
        raise ValueError("--torch-threads must be positive.")
    torch.set_num_threads(arguments.torch_threads)
    contract_config = _load_json(arguments.contract_config)
    experiment_config = _load_json(arguments.experiment_config)
    if arguments.contract_seed is not None:
        contract_config["seed"] = int(arguments.contract_seed)
    if arguments.seed is not None:
        experiment_config["seed"] = int(arguments.seed)
    if arguments.outer_alpha is not None:
        experiment_config["calibration"]["outer_alpha"] = float(arguments.outer_alpha)
    summary, checkpoint = run_experiment(contract_config, experiment_config)
    write_result(arguments.output_dir, summary, checkpoint)
    print(
        f"A2-M1 complete: gates_passed={summary['all_gates_passed']} "
        f"summary={arguments.output_dir / 'summary.json'}"
    )
    return 0 if summary["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
