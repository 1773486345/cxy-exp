"""Run the pre-registered fixed reconstruction-error decomposition gate v0."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.decomp_gate_v0_data import (
    ANOMALY_TYPES,
    SCORING_SEED,
    TEST_LENGTH,
    TRAIN_SEEDS,
    fixed_generator_parameters,
    generate_test_baseline,
    generate_training_series,
    inject_anomaly,
    precompute_anomaly_events,
    split_training_validation,
)
from ts_benchmark.baselines.catch.CATCH import CATCH
from ts_benchmark.baselines.catch.CATCH import DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS
from ts_benchmark.baselines.decomp_catch.scoring import CATCHDecompositionScorer


CATCH_CONFIG = {
    "seq_len": 192,
    "patch_size": 16,
    "patch_stride": 8,
    "inference_patch_size": 32,
    "inference_patch_stride": 1,
    "num_epochs": 3,
    "batch_size": 128,
}
SCORE_NAMES = (
    "original_score",
    "time_score",
    "frequency_score",
    "slow_score",
    "fast_score",
    "fusion_score",
)
BOOTSTRAP_SEED = 20260717
BOOTSTRAP_SAMPLES = 10_000


def create_run_directory(root: Path, run_id: str) -> Path:
    """Create one immutable run directory, refusing to overwrite any existing ID."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single non-empty path component")
    run_directory = root / run_id
    if run_directory.exists():
        raise FileExistsError(f"run ID already exists: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def bootstrap_fusion_delta(deltas: Sequence[float], seed: int, samples: int = BOOTSTRAP_SAMPLES) -> Dict[str, Any]:
    """Bootstrap paired seed-category deltas, never individual time points."""
    values = np.asarray(deltas, dtype=np.float64)
    if len(values) != 18:
        raise ValueError("bootstrap requires exactly 18 seed-category paired units")
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    resampled_means = values[indices].mean(axis=1)
    return {
        "unit": "(seed, anomaly_type)",
        "unit_count": int(len(values)),
        "resamples": int(samples),
        "seed": int(seed),
        "mean_delta": float(values.mean()),
        "one_sided_95_lower_bound": float(np.quantile(resampled_means, 0.05)),
    }


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "result" / "decomposition_study_v0",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()

    run_id = args.run_id or _default_run_id()
    run_directory = create_run_directory(args.output_root, run_id)
    for directory in (run_directory / "checkpoints", run_directory / "scores"):
        directory.mkdir()

    # These inputs are frozen before any model training or scoring begins.
    baselines = {seed: generate_test_baseline(seed) for seed in TRAIN_SEEDS}
    events_by_seed = {seed: precompute_anomaly_events(seed) for seed in TRAIN_SEEDS}
    manifest = _initial_manifest(
        run_id,
        args.device,
        normal_test_baseline_hashes={str(seed): baseline.baseline_hash for seed, baseline in baselines.items()},
        anomaly_events={str(seed): events for seed, events in events_by_seed.items()},
    )
    _write_json(run_directory / "manifest.json", manifest)
    _write_json(
        run_directory / "config.json",
        {
            "catch_overrides": CATCH_CONFIG,
            "catch_original_defaults": DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS,
            "generator": fixed_generator_parameters(),
            "moving_average_window": 15,
            "bootstrap": {"seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES},
        },
    )
    log_path = run_directory / "run.log"
    metrics_rows: List[Dict[str, Any]] = []
    branch_rows: List[Dict[str, Any]] = []
    checkpoint_records: List[Dict[str, Any]] = []
    normalization_records: Dict[str, Dict[str, Any]] = {}

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            for seed in TRAIN_SEEDS:
                _log(log_file, f"seed={seed}: generating fixed normal data")
                set_training_seed(seed)
                train_series = generate_training_series(seed)
                train_core, validation = split_training_validation(train_series)
                baseline = baselines[seed]
                events = events_by_seed[seed]
                train_std = train_series.frame.to_numpy().std(axis=0)

                _log(log_file, f"seed={seed}: one original CATCH training run")
                detector = CATCH(**CATCH_CONFIG)
                detector.device = torch.device(args.device)
                with redirect_stdout(log_file), redirect_stderr(log_file):
                    detector.detect_fit(train_series.frame, baseline.frame)

                checkpoint = detector.early_stopping.check_point
                checkpoint_path = run_directory / "checkpoints" / f"seed_{seed}" / "checkpoint.pt"
                checkpoint_path.parent.mkdir()
                torch.save(checkpoint, checkpoint_path)
                checkpoint_hash = _sha256_file(checkpoint_path)
                checkpoint_records.append(
                    {
                        "seed": seed,
                        "path": str(checkpoint_path.relative_to(run_directory)),
                        "sha256": checkpoint_hash,
                        "parameter_count": int(sum(parameter.numel() for parameter in detector.model.parameters())),
                    }
                )

                scorer = CATCHDecompositionScorer(detector)
                normalization_stats = scorer.fit_normalization_stats(
                    validation, source_name="validation", scoring_seed=SCORING_SEED
                )
                _write_json(run_directory / f"normalization_stats_seed_{seed}.json", normalization_stats)
                normalization_records[str(seed)] = normalization_stats

                for anomaly_type in ANOMALY_TYPES:
                    test_frame, labels = inject_anomaly(
                        baseline, train_std, anomaly_type, events, seed
                    )
                    result = scorer.score_dataframe(
                        test_frame,
                        normalization_stats=normalization_stats,
                        scoring_seed=SCORING_SEED,
                    )
                    aligned_labels = labels[: result["scored_length"]]
                    if len(aligned_labels) != len(result["time_index"]):
                        raise RuntimeError("labels and scorer time index are not aligned")
                    _save_scores(
                        run_directory / "scores" / f"seed_{seed}_{anomaly_type}.npz",
                        result,
                        aligned_labels,
                    )
                    rows, branch = _evaluate_scores(
                        seed, anomaly_type, aligned_labels, result, normalization_stats
                    )
                    metrics_rows.extend(rows)
                    branch_rows.append(branch)

        metrics = pd.DataFrame(metrics_rows)
        branch_matrix = pd.DataFrame(branch_rows)
        metrics.to_csv(run_directory / "metrics_by_seed_type.csv", index=False)
        branch_matrix.to_csv(run_directory / "branch_response_matrix.csv", index=False)
        bootstrap = _bootstrap_from_metrics(metrics)
        _write_json(run_directory / "bootstrap.json", bootstrap)
        gate = _gate_decision(branch_matrix, bootstrap)
        _write_json(run_directory / "gate_decision.json", gate)
        _write_json(run_directory / "checkpoint.sha256", {"checkpoints": checkpoint_records})
        _write_json(run_directory / "normalization_stats.json", {"by_seed": normalization_records})
        manifest["status"] = "completed"
        manifest["checkpoints"] = checkpoint_records
        manifest["gate_decision"] = gate["decision"]
        _write_json(run_directory / "manifest.json", manifest)
        _write_run_report(run_directory, metrics, branch_matrix, bootstrap, gate)
        print(f"run_id: {run_id}")
        print(f"result_dir: {run_directory}")
        print(f"gate_decision: {gate['decision']}")
    except Exception:
        manifest["status"] = "failed"
        _write_json(run_directory / "manifest.json", manifest)
        raise


def _evaluate_scores(
    seed: int,
    anomaly_type: str,
    labels: np.ndarray,
    result: Dict[str, Any],
    normalization_stats: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for score_name in SCORE_NAMES:
        rows.append(
            {
                "seed": seed,
                "anomaly_type": anomaly_type,
                "score": score_name,
                "auc_pr": float(average_precision_score(labels, result[score_name])),
                "anomaly_points": int(labels.sum()),
                "scored_length": result["scored_length"],
            }
        )

    anomaly_mask = labels == 1
    normal_mask = ~anomaly_mask
    anomaly_spearman = _spearman(result["slow_score"][anomaly_mask], result["fast_score"][anomaly_mask])
    normal_spearman = _spearman(result["slow_score"][normal_mask], result["fast_score"][normal_mask])
    top_k = int(labels.sum())
    original_top = _top_k_mask(result["original_score"], top_k)
    component_top = _top_k_mask(result["slow_score"], top_k) | _top_k_mask(result["fast_score"], top_k)
    component_discoveries = int(np.sum(anomaly_mask & ~original_top & component_top))
    slow_z = (result["slow_score"] - normalization_stats["slow_location"]) / (
        normalization_stats["slow_scale"] + 1e-8
    )
    fast_z = (result["fast_score"] - normalization_stats["fast_location"]) / (
        normalization_stats["fast_scale"] + 1e-8
    )
    lookup = {row["score"]: row["auc_pr"] for row in rows}
    return rows, {
        "seed": seed,
        "anomaly_type": anomaly_type,
        "anomaly_points": top_k,
        "slow_fast_anomaly_spearman": anomaly_spearman,
        "slow_fast_normal_spearman": normal_spearman,
        "original_top_k_out_component_top_k_in": component_discoveries,
        "slow_anomaly_mean_z": float(np.mean(slow_z[anomaly_mask])),
        "fast_anomaly_mean_z": float(np.mean(fast_z[anomaly_mask])),
        "slow_auc_pr": lookup["slow_score"],
        "fast_auc_pr": lookup["fast_score"],
        "original_auc_pr": lookup["original_score"],
        "fusion_auc_pr": lookup["fusion_score"],
        "time_auc_pr": lookup["time_score"],
        "frequency_auc_pr": lookup["frequency_score"],
        "time_original_mean_absolute_difference": float(
            np.mean(np.abs(result["time_score"] - result["original_score"]))
        ),
        "baseline_hash": result.get("baseline_hash", "recorded_in_manifest"),
    }


def _bootstrap_from_metrics(metrics: pd.DataFrame) -> Dict[str, Any]:
    pivot = metrics.pivot(index=["seed", "anomaly_type"], columns="score", values="auc_pr")
    deltas = (pivot["fusion_score"] - pivot["original_score"]).to_numpy()
    return bootstrap_fusion_delta(deltas, seed=BOOTSTRAP_SEED)


def _gate_decision(branch: pd.DataFrame, bootstrap: Dict[str, Any]) -> Dict[str, Any]:
    evaluable = branch[branch["anomaly_points"] >= 10]
    if evaluable.empty:
        statuses = {
            "condition_1_component_ranking_divergence": "NOT_EVALUABLE",
            "condition_2_different_branch_response": "NOT_EVALUABLE",
            "condition_3_component_discovers_original_low_rank": "NOT_EVALUABLE",
            "condition_4_equal_fusion_noninferior": "NOT_EVALUABLE",
            "condition_5_failure_scope_applied": "NOT_EVALUABLE",
        }
        return {"decision": "GATE_FAILED", "conditions": statuses, "bootstrap": bootstrap}

    condition_1 = "FAIL" if bool((evaluable["slow_fast_anomaly_spearman"] >= 0.98).all()) else "PASS"
    slow_wins = bool(((evaluable["slow_auc_pr"] - evaluable["fast_auc_pr"]) >= 0.01).any())
    fast_wins = bool(((evaluable["fast_auc_pr"] - evaluable["slow_auc_pr"]) >= 0.01).any())
    condition_2 = "PASS" if slow_wins and fast_wins else "FAIL"
    condition_3 = "PASS" if bool((evaluable["original_top_k_out_component_top_k_in"] > 0).any()) else "FAIL"
    condition_4 = "PASS" if bootstrap["one_sided_95_lower_bound"] >= -0.01 else "FAIL"
    primary = (condition_1, condition_2, condition_3, condition_4)
    decision = "GATE_PASSED" if all(status == "PASS" for status in primary) else "GATE_FAILED"
    return {
        "decision": decision,
        "conditions": {
            "condition_1_component_ranking_divergence": condition_1,
            "condition_2_different_branch_response": condition_2,
            "condition_3_component_discovers_original_low_rank": condition_3,
            "condition_4_equal_fusion_noninferior": condition_4,
            "condition_5_failure_scope_applied": "NOT_EVALUABLE" if decision == "GATE_PASSED" else "PASS",
        },
        "bootstrap": bootstrap,
    }


def _initial_manifest(
    run_id: str,
    device: str,
    normal_test_baseline_hashes: Dict[str, str],
    anomaly_events: Dict[str, Dict[str, object]],
) -> Dict[str, Any]:
    return {
        "experiment": "decomp_score_gate_v0",
        "run_id": run_id,
        "status": "running",
        "git_commit": _git_output("rev-parse", "HEAD"),
        "dirty_working_tree": _git_output("status", "--porcelain=v1"),
        "train_seeds": list(TRAIN_SEEDS),
        "scoring_seed": SCORING_SEED,
        "catch_config": CATCH_CONFIG,
        "moving_average_window": 15,
        "device": device,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "generator_parameters": fixed_generator_parameters(),
        "normal_test_baseline_hashes": normal_test_baseline_hashes,
        "anomaly_events": anomaly_events,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _save_scores(path: Path, result: Dict[str, Any], labels: np.ndarray) -> None:
    np.savez_compressed(
        path,
        labels=labels,
        time_index=result["time_index"],
        **{name: result[name] for name in SCORE_NAMES},
    )


def _write_run_report(
    run_directory: Path,
    metrics: pd.DataFrame,
    branch: pd.DataFrame,
    bootstrap: Dict[str, Any],
    gate: Dict[str, Any],
) -> None:
    lines = [
        "# Decomposition Score Gate v0 Run Report",
        "",
        f"Run: `{run_directory.name}`",
        f"Decision: `{gate['decision']}`",
        "",
        "## Conditions",
        "",
    ]
    lines.extend(f"- `{name}`: `{status}`" for name, status in gate["conditions"].items())
    lines.extend(["", "## Bootstrap", "", "```json", json.dumps(bootstrap, indent=2), "```", "", "## Metrics", "", "```csv"])
    lines.append(metrics.to_csv(index=False).rstrip())
    lines.extend(["```", "", "## Branch Response Matrix", "", "```csv", branch.to_csv(index=False).rstrip(), "```", ""])
    (run_directory / "gate_report.md").write_text("\n".join(lines), encoding="utf-8")


def _spearman(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    return float(pd.Series(left).corr(pd.Series(right), method="spearman"))


def _top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    if k <= 0:
        return mask
    indices = np.argpartition(-scores, k - 1)[:k]
    mask[indices] = True
    return mask


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT.parent, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value)!r}")


def _log(handle: Any, message: str) -> None:
    handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    handle.flush()


def _default_run_id() -> str:
    return "decomp_gate_v0_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


if __name__ == "__main__":
    main()
