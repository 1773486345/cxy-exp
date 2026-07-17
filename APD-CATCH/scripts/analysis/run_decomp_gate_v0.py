"""Run the pre-registered fixed reconstruction-error decomposition gate v0."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
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
    parser.parse_args()
    raise SystemExit(
        "The serial Gate v0 runner is retired. Run one immutable seed shard with "
        "scripts/analysis/run_decomp_gate_v0_seed.py, then use "
        "scripts/analysis/finalize_decomp_gate_v0.py."
    )


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


def _not_evaluable_gate(reasons: Sequence[str]) -> Dict[str, Any]:
    return {
        "decision": "GATE_NOT_EVALUABLE",
        "conditions": {
            "condition_1_component_ranking_divergence": "NOT_EVALUABLE",
            "condition_2_different_branch_response": "NOT_EVALUABLE",
            "condition_3_component_discovers_original_low_rank": "NOT_EVALUABLE",
            "condition_4_equal_fusion_noninferior": "NOT_EVALUABLE",
            "condition_5_failure_scope_applied": "NOT_EVALUABLE",
        },
        "not_evaluable_reasons": list(reasons),
    }


def _gate_decision(branch: pd.DataFrame, bootstrap: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    required_units = {(seed, anomaly_type) for seed in TRAIN_SEEDS for anomaly_type in ANOMALY_TYPES}
    required_columns = {
        "seed", "anomaly_type", "anomaly_points", "slow_fast_anomaly_spearman",
        "slow_auc_pr", "fast_auc_pr", "original_top_k_out_component_top_k_in",
    }
    if not required_columns.issubset(branch.columns):
        return _not_evaluable_gate(["branch response matrix is missing required columns"])
    units = list(zip(branch["seed"], branch["anomaly_type"]))
    if len(units) != 18 or set(units) != required_units or len(set(units)) != len(units):
        return _not_evaluable_gate(["expected exactly 18 unique pre-registered seed-category units"])
    if bootstrap is None:
        return _not_evaluable_gate(["bootstrap was not executed for a complete unit set"])
    if (
        bootstrap.get("unit_count") != 18
        or bootstrap.get("seed") != BOOTSTRAP_SEED
        or bootstrap.get("resamples") != BOOTSTRAP_SAMPLES
        or "one_sided_95_lower_bound" not in bootstrap
    ):
        return _not_evaluable_gate(["bootstrap metadata is incomplete or inconsistent"])

    evaluable = branch[branch["anomaly_points"] >= 10]
    if len(evaluable) != 18:
        return _not_evaluable_gate(["one or more pre-registered units has fewer than 10 anomaly points"])

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


if __name__ == "__main__":
    main()
