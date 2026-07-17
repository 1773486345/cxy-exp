"""Read completed Gate v0 shards and finalize only a fully validated study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.decomp_gate_v0_data import ANOMALY_TYPES, SCORING_SEED, TRAIN_SEEDS
from scripts.analysis.decomp_gate_v0_runtime import atomic_write_bytes, atomic_write_csv, atomic_write_json, sha256_file
from scripts.analysis.run_decomp_gate_v0 import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    SCORE_NAMES,
    _bootstrap_from_metrics,
    _gate_decision,
    _not_evaluable_gate,
)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _required_paths(shard: Path) -> List[Path]:
    return [
        shard / "checkpoint.pt",
        shard / "checkpoint.sha256",
        shard / "normalization_stats.json",
        shard / "metrics_by_seed_type.csv",
        shard / "branch_response_matrix.csv",
        shard / "artifact_hashes.json",
    ] + [shard / "scores" / f"{anomaly_type}.npz" for anomaly_type in ANOMALY_TYPES]


def _validate_shard(shard: Path, expected_seed: int) -> Tuple[List[str], pd.DataFrame | None, pd.DataFrame | None]:
    reasons: List[str] = []
    if not shard.is_dir():
        return [f"missing shard directory: {shard}"], None, None
    temporary = list(shard.rglob("*.tmp"))
    if temporary:
        reasons.append(f"shard contains temporary artifacts: {[path.name for path in temporary]}")
    manifest_path = shard / "manifest.json"
    if not manifest_path.is_file():
        return reasons + [f"missing manifest: {shard}"], None, None
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as error:
        return reasons + [f"unreadable shard manifest: {error}"], None, None
    if manifest.get("status") != "completed":
        reasons.append(f"shard status is not completed: {manifest.get('status')}")
    if manifest.get("current_seed") != expected_seed:
        reasons.append(f"shard seed mismatch: expected {expected_seed}, got {manifest.get('current_seed')}")
    for path in _required_paths(shard):
        if not path.is_file():
            reasons.append(f"missing required artifact: {path.name}")
    if reasons:
        return reasons, None, None
    try:
        hashes = _read_json(shard / "artifact_hashes.json")
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable artifact hash index: {error}"], None, None
    expected_hash_paths = {str(path.relative_to(shard)) for path in _required_paths(shard) if path.name != "artifact_hashes.json"}
    if set(hashes) != expected_hash_paths:
        reasons.append("artifact hash index does not cover exactly the required artifacts")
    for relative, expected_hash in hashes.items():
        path = shard / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            reasons.append(f"artifact hash mismatch: {relative}")
    checkpoint_metadata = _read_json(shard / "checkpoint.sha256")
    if checkpoint_metadata.get("sha256") != sha256_file(shard / "checkpoint.pt"):
        reasons.append("checkpoint.sha256 does not match checkpoint.pt")
    try:
        metrics = pd.read_csv(shard / "metrics_by_seed_type.csv")
        branch = pd.read_csv(shard / "branch_response_matrix.csv")
    except Exception as error:
        return reasons + [f"unreadable metrics artifact: {type(error).__name__}: {error}"], None, None
    expected_scores = set(SCORE_NAMES)
    if len(metrics) != len(ANOMALY_TYPES) * len(SCORE_NAMES):
        reasons.append("shard metrics does not contain six anomaly types times six scores")
    if set(metrics.get("seed", pd.Series(dtype=int))) != {expected_seed}:
        reasons.append("shard metrics has an unexpected seed")
    if set(metrics.get("anomaly_type", pd.Series(dtype=str))) != set(ANOMALY_TYPES):
        reasons.append("shard metrics has missing or duplicate anomaly types")
    if set(metrics.get("score", pd.Series(dtype=str))) != expected_scores:
        reasons.append("shard metrics has missing continuous-score metrics")
    expected_metric_units = {(anomaly_type, score) for anomaly_type in ANOMALY_TYPES for score in SCORE_NAMES}
    metric_units = list(zip(metrics.get("anomaly_type", pd.Series(dtype=str)), metrics.get("score", pd.Series(dtype=str))))
    if set(metric_units) != expected_metric_units or len(set(metric_units)) != len(metric_units):
        reasons.append("shard metrics must contain each anomaly-type and score pair exactly once")
    if len(branch) != len(ANOMALY_TYPES) or set(branch.get("anomaly_type", pd.Series(dtype=str))) != set(ANOMALY_TYPES):
        reasons.append("shard branch matrix must contain each anomaly type exactly once")
    for anomaly_type in ANOMALY_TYPES:
        score_path = shard / "scores" / f"{anomaly_type}.npz"
        try:
            with np.load(score_path, allow_pickle=False) as scores:
                required = set(SCORE_NAMES) | {"labels", "time_index"}
                if not required.issubset(scores.files):
                    reasons.append(f"score file misses continuous score: {anomaly_type}")
                elif len({len(scores[name]) for name in required}) != 1:
                    reasons.append(f"score array length mismatch: {anomaly_type}")
        except Exception as error:
            reasons.append(f"unreadable score file {anomaly_type}: {type(error).__name__}: {error}")
    return reasons, metrics, branch


def _shared_metadata_reasons(manifests: List[Dict[str, Any]]) -> List[str]:
    reasons: List[str] = []
    fields = ("git_commit", "dirty_working_tree")
    for field in fields:
        if len({json.dumps(manifest.get(field), sort_keys=True) for manifest in manifests}) != 1:
            reasons.append(f"shards disagree on {field}")
    locks = [manifest.get("config_lock", {}) for manifest in manifests]
    for field, expected in (
        ("catch_config_hash", None),
        ("generator_parameters_hash", None),
        ("moving_average_window", 15),
        ("scoring_seed", SCORING_SEED),
    ):
        values = {json.dumps(lock.get(field), sort_keys=True) for lock in locks}
        if len(values) != 1:
            reasons.append(f"shards disagree on {field}")
        elif expected is not None and locks[0].get(field) != expected:
            reasons.append(f"invalid {field}")
    bootstrap_values = {json.dumps(lock.get("bootstrap"), sort_keys=True) for lock in locks}
    if len(bootstrap_values) != 1 or locks[0].get("bootstrap") != {"seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES}:
        reasons.append("shards disagree on bootstrap seed or samples")
    return reasons


def _write_not_evaluable(study: Path, reasons: List[str]) -> Dict[str, Any]:
    decision = _not_evaluable_gate(reasons)
    atomic_write_json(study / "gate_decision.json", decision)
    return decision


def finalize(study_id: str, attempts: Dict[int, str], output_root: Path) -> Dict[str, Any]:
    study = output_root / study_id
    if (study / "gate_decision.json").exists():
        raise FileExistsError("study already has gate_decision.json; finalization is immutable")
    if not study.is_dir() or not (study / "study_manifest.json").is_file():
        raise FileNotFoundError("study directory or study_manifest.json is missing")
    study_manifest = _read_json(study / "study_manifest.json")
    reasons: List[str] = []
    if set(attempts) != set(TRAIN_SEEDS) or len(set(attempts.values())) != len(TRAIN_SEEDS):
        reasons.append("must specify one distinct attempt for every pre-registered seed")
    manifests: List[Dict[str, Any]] = []
    metrics_frames: List[pd.DataFrame] = []
    branch_frames: List[pd.DataFrame] = []
    for seed in TRAIN_SEEDS:
        attempt = attempts.get(seed)
        if not attempt or Path(attempt).name != attempt:
            reasons.append(f"invalid attempt id for seed {seed}")
            continue
        shard = study / "shards" / attempt
        shard_reasons, metrics, branch = _validate_shard(shard, seed)
        reasons.extend(shard_reasons)
        if metrics is not None and branch is not None:
            metrics_frames.append(metrics)
            branch_frames.append(branch)
            manifests.append(_read_json(shard / "manifest.json"))
    if not reasons and len(manifests) == len(TRAIN_SEEDS):
        reasons.extend(_shared_metadata_reasons(manifests))
        if any(manifest.get("config_lock") != study_manifest.get("config_lock") for manifest in manifests):
            reasons.append("shard config lock differs from the study manifest")
    if reasons:
        return _write_not_evaluable(study, reasons)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    branch = pd.concat(branch_frames, ignore_index=True)
    if len(branch) != 18 or len(metrics) != 18 * len(SCORE_NAMES):
        return _write_not_evaluable(study, ["completed shards did not provide exactly 18 paired units"])
    try:
        bootstrap = _bootstrap_from_metrics(metrics)
    except Exception as error:
        return _write_not_evaluable(study, [f"bootstrap was not executed: {type(error).__name__}: {error}"])
    gate = _gate_decision(branch, bootstrap)
    if gate["decision"] == "GATE_NOT_EVALUABLE":
        return _write_not_evaluable(study, gate["not_evaluable_reasons"])
    atomic_write_csv(study / "metrics_by_seed_type.csv", metrics)
    atomic_write_csv(study / "branch_response_matrix.csv", branch)
    atomic_write_json(study / "bootstrap.json", bootstrap)
    atomic_write_json(study / "gate_decision.json", gate)
    report = "\n".join([
        "# Decomposition Score Gate v0 Final Report", "", f"Decision: `{gate['decision']}`", "",
        "## Conditions", "", *[f"- `{name}`: `{status}`" for name, status in gate["conditions"].items()],
        "", "## Bootstrap", "", "```json", json.dumps(bootstrap, indent=2), "```", "",
    ])
    atomic_write_bytes(study / "gate_report.md", report.encode("utf-8"))
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--seed-20260717-attempt", required=True)
    parser.add_argument("--seed-20260718-attempt", required=True)
    parser.add_argument("--seed-20260719-attempt", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "result" / "decomposition_study_v0")
    args = parser.parse_args()
    decision = finalize(
        args.study_id,
        {20260717: args.seed_20260717_attempt, 20260718: args.seed_20260718_attempt, 20260719: args.seed_20260719_attempt},
        args.output_root,
    )
    print(decision["decision"])


if __name__ == "__main__":
    main()
