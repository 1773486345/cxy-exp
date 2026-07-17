"""Run one immutable pre-registered Gate v0 seed shard."""

from __future__ import annotations

import argparse
import platform
import signal
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.decomp_gate_v0_data import (
    ANOMALY_TYPES,
    SCORING_SEED,
    TRAIN_SEEDS,
    fixed_generator_parameters,
    generate_test_baseline,
    generate_training_series,
    inject_anomaly,
    precompute_anomaly_events,
    split_training_validation,
)
from scripts.analysis.decomp_gate_v0_runtime import (
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npz,
    canonical_json_hash,
    runtime_identity,
    sha256_file,
    stage_state,
    write_progress,
)
from scripts.analysis.run_decomp_gate_v0 import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CATCH_CONFIG,
    SCORE_NAMES,
    _evaluate_scores,
    set_training_seed,
)
from ts_benchmark.baselines.catch.CATCH import CATCH, DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS
from ts_benchmark.baselines.decomp_catch.scoring import CATCHDecompositionScorer


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT.parent, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _read_json(path: Path) -> Dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _valid_component(value: str, name: str) -> str:
    if not value or Path(value).name != value:
        raise ValueError(f"{name} must be one non-empty path component")
    return value


def _config_lock() -> Dict[str, Any]:
    generator = fixed_generator_parameters()
    return {
        "catch_overrides": CATCH_CONFIG,
        "catch_original_defaults": DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS,
        "generator": generator,
        "catch_config_hash": canonical_json_hash(CATCH_CONFIG),
        "generator_parameters_hash": canonical_json_hash(generator),
        "moving_average_window": 15,
        "scoring_seed": SCORING_SEED,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES},
    }


def _create_study_directory(output_root: Path, study_id: str, config_lock: Dict[str, Any]) -> Path:
    study_directory = output_root / study_id
    if not study_directory.exists():
        study_directory.mkdir(parents=True, exist_ok=False)
        (study_directory / "shards").mkdir()
        atomic_write_json(
            study_directory / "study_manifest.json",
            {
                "experiment": "decomp_score_gate_v0",
                "study_id": study_id,
                "status": "created",
                "config_lock": config_lock,
                "git_commit": _git_output("rev-parse", "HEAD"),
                "dirty_working_tree": _git_output("status", "--porcelain=v1"),
            },
        )
    elif not (study_directory / "study_manifest.json").is_file():
        raise RuntimeError("existing study directory lacks study_manifest.json")
    elif _read_json(study_directory / "study_manifest.json").get("config_lock") != config_lock:
        raise RuntimeError("existing study has a different immutable configuration lock")
    return study_directory


def _new_shard_manifest(study_id: str, attempt_id: str, seed: int, device: str, config_lock: Dict[str, Any]) -> Dict[str, Any]:
    identity = runtime_identity()
    return {
        "experiment": "decomp_score_gate_v0",
        "study_id": study_id,
        "attempt_id": attempt_id,
        "status": "created",
        **identity,
        "updated_at_utc": identity["started_at_utc"],
        "current_stage": "preflight",
        "current_seed": seed,
        "current_anomaly_type": None,
        "last_completed_stage": None,
        "exit_reason": None,
        "signal_note": "SIGKILL cannot be caught",
        "git_commit": _git_output("rev-parse", "HEAD"),
        "dirty_working_tree": _git_output("status", "--porcelain=v1"),
        "device": device,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "config_lock": config_lock,
    }


def _save_scores(path: Path, result: Dict[str, Any], labels: np.ndarray) -> None:
    atomic_write_npz(
        path,
        labels=labels,
        time_index=result["time_index"],
        **{name: result[name] for name in SCORE_NAMES},
    )


def _artifact_hashes(shard_directory: Path) -> Dict[str, str]:
    paths = [
        shard_directory / "checkpoint.pt",
        shard_directory / "checkpoint.sha256",
        shard_directory / "normalization_stats.json",
        shard_directory / "metrics_by_seed_type.csv",
        shard_directory / "branch_response_matrix.csv",
    ] + [shard_directory / "scores" / f"{anomaly_type}.npz" for anomaly_type in ANOMALY_TYPES]
    return {str(path.relative_to(shard_directory)): sha256_file(path) for path in paths}


def run_seed_shard(seed: int, study_id: str, attempt_id: str, output_root: Path, device: str) -> Path:
    if seed not in TRAIN_SEEDS:
        raise ValueError(f"seed must be one of {TRAIN_SEEDS}")
    if device != "cpu":
        raise ValueError("Gate v0 shard runner only permits the pre-registered CPU device")
    study_id = _valid_component(study_id, "study_id")
    attempt_id = _valid_component(attempt_id, "attempt_id")
    config_lock = _config_lock()
    study_directory = _create_study_directory(output_root, study_id, config_lock)
    shard_directory = study_directory / "shards" / attempt_id
    if shard_directory.exists():
        raise FileExistsError(f"attempt ID already exists: {shard_directory}")
    shard_directory.mkdir(parents=False, exist_ok=False)
    (shard_directory / "scores").mkdir()

    manifest = _new_shard_manifest(study_id, attempt_id, seed, device, config_lock)
    write_progress(shard_directory, manifest)

    def transition(stage: str, **kwargs: Any) -> None:
        stage_state(manifest, stage, **kwargs)
        write_progress(shard_directory, manifest)

    def handle_signal(signum: int, _frame: Any) -> None:
        transition(
            manifest["current_stage"],
            status="interrupted",
            exit_reason=f"received {signal.Signals(signum).name}",
        )
        raise KeyboardInterrupt

    old_int = signal.signal(signal.SIGINT, handle_signal)
    old_term = signal.signal(signal.SIGTERM, handle_signal)
    try:
        if torch.cuda.is_available():
            transition("preflight", status="failed", exit_reason="CUDA is visible during CPU preflight")
            raise RuntimeError("CPU formal run requires torch.cuda.is_available() is False")
        transition("preflight", status="running", last_completed_stage="preflight")

        transition("data_generated")
        set_training_seed(seed)
        train_series = generate_training_series(seed)
        _, validation = split_training_validation(train_series)
        baseline = generate_test_baseline(seed)
        events = precompute_anomaly_events(seed)
        train_std = train_series.frame.to_numpy().std(axis=0)
        manifest["normal_test_baseline_hash"] = baseline.baseline_hash
        manifest["anomaly_events"] = events
        transition("data_generated", last_completed_stage="data_generated")

        transition("training")
        detector = CATCH(**CATCH_CONFIG)
        detector.device = torch.device(device)
        with (shard_directory / "run.log").open("x", encoding="utf-8") as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                detector.detect_fit(train_series.frame, baseline.frame)

        checkpoint_path = shard_directory / "checkpoint.pt"
        atomic_torch_save(checkpoint_path, detector.early_stopping.check_point)
        checkpoint_metadata = {
            "path": "checkpoint.pt",
            "sha256": sha256_file(checkpoint_path),
            "parameter_count": int(sum(parameter.numel() for parameter in detector.model.parameters())),
        }
        atomic_write_json(shard_directory / "checkpoint.sha256", checkpoint_metadata)
        manifest["checkpoint"] = checkpoint_metadata
        transition("checkpoint_saved", last_completed_stage="checkpoint_saved")

        scorer = CATCHDecompositionScorer(detector)
        normalization_stats = scorer.fit_normalization_stats(
            validation, source_name="validation", scoring_seed=SCORING_SEED
        )
        atomic_write_json(shard_directory / "normalization_stats.json", normalization_stats)
        transition("normalization_saved", last_completed_stage="normalization_saved")

        metrics_rows: List[Dict[str, Any]] = []
        branch_rows: List[Dict[str, Any]] = []
        for anomaly_type in ANOMALY_TYPES:
            transition(f"scoring_{anomaly_type}", anomaly_type=anomaly_type)
            test_frame, labels = inject_anomaly(baseline, train_std, anomaly_type, events, seed)
            result = scorer.score_dataframe(test_frame, normalization_stats=normalization_stats, scoring_seed=SCORING_SEED)
            aligned_labels = labels[: result["scored_length"]]
            if len(aligned_labels) != len(result["time_index"]):
                raise RuntimeError("labels and scorer time index are not aligned")
            _save_scores(shard_directory / "scores" / f"{anomaly_type}.npz", result, aligned_labels)
            rows, branch = _evaluate_scores(seed, anomaly_type, aligned_labels, result, normalization_stats)
            metrics_rows.extend(rows)
            branch_rows.append(branch)
            transition(f"scoring_{anomaly_type}", anomaly_type=anomaly_type, last_completed_stage=f"scoring_{anomaly_type}")

        atomic_write_csv(shard_directory / "metrics_by_seed_type.csv", pd.DataFrame(metrics_rows))
        atomic_write_csv(shard_directory / "branch_response_matrix.csv", pd.DataFrame(branch_rows))
        transition("metrics_saved", last_completed_stage="metrics_saved")
        hashes = _artifact_hashes(shard_directory)
        atomic_write_json(shard_directory / "artifact_hashes.json", hashes)
        manifest["artifact_hashes"] = hashes
        transition("completed", status="completed", anomaly_type=None, last_completed_stage="completed")
        return shard_directory
    except KeyboardInterrupt:
        if manifest["status"] != "interrupted":
            transition(manifest["current_stage"], status="interrupted", exit_reason="KeyboardInterrupt")
        raise
    except Exception as error:
        if manifest["status"] != "failed":
            transition(manifest["current_stage"], status="failed", exit_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "result" / "decomposition_study_v0")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    result = run_seed_shard(args.seed, args.study_id, args.attempt_id, args.output_root, args.device)
    print(result)


if __name__ == "__main__":
    main()
