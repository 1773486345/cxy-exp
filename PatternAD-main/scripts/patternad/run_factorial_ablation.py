#!/usr/bin/env python3
"""Generate and execute PatternAD factorial runs from one manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "patternad" / "factorial_ablation.json"
DEFAULT_DATASETS = REPO_ROOT / "config" / "patternad" / "dataset_groups.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "patternad_strict"
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_component(value: str, name: str) -> str:
    if not SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"Invalid {name} {value!r}; use only letters, digits, '.', '_' and '-'."
        )
    return value


def _require_keys(mapping: Mapping[str, Any], keys: Iterable[str], owner: str) -> None:
    missing = sorted(set(keys) - set(mapping))
    if missing:
        raise ValueError(f"{owner} is missing required keys: {', '.join(missing)}")


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    _require_keys(
        manifest,
        {
            "benchmark",
            "default_group",
            "development_seeds",
            "confirmation_seeds",
            "score_metrics",
            "shared_hyperparameters",
            "variants",
            "comparisons",
        },
        "factorial manifest",
    )
    variants = manifest["variants"]
    if not isinstance(variants, dict):
        raise ValueError("manifest.variants must be an object.")
    expected = {"A00", "A10", "A01", "A11", "B00", "B11"}
    if set(variants) != expected:
        raise ValueError(
            "The minimal factorial manifest must define exactly "
            f"{sorted(expected)}; got {sorted(variants)}."
        )

    benchmark = manifest["benchmark"]
    _require_keys(
        benchmark,
        {"max_scale_boundary_fraction", "require_model_diagnostics"},
        "manifest.benchmark",
    )
    maximum_boundary_fraction = float(benchmark["max_scale_boundary_fraction"])
    if not 0 <= maximum_boundary_fraction <= 1:
        raise ValueError("max_scale_boundary_fraction must be between 0 and 1.")
    if benchmark["require_model_diagnostics"] is not True:
        raise ValueError("Strict PatternAD runs must require model diagnostics.")

    expected_factors = {
        "A00": ("C0", "D0", "M1"),
        "A10": ("C1", "D0", "M1"),
        "A01": ("C0", "D1", "M1"),
        "A11": ("C1", "D1", "M1"),
        "B00": ("C0", "D0", "M0"),
        "B11": ("C1", "D1", "M0"),
    }
    for variant_id, expected_factor_tuple in expected_factors.items():
        variant = variants[variant_id]
        _require_keys(variant, {"factors", "hyperparameters"}, variant_id)
        factors = variant["factors"]
        actual = (
            factors.get("context"),
            factors.get("distribution"),
            factors.get("mask"),
        )
        if actual != expected_factor_tuple:
            raise ValueError(
                f"{variant_id} factors must be {expected_factor_tuple}, got {actual}."
            )
        hyper = variant["hyperparameters"]
        if factors["distribution"] == "D1" and hyper.get(
            "reconstruction_distribution"
        ) != "gaussian":
            raise ValueError(f"{variant_id} is D1 and must use Gaussian first.")
        expected_conditional = factors["mask"] == "M1"
        if hyper.get("use_conditional_scoring") is not expected_conditional:
            raise ValueError(
                f"{variant_id} mask factor and use_conditional_scoring disagree."
            )

    score_ratio = float(manifest["shared_hyperparameters"].get("score_mask_ratio", 0))
    if not 0 < score_ratio <= 0.5:
        raise ValueError("score_mask_ratio must define at least two complementary passes.")

    comparison_names = set()
    for comparison in manifest["comparisons"]:
        _require_keys(comparison, {"name", "lhs", "rhs"}, "comparison")
        if comparison["name"] in comparison_names:
            raise ValueError(f"Duplicate comparison name: {comparison['name']}")
        comparison_names.add(comparison["name"])
        if comparison["lhs"] not in variants or comparison["rhs"] not in variants:
            raise ValueError(f"Unknown variant in comparison {comparison['name']!r}.")


def _validate_dataset_config(config: Mapping[str, Any]) -> None:
    _require_keys(config, {"datasets", "groups", "resource_profiles"}, "dataset config")
    datasets = config["datasets"]
    groups = config["groups"]
    for required_group in ("smoke", "motivation", "robustness", "confirmation"):
        if required_group not in groups:
            raise ValueError(f"Missing dataset group {required_group!r}.")
    if not groups["confirmation"].get("locked", False):
        raise ValueError("The confirmation group must be marked locked=true.")

    for dataset_id, dataset in datasets.items():
        _safe_component(dataset_id, "dataset id")
        _require_keys(
            dataset,
            {"data_name", "text_name", "family", "entity", "resource_profile"},
            f"dataset {dataset_id}",
        )
        if dataset["resource_profile"] not in config["resource_profiles"]:
            raise ValueError(
                f"Dataset {dataset_id} uses unknown resource profile "
                f"{dataset['resource_profile']!r}."
            )
    for group_id, group in groups.items():
        _safe_component(group_id, "group id")
        _require_keys(group, {"locked", "default_variants", "datasets"}, f"group {group_id}")
        unknown = sorted(set(group["datasets"]) - set(datasets))
        if unknown:
            raise ValueError(f"Group {group_id} has unknown datasets: {unknown}")

    unlocked = {
        dataset_id
        for group in groups.values()
        if not group.get("locked", False)
        for dataset_id in group["datasets"]
    }
    overlap = unlocked.intersection(groups["confirmation"]["datasets"])
    if overlap:
        raise ValueError(
            "Locked confirmation datasets also occur in development groups: "
            f"{sorted(overlap)}"
        )


def _git_commit() -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> Optional[bool]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "."],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _critical_source_hashes(benchmark_config: Path) -> Dict[str, str]:
    paths = [
        REPO_ROOT / "scripts" / "run_benchmark.py",
        REPO_ROOT / "scripts" / "patternad" / "run_factorial_ablation.py",
        REPO_ROOT / "scripts" / "patternad" / "summarize_factorial.py",
        REPO_ROOT / "ts_benchmark" / "baselines" / "utils.py",
        REPO_ROOT / "ts_benchmark" / "baselines" / "PatternAD" / "PatternAD.py",
        REPO_ROOT
        / "ts_benchmark"
        / "baselines"
        / "PatternAD"
        / "utils"
        / "pattern_scoring.py",
        REPO_ROOT / "ts_benchmark" / "evaluation" / "evaluate_model.py",
        REPO_ROOT / "ts_benchmark" / "evaluation" / "strategy" / "constants.py",
        REPO_ROOT / "ts_benchmark" / "evaluation" / "strategy" / "anomaly_detect.py",
        REPO_ROOT / "ts_benchmark" / "pipeline.py",
        REPO_ROOT / "ts_benchmark" / "recording.py",
        REPO_ROOT / "ts_benchmark" / "report" / "report_csv.py",
        REPO_ROOT / "ts_benchmark" / "report" / "utils" / "leaderboard.py",
        benchmark_config,
        REPO_ROOT / "dataset" / "anomaly_detect" / "DETECT_META.csv",
    ]
    paths.extend(sorted((REPO_ROOT / "scripts" / "patternad").glob("*.py")))
    paths.extend(sorted((REPO_ROOT / "config" / "patternad").glob("*.json")))
    paths.extend(
        sorted(
            (REPO_ROOT / "ts_benchmark" / "baselines" / "PatternAD").rglob("*.py")
        )
    )
    paths.extend(
        sorted((REPO_ROOT / "ts_benchmark" / "evaluation" / "metrics").glob("*.py"))
    )
    paths = list(dict.fromkeys(paths))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing critical source file: {missing[0]}")
    return {
        str(path.relative_to(REPO_ROOT)): _file_sha256(path)
        for path in paths
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    os.replace(temporary, path)


def _completed_attempt(
    seed_dir: Path, expected_config_hash: Optional[str] = None
) -> Optional[Path]:
    completed = []
    if not seed_dir.exists():
        return None
    for metadata_path in sorted(seed_dir.glob("attempt_*/run_metadata.json")):
        try:
            metadata = _load_json(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        attempt_dir = metadata_path.parent
        if metadata.get("status") == "completed" and list(attempt_dir.glob("*.csv.tar.gz")):
            if (
                expected_config_hash is not None
                and metadata.get("config_hash") != expected_config_hash
            ):
                raise RuntimeError(
                    "Completed attempt does not match the current frozen config: "
                    f"{attempt_dir}. Use a new --run-name."
                )
            if not isinstance(metadata.get("model_diagnostics"), dict):
                raise RuntimeError(
                    "Completed attempt lacks validated model diagnostics: "
                    f"{attempt_dir}. Use a new --run-name."
                )
            completed.append(attempt_dir)
    if len(completed) > 1:
        raise RuntimeError(
            f"Multiple completed attempts exist for one run identity: {seed_dir}"
        )
    return completed[0] if completed else None


def _next_attempt_dir(seed_dir: Path) -> Path:
    numbers = []
    if seed_dir.exists():
        for path in seed_dir.glob("attempt_*"):
            suffix = path.name[len("attempt_") :]
            if suffix.isdigit():
                numbers.append(int(suffix))
    return seed_dir / f"attempt_{max(numbers, default=0) + 1:03d}"


def _read_detail_rows(tar_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with tarfile.open(tar_path, mode="r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".csv")
        ]
        if not members:
            raise RuntimeError(f"No CSV member found in {tar_path}.")
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read {member.name} in {tar_path}.")
            text = (line.decode("utf-8-sig") for line in extracted)
            rows.extend(csv.DictReader(text))
    return rows


def _finite_number(value: Any, name: str, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"Model diagnostic {name} must be numeric, not boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Model diagnostic {name} is not numeric.") from error
    if not math.isfinite(numeric):
        raise RuntimeError(f"Model diagnostic {name} is not finite.")
    if minimum is not None and numeric < minimum:
        raise RuntimeError(f"Model diagnostic {name} must be at least {minimum}.")
    return numeric


def _validate_model_diagnostics(
    diagnostics: Mapping[str, Any],
    hyperparameters: Mapping[str, Any],
    maximum_boundary_fraction: float,
) -> Dict[str, Any]:
    if diagnostics.get("schema_version") != 1 or diagnostics.get("model") != "PatternAD":
        raise RuntimeError("Detailed result has unsupported PatternAD diagnostics.")
    distribution = str(hyperparameters.get("reconstruction_distribution", "mse"))
    if diagnostics.get("distribution") != distribution:
        raise RuntimeError("Model diagnostic distribution differs from the frozen cell.")
    expected_score_mode = str(hyperparameters.get("pattern_score_mode", "raw"))
    if diagnostics.get("score_mode") != expected_score_mode:
        raise RuntimeError("Model diagnostic score mode differs from the frozen cell.")

    training = diagnostics.get("training")
    if not isinstance(training, dict):
        raise RuntimeError("Model diagnostics are missing training details.")
    for name in ("fit_seconds", "training_seconds", "scorer_fit_seconds"):
        _finite_number(training.get(name), f"training.{name}", minimum=0.0)
    epochs_completed = int(
        _finite_number(
            training.get("epochs_completed"), "training.epochs_completed", minimum=1
        )
    )
    epochs_requested = int(
        _finite_number(
            training.get("epochs_requested"), "training.epochs_requested", minimum=1
        )
    )
    if epochs_completed > epochs_requested:
        raise RuntimeError("Model diagnostics report too many completed epochs.")
    history = training.get("epoch_history")
    if not isinstance(history, list) or len(history) != epochs_completed:
        raise RuntimeError("Model diagnostic epoch history is incomplete.")
    for index, epoch in enumerate(history, start=1):
        if not isinstance(epoch, dict) or int(epoch.get("epoch", -1)) != index:
            raise RuntimeError("Model diagnostic epoch history is out of order.")
        for name in ("train_loss", "validation_loss", "learning_rate"):
            _finite_number(epoch.get(name), f"epoch[{index}].{name}")
        _finite_number(
            epoch.get("elapsed_seconds"),
            f"epoch[{index}].elapsed_seconds",
            minimum=0.0,
        )
    best_epoch = int(
        _finite_number(training.get("best_epoch"), "training.best_epoch", minimum=1)
    )
    if best_epoch > epochs_completed:
        raise RuntimeError("Model diagnostic best_epoch exceeds completed epochs.")
    _finite_number(training.get("best_validation_loss"), "training.best_validation_loss")
    _finite_number(training.get("parameter_count"), "training.parameter_count", minimum=1)
    _finite_number(
        training.get("optimization_train_points"),
        "training.optimization_train_points",
        minimum=1,
    )
    _finite_number(
        training.get("validation_points"), "training.validation_points", minimum=1
    )
    if not isinstance(training.get("stopped_early"), bool):
        raise RuntimeError("Model diagnostic stopped_early must be boolean.")

    score_calls = diagnostics.get("score_calls")
    if not isinstance(score_calls, list) or len(score_calls) != 2:
        raise RuntimeError(
            "Strict diagnostics require exactly calibration and test score calls."
        )
    for call_index, (call, expected_phase) in enumerate(
        zip(score_calls, ("calibration", "test"))
    ):
        if not isinstance(call, dict):
            raise RuntimeError("Model score diagnostics must be objects.")
        if int(call.get("call_index", -1)) != call_index:
            raise RuntimeError("Model score diagnostic call_index is invalid.")
        if call.get("phase") != expected_phase:
            raise RuntimeError("Model score diagnostic phases are incomplete or out of order.")
        for name in ("input_length", "batch_count", "window_count"):
            _finite_number(call.get(name), f"{expected_phase}.{name}", minimum=1)
        _finite_number(
            call.get("elapsed_seconds"),
            f"{expected_phase}.elapsed_seconds",
            minimum=0.0,
        )
        score = call.get("score")
        if not isinstance(score, dict):
            raise RuntimeError(f"{expected_phase} score diagnostics are missing.")
        score_count = int(_finite_number(score.get("count"), "score.count", minimum=1))
        finite_count = int(
            _finite_number(score.get("finite_count"), "score.finite_count", minimum=0)
        )
        nonfinite_count = int(
            _finite_number(
                score.get("nonfinite_count"), "score.nonfinite_count", minimum=0
            )
        )
        if score_count != finite_count + nonfinite_count or nonfinite_count:
            raise RuntimeError("Model score diagnostics contain non-finite values.")
        if score_count != int(call["input_length"]):
            raise RuntimeError("Model score count differs from the score input length.")
        score_min = _finite_number(score.get("min"), f"{expected_phase}.score.min")
        score_mean = _finite_number(score.get("mean"), f"{expected_phase}.score.mean")
        score_max = _finite_number(score.get("max"), f"{expected_phase}.score.max")
        if not score_min <= score_mean <= score_max:
            raise RuntimeError("Model score min/mean/max are inconsistent.")

        scale = call.get("scale")
        if distribution == "mse":
            if scale is not None:
                raise RuntimeError("MSE diagnostics must mark scale as not applicable.")
            continue
        if not isinstance(scale, dict):
            raise RuntimeError("Probabilistic diagnostics are missing scale statistics.")
        scale_count = int(
            _finite_number(scale.get("count"), "scale.count", minimum=1)
        )
        scale_finite = int(
            _finite_number(scale.get("finite_count"), "scale.finite_count", minimum=0)
        )
        scale_nonfinite = int(
            _finite_number(
                scale.get("nonfinite_count"), "scale.nonfinite_count", minimum=0
            )
        )
        if scale_count != scale_finite + scale_nonfinite or scale_nonfinite:
            raise RuntimeError("Predicted scale contains non-finite values.")
        scale_min = _finite_number(scale.get("min"), "scale.min", minimum=0.0)
        scale_mean = _finite_number(scale.get("mean"), "scale.mean", minimum=0.0)
        scale_max = _finite_number(scale.get("max"), "scale.max", minimum=0.0)
        _finite_number(scale.get("std"), "scale.std", minimum=0.0)
        lower_bound = _finite_number(
            scale.get("lower_bound"), "scale.lower_bound", minimum=0.0
        )
        upper_bound = _finite_number(
            scale.get("upper_bound"), "scale.upper_bound", minimum=lower_bound
        )
        if not scale_min <= scale_mean <= scale_max:
            raise RuntimeError("Predicted scale min/mean/max are inconsistent.")
        if scale_min < lower_bound or scale_max > upper_bound * (1.0 + 1e-12):
            raise RuntimeError("Predicted scale falls outside its configured bounds.")
        for side in ("lower", "upper"):
            count = int(
                _finite_number(
                    scale.get(f"{side}_bound_count"),
                    f"scale.{side}_bound_count",
                    minimum=0,
                )
            )
            fraction = _finite_number(
                scale.get(f"{side}_bound_fraction"),
                f"scale.{side}_bound_fraction",
                minimum=0.0,
            )
            if count > scale_count or fraction > 1:
                raise RuntimeError("Predicted scale boundary diagnostics are invalid.")
            if not math.isclose(fraction, count / scale_count, abs_tol=1e-12):
                raise RuntimeError("Predicted scale boundary count/fraction disagree.")
            if (
                expected_phase == "calibration"
                and fraction >= maximum_boundary_fraction
            ):
                raise RuntimeError(
                    f"Calibration scale {side}-boundary fraction {fraction:.6f} "
                    f"exceeds the frozen limit {maximum_boundary_fraction:.6f}. "
                    "Test boundary fractions are reported but never tune this gate."
                )
    return dict(diagnostics)


def _validate_artifact(
    attempt_dir: Path,
    benchmark: Mapping[str, Any],
    score_metrics: Sequence[str],
    seed: int,
    hyperparameters: Mapping[str, Any],
) -> Tuple[Path, Dict[str, Any]]:
    tar_files = sorted(attempt_dir.glob("*.csv.tar.gz"))
    if len(tar_files) != 1:
        raise RuntimeError(
            f"Expected one detailed CSV tar in {attempt_dir}, found {len(tar_files)}."
        )
    rows = _read_detail_rows(tar_files[0])
    if not rows:
        raise RuntimeError(f"Detailed result is empty: {tar_files[0]}")
    errors = [row.get("log_info", "").strip() for row in rows if row.get("log_info", "").strip()]
    if errors:
        raise RuntimeError(
            "Benchmark returned error rows; first error: " + errors[0].splitlines()[0]
        )
    expected_ratios = {float(value) for value in benchmark["anomaly_ratios"]}
    observed_ratios = set()
    diagnostic_payloads = set()
    for row in rows:
        try:
            strategy_args = json.loads(row.get("strategy_args", "{}"))
        except json.JSONDecodeError as error:
            raise RuntimeError("Detailed result has invalid strategy_args JSON.") from error
        if strategy_args.get("evaluation_protocol") != benchmark["evaluation_protocol"]:
            raise RuntimeError("Detailed result used the wrong evaluation protocol.")
        if float(strategy_args.get("calibration_fraction", -1)) != float(
            benchmark["calibration_fraction"]
        ):
            raise RuntimeError("Detailed result used the wrong calibration fraction.")
        if int(strategy_args.get("seed", -1)) != int(seed):
            raise RuntimeError("Detailed result used the wrong seed.")
        observed_ratios.add(float(row.get("typical_anomaly_ratio", "nan")))
        diagnostic_payload = row.get("model_diagnostics", "").strip()
        if not diagnostic_payload:
            raise RuntimeError("Detailed result is missing model diagnostics.")
        diagnostic_payloads.add(diagnostic_payload)
        for metric in score_metrics:
            try:
                value = float(row.get(metric, "nan"))
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Detailed result metric {metric!r} is non-numeric."
                ) from error
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Detailed result metric {metric!r} is not finite."
                )
    if observed_ratios != expected_ratios:
        raise RuntimeError(
            f"Detailed result anomaly ratios differ: {observed_ratios} != {expected_ratios}."
        )
    if len(diagnostic_payloads) != 1:
        raise RuntimeError("Detailed threshold rows disagree on model diagnostics.")
    try:
        diagnostics = json.loads(next(iter(diagnostic_payloads)))
    except (json.JSONDecodeError, StopIteration) as error:
        raise RuntimeError("Detailed result has invalid model diagnostics JSON.") from error
    if not isinstance(diagnostics, dict):
        raise RuntimeError("Detailed model diagnostics must be a JSON object.")
    diagnostics = _validate_model_diagnostics(
        diagnostics,
        hyperparameters,
        float(benchmark["max_scale_boundary_fraction"]),
    )
    return tar_files[0], diagnostics


def _build_command(
    python: str,
    manifest: Mapping[str, Any],
    dataset: Mapping[str, Any],
    hyperparameters: Mapping[str, Any],
    seed: int,
    attempt_dir: Path,
    gpu_ids: Sequence[int],
) -> List[str]:
    benchmark = manifest["benchmark"]
    strategy_args = {
        "evaluation_protocol": benchmark["evaluation_protocol"],
        "anomaly_ratios": benchmark["anomaly_ratios"],
        "calibration_fraction": benchmark["calibration_fraction"],
    }
    command = [
        python,
        str(REPO_ROOT / benchmark["script"]),
        "--config-path",
        benchmark["config_path"],
        "--data-name-list",
        dataset["data_name"],
        "--model-name",
        benchmark["model_name"],
        "--model-hyper-params",
        json.dumps(hyperparameters, sort_keys=True, separators=(",", ":")),
        "--strategy-args",
        json.dumps(strategy_args, sort_keys=True, separators=(",", ":")),
        "--seed",
        str(seed),
        "--eval-backend",
        benchmark["eval_backend"],
        "--num-workers",
        str(benchmark["num_workers"]),
        "--num-cpus",
        str(benchmark["num_cpus"]),
        "--timeout",
        str(benchmark["timeout_seconds"]),
        "--aggregate_type",
        benchmark["aggregate_type"],
        "--save-path",
        str(attempt_dir),
        "--text-name-list",
        dataset["text_name"],
    ]
    if gpu_ids:
        command.extend(["--gpus", *map(str, gpu_ids)])
    return command


def _merged_hyperparameters(
    manifest: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = dict(manifest["shared_hyperparameters"])
    merged.update(dataset_config["resource_profiles"][dataset["resource_profile"]])
    merged.update(dataset.get("hyperparameters", {}))
    merged.update(variant["hyperparameters"])
    return merged


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PatternAD factorial cells from the canonical manifests."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--group", help="Dataset group; defaults to manifest.default_group.")
    parser.add_argument("--dataset", nargs="+", help="Optional subset of dataset IDs in the group.")
    parser.add_argument("--variant", nargs="+", help="Variant IDs; defaults to the group list.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Model seeds.")
    parser.add_argument("--gpus", nargs="+", type=int, help="Physical GPUs exposed to each run.")
    parser.add_argument("--run-name", help="Result namespace; defaults to experiment_name.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable, help="Python used for run_benchmark.py.")
    parser.add_argument("--resume", action="store_true", help="Skip completed identities and retry incomplete ones in a new attempt directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without writing or executing.")
    parser.add_argument("--allow-locked", action="store_true", help="Required acknowledgement for a locked group.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed run.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    manifest = _load_json(args.manifest.resolve())
    dataset_config = _load_json(args.dataset_config.resolve())
    _validate_manifest(manifest)
    _validate_dataset_config(dataset_config)

    group_id = args.group or manifest["default_group"]
    if group_id not in dataset_config["groups"]:
        raise ValueError(
            f"Unknown group {group_id!r}; choose from {sorted(dataset_config['groups'])}."
        )
    group = dataset_config["groups"][group_id]
    explicit_variants = args.variant is not None
    explicit_seeds = args.seeds is not None
    explicit_run_name = args.run_name is not None
    if group["locked"] and not (
        args.allow_locked and explicit_variants and explicit_seeds and explicit_run_name
    ):
        raise ValueError(
            "Locked group refused. Confirmation runs require --allow-locked and explicit "
            "--variant, --seeds, and --run-name."
        )

    variant_ids = args.variant or group["default_variants"]
    unknown_variants = sorted(set(variant_ids) - set(manifest["variants"]))
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")
    if len(set(variant_ids)) != len(variant_ids):
        raise ValueError("Duplicate --variant values are not allowed.")

    seeds = args.seeds or (
        manifest["confirmation_seeds"] if group["locked"] else manifest["development_seeds"]
    )
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds are not allowed.")

    dataset_ids = args.dataset or group["datasets"]
    outside_group = sorted(set(dataset_ids) - set(group["datasets"]))
    if outside_group:
        raise ValueError(f"Datasets are not members of group {group_id}: {outside_group}")
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("Duplicate dataset IDs are not allowed.")
    if group["locked"]:
        if dataset_ids != group["datasets"]:
            raise ValueError(
                "Locked confirmation must run the complete predeclared dataset list."
            )
        if variant_ids != group["default_variants"]:
            raise ValueError(
                "Locked confirmation variants must exactly match the frozen candidate list."
            )
        if seeds != manifest["confirmation_seeds"]:
            raise ValueError(
                "Locked confirmation seeds must exactly match confirmation_seeds."
            )

    run_name = _safe_component(args.run_name or manifest["experiment_name"], "run name")
    output_root = args.output_root.resolve()
    benchmark_script = REPO_ROOT / manifest["benchmark"]["script"]
    benchmark_config = REPO_ROOT / "config" / manifest["benchmark"]["config_path"]
    if not benchmark_script.is_file():
        raise FileNotFoundError(benchmark_script)
    if not benchmark_config.is_file():
        raise FileNotFoundError(benchmark_config)

    manifest_hash = _canonical_hash(manifest)
    dataset_config_hash = _canonical_hash(dataset_config)
    critical_source_hashes = _critical_source_hashes(benchmark_config)
    source_bundle_hash = _canonical_hash(critical_source_hashes)
    git_commit = _git_commit()
    git_dirty = _git_dirty()
    if group["locked"] and git_dirty is not False:
        raise ValueError(
            "Locked confirmation requires a clean PatternAD-main worktree so the "
            "recorded commit can reconstruct the frozen sources."
        )

    data_root = REPO_ROOT / "dataset" / "anomaly_detect" / "data"
    dataset_artifact_hashes: Dict[str, Dict[str, str]] = {}
    for dataset_id in dataset_ids:
        dataset = dataset_config["datasets"][dataset_id]
        dataset_artifact_hashes[dataset_id] = {}
        for file_key in ("data_name", "text_name"):
            file_path = data_root / dataset[file_key]
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"Dataset {dataset_id} references missing {file_key}: {file_path}"
                )
            dataset_artifact_hashes[dataset_id][file_key] = _file_sha256(file_path)

    gpu_ids = args.gpus or []
    if len(set(gpu_ids)) != len(gpu_ids) or any(gpu < 0 for gpu in gpu_ids):
        raise ValueError("--gpus must contain unique non-negative device IDs.")
    environment = os.environ.copy()

    jobs = []
    planned_identities = []
    skipped = 0
    conflicts = []
    for dataset_id in dataset_ids:
        dataset = dataset_config["datasets"][dataset_id]
        for variant_id in variant_ids:
            variant = manifest["variants"][variant_id]
            for seed in seeds:
                hyperparameters = _merged_hyperparameters(
                    manifest, dataset_config, dataset, variant
                )
                config_hash = _canonical_hash(
                    {
                        "benchmark": manifest["benchmark"],
                        "dataset": dataset,
                        "dataset_artifacts": dataset_artifact_hashes[dataset_id],
                        "dataset_config_hash": dataset_config_hash,
                        "hyperparameters": hyperparameters,
                        "manifest_hash": manifest_hash,
                        "seed": seed,
                        "source_bundle_hash": source_bundle_hash,
                        "variant": variant,
                    }
                )
                seed_dir = (
                    output_root
                    / run_name
                    / group_id
                    / dataset_id
                    / variant_id
                    / f"seed_{seed}"
                )
                planned_identities.append(
                    {
                        "dataset_id": dataset_id,
                        "variant": variant_id,
                        "seed": seed,
                        "config_hash": config_hash,
                    }
                )
                completed = _completed_attempt(seed_dir, config_hash)
                if args.resume and completed is not None:
                    print(f"SKIP completed: {completed}")
                    skipped += 1
                    continue
                if seed_dir.exists() and not args.resume:
                    conflicts.append(seed_dir)
                    continue
                jobs.append(
                    (
                        dataset_id,
                        dataset,
                        variant_id,
                        variant,
                        seed,
                        seed_dir,
                        hyperparameters,
                        config_hash,
                    )
                )

    if conflicts:
        examples = "\n".join(f"  {path}" for path in conflicts[:5])
        raise FileExistsError(
            "Refusing to append to existing run identities. Use --resume or a new "
            f"--run-name. Examples:\n{examples}"
        )

    plan_core = {
        "schema_version": 1,
        "experiment": manifest["experiment_name"],
        "run_name": run_name,
        "group": group_id,
        "locked_group": bool(group["locked"]),
        "manifest_hash": manifest_hash,
        "dataset_config_hash": dataset_config_hash,
        "source_bundle_hash": source_bundle_hash,
        "critical_source_hashes": critical_source_hashes,
        "dataset_artifact_hashes": dataset_artifact_hashes,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "expected_identities": planned_identities,
    }
    plan_hash = _canonical_hash(plan_core)
    run_root = output_root / run_name / group_id
    plan_path = run_root / "run_plan.json"
    if not args.dry_run:
        if plan_path.exists():
            existing_plan = _load_json(plan_path)
            if existing_plan.get("plan_hash") != plan_hash:
                raise RuntimeError(
                    f"Existing run plan differs from the current frozen plan: {plan_path}. "
                    "Use a new --run-name."
                )
        else:
            if args.resume and run_root.exists():
                raise RuntimeError(
                    f"Cannot resume a result tree without run_plan.json: {run_root}."
                )
            run_root.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                plan_path,
                {**plan_core, "plan_hash": plan_hash, "created_at": _utc_now()},
            )

    print(
        f"Plan: group={group_id} datasets={len(dataset_ids)} variants={len(variant_ids)} "
        f"seeds={len(seeds)} jobs={len(jobs)} skipped={skipped} plan_hash={plan_hash[:12]}"
    )
    failures = []
    for index, (
        dataset_id,
        dataset,
        variant_id,
        variant,
        seed,
        seed_dir,
        hyperparameters,
        config_hash,
    ) in enumerate(jobs, start=1):
        attempt_dir = _next_attempt_dir(seed_dir)
        command = _build_command(
            args.python,
            manifest,
            dataset,
            hyperparameters,
            seed,
            attempt_dir,
            gpu_ids,
        )
        print(
            f"[{index}/{len(jobs)}] {dataset_id} {variant_id} seed={seed}\n"
            f"  output: {attempt_dir}\n"
            f"  command: {shlex.join(command)}"
        )
        if args.dry_run:
            continue

        attempt_dir.mkdir(parents=True, exist_ok=False)
        metadata: Dict[str, Any] = {
            "schema_version": 2,
            "status": "running",
            "started_at": _utc_now(),
            "experiment": manifest["experiment_name"],
            "run_name": run_name,
            "group": group_id,
            "locked_group": bool(group["locked"]),
            "dataset_id": dataset_id,
            "data_name": dataset["data_name"],
            "text_name": dataset["text_name"],
            "family": dataset["family"],
            "entity": dataset["entity"],
            "variant": variant_id,
            "factors": variant["factors"],
            "seed": seed,
            "hyperparameters": hyperparameters,
            "config_hash": config_hash,
            "plan_hash": plan_hash,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "critical_source_hashes": critical_source_hashes,
            "source_bundle_hash": source_bundle_hash,
            "dataset_artifact_hashes": dataset_artifact_hashes[dataset_id],
            "manifest_path": str(args.manifest.resolve()),
            "manifest_hash": manifest_hash,
            "dataset_config_path": str(args.dataset_config.resolve()),
            "dataset_config_hash": dataset_config_hash,
            "gpus": gpu_ids,
            "command": command,
        }
        metadata_path = attempt_dir / "run_metadata.json"
        benchmark_log_path = attempt_dir / "benchmark.log"
        metadata["benchmark_log"] = benchmark_log_path.name
        _write_json_atomic(metadata_path, metadata)
        run_started_at = time.perf_counter()
        try:
            with benchmark_log_path.open("w", encoding="utf-8") as benchmark_log:
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    check=False,
                    timeout=float(manifest["benchmark"]["timeout_seconds"]),
                    stdout=benchmark_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"run_benchmark.py exited with status {result.returncode}; "
                    f"inspect {benchmark_log_path}."
                )
            artifact, model_diagnostics = _validate_artifact(
                attempt_dir,
                manifest["benchmark"],
                manifest["score_metrics"],
                seed,
                hyperparameters,
            )
            metadata.update(
                {
                    "status": "completed",
                    "completed_at": _utc_now(),
                    "return_code": result.returncode,
                    "runner_wall_seconds": float(
                        time.perf_counter() - run_started_at
                    ),
                    "detail_artifact": artifact.name,
                    "model_diagnostics": model_diagnostics,
                }
            )
            _write_json_atomic(metadata_path, metadata)
        except BaseException as error:
            metadata.update(
                {
                    "status": "interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "failed",
                    "completed_at": _utc_now(),
                    "runner_wall_seconds": float(
                        time.perf_counter() - run_started_at
                    ),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            _write_json_atomic(metadata_path, metadata)
            if isinstance(error, KeyboardInterrupt):
                raise
            failures.append((dataset_id, variant_id, seed, str(error)))
            print(f"FAILED: {dataset_id} {variant_id} seed={seed}: {error}", file=sys.stderr)
            if args.fail_fast:
                break

    if args.dry_run:
        print("Dry run complete; no directories or benchmark processes were created.")
        return 0
    if failures:
        print(f"Completed with {len(failures)} failed run(s).", file=sys.stderr)
        return 1
    print(f"Completed {len(jobs)} run(s); skipped {skipped} completed run(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
