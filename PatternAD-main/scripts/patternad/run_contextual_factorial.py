#!/usr/bin/env python3
"""Run the frozen PatternAD contextual synthetic factorial grid."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYNTHETIC_CONFIG = REPO_ROOT / "config/patternad/synthetic_suite.json"
DEFAULT_FACTORIAL_MANIFEST = REPO_ROOT / "config/patternad/factorial_ablation.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result/patternad_synthetic"
GENERATOR = REPO_ROOT / "scripts/patternad/generate_contextual_synthetic.py"
EVALUATOR = REPO_ROOT / "scripts/patternad/evaluate_contextual_mechanisms.py"
MAIN_VARIANTS = ("A00", "A10", "A01", "A11")
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


def _safe_component(value: str, owner: str) -> str:
    if not SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"Invalid {owner} {value!r}; use only letters, digits, '.', '_' and '-'."
        )
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    os.replace(temporary, path)


def _git_commit() -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> Optional[bool]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "."],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _critical_source_hashes() -> Dict[str, str]:
    paths = [GENERATOR, EVALUATOR, Path(__file__).resolve()]
    summary = Path(__file__).resolve().with_name("summarize_contextual_factorial.py")
    if summary.is_file():
        paths.append(summary)
    paths.extend(
        sorted((REPO_ROOT / "ts_benchmark/baselines/PatternAD").rglob("*.py"))
    )
    paths.extend(
        [
            REPO_ROOT / "ts_benchmark/baselines/utils.py",
            REPO_ROOT / "ts_benchmark/utils/random_utils.py",
        ]
    )
    paths = list(dict.fromkeys(paths))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing critical source file: {missing[0]}")
    return {
        str(path.relative_to(REPO_ROOT)): _file_sha256(path) for path in paths
    }


def _resolved_synthetic_config(
    base_config: Mapping[str, Any], generator_seed: int
) -> Dict[str, Any]:
    resolved = copy.deepcopy(dict(base_config))
    resolved["seed"] = int(generator_seed)
    return resolved


def _artifact_status(
    artifact_dir: Path,
    resolved_config: Mapping[str, Any],
    generator_hash: str,
) -> Tuple[bool, str]:
    required = [
        "resolved_config.json",
        "suite_manifest.json",
        "same_deviation_different_context.npz",
        "same_deviation_different_context.metadata.json",
        "slow_drift_vs_abrupt_shift.npz",
        "slow_drift_vs_abrupt_shift.metadata.json",
        "dependency_break.npz",
        "dependency_break.metadata.json",
        "context_ood.npz",
        "context_ood.metadata.json",
    ]
    missing = [name for name in required if not (artifact_dir / name).is_file()]
    if missing:
        return False, f"missing {missing[0]}"
    observed_config = _load_json(artifact_dir / "resolved_config.json")
    if _canonical_hash(observed_config) != _canonical_hash(resolved_config):
        return False, "resolved config differs"
    suite_manifest = _load_json(artifact_dir / "suite_manifest.json")
    if suite_manifest.get("config_hash") != _canonical_hash(resolved_config):
        return False, "suite manifest config hash differs"
    if suite_manifest.get("source_hashes", {}).get("generator") != generator_hash:
        return False, "suite was generated by a different generator source"
    return True, "complete"


def _identity_config_hash(
    *,
    factorial_hash: str,
    source_bundle_hash: str,
    synthetic_config_hash: str,
    variant_id: str,
    variant: Mapping[str, Any],
    shared_hyperparameters: Mapping[str, Any],
    generator_seed: int,
    model_seed: int,
) -> str:
    merged_hyperparameters = dict(shared_hyperparameters)
    merged_hyperparameters.update(variant["hyperparameters"])
    return _canonical_hash(
        {
            "factorial_manifest_hash": factorial_hash,
            "source_bundle_hash": source_bundle_hash,
            "synthetic_config_hash": synthetic_config_hash,
            "variant": variant_id,
            "variant_definition": variant,
            "hyperparameters": merged_hyperparameters,
            "generator_seed": int(generator_seed),
            "model_seed": int(model_seed),
        }
    )


def _validate_completed_identity(
    result_dir: Path,
    identity: Mapping[str, Any],
    plan_hash: str,
    factorial_hash: str,
) -> bool:
    metadata_path = result_dir / "identity_metadata.json"
    summary_path = result_dir / "contextual_evaluation.json"
    score_metadata_path = result_dir / "scores/score_run_metadata.json"
    if not all(path.is_file() for path in (metadata_path, summary_path, score_metadata_path)):
        return False
    metadata = _load_json(metadata_path)
    if metadata.get("status") != "completed":
        return False
    if metadata.get("plan_hash") != plan_hash:
        return False
    if metadata.get("config_hash") != identity["config_hash"]:
        return False
    for key in ("variant", "generator_seed", "model_seed"):
        if metadata.get(key) != identity[key]:
            return False
    output_hashes = metadata.get("output_hashes")
    if not isinstance(output_hashes, dict) or not output_hashes:
        return False
    for relative_path, expected_hash in output_hashes.items():
        output_path = result_dir / relative_path
        if not output_path.is_file() or _file_sha256(output_path) != expected_hash:
            return False
    summary = _load_json(summary_path)
    score_metadata = _load_json(score_metadata_path)
    mechanisms = summary.get("mechanisms")
    if not isinstance(mechanisms, list) or len(mechanisms) != 4:
        return False
    for row in mechanisms:
        score_path = result_dir / "scores" / f"{row.get('mechanism')}.npz"
        if not score_path.is_file() or _file_sha256(score_path) != row.get(
            "score_sha256"
        ):
            return False
    expected_method = f"{identity['variant']}_seed_{identity['model_seed']}"
    return bool(
        summary.get("method") == expected_method
        and summary.get("config_hash") == identity["synthetic_config_hash"]
        and score_metadata.get("variant") == identity["variant"]
        and int(score_metadata.get("seed", -1)) == int(identity["model_seed"])
        and score_metadata.get("synthetic_config_hash")
        == identity["synthetic_config_hash"]
        and score_metadata.get("factorial_manifest_hash") == factorial_hash
        and _canonical_hash(score_metadata.get("hyperparameters"))
        == identity["hyperparameters_hash"]
    )


def _terminate_process_group(process: subprocess.Popen, grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_logged(
    command: Sequence[str],
    log_path: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> float:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except BaseException:
            _terminate_process_group(process)
            raise
    if return_code != 0:
        raise RuntimeError(
            f"Command exited with status {return_code}; inspect {log_path}."
        )
    return float(time.perf_counter() - started)


def _select_grid(
    args: argparse.Namespace,
    synthetic_config: Mapping[str, Any],
    factorial: Mapping[str, Any],
) -> Tuple[list[int], list[int], list[str], bool]:
    phase = args.seed_group
    if phase not in ("development", "confirmation"):
        raise ValueError("--seed-group must be development or confirmation.")
    seed_groups = synthetic_config.get("seed_groups", {})
    if phase not in seed_groups:
        raise ValueError(f"Synthetic config has no seed group {phase!r}.")
    expected_generator_seeds = [int(value) for value in seed_groups[phase]]
    expected_model_seeds = [
        int(value)
        for value in factorial[
            "development_seeds" if phase == "development" else "confirmation_seeds"
        ]
    ]
    generator_seeds = args.generator_seeds or expected_generator_seeds
    model_seeds = args.model_seeds or expected_model_seeds
    variants = args.variant or list(MAIN_VARIANTS)
    for owner, values in (
        ("generator seeds", generator_seeds),
        ("model seeds", model_seeds),
        ("variants", variants),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {owner} are not allowed.")
    unknown_variants = sorted(set(variants) - set(factorial["variants"]))
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")
    if not set(variants).issubset(MAIN_VARIANTS):
        raise ValueError("P1 accepts only A00/A10/A01/A11.")

    complete_phase_grid = bool(
        generator_seeds == expected_generator_seeds
        and model_seeds == expected_model_seeds
        and variants == list(MAIN_VARIANTS)
    )
    if phase == "confirmation":
        if not (
            args.allow_locked
            and args.run_name is not None
            and args.generator_seeds is not None
            and args.model_seeds is not None
            and args.variant is not None
        ):
            raise ValueError(
                "Locked confirmation requires --allow-locked plus explicit "
                "--run-name, --generator-seeds, --model-seeds, and --variant."
            )
        if not complete_phase_grid:
            raise ValueError(
                "Locked confirmation must use the complete predeclared generator/model "
                "seed grid and A00/A10/A01/A11 in canonical order."
            )
    return generator_seeds, model_seeds, variants, complete_phase_grid


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-config", type=Path, default=DEFAULT_SYNTHETIC_CONFIG)
    parser.add_argument("--factorial-manifest", type=Path, default=DEFAULT_FACTORIAL_MANIFEST)
    parser.add_argument(
        "--seed-group", choices=("development", "confirmation"), default="development"
    )
    parser.add_argument("--generator-seeds", nargs="+", type=int)
    parser.add_argument("--model-seeds", nargs="+", type=int)
    parser.add_argument("--variant", nargs="+")
    parser.add_argument("--run-name")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-locked", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    synthetic_config_path = args.synthetic_config.resolve()
    factorial_path = args.factorial_manifest.resolve()
    synthetic_config = _load_json(synthetic_config_path)
    factorial = _load_json(factorial_path)
    generator_seeds, model_seeds, variants, complete_phase_grid = _select_grid(
        args, synthetic_config, factorial
    )

    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")
    timeout_seconds = float(
        args.timeout_seconds
        if args.timeout_seconds is not None
        else factorial.get("benchmark", {}).get("timeout_seconds", 60000)
    )
    gpu_ids = args.gpus or []
    if len(gpu_ids) != len(set(gpu_ids)) or any(gpu < 0 for gpu in gpu_ids):
        raise ValueError("--gpus must contain unique non-negative physical GPU IDs.")

    run_name = _safe_component(
        args.run_name or f"p1_contextual_{args.seed_group}", "run name"
    )
    output_root = args.output_root.resolve()
    run_root = output_root / run_name / args.seed_group
    run_root_preexisting = run_root.exists()
    frozen_input_dir = run_root / "frozen_inputs"
    frozen_synthetic_path = frozen_input_dir / "synthetic_suite.json"
    frozen_factorial_path = frozen_input_dir / "factorial_ablation.json"
    configured_artifact_root = Path(synthetic_config["output"]["artifact_dir"])
    if not configured_artifact_root.is_absolute():
        configured_artifact_root = REPO_ROOT / configured_artifact_root
    artifact_root = (
        args.artifact_root.resolve()
        if args.artifact_root is not None
        else configured_artifact_root.resolve()
    )

    factorial_hash = _canonical_hash(factorial)
    synthetic_config_hash = _canonical_hash(synthetic_config)
    source_hashes = _critical_source_hashes()
    source_bundle_hash = _canonical_hash(source_hashes)
    git_commit = _git_commit()
    git_dirty = _git_dirty()
    if args.seed_group == "confirmation" and git_dirty is not False:
        raise ValueError(
            "Locked confirmation requires a clean PatternAD-main worktree."
        )

    expected_identities = []
    for variant_id in variants:
        variant = factorial["variants"][variant_id]
        for generator_seed in generator_seeds:
            resolved_config = _resolved_synthetic_config(
                synthetic_config, generator_seed
            )
            resolved_hash = _canonical_hash(resolved_config)
            for model_seed in model_seeds:
                expected_hyperparameters = dict(
                    factorial["shared_hyperparameters"]
                )
                expected_hyperparameters.update(variant["hyperparameters"])
                expected_hyperparameters["train_mask_seed"] = int(model_seed)
                relative_dir = (
                    Path(variant_id)
                    / f"generator_seed_{generator_seed}"
                    / f"model_seed_{model_seed}"
                )
                expected_identities.append(
                    {
                        "variant": variant_id,
                        "generator_seed": int(generator_seed),
                        "model_seed": int(model_seed),
                        "result_dir": relative_dir.as_posix(),
                        "synthetic_config_hash": resolved_hash,
                        "hyperparameters_hash": _canonical_hash(
                            expected_hyperparameters
                        ),
                        "config_hash": _identity_config_hash(
                            factorial_hash=factorial_hash,
                            source_bundle_hash=source_bundle_hash,
                            synthetic_config_hash=resolved_hash,
                            variant_id=variant_id,
                            variant=variant,
                            shared_hyperparameters=factorial[
                                "shared_hyperparameters"
                            ],
                            generator_seed=generator_seed,
                            model_seed=model_seed,
                        ),
                    }
                )

    plan_core = {
        "schema_version": 1,
        "experiment": "patternad_contextual_factorial_v1",
        "run_name": run_name,
        "phase": args.seed_group,
        "locked_phase": args.seed_group == "confirmation",
        "complete_phase_grid": complete_phase_grid,
        "generator_seeds": generator_seeds,
        "model_seeds": model_seeds,
        "variants": variants,
        "synthetic_config_path": "frozen_inputs/synthetic_suite.json",
        "synthetic_config_hash": synthetic_config_hash,
        "factorial_manifest_path": "frozen_inputs/factorial_ablation.json",
        "factorial_manifest_hash": factorial_hash,
        "artifact_root": str(artifact_root),
        "source_hashes": source_hashes,
        "source_bundle_hash": source_bundle_hash,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "gpus": gpu_ids,
        "expected_identities": expected_identities,
    }
    plan_hash = _canonical_hash(plan_core)
    plan_path = run_root / "run_plan.json"

    completed = 0
    conflicts = []
    jobs = []
    for identity in expected_identities:
        result_dir = run_root / identity["result_dir"]
        if args.resume and _validate_completed_identity(
            result_dir, identity, plan_hash, factorial_hash
        ):
            completed += 1
            continue
        if result_dir.exists() and not args.resume:
            conflicts.append(result_dir)
        else:
            jobs.append((identity, result_dir))
    if conflicts:
        examples = "\n".join(f"  {path}" for path in conflicts[:5])
        raise FileExistsError(
            "Refusing to overwrite existing identities. Use --resume or a new "
            f"--run-name. Examples:\n{examples}"
        )

    if not args.dry_run:
        frozen_input_dir.mkdir(parents=True, exist_ok=True)
        for frozen_path, payload, expected_hash in (
            (frozen_synthetic_path, synthetic_config, synthetic_config_hash),
            (frozen_factorial_path, factorial, factorial_hash),
        ):
            if frozen_path.is_file():
                if _canonical_hash(_load_json(frozen_path)) != expected_hash:
                    raise RuntimeError(
                        f"Frozen input differs from the current run plan: {frozen_path}."
                    )
            elif plan_path.is_file():
                raise RuntimeError(
                    f"Existing run plan is missing its frozen input: {frozen_path}."
                )
            else:
                _write_json_atomic(frozen_path, payload)
        if plan_path.is_file():
            existing = _load_json(plan_path)
            if existing.get("plan_hash") != plan_hash:
                raise RuntimeError(
                    f"Existing run plan differs from the current plan: {plan_path}."
                )
        else:
            if args.resume and run_root_preexisting:
                raise RuntimeError(
                    f"Cannot resume a result tree without run_plan.json: {run_root}."
                )
            run_root.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                plan_path,
                {**plan_core, "plan_hash": plan_hash, "created_at": _utc_now()},
            )

    generator_hash = _file_sha256(GENERATOR)
    generator_jobs = []
    for generator_seed in generator_seeds:
        artifact_dir = artifact_root / f"seed_{generator_seed}"
        resolved = _resolved_synthetic_config(synthetic_config, generator_seed)
        ready, reason = _artifact_status(artifact_dir, resolved, generator_hash)
        if ready:
            continue
        if artifact_dir.exists():
            raise RuntimeError(
                f"Synthetic artifact directory is stale or partial ({reason}): "
                f"{artifact_dir}. Move it aside or use a new --artifact-root."
            )
        generator_jobs.append((generator_seed, artifact_dir))

    print(
        f"Plan: phase={args.seed_group} generators={len(generator_seeds)} "
        f"model_seeds={len(model_seeds)} variants={len(variants)} "
        f"generate={len(generator_jobs)} run={len(jobs)} completed={completed} "
        f"plan_hash={plan_hash[:12]}"
    )
    environment = os.environ.copy()
    if gpu_ids:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    for generator_seed, artifact_dir in generator_jobs:
        command = [
            args.python,
            "-u",
            str(GENERATOR),
            "--config",
            str(
                synthetic_config_path
                if args.dry_run
                else frozen_synthetic_path
            ),
            "--seed",
            str(generator_seed),
            "--output-dir",
            str(artifact_dir),
        ]
        print(f"GENERATE seed={generator_seed}: {shlex.join(command)}")
        if args.dry_run:
            continue
        log_dir = run_root / "generator_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _run_logged(
            command,
            log_dir / f"seed_{generator_seed}.log",
            environment,
            timeout_seconds,
        )
        ready, reason = _artifact_status(
            artifact_dir,
            _resolved_synthetic_config(synthetic_config, generator_seed),
            generator_hash,
        )
        if not ready:
            raise RuntimeError(
                f"Generated artifact validation failed ({reason}): {artifact_dir}"
            )

    failures = []
    for index, (identity, result_dir) in enumerate(jobs, start=1):
        artifact_dir = artifact_root / f"seed_{identity['generator_seed']}"
        command = [
            args.python,
            "-u",
            str(EVALUATOR),
            "--artifact-dir",
            str(artifact_dir),
            "--factorial-manifest",
            str(factorial_path if args.dry_run else frozen_factorial_path),
            "--patternad-variant",
            identity["variant"],
            "--seed",
            str(identity["model_seed"]),
            "--output-dir",
            str(result_dir),
        ]
        if result_dir.exists():
            command.append("--overwrite")
        print(
            f"[{index}/{len(jobs)}] {identity['variant']} "
            f"generator={identity['generator_seed']} model={identity['model_seed']}\n"
            f"  output: {result_dir}\n  command: {shlex.join(command)}"
        )
        if args.dry_run:
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "status": "running",
            "started_at": _utc_now(),
            "plan_hash": plan_hash,
            "config_hash": identity["config_hash"],
            "factorial_manifest_hash": factorial_hash,
            "synthetic_config_hash": identity["synthetic_config_hash"],
            "variant": identity["variant"],
            "generator_seed": identity["generator_seed"],
            "model_seed": identity["model_seed"],
            "gpus": gpu_ids,
            "command": command,
            "run_log": "run.log",
        }
        metadata_path = result_dir / "identity_metadata.json"
        _write_json_atomic(metadata_path, metadata)
        try:
            wall_seconds = _run_logged(
                command,
                result_dir / "run.log",
                environment,
                timeout_seconds,
            )
            if not _validate_completed_identity(
                result_dir,
                identity,
                plan_hash,
                factorial_hash,
            ):
                # Validation expects completed metadata, so validate artifacts first here.
                summary = _load_json(result_dir / "contextual_evaluation.json")
                score_metadata = _load_json(
                    result_dir / "scores/score_run_metadata.json"
                )
                if not (
                    summary.get("config_hash") == identity["synthetic_config_hash"]
                    and score_metadata.get("factorial_manifest_hash")
                    == factorial_hash
                    and score_metadata.get("variant") == identity["variant"]
                    and int(score_metadata.get("seed", -1))
                    == int(identity["model_seed"])
                    and _canonical_hash(score_metadata.get("hyperparameters"))
                    == identity["hyperparameters_hash"]
                ):
                    raise RuntimeError("Evaluator outputs failed frozen identity checks.")
            metadata.update(
                {
                    "status": "completed",
                    "completed_at": _utc_now(),
                    "runner_wall_seconds": wall_seconds,
                    "output_hashes": {
                        name: _file_sha256(result_dir / name)
                        for name in (
                            "contextual_evaluation.json",
                            "mechanism_metrics.csv",
                            "matched_orderings.csv",
                            "score_component_orderings.csv",
                            "scores/score_run_metadata.json",
                        )
                    },
                }
            )
            _write_json_atomic(metadata_path, metadata)
            if not _validate_completed_identity(
                result_dir, identity, plan_hash, factorial_hash
            ):
                raise RuntimeError("Completed identity failed final validation.")
        except BaseException as error:
            metadata.update(
                {
                    "status": "interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "failed",
                    "completed_at": _utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            _write_json_atomic(metadata_path, metadata)
            if isinstance(error, KeyboardInterrupt):
                raise
            failures.append(
                (
                    identity["variant"],
                    identity["generator_seed"],
                    identity["model_seed"],
                    str(error),
                )
            )
            print(
                "FAILED: {} generator={} model={}: {}".format(*failures[-1]),
                file=sys.stderr,
            )
            if args.fail_fast:
                break

    if args.dry_run:
        print("Dry run complete; no files or subprocesses were created.")
        return 0
    if failures:
        print(f"Completed with {len(failures)} failed identity(s).", file=sys.stderr)
        return 1
    print(f"Completed {len(jobs)} identity run(s); skipped {completed} completed run(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
