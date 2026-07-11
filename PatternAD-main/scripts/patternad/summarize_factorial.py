#!/usr/bin/env python3
"""Summarize detailed PatternAD CSV tar artifacts without test-oracle selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tarfile
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "patternad" / "factorial_ablation.json"
IDENTITY_FIELDS = ("run_name", "group", "family", "entity", "seed")


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


def _load_run_plan(input_root: Path) -> Dict[str, Any]:
    plan_paths = sorted(input_root.rglob("run_plan.json"))
    if len(plan_paths) != 1:
        raise ValueError(
            f"Expected exactly one run_plan.json below {input_root}, found "
            f"{len(plan_paths)}. Summarize one frozen run tree at a time."
        )
    plan = _load_json(plan_paths[0])
    required = {"plan_hash", "expected_identities", "run_name", "group"}
    missing = sorted(required - set(plan))
    if missing:
        raise ValueError(f"{plan_paths[0]} is missing plan fields: {missing}")
    return plan


def _read_detail_rows(tar_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with tarfile.open(tar_path, mode="r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".csv")
        ]
        if not members:
            raise ValueError(f"No CSV member found in {tar_path}.")
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not read {member.name} in {tar_path}.")
            decoded_lines = (line.decode("utf-8-sig") for line in extracted)
            rows.extend(csv.DictReader(decoded_lines))
    if not rows:
        raise ValueError(f"Detailed result contains no rows: {tar_path}")
    return rows


def _artifact_for_attempt(attempt_dir: Path, metadata: Mapping[str, Any]) -> Path:
    recorded = metadata.get("detail_artifact")
    if recorded:
        artifact = attempt_dir / str(recorded)
        if not artifact.is_file():
            raise FileNotFoundError(
                f"Recorded detailed artifact does not exist: {artifact}"
            )
        return artifact
    artifacts = sorted(attempt_dir.glob("*.csv.tar.gz"))
    if len(artifacts) != 1:
        raise ValueError(
            f"Expected one detailed CSV tar in {attempt_dir}, found {len(artifacts)}."
        )
    return artifacts[0]


def _parse_finite(raw: str, metric: str, artifact: Path) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Metric {metric} is missing or non-numeric in {artifact}: {raw!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"Metric {metric} is not finite in {artifact}: {raw!r}")
    return value


def _constant_score_metric(
    rows: Sequence[Mapping[str, str]], metric: str, artifact: Path
) -> float:
    """Return a score metric only when every threshold row agrees.

    Rank/score metrics are independent of the threshold. Historical detailed files
    repeat them for each anomaly ratio. We validate equality and retain the first
    value. We never optimize or take a maximum across threshold rows.
    """

    if metric not in rows[0]:
        raise ValueError(f"Metric column {metric!r} is absent from {artifact}.")
    values = [_parse_finite(row.get(metric, ""), metric, artifact) for row in rows]
    reference = values[0]
    inconsistent = [
        value
        for value in values[1:]
        if not math.isclose(value, reference, rel_tol=1e-12, abs_tol=1e-12)
    ]
    if inconsistent:
        raise ValueError(
            f"Score metric {metric} varies across anomaly-ratio rows in {artifact}. "
            "Refusing to select a test result."
        )
    return reference


def _verify_run_rows(
    rows: Sequence[Mapping[str, str]], metadata: Mapping[str, Any], artifact: Path
) -> None:
    errors = [
        row.get("log_info", "").strip()
        for row in rows
        if row.get("log_info", "").strip()
    ]
    if errors:
        raise ValueError(
            f"Benchmark error row in {artifact}: {errors[0].splitlines()[0]}"
        )
    file_names = {row.get("file_name", "") for row in rows}
    if file_names != {metadata["data_name"]}:
        raise ValueError(
            f"Artifact/metadata dataset mismatch in {artifact}: "
            f"rows={sorted(file_names)}, metadata={metadata['data_name']!r}."
        )
    for row in rows:
        try:
            strategy_args = json.loads(row.get("strategy_args", "{}"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid strategy_args JSON in {artifact}.") from error
        if int(strategy_args.get("seed", -1)) != int(metadata["seed"]):
            raise ValueError(
                f"Artifact/metadata seed mismatch in {artifact}: "
                f"{strategy_args.get('seed')} != {metadata['seed']}."
            )
    diagnostic_payloads = {
        row.get("model_diagnostics", "").strip()
        for row in rows
        if row.get("model_diagnostics", "").strip()
    }
    if len(diagnostic_payloads) != 1:
        raise ValueError(
            f"Artifact diagnostics are missing or inconsistent in {artifact}."
        )
    try:
        artifact_diagnostics = json.loads(next(iter(diagnostic_payloads)))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid model diagnostics JSON in {artifact}.") from error
    if artifact_diagnostics != metadata.get("model_diagnostics"):
        raise ValueError(
            f"Artifact/metadata model diagnostics mismatch in {artifact}."
        )


def _flatten_run_diagnostics(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    diagnostics = metadata["model_diagnostics"]
    training = diagnostics["training"]
    calls = {call["phase"]: call for call in diagnostics["score_calls"]}
    flattened: Dict[str, Any] = {
        "runner_wall_seconds": metadata["runner_wall_seconds"],
        "fit_seconds": training["fit_seconds"],
        "training_seconds": training["training_seconds"],
        "scorer_fit_seconds": training["scorer_fit_seconds"],
        "epochs_requested": training["epochs_requested"],
        "epochs_completed": training["epochs_completed"],
        "best_epoch": training["best_epoch"],
        "best_validation_loss": training["best_validation_loss"],
        "stopped_early": training["stopped_early"],
        "parameter_count": training["parameter_count"],
        "optimization_train_points": training["optimization_train_points"],
        "validation_points": training["validation_points"],
        "benchmark_log": str(metadata["benchmark_log"]),
    }
    for name in ("scorer_reference_points",):
        if name in training:
            flattened[name] = training[name]
    fit_partition = training.get("fit_partition")
    if isinstance(fit_partition, dict):
        for name in (
            "reference_source",
            "inter_partition_gap_points",
            "validation_fraction",
            "reference_fraction",
        ):
            if name in fit_partition:
                flattened[f"fit_{name}"] = fit_partition[name]
    score_calibration = diagnostics.get("score_calibration")
    if isinstance(score_calibration, dict):
        for name in ("global_count", "minimum_bin_size", "shrinkage"):
            if name in score_calibration:
                flattened[f"score_calibration_{name}"] = score_calibration[name]
    for phase in ("calibration", "test"):
        call = calls[phase]
        flattened[f"{phase}_input_length"] = call["input_length"]
        flattened[f"{phase}_score_seconds"] = call["elapsed_seconds"]
        flattened[f"{phase}_score_min"] = call["score"]["min"]
        flattened[f"{phase}_score_max"] = call["score"]["max"]
        flattened[f"{phase}_score_mean"] = call["score"]["mean"]
        scale = call["scale"]
        for statistic in (
            "min",
            "max",
            "mean",
            "std",
            "lower_bound_fraction",
            "upper_bound_fraction",
        ):
            flattened[f"{phase}_scale_{statistic}"] = (
                "" if scale is None else scale[statistic]
            )
    return flattened


def _scan_completed_runs(
    input_root: Path, score_metrics: Sequence[str]
) -> List[Dict[str, Any]]:
    metadata_paths = sorted(input_root.rglob("run_metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"No run_metadata.json files found below {input_root}. "
            "Point --input at a run_factorial_ablation.py result tree."
        )

    result_rows = []
    seen_identities: Dict[Tuple[Any, ...], Path] = {}
    for metadata_path in metadata_paths:
        metadata = _load_json(metadata_path)
        if metadata.get("status") != "completed":
            continue
        required = {
            "run_name",
            "group",
            "family",
            "entity",
            "dataset_id",
            "data_name",
            "variant",
            "factors",
            "seed",
            "config_hash",
            "plan_hash",
            "runner_wall_seconds",
            "benchmark_log",
            "model_diagnostics",
        }
        missing = sorted(required - set(metadata))
        if missing:
            raise ValueError(f"{metadata_path} is missing metadata fields: {missing}")
        identity = tuple(metadata[field] for field in IDENTITY_FIELDS) + (
            metadata["variant"],
        )
        if identity in seen_identities:
            raise ValueError(
                "Multiple completed artifacts share one entity/seed/variant identity: "
                f"{seen_identities[identity]} and {metadata_path}."
            )
        seen_identities[identity] = metadata_path

        artifact = _artifact_for_attempt(metadata_path.parent, metadata)
        rows = _read_detail_rows(artifact)
        _verify_run_rows(rows, metadata, artifact)
        ratio_values = {
            row.get("typical_anomaly_ratio", "")
            for row in rows
            if row.get("typical_anomaly_ratio", "") != ""
        }
        try:
            ratios = sorted(ratio_values, key=float)
        except ValueError:
            ratios = sorted(ratio_values)
        factors = metadata["factors"]
        result: Dict[str, Any] = {
            "run_name": metadata["run_name"],
            "group": metadata["group"],
            "family": metadata["family"],
            "entity": metadata["entity"],
            "dataset_id": metadata["dataset_id"],
            "data_name": metadata["data_name"],
            "variant": metadata["variant"],
            "seed": int(metadata["seed"]),
            "context": factors["context"],
            "distribution": factors["distribution"],
            "mask": factors["mask"],
            "config_hash": metadata["config_hash"],
            "plan_hash": metadata["plan_hash"],
            "source_threshold_rows": len(rows),
            "source_anomaly_ratios": ";".join(ratios),
            "detail_artifact": str(artifact),
        }
        result.update(_flatten_run_diagnostics(metadata))
        for metric in score_metrics:
            result[metric] = _constant_score_metric(rows, metric, artifact)
        result_rows.append(result)

    if not result_rows:
        raise ValueError(f"No completed runs found below {input_root}.")
    return sorted(
        result_rows,
        key=lambda row: (
            row["run_name"],
            row["group"],
            row["family"],
            row["entity"],
            row["variant"],
            row["seed"],
        ),
    )


def _verify_complete_plan(
    rows: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]
) -> None:
    expected = {
        (item["dataset_id"], item["variant"], int(item["seed"])): item[
            "config_hash"
        ]
        for item in plan["expected_identities"]
    }
    if len(expected) != len(plan["expected_identities"]):
        raise ValueError("run_plan.json contains duplicate expected identities.")
    actual = {
        (row["dataset_id"], row["variant"], int(row["seed"])): row["config_hash"]
        for row in rows
    }
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        key for key in set(expected) & set(actual) if expected[key] != actual[key]
    )
    bad_plan_hash = [
        row for row in rows if row.get("plan_hash") != plan["plan_hash"]
    ]
    if missing or unexpected or mismatched or bad_plan_hash:
        raise ValueError(
            "Frozen run is incomplete or inconsistent; refusing a selective summary. "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"config_mismatch={mismatched[:5]}, plan_hash_mismatch={len(bad_plan_hash)}."
        )


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("Cannot average an empty group.")
    return math.fsum(materialized) / len(materialized)


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
    metrics: Sequence[str],
    count_field: str,
    count_key: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group_rows in sorted(grouped.items()):
        aggregate_row = dict(zip(group_fields, key))
        aggregate_row[count_field] = len({row[count_key] for row in group_rows})
        for metric in metrics:
            aggregate_row[metric] = _mean(float(row[metric]) for row in group_rows)
        output.append(aggregate_row)
    return output


def _paired_deltas(
    rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, str]],
    metrics: Sequence[str],
) -> List[Dict[str, Any]]:
    index: Dict[Tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(row[field] for field in IDENTITY_FIELDS) + (row["variant"],)
        if key in index:
            raise ValueError(f"Duplicate paired index: {key}")
        index[key] = row

    base_identities = sorted(
        {tuple(row[field] for field in IDENTITY_FIELDS) for row in rows}
    )
    output = []
    for comparison in comparisons:
        lhs = comparison["lhs"]
        rhs = comparison["rhs"]
        for identity in base_identities:
            lhs_row = index.get(identity + (lhs,))
            rhs_row = index.get(identity + (rhs,))
            if lhs_row is None or rhs_row is None:
                continue
            paired = dict(zip(IDENTITY_FIELDS, identity))
            paired.update(
                {
                    "comparison": comparison["name"],
                    "lhs": lhs,
                    "rhs": rhs,
                    "lhs_config_hash": lhs_row["config_hash"],
                    "rhs_config_hash": rhs_row["config_hash"],
                }
            )
            for metric in metrics:
                paired[metric] = float(lhs_row[metric]) - float(rhs_row[metric])
            output.append(paired)
    return sorted(
        output,
        key=lambda row: (
            row["run_name"],
            row["group"],
            row["comparison"],
            row["family"],
            row["entity"],
            row["seed"],
        ),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {
                field: format(row[field], ".17g")
                if isinstance(row.get(field), float)
                else row.get(field, "")
                for field in fields
            }
            writer.writerow(serialized)
    os.replace(temporary, path)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize detailed CSV tar files. Only threshold-independent score "
            "metrics are accepted; no test-oracle maximum is computed."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Factorial result tree.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Summary directory; defaults to INPUT/summary.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    input_root = args.input.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    plan = _load_run_plan(input_root)
    manifest = _load_json(args.manifest.resolve())
    if _canonical_hash(manifest) != plan.get("manifest_hash"):
        raise ValueError(
            "The summarizer manifest does not match the frozen run plan. Use the "
            "exact manifest recorded for this run."
        )
    metrics = manifest.get("score_metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("Manifest score_metrics must be a non-empty list.")
    comparisons = manifest.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("Manifest comparisons must be a list.")

    entity_rows = _scan_completed_runs(input_root, metrics)
    _verify_complete_plan(entity_rows, plan)
    family_seed_rows = _aggregate(
        entity_rows,
        ("run_name", "group", "family", "variant", "seed"),
        metrics,
        "n_entities",
        "entity",
    )
    family_rows = _aggregate(
        family_seed_rows,
        ("run_name", "group", "family", "variant"),
        metrics,
        "n_seeds",
        "seed",
    )
    overall_family_seed_rows = _aggregate(
        family_seed_rows,
        ("run_name", "group", "variant", "seed"),
        metrics,
        "n_families",
        "family",
    )

    paired_rows = _paired_deltas(entity_rows, comparisons, metrics)
    paired_family_seed_rows = _aggregate(
        paired_rows,
        ("run_name", "group", "family", "comparison", "lhs", "rhs", "seed"),
        metrics,
        "n_entities",
        "entity",
    )
    paired_family_rows = _aggregate(
        paired_family_seed_rows,
        ("run_name", "group", "family", "comparison", "lhs", "rhs"),
        metrics,
        "n_seeds",
        "seed",
    )
    paired_overall_rows = _aggregate(
        paired_family_rows,
        ("run_name", "group", "comparison", "lhs", "rhs"),
        metrics,
        "n_families",
        "family",
    )

    output_dir = (args.output_dir or (input_root / "summary")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entity_fields = [
        "run_name",
        "group",
        "family",
        "entity",
        "dataset_id",
        "data_name",
        "variant",
        "seed",
        "context",
        "distribution",
        "mask",
        "config_hash",
        "plan_hash",
        "source_threshold_rows",
        "source_anomaly_ratios",
        "detail_artifact",
        *metrics,
    ]
    _write_csv(
        output_dir / "entity_seed_score_metrics.csv", entity_rows, entity_fields
    )
    diagnostic_fields = [
        "run_name",
        "group",
        "family",
        "entity",
        "dataset_id",
        "variant",
        "seed",
        "context",
        "distribution",
        "mask",
        "config_hash",
        "plan_hash",
        "runner_wall_seconds",
        "fit_seconds",
        "training_seconds",
        "scorer_fit_seconds",
        "epochs_requested",
        "epochs_completed",
        "best_epoch",
        "best_validation_loss",
        "stopped_early",
        "parameter_count",
        "calibration_input_length",
        "calibration_score_seconds",
        "calibration_score_min",
        "calibration_score_max",
        "calibration_score_mean",
        "calibration_scale_min",
        "calibration_scale_max",
        "calibration_scale_mean",
        "calibration_scale_std",
        "calibration_scale_lower_bound_fraction",
        "calibration_scale_upper_bound_fraction",
        "test_input_length",
        "test_score_seconds",
        "test_score_min",
        "test_score_max",
        "test_score_mean",
        "test_scale_min",
        "test_scale_max",
        "test_scale_mean",
        "test_scale_std",
        "test_scale_lower_bound_fraction",
        "test_scale_upper_bound_fraction",
        "benchmark_log",
    ]
    _write_csv(
        output_dir / "entity_seed_run_diagnostics.csv",
        entity_rows,
        diagnostic_fields,
    )
    _write_csv(
        output_dir / "family_seed_macro.csv",
        family_seed_rows,
        ["run_name", "group", "family", "variant", "seed", "n_entities", *metrics],
    )
    _write_csv(
        output_dir / "family_macro.csv",
        family_rows,
        ["run_name", "group", "family", "variant", "n_seeds", *metrics],
    )
    _write_csv(
        output_dir / "overall_family_seed_macro.csv",
        overall_family_seed_rows,
        ["run_name", "group", "variant", "seed", "n_families", *metrics],
    )
    _write_csv(
        output_dir / "paired_entity_seed_delta.csv",
        paired_rows,
        [
            "run_name",
            "group",
            "family",
            "entity",
            "seed",
            "comparison",
            "lhs",
            "rhs",
            "lhs_config_hash",
            "rhs_config_hash",
            *metrics,
        ],
    )
    _write_csv(
        output_dir / "paired_family_seed_delta.csv",
        paired_family_seed_rows,
        [
            "run_name",
            "group",
            "family",
            "comparison",
            "lhs",
            "rhs",
            "seed",
            "n_entities",
            *metrics,
        ],
    )
    _write_csv(
        output_dir / "paired_family_macro_delta.csv",
        paired_family_rows,
        [
            "run_name",
            "group",
            "family",
            "comparison",
            "lhs",
            "rhs",
            "n_seeds",
            *metrics,
        ],
    )
    _write_csv(
        output_dir / "paired_overall_family_delta.csv",
        paired_overall_rows,
        [
            "run_name",
            "group",
            "comparison",
            "lhs",
            "rhs",
            "n_families",
            *metrics,
        ],
    )
    print(
        f"Wrote {len(entity_rows)} entity/seed rows and {len(paired_rows)} paired "
        f"rows to {output_dir}."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
