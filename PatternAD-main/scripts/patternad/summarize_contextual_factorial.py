#!/usr/bin/env python3
"""Strictly summarize a frozen PatternAD contextual factorial run."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MAIN_VARIANTS = ("A00", "A10", "A01", "A11")
MECHANISMS = (
    "same_deviation_different_context",
    "slow_drift_vs_abrupt_shift",
    "dependency_break",
    "context_ood",
)
ORDERING_MECHANISMS = (
    "same_deviation_different_context",
    "slow_drift_vs_abrupt_shift",
)
ROOT_METRICS = (
    "macro_average_precision",
    "matched_ordering_rate",
    "maximum_regime_fpr_gap",
)
MECHANISM_METRICS = (
    "average_precision",
    "ap_over_prevalence",
    "regime_fpr_gap",
)
METRICS = (
    ROOT_METRICS
    + tuple(
        f"{mechanism}_{metric}"
        for mechanism in MECHANISMS
        for metric in MECHANISM_METRICS
    )
    + tuple(f"{mechanism}_matched_ordering_rate" for mechanism in ORDERING_MECHANISMS)
)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
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


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized: Dict[str, Any] = {}
            for field in fields:
                value = row.get(field, "")
                if value is None:
                    value = ""
                elif isinstance(value, float):
                    value = format(value, ".17g")
                elif isinstance(value, bool):
                    value = "true" if value else "false"
                serialized[field] = value
            writer.writerow(serialized)
    os.replace(temporary, path)


def _required_fields(
    value: Mapping[str, Any], fields: Iterable[str], source: Path
) -> None:
    missing = sorted(field for field in fields if field not in value)
    if missing:
        raise ValueError(f"{source} is missing required fields: {missing}")


def _integer(value: Any, field: str, source: Path) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer in {source}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field} must be an integer in {source}: {value!r}"
        ) from error
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer in {source}: {value!r}")
    return parsed


def _finite(value: Any, field: str, source: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Metric {field} is not numeric in {source}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Metric {field} is missing or non-numeric in {source}: {value!r}"
        ) from error
    if not math.isfinite(parsed):
        raise ValueError(f"Metric {field} is not finite in {source}: {value!r}")
    return parsed


def _bounded(value: Any, field: str, source: Path, lower: float, upper: float) -> float:
    parsed = _finite(value, field, source)
    if not lower <= parsed <= upper:
        raise ValueError(
            f"Metric {field} is outside [{lower}, {upper}] in {source}: {parsed}"
        )
    return parsed


def _resolve_recorded_path(input_root: Path, raw: Any, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"run_plan.json field {field} must be a non-empty path.")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (input_root / path).resolve()


def _resolved_config_hash(
    base_synthetic_config: Mapping[str, Any], generator_seed: int
) -> str:
    resolved = copy.deepcopy(dict(base_synthetic_config))
    resolved["seed"] = int(generator_seed)
    return _canonical_hash(resolved)


def _load_frozen_inputs(
    input_root: Path,
) -> Tuple[Dict[str, Any], Path, Dict[str, Any], Path, Dict[str, Any], Path]:
    plan_path = input_root / "run_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(
            f"Missing run plan: {plan_path}. Point --input at one contextual run root."
        )
    plan = _load_json(plan_path)
    _required_fields(
        plan,
        (
            "phase",
            "plan_hash",
            "expected_identities",
            "synthetic_config_path",
            "synthetic_config_hash",
            "factorial_manifest_path",
            "factorial_manifest_hash",
        ),
        plan_path,
    )
    if (
        not isinstance(plan["expected_identities"], list)
        or not plan["expected_identities"]
    ):
        raise ValueError("run_plan.json expected_identities must be a non-empty list.")
    if not isinstance(plan["plan_hash"], str) or not plan["plan_hash"]:
        raise ValueError("run_plan.json plan_hash must be a non-empty string.")
    plan_core = {
        key: value
        for key, value in plan.items()
        if key not in {"plan_hash", "created_at"}
    }
    observed_plan_hash = _canonical_hash(plan_core)
    if observed_plan_hash != plan["plan_hash"]:
        raise ValueError(
            "run_plan.json plan_hash does not match its canonical plan core: "
            f"{observed_plan_hash} != {plan['plan_hash']}."
        )

    synthetic_path = _resolve_recorded_path(
        input_root, plan["synthetic_config_path"], "synthetic_config_path"
    )
    factorial_path = _resolve_recorded_path(
        input_root, plan["factorial_manifest_path"], "factorial_manifest_path"
    )
    if not synthetic_path.is_file():
        raise FileNotFoundError(
            f"Frozen synthetic config does not exist: {synthetic_path}"
        )
    if not factorial_path.is_file():
        raise FileNotFoundError(
            f"Frozen factorial manifest does not exist: {factorial_path}"
        )
    synthetic = _load_json(synthetic_path)
    factorial = _load_json(factorial_path)
    observed_synthetic_hash = _canonical_hash(synthetic)
    observed_factorial_hash = _canonical_hash(factorial)
    if observed_synthetic_hash != plan["synthetic_config_hash"]:
        raise ValueError(
            "Frozen synthetic config hash differs from run_plan.json: "
            f"{observed_synthetic_hash} != {plan['synthetic_config_hash']}."
        )
    if observed_factorial_hash != plan["factorial_manifest_hash"]:
        raise ValueError(
            "Frozen factorial manifest hash differs from run_plan.json: "
            f"{observed_factorial_hash} != {plan['factorial_manifest_hash']}."
        )
    return plan, plan_path, synthetic, synthetic_path, factorial, factorial_path


def _normalize_expected_identities(
    input_root: Path,
    plan: Mapping[str, Any],
    plan_path: Path,
    synthetic: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    keys: Dict[Tuple[str, int, int], int] = {}
    result_paths: Dict[Path, int] = {}
    root_resolved = input_root.resolve()
    for index, raw in enumerate(plan["expected_identities"]):
        if not isinstance(raw, dict):
            raise ValueError(
                f"run_plan.json expected_identities[{index}] must be an object."
            )
        _required_fields(
            raw,
            (
                "variant",
                "generator_seed",
                "model_seed",
                "result_dir",
                "config_hash",
            ),
            plan_path,
        )
        variant = raw["variant"]
        if not isinstance(variant, str) or not variant:
            raise ValueError(
                f"expected_identities[{index}].variant must be a non-empty string."
            )
        generator_seed = _integer(
            raw["generator_seed"],
            f"expected_identities[{index}].generator_seed",
            plan_path,
        )
        model_seed = _integer(
            raw["model_seed"],
            f"expected_identities[{index}].model_seed",
            plan_path,
        )
        if not isinstance(raw["result_dir"], str) or not raw["result_dir"]:
            raise ValueError(
                f"expected_identities[{index}].result_dir must be a relative path."
            )
        relative_dir = Path(raw["result_dir"])
        if relative_dir.is_absolute():
            raise ValueError(
                f"expected_identities[{index}].result_dir must be relative: "
                f"{relative_dir}"
            )
        result_path = (input_root / relative_dir).resolve()
        try:
            result_path.relative_to(root_resolved)
        except ValueError as error:
            raise ValueError(
                f"expected_identities[{index}].result_dir escapes the run root: "
                f"{relative_dir}"
            ) from error
        if result_path == root_resolved:
            raise ValueError(
                f"expected_identities[{index}].result_dir cannot be the run root."
            )
        config_hash = raw["config_hash"]
        if not isinstance(config_hash, str) or not config_hash:
            raise ValueError(
                f"expected_identities[{index}].config_hash must be non-empty."
            )
        identity_key = (variant, generator_seed, model_seed)
        if identity_key in keys:
            raise ValueError(
                "run_plan.json contains duplicate expected identity "
                f"{identity_key} at indexes {keys[identity_key]} and {index}."
            )
        if result_path in result_paths:
            raise ValueError(
                "run_plan.json assigns one result_dir to multiple identities at "
                f"indexes {result_paths[result_path]} and {index}: {relative_dir}"
            )
        expected_synthetic_hash = _resolved_config_hash(synthetic, generator_seed)
        recorded_synthetic_hash = raw.get("synthetic_config_hash")
        if (
            recorded_synthetic_hash is not None
            and recorded_synthetic_hash != expected_synthetic_hash
        ):
            raise ValueError(
                f"Expected identity {identity_key} has a stale synthetic_config_hash."
            )
        hyperparameters_hash = raw.get("hyperparameters_hash")
        if hyperparameters_hash is not None and (
            not isinstance(hyperparameters_hash, str) or not hyperparameters_hash
        ):
            raise ValueError(
                f"Expected identity {identity_key} has an invalid "
                "hyperparameters_hash."
            )
        keys[identity_key] = index
        result_paths[result_path] = index
        normalized.append(
            {
                "variant": variant,
                "generator_seed": generator_seed,
                "model_seed": model_seed,
                "result_dir": relative_dir.as_posix(),
                "result_path": result_path,
                "config_hash": config_hash,
                "synthetic_config_hash": expected_synthetic_hash,
                "hyperparameters_hash": hyperparameters_hash,
            }
        )

    normalized.sort(
        key=lambda row: (
            row["variant"],
            row["generator_seed"],
            row["model_seed"],
        )
    )
    return normalized


def _verify_identity_metadata_set(
    input_root: Path, identities: Sequence[Mapping[str, Any]]
) -> None:
    expected = {
        (Path(identity["result_path"]) / "identity_metadata.json").resolve()
        for identity in identities
    }
    actual = {path.resolve() for path in input_root.rglob("identity_metadata.json")}
    missing = sorted(str(path) for path in expected - actual)
    unexpected = sorted(str(path) for path in actual - expected)
    if missing or unexpected:
        raise ValueError(
            "Frozen contextual run has missing or unexpected identity metadata; "
            "refusing a selective summary. "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}."
        )


def _validate_recorded_output_hashes(
    result_dir: Path, metadata: Mapping[str, Any], metadata_path: Path
) -> None:
    hashes = metadata.get("output_hashes")
    if hashes is None:
        return
    if not isinstance(hashes, dict):
        raise ValueError(f"output_hashes must be an object in {metadata_path}.")
    for raw_name, expected_hash in sorted(hashes.items()):
        if not isinstance(raw_name, str) or not isinstance(expected_hash, str):
            raise ValueError(f"Invalid output_hashes entry in {metadata_path}.")
        output_path = (result_dir / raw_name).resolve()
        try:
            output_path.relative_to(result_dir.resolve())
        except ValueError as error:
            raise ValueError(
                f"Recorded output hash escapes its identity directory: {raw_name!r}"
            ) from error
        if not output_path.is_file():
            raise FileNotFoundError(
                f"Recorded completed output does not exist: {output_path}"
            )
        observed_hash = _file_sha256(output_path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Recorded output hash mismatch for {output_path}: "
                f"{observed_hash} != {expected_hash}."
            )


def _flatten_metrics(
    evaluation: Mapping[str, Any], source: Path, result_dir: Path
) -> Dict[str, float]:
    result = {
        "macro_average_precision": _bounded(
            evaluation.get("macro_average_precision"),
            "macro_average_precision",
            source,
            0.0,
            1.0,
        ),
        "matched_ordering_rate": _bounded(
            evaluation.get("matched_ordering_rate"),
            "matched_ordering_rate",
            source,
            0.0,
            1.0,
        ),
        "maximum_regime_fpr_gap": _bounded(
            evaluation.get("maximum_regime_fpr_gap"),
            "maximum_regime_fpr_gap",
            source,
            0.0,
            1.0,
        ),
    }
    mechanism_rows = evaluation.get("mechanisms")
    if not isinstance(mechanism_rows, list):
        raise ValueError(f"mechanisms must be a list in {source}.")
    mechanism_index: Dict[str, Mapping[str, Any]] = {}
    for row_index, row in enumerate(mechanism_rows):
        if not isinstance(row, dict) or not isinstance(row.get("mechanism"), str):
            raise ValueError(f"Invalid mechanisms[{row_index}] in {source}.")
        mechanism = row["mechanism"]
        if mechanism in mechanism_index:
            raise ValueError(f"Duplicate mechanism {mechanism!r} in {source}.")
        mechanism_index[mechanism] = row
    missing = sorted(set(MECHANISMS) - set(mechanism_index))
    unexpected = sorted(set(mechanism_index) - set(MECHANISMS))
    if missing or unexpected:
        raise ValueError(
            f"Contextual mechanism set mismatch in {source}: "
            f"missing={missing}, unexpected={unexpected}."
        )

    for mechanism in MECHANISMS:
        row = mechanism_index[mechanism]
        score_path = result_dir / "scores" / f"{mechanism}.npz"
        if not score_path.is_file():
            raise FileNotFoundError(f"Missing aligned mechanism score: {score_path}")
        recorded_score_path = row.get("score_file")
        if not isinstance(recorded_score_path, str) or not recorded_score_path:
            raise ValueError(f"Missing {mechanism}.score_file in {source}.")
        if Path(recorded_score_path).resolve() != score_path.resolve():
            raise ValueError(
                f"{mechanism}.score_file does not identify {score_path} in {source}."
            )
        recorded_score_hash = row.get("score_sha256")
        if not isinstance(recorded_score_hash, str) or not recorded_score_hash:
            raise ValueError(f"Missing {mechanism}.score_sha256 in {source}.")
        observed_score_hash = _file_sha256(score_path)
        if observed_score_hash != recorded_score_hash:
            raise ValueError(
                f"Score SHA-256 mismatch for {score_path}: "
                f"{observed_score_hash} != {recorded_score_hash}."
            )
        result[f"{mechanism}_average_precision"] = _bounded(
            row.get("average_precision"),
            f"{mechanism}.average_precision",
            source,
            0.0,
            1.0,
        )
        result[f"{mechanism}_ap_over_prevalence"] = _bounded(
            row.get("ap_over_prevalence"),
            f"{mechanism}.ap_over_prevalence",
            source,
            -1.0,
            1.0,
        )
        result[f"{mechanism}_regime_fpr_gap"] = _bounded(
            row.get("regime_fpr_gap"),
            f"{mechanism}.regime_fpr_gap",
            source,
            0.0,
            1.0,
        )
    for mechanism in ORDERING_MECHANISMS:
        result[f"{mechanism}_matched_ordering_rate"] = _bounded(
            mechanism_index[mechanism].get("matched_ordering_rate"),
            f"{mechanism}.matched_ordering_rate",
            source,
            0.0,
            1.0,
        )

    mechanism_macro = math.fsum(
        result[f"{mechanism}_average_precision"] for mechanism in MECHANISMS
    ) / len(MECHANISMS)
    mechanism_maximum_gap = max(
        result[f"{mechanism}_regime_fpr_gap"] for mechanism in MECHANISMS
    )
    if not math.isclose(
        mechanism_macro,
        result["macro_average_precision"],
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"macro_average_precision disagrees with mechanism rows in {source}."
        )
    if not math.isclose(
        mechanism_maximum_gap,
        result["maximum_regime_fpr_gap"],
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"maximum_regime_fpr_gap disagrees with mechanism rows in {source}."
        )

    orderings = evaluation.get("matched_orderings")
    if orderings is not None:
        if not isinstance(orderings, list) or not orderings:
            raise ValueError(f"matched_orderings must be a non-empty list in {source}.")
        correct: List[bool] = []
        by_mechanism: Dict[str, List[bool]] = {
            mechanism: [] for mechanism in ORDERING_MECHANISMS
        }
        for row_index, row in enumerate(orderings):
            if not isinstance(row, dict) or not isinstance(row.get("correct"), bool):
                raise ValueError(f"Invalid matched_orderings[{row_index}] in {source}.")
            mechanism = row.get("mechanism")
            if mechanism not in by_mechanism:
                raise ValueError(
                    f"Unexpected ordering mechanism {mechanism!r} in {source}."
                )
            correct.append(row["correct"])
            by_mechanism[mechanism].append(row["correct"])
        observed_rate = math.fsum(int(value) for value in correct) / len(correct)
        if not math.isclose(
            observed_rate,
            result["matched_ordering_rate"],
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"matched_ordering_rate disagrees with ordering rows in {source}."
            )
        for mechanism, values in by_mechanism.items():
            if not values:
                raise ValueError(
                    f"No matched ordering rows for {mechanism} in {source}."
                )
            observed = math.fsum(int(value) for value in values) / len(values)
            field = f"{mechanism}_matched_ordering_rate"
            if not math.isclose(observed, result[field], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"{field} disagrees with ordering rows in {source}.")
    return result


def _load_identity_metrics(
    identity: Mapping[str, Any], plan: Mapping[str, Any]
) -> Dict[str, Any]:
    result_dir = Path(identity["result_path"])
    metadata_path = result_dir / "identity_metadata.json"
    evaluation_path = result_dir / "contextual_evaluation.json"
    score_metadata_path = result_dir / "scores" / "score_run_metadata.json"
    metadata = _load_json(metadata_path)
    _required_fields(
        metadata,
        (
            "status",
            "plan_hash",
            "config_hash",
            "variant",
            "generator_seed",
            "model_seed",
        ),
        metadata_path,
    )
    key = (
        identity["variant"],
        identity["generator_seed"],
        identity["model_seed"],
    )
    if metadata["status"] != "completed":
        raise ValueError(
            f"Identity {key} is not completed in {metadata_path}: "
            f"status={metadata['status']!r}."
        )
    if metadata["plan_hash"] != plan["plan_hash"]:
        raise ValueError(f"Identity {key} has a plan_hash mismatch in {metadata_path}.")
    if metadata["config_hash"] != identity["config_hash"]:
        raise ValueError(
            f"Identity {key} has a config_hash mismatch in {metadata_path}."
        )
    observed_key = (
        metadata["variant"],
        _integer(metadata["generator_seed"], "generator_seed", metadata_path),
        _integer(metadata["model_seed"], "model_seed", metadata_path),
    )
    if observed_key != key:
        raise ValueError(
            f"Identity metadata/path mismatch in {metadata_path}: "
            f"observed={observed_key}, expected={key}."
        )
    for field, expected in (
        ("synthetic_config_hash", identity["synthetic_config_hash"]),
        ("factorial_manifest_hash", plan["factorial_manifest_hash"]),
    ):
        if field in metadata and metadata[field] != expected:
            raise ValueError(
                f"Identity {key} has a {field} mismatch in {metadata_path}."
            )
    _validate_recorded_output_hashes(result_dir, metadata, metadata_path)

    if not evaluation_path.is_file():
        raise FileNotFoundError(f"Missing contextual evaluation: {evaluation_path}")
    if not score_metadata_path.is_file():
        raise FileNotFoundError(f"Missing score run metadata: {score_metadata_path}")
    evaluation = _load_json(evaluation_path)
    score_metadata = _load_json(score_metadata_path)
    expected_method = f"{identity['variant']}_seed_{identity['model_seed']}"
    if evaluation.get("method") != expected_method:
        raise ValueError(
            f"Evaluation method/identity mismatch in {evaluation_path}: "
            f"{evaluation.get('method')!r} != {expected_method!r}."
        )
    if evaluation.get("config_hash") != identity["synthetic_config_hash"]:
        raise ValueError(f"Synthetic config hash mismatch in {evaluation_path}.")
    if "variant" in evaluation and evaluation["variant"] != identity["variant"]:
        raise ValueError(f"Variant mismatch in {evaluation_path}.")
    if (
        "model_seed" in evaluation
        and _integer(evaluation["model_seed"], "model_seed", evaluation_path)
        != identity["model_seed"]
    ):
        raise ValueError(f"Model seed mismatch in {evaluation_path}.")

    _required_fields(
        score_metadata,
        ("variant", "seed", "synthetic_config_hash", "factorial_manifest_hash"),
        score_metadata_path,
    )
    if score_metadata["variant"] != identity["variant"]:
        raise ValueError(f"Variant mismatch in {score_metadata_path}.")
    if (
        _integer(score_metadata["seed"], "seed", score_metadata_path)
        != identity["model_seed"]
    ):
        raise ValueError(f"Model seed mismatch in {score_metadata_path}.")
    if score_metadata["synthetic_config_hash"] != identity["synthetic_config_hash"]:
        raise ValueError(f"Synthetic config hash mismatch in {score_metadata_path}.")
    if score_metadata["factorial_manifest_hash"] != plan["factorial_manifest_hash"]:
        raise ValueError(f"Factorial manifest hash mismatch in {score_metadata_path}.")
    if identity.get("hyperparameters_hash") is not None:
        hyperparameters = score_metadata.get("hyperparameters")
        if not isinstance(hyperparameters, dict):
            raise ValueError(
                f"Missing hyperparameters object in {score_metadata_path}."
            )
        observed_hyperparameters_hash = _canonical_hash(hyperparameters)
        if observed_hyperparameters_hash != identity["hyperparameters_hash"]:
            raise ValueError(
                f"Hyperparameters hash mismatch in {score_metadata_path}: "
                f"{observed_hyperparameters_hash} != "
                f"{identity['hyperparameters_hash']}."
            )

    row: Dict[str, Any] = {
        "variant": identity["variant"],
        "generator_seed": identity["generator_seed"],
        "model_seed": identity["model_seed"],
        "config_hash": identity["config_hash"],
        "plan_hash": plan["plan_hash"],
        "synthetic_config_hash": identity["synthetic_config_hash"],
        "factorial_manifest_hash": plan["factorial_manifest_hash"],
        "result_dir": identity["result_dir"],
        "evaluation_file": str(evaluation_path),
        "score_metadata_file": str(score_metadata_path),
    }
    row.update(_flatten_metrics(evaluation, evaluation_path, result_dir))
    return row


def _comparison_definitions(
    factorial: Mapping[str, Any], running_variants: Sequence[str], source: Path
) -> List[Dict[str, str]]:
    raw_comparisons = factorial.get("comparisons")
    if not isinstance(raw_comparisons, list):
        raise ValueError(f"Factorial manifest comparisons must be a list in {source}.")
    running = set(running_variants)
    output: List[Dict[str, str]] = []
    names: set[str] = set()
    for index, comparison in enumerate(raw_comparisons):
        if not isinstance(comparison, dict):
            raise ValueError(f"comparisons[{index}] must be an object in {source}.")
        required = ("name", "lhs", "rhs")
        if any(
            not isinstance(comparison.get(field), str) or not comparison[field]
            for field in required
        ):
            raise ValueError(f"comparisons[{index}] is invalid in {source}.")
        name = comparison["name"]
        if name in names:
            raise ValueError(f"Duplicate comparison name {name!r} in {source}.")
        names.add(name)
        lhs = comparison["lhs"]
        rhs = comparison["rhs"]
        if lhs == rhs:
            raise ValueError(f"Comparison {name!r} has identical lhs and rhs.")
        if lhs in running and rhs in running:
            output.append({"name": name, "lhs": lhs, "rhs": rhs})
    return output


def _paired_deltas(
    rows: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, str]]
) -> List[Dict[str, Any]]:
    index: Dict[Tuple[str, int, int], Mapping[str, Any]] = {}
    axes_by_variant: Dict[str, set[Tuple[int, int]]] = {}
    for row in rows:
        key = (row["variant"], row["generator_seed"], row["model_seed"])
        if key in index:
            raise ValueError(f"Duplicate identity metrics row: {key}")
        index[key] = row
        axes_by_variant.setdefault(row["variant"], set()).add(
            (row["generator_seed"], row["model_seed"])
        )

    output: List[Dict[str, Any]] = []
    for comparison in comparisons:
        lhs = comparison["lhs"]
        rhs = comparison["rhs"]
        lhs_axes = axes_by_variant[lhs]
        rhs_axes = axes_by_variant[rhs]
        if lhs_axes != rhs_axes:
            missing_lhs = sorted(rhs_axes - lhs_axes)
            missing_rhs = sorted(lhs_axes - rhs_axes)
            raise ValueError(
                f"Comparison {comparison['name']!r} is not completely paired by "
                "generator/model seed; refusing selective deltas. "
                f"missing_lhs={missing_lhs[:5]}, missing_rhs={missing_rhs[:5]}."
            )
        for generator_seed, model_seed in sorted(lhs_axes):
            lhs_row = index[(lhs, generator_seed, model_seed)]
            rhs_row = index[(rhs, generator_seed, model_seed)]
            paired: Dict[str, Any] = {
                "comparison": comparison["name"],
                "lhs": lhs,
                "rhs": rhs,
                "generator_seed": generator_seed,
                "model_seed": model_seed,
                "lhs_config_hash": lhs_row["config_hash"],
                "rhs_config_hash": rhs_row["config_hash"],
                "plan_hash": lhs_row["plan_hash"],
            }
            for metric in METRICS:
                paired[metric] = float(lhs_row[metric]) - float(rhs_row[metric])
            output.append(paired)
    return output


def _stream_seed(base_seed: int, owner: str) -> int:
    payload = f"{int(base_seed)}\0{owner}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _is_crossed_grid(axes: Iterable[Tuple[int, int]]) -> bool:
    materialized = set(axes)
    if not materialized:
        return False
    generator_seeds = {key[0] for key in materialized}
    model_seeds = {key[1] for key in materialized}
    return materialized == {
        (generator_seed, model_seed)
        for generator_seed in generator_seeds
        for model_seed in model_seeds
    }


def _crossed_bootstrap_cube(
    cube: np.ndarray, n_bootstrap: int, stream_seed: int
) -> np.ndarray:
    """Resample generator and model axes independently and retain their crossing."""
    values = np.asarray(cube, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("Crossed bootstrap input must be a finite G x M x K cube.")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")
    n_generators, n_models, n_metrics = values.shape
    rng = np.random.default_rng(stream_seed)
    samples = np.empty((n_bootstrap, n_metrics), dtype=np.float64)
    chunk_size = min(1024, n_bootstrap)
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        count = stop - start
        if n_generators > 1:
            generator_draws = rng.integers(0, n_generators, size=(count, n_generators))
        else:
            generator_draws = np.zeros((count, 1), dtype=np.int64)
        if n_models > 1:
            model_draws = rng.integers(0, n_models, size=(count, n_models))
        else:
            model_draws = np.zeros((count, 1), dtype=np.int64)
        crossed = values[generator_draws[:, :, None], model_draws[:, None, :], :]
        samples[start:stop] = crossed.mean(axis=(1, 2))
    return samples


def _bootstrap_summaries(
    paired_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, str]],
    n_bootstrap: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], np.ndarray]]:
    rows_by_comparison: Dict[str, List[Mapping[str, Any]]] = {}
    for row in paired_rows:
        rows_by_comparison.setdefault(row["comparison"], []).append(row)
    summaries: List[Dict[str, Any]] = []
    samples_by_metric: Dict[Tuple[str, str], np.ndarray] = {}
    for comparison in comparisons:
        name = comparison["name"]
        rows = rows_by_comparison.get(name, [])
        axes = {(int(row["generator_seed"]), int(row["model_seed"])) for row in rows}
        generator_seeds = sorted({axis[0] for axis in axes})
        model_seeds = sorted({axis[1] for axis in axes})
        crossed = _is_crossed_grid(axes)
        if not rows:
            raise ValueError(f"Comparison {name!r} has no paired identities.")
        if not crossed:
            for metric in METRICS:
                values = [float(row[metric]) for row in rows]
                summaries.append(
                    {
                        "comparison": name,
                        "lhs": comparison["lhs"],
                        "rhs": comparison["rhs"],
                        "metric": metric,
                        "mean": math.fsum(values) / len(values),
                        "bootstrap_mean": None,
                        "std": None,
                        "ci_lower": None,
                        "ci_upper": None,
                        "n_generator_seeds": len(generator_seeds),
                        "n_model_seeds": len(model_seeds),
                        "n_pairs": len(rows),
                        "n_bootstrap": n_bootstrap,
                        "seed": seed,
                        "stream_seed": None,
                        "sampling": "unavailable_unbalanced_crossed_grid",
                        "status": "insufficient_data",
                    }
                )
            continue

        row_index = {
            (int(row["generator_seed"]), int(row["model_seed"])): row for row in rows
        }
        cube = np.asarray(
            [
                [
                    [
                        float(row_index[(generator_seed, model_seed)][metric])
                        for metric in METRICS
                    ]
                    for model_seed in model_seeds
                ]
                for generator_seed in generator_seeds
            ],
            dtype=np.float64,
        )
        derived_seed = _stream_seed(seed, f"comparison:{name}")
        bootstrap = _crossed_bootstrap_cube(cube, n_bootstrap, derived_seed)
        point_means = cube.mean(axis=(0, 1))
        for metric_index, metric in enumerate(METRICS):
            metric_samples = bootstrap[:, metric_index].copy()
            samples_by_metric[(name, metric)] = metric_samples
            summaries.append(
                {
                    "comparison": name,
                    "lhs": comparison["lhs"],
                    "rhs": comparison["rhs"],
                    "metric": metric,
                    "mean": float(point_means[metric_index]),
                    "bootstrap_mean": float(metric_samples.mean()),
                    "std": float(metric_samples.std(ddof=1)),
                    "ci_lower": float(np.quantile(metric_samples, 0.025)),
                    "ci_upper": float(np.quantile(metric_samples, 0.975)),
                    "n_generator_seeds": len(generator_seeds),
                    "n_model_seeds": len(model_seeds),
                    "n_pairs": len(rows),
                    "n_bootstrap": n_bootstrap,
                    "seed": seed,
                    "stream_seed": derived_seed,
                    "sampling": (
                        "generator_only"
                        if len(model_seeds) == 1 and len(generator_seeds) > 1
                        else "model_only"
                        if len(generator_seeds) == 1 and len(model_seeds) > 1
                        else "constant_single_cell"
                        if len(generator_seeds) == len(model_seeds) == 1
                        else "independent_generator_and_model_crossed"
                    ),
                    "status": "ok",
                }
            )
    return summaries, samples_by_metric


def _grid_diagnostics(
    identities: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    factorial: Mapping[str, Any],
) -> Dict[str, Any]:
    variants = sorted({str(row["variant"]) for row in identities})
    generator_seeds = sorted({int(row["generator_seed"]) for row in identities})
    model_seeds = sorted({int(row["model_seed"]) for row in identities})
    keys = {
        (row["variant"], row["generator_seed"], row["model_seed"]) for row in identities
    }
    expected_cross = {
        (variant, generator_seed, model_seed)
        for variant in variants
        for generator_seed in generator_seeds
        for model_seed in model_seeds
    }
    balanced = keys == expected_cross
    seed_groups = synthetic.get("seed_groups", {})
    development_raw = (
        seed_groups.get("development") if isinstance(seed_groups, dict) else None
    )
    valid_development_group = isinstance(development_raw, list)
    development_seeds: List[int] = []
    if valid_development_group:
        try:
            development_seeds = [int(value) for value in development_raw]
        except (TypeError, ValueError):
            valid_development_group = False
        if len(development_seeds) != len(set(development_seeds)):
            valid_development_group = False
    complete_development = bool(
        valid_development_group
        and set(generator_seeds) == set(development_seeds)
        and len(generator_seeds) == len(development_seeds)
    )
    factorial_model_raw = factorial.get("development_seeds")
    valid_factorial_models = isinstance(factorial_model_raw, list)
    factorial_model_seeds: List[int] = []
    if valid_factorial_models:
        try:
            factorial_model_seeds = [int(value) for value in factorial_model_raw]
        except (TypeError, ValueError):
            valid_factorial_models = False
        if len(factorial_model_seeds) != len(set(factorial_model_seeds)):
            valid_factorial_models = False
    complete_model_seeds = bool(
        valid_factorial_models
        and set(model_seeds) == set(factorial_model_seeds)
        and len(model_seeds) == len(factorial_model_seeds)
    )

    def plan_axis_matches(field: str, observed: Sequence[Any], cast: Any) -> bool:
        raw = plan.get(field)
        if not isinstance(raw, list):
            return False
        try:
            normalized = [cast(value) for value in raw]
        except (TypeError, ValueError):
            return False
        return len(normalized) == len(set(normalized)) and set(normalized) == set(
            observed
        )

    plan_generator_axis_matches = plan_axis_matches(
        "generator_seeds", generator_seeds, int
    )
    plan_model_axis_matches = plan_axis_matches("model_seeds", model_seeds, int)
    plan_variant_axis_matches = plan_axis_matches("variants", variants, str)
    plan_axes_match = bool(
        plan_generator_axis_matches
        and plan_model_axis_matches
        and plan_variant_axis_matches
    )
    complete_phase_grid = plan.get("complete_phase_grid") is True
    four_cells = set(variants) == set(MAIN_VARIANTS)
    reasons = []
    if plan.get("phase") != "development":
        reasons.append("phase_is_not_development")
    if not complete_development:
        reasons.append("development_generator_seed_group_is_incomplete")
    if not complete_model_seeds:
        reasons.append("factorial_development_model_seed_group_is_incomplete")
    if not four_cells:
        reasons.append("main_four_variant_cells_are_incomplete")
    if len(generator_seeds) < 10:
        reasons.append("fewer_than_10_generator_seeds")
    if not balanced:
        reasons.append("identity_grid_is_not_balanced_crossed")
    if not plan_axes_match:
        reasons.append("plan_axes_do_not_match_expected_identities")
    if not complete_phase_grid:
        reasons.append("plan_complete_phase_grid_is_not_true")
    return {
        "phase": plan.get("phase"),
        "variants": variants,
        "generator_seeds": generator_seeds,
        "model_seeds": model_seeds,
        "balanced_crossed_grid": balanced,
        "complete_development_generator_seed_group": complete_development,
        "complete_development_model_seed_group": complete_model_seeds,
        "main_four_variants_complete": four_cells,
        "plan_axes_match_expected_identities": plan_axes_match,
        "plan_complete_phase_grid": complete_phase_grid,
        "gate_eligible": not reasons,
        "gate_ineligibility_reasons": reasons,
    }


def _summary_index(
    summaries: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {(str(row["comparison"]), str(row["metric"])): row for row in summaries}


def _comparison_for_pair(
    comparisons: Sequence[Mapping[str, str]], lhs: str, rhs: str
) -> Optional[Mapping[str, str]]:
    matches = [
        comparison
        for comparison in comparisons
        if comparison["lhs"] == lhs and comparison["rhs"] == rhs
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple manifest comparisons define {lhs}-{rhs}.")
    return matches[0] if matches else None


def _gate_row(
    criterion: str,
    comparison: str,
    metric: str,
    observed_value: Optional[float],
    ci_lower: Optional[float],
    ci_upper: Optional[float],
    operator: str,
    threshold: str,
    status: str,
    detail: str,
) -> Dict[str, Any]:
    return {
        "profile": "p1",
        "criterion": criterion,
        "comparison": comparison,
        "metric": metric,
        "observed_value": observed_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "operator": operator,
        "threshold": threshold,
        "status": status,
        "detail": detail,
    }


def _fpr_relative_reduction(
    identity_rows: Sequence[Mapping[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    index = {
        (row["variant"], row["generator_seed"], row["model_seed"]): row
        for row in identity_rows
    }
    axes_a00 = {
        (row["generator_seed"], row["model_seed"])
        for row in identity_rows
        if row["variant"] == "A00"
    }
    axes_a11 = {
        (row["generator_seed"], row["model_seed"])
        for row in identity_rows
        if row["variant"] == "A11"
    }
    if axes_a00 != axes_a11 or not _is_crossed_grid(axes_a00):
        return None, None, None, "A00/A11 do not form one complete crossed grid"
    generator_seeds = sorted({axis[0] for axis in axes_a00})
    model_seeds = sorted({axis[1] for axis in axes_a00})
    metric = "maximum_regime_fpr_gap"
    cube = np.asarray(
        [
            [
                [
                    float(index[("A00", generator_seed, model_seed)][metric]),
                    float(index[("A11", generator_seed, model_seed)][metric]),
                ]
                for model_seed in model_seeds
            ]
            for generator_seed in generator_seeds
        ],
        dtype=np.float64,
    )
    means = cube.mean(axis=(0, 1))
    if means[0] <= 0.0:
        return None, None, None, "A00 mean maximum regime FPR gap is zero"
    observed = float((means[0] - means[1]) / means[0])
    samples = _crossed_bootstrap_cube(
        cube, n_bootstrap, _stream_seed(seed, "gate:fpr_relative_reduction")
    )
    valid = samples[:, 0] > 0.0
    if not np.any(valid):
        return observed, None, None, "No bootstrap replicate has positive A00 mean"
    ratios = (samples[valid, 0] - samples[valid, 1]) / samples[valid, 0]
    detail = (
        "relative reduction=(mean_A00-mean_A11)/mean_A00;"
        f"valid_bootstrap={int(valid.sum())}/{n_bootstrap}"
    )
    return (
        observed,
        float(np.quantile(ratios, 0.025)),
        float(np.quantile(ratios, 0.975)),
        detail,
    )


def _gate_diagnostics(
    identity_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, str]],
    summaries: Sequence[Mapping[str, Any]],
    grid: Mapping[str, Any],
    n_bootstrap: int,
    seed: int,
) -> List[Dict[str, Any]]:
    eligible = bool(grid["gate_eligible"])
    insufficiency = ";".join(grid["gate_ineligibility_reasons"])
    output = [
        _gate_row(
            "gate_preconditions",
            "",
            "",
            None,
            None,
            None,
            "all_required",
            "development_complete_four_cell_balanced_grid",
            "eligible" if eligible else "insufficient_data",
            "all_gate_preconditions_met" if eligible else insufficiency,
        )
    ]
    summary_index = _summary_index(summaries)

    context_comparison = _comparison_for_pair(comparisons, "A11", "A01")
    context_summary = (
        summary_index.get((context_comparison["name"], "matched_ordering_rate"))
        if context_comparison is not None
        else None
    )
    if not eligible or context_summary is None or context_summary["status"] != "ok":
        output.append(
            _gate_row(
                "matched_ordering_improvement",
                "" if context_comparison is None else context_comparison["name"],
                "matched_ordering_rate",
                None if context_summary is None else float(context_summary["mean"]),
                None if context_summary is None else context_summary["ci_lower"],
                None if context_summary is None else context_summary["ci_upper"],
                "mean>=0.05 and ci_lower>0",
                "0.05;0",
                "insufficient_data",
                insufficiency or "A11-A01 comparison is unavailable",
            )
        )
    else:
        mean = float(context_summary["mean"])
        lower = float(context_summary["ci_lower"])
        output.append(
            _gate_row(
                "matched_ordering_improvement",
                context_comparison["name"],
                "matched_ordering_rate",
                mean,
                lower,
                float(context_summary["ci_upper"]),
                "mean>=0.05 and ci_lower>0",
                "0.05;0",
                "pass" if mean >= 0.05 and lower > 0.0 else "fail",
                "A11 minus A01 on matched generator/model seeds",
            )
        )

    full_comparison = _comparison_for_pair(comparisons, "A11", "A00")
    fpr_value, fpr_lower, fpr_upper, fpr_detail = _fpr_relative_reduction(
        identity_rows, n_bootstrap, seed
    )
    if not eligible or full_comparison is None or fpr_value is None:
        output.append(
            _gate_row(
                "maximum_regime_fpr_gap_relative_reduction",
                "" if full_comparison is None else full_comparison["name"],
                "maximum_regime_fpr_gap",
                fpr_value,
                fpr_lower,
                fpr_upper,
                ">=",
                "0.25",
                "insufficient_data",
                insufficiency or fpr_detail or "A11-A00 comparison is unavailable",
            )
        )
    else:
        output.append(
            _gate_row(
                "maximum_regime_fpr_gap_relative_reduction",
                full_comparison["name"],
                "maximum_regime_fpr_gap",
                fpr_value,
                fpr_lower,
                fpr_upper,
                ">=",
                "0.25",
                "pass" if fpr_value >= 0.25 else "fail",
                fpr_detail,
            )
        )

    dependency_metric = "dependency_break_average_precision"
    dependency_summary = (
        summary_index.get((full_comparison["name"], dependency_metric))
        if full_comparison is not None
        else None
    )
    if (
        not eligible
        or dependency_summary is None
        or dependency_summary["status"] != "ok"
    ):
        output.append(
            _gate_row(
                "dependency_break_average_precision",
                "" if full_comparison is None else full_comparison["name"],
                dependency_metric,
                None
                if dependency_summary is None
                else float(dependency_summary["mean"]),
                None if dependency_summary is None else dependency_summary["ci_lower"],
                None if dependency_summary is None else dependency_summary["ci_upper"],
                ">=",
                "0",
                "insufficient_data",
                insufficiency or "A11-A00 dependency AP comparison is unavailable",
            )
        )
    else:
        dependency_mean = float(dependency_summary["mean"])
        output.append(
            _gate_row(
                "dependency_break_average_precision",
                full_comparison["name"],
                dependency_metric,
                dependency_mean,
                float(dependency_summary["ci_lower"]),
                float(dependency_summary["ci_upper"]),
                ">=",
                "0",
                "pass" if dependency_mean >= 0.0 else "fail",
                "A11 minus A00 on matched generator/model seeds",
            )
        )

    output.append(
        _gate_row(
            "ordinary_large_spike",
            "",
            "",
            None,
            None,
            None,
            "not_defined",
            "",
            "not_evaluated",
            "The current contextual synthetic suite defines no ordinary-large-spike criterion.",
        )
    )
    return output


def summarize(
    input_root: Path,
    n_bootstrap: int = 10000,
    seed: int = 2021,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    input_root = input_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if n_bootstrap < 2:
        raise ValueError("--n-bootstrap must be at least 2.")
    plan, plan_path, synthetic, synthetic_path, factorial, factorial_path = (
        _load_frozen_inputs(input_root)
    )
    identities = _normalize_expected_identities(input_root, plan, plan_path, synthetic)
    _verify_identity_metadata_set(input_root, identities)
    identity_rows = [_load_identity_metrics(identity, plan) for identity in identities]
    identity_rows.sort(
        key=lambda row: (row["variant"], row["generator_seed"], row["model_seed"])
    )
    variants = sorted({row["variant"] for row in identity_rows})
    comparisons = _comparison_definitions(factorial, variants, factorial_path)
    paired_rows = _paired_deltas(identity_rows, comparisons)
    bootstrap_rows, _ = _bootstrap_summaries(
        paired_rows, comparisons, n_bootstrap, seed
    )
    grid = _grid_diagnostics(identities, plan, synthetic, factorial)
    gate_rows = _gate_diagnostics(
        identity_rows, comparisons, bootstrap_rows, grid, n_bootstrap, seed
    )

    destination = (output_dir or (input_root / "summary")).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    identity_fields = [
        "variant",
        "generator_seed",
        "model_seed",
        "config_hash",
        "plan_hash",
        "synthetic_config_hash",
        "factorial_manifest_hash",
        "result_dir",
        "evaluation_file",
        "score_metadata_file",
        *METRICS,
    ]
    paired_fields = [
        "comparison",
        "lhs",
        "rhs",
        "generator_seed",
        "model_seed",
        "lhs_config_hash",
        "rhs_config_hash",
        "plan_hash",
        *METRICS,
    ]
    bootstrap_fields = [
        "comparison",
        "lhs",
        "rhs",
        "metric",
        "mean",
        "bootstrap_mean",
        "std",
        "ci_lower",
        "ci_upper",
        "n_generator_seeds",
        "n_model_seeds",
        "n_pairs",
        "n_bootstrap",
        "seed",
        "stream_seed",
        "sampling",
        "status",
    ]
    gate_fields = [
        "profile",
        "criterion",
        "comparison",
        "metric",
        "observed_value",
        "ci_lower",
        "ci_upper",
        "operator",
        "threshold",
        "status",
        "detail",
    ]
    output_paths = {
        "identity_metrics.csv": destination / "identity_metrics.csv",
        "paired_deltas.csv": destination / "paired_deltas.csv",
        "paired_bootstrap.csv": destination / "paired_bootstrap.csv",
        "gate_diagnostics.csv": destination / "gate_diagnostics.csv",
    }
    _write_csv(output_paths["identity_metrics.csv"], identity_rows, identity_fields)
    _write_csv(output_paths["paired_deltas.csv"], paired_rows, paired_fields)
    _write_csv(output_paths["paired_bootstrap.csv"], bootstrap_rows, bootstrap_fields)
    _write_csv(output_paths["gate_diagnostics.csv"], gate_rows, gate_fields)

    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "input_root": str(input_root),
        "plan_path": str(plan_path),
        "plan_hash": plan["plan_hash"],
        "phase": plan["phase"],
        "synthetic_config_path": str(synthetic_path),
        "synthetic_config_hash": plan["synthetic_config_hash"],
        "factorial_manifest_path": str(factorial_path),
        "factorial_manifest_hash": plan["factorial_manifest_hash"],
        "variants": grid["variants"],
        "generator_seeds": grid["generator_seeds"],
        "model_seeds": grid["model_seeds"],
        "n_identities": len(identity_rows),
        "n_paired_rows": len(paired_rows),
        "comparisons": comparisons,
        "metrics": list(METRICS),
        "bootstrap": {
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "ci_level": 0.95,
            "std_definition": "sample standard deviation of bootstrap statistics",
            "statistic": (
                "mean of the crossed grid after independent generator/model seed "
                "resampling; with one model seed, resample generator seeds only"
            ),
        },
        "grid_diagnostics": grid,
        "gate_statuses": {row["criterion"]: row["status"] for row in gate_rows},
        "selection_policy": (
            "all frozen identities and all manifest comparisons whose variants ran; "
            "no value-based selection"
        ),
        "output_sha256": {
            name: _file_sha256(path) for name, path in sorted(output_paths.items())
        },
    }
    _write_json_atomic(destination / "summary_metadata.json", metadata)
    return metadata


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Frozen contextual run root."
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--output-dir", type=Path, help="Defaults to INPUT/summary.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    metadata = summarize(args.input, args.n_bootstrap, args.seed, args.output_dir)
    print(
        "Wrote contextual summary: identities={} pairs={} gate_eligible={}".format(
            metadata["n_identities"],
            metadata["n_paired_rows"],
            metadata["grid_diagnostics"]["gate_eligible"],
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
