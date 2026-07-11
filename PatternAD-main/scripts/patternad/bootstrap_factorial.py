#!/usr/bin/env python3
"""Deterministic hierarchical bootstrap for frozen PatternAD comparisons.

The script consumes one of the long tables emitted by summarize_factorial.py.
It never searches anomaly ratios, variants, comparisons, or metrics for the
best observed test result. Entity tables are converted to paired deltas using
comparison definitions fixed in the experiment manifest (or explicitly on the
command line) before any statistic is computed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "patternad" / "factorial_ablation.json"
DEFAULT_SCORE_METRICS = (
    "auc_pr",
    "VUS_PR",
    "auc_roc",
    "VUS_ROC",
    "R_AUC_PR",
    "R_AUC_ROC",
)
BASE_FIELDS = ("run_name", "group", "family", "entity", "seed")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return list(reader.fieldnames), rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {}
            for field in fields:
                value = row.get(field, "")
                serialized[field] = format(value, ".17g") if isinstance(value, float) else value
            writer.writerow(serialized)
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_fields(headers: Iterable[str], required: Iterable[str], source: Path) -> None:
    missing = sorted(set(required) - set(headers))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _finite(raw: Any, field: str, context: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric {field} in {context}: {raw!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} in {context}: {raw!r}")
    return value


def _seed(raw: Any, context: str) -> int:
    value = _finite(raw, "seed", context)
    integer = int(value)
    if value != integer:
        raise ValueError(f"Seed must be an integer in {context}: {raw!r}")
    return integer


def _metric_names(headers: Sequence[str], requested: Optional[Sequence[str]]) -> List[str]:
    metrics = list(requested) if requested else [m for m in DEFAULT_SCORE_METRICS if m in headers]
    if not metrics:
        raise ValueError(
            "No score metric was selected. Pass --metrics explicitly or provide one of "
            f"{list(DEFAULT_SCORE_METRICS)}."
        )
    duplicates = sorted({metric for metric in metrics if metrics.count(metric) > 1})
    if duplicates:
        raise ValueError(f"Duplicate metric names: {duplicates}")
    missing = sorted(set(metrics) - set(headers))
    if missing:
        raise ValueError(f"Requested metric columns are absent: {missing}")
    return metrics


def _parse_comparison(raw: str) -> Dict[str, str]:
    parts = raw.split(":")
    if len(parts) != 3 or any(not part.strip() for part in parts):
        raise ValueError(
            f"Invalid comparison {raw!r}; expected NAME:LHS_VARIANT:RHS_VARIANT."
        )
    name, lhs, rhs = (part.strip() for part in parts)
    if lhs == rhs:
        raise ValueError(f"Comparison {name!r} uses the same lhs and rhs variant.")
    return {"name": name, "lhs": lhs, "rhs": rhs}


def _comparison_definitions(
    explicit: Optional[Sequence[str]], manifest_path: Path
) -> Tuple[List[Dict[str, str]], bool]:
    if explicit:
        comparisons = [_parse_comparison(raw) for raw in explicit]
        is_explicit = True
    else:
        manifest = _load_json(manifest_path)
        raw_comparisons = manifest.get("comparisons")
        if not isinstance(raw_comparisons, list) or not raw_comparisons:
            raise ValueError("Manifest comparisons must be a non-empty list.")
        comparisons = []
        for item in raw_comparisons:
            if not isinstance(item, dict) or not all(item.get(k) for k in ("name", "lhs", "rhs")):
                raise ValueError(f"Invalid comparison in {manifest_path}: {item!r}")
            comparisons.append({key: str(item[key]) for key in ("name", "lhs", "rhs")})
        is_explicit = False
    names = [item["name"] for item in comparisons]
    if len(names) != len(set(names)):
        raise ValueError("Comparison names must be unique.")
    return comparisons, is_explicit


def _normalize_common(row: Mapping[str, str], row_number: int) -> Dict[str, Any]:
    context = f"CSV row {row_number}"
    normalized: Dict[str, Any] = {}
    for field in BASE_FIELDS[:-1]:
        value = str(row.get(field, "")).strip()
        if not value:
            raise ValueError(f"Missing {field} in {context}.")
        normalized[field] = value
    normalized["seed"] = _seed(row.get("seed"), context)
    return normalized


def _paired_from_entity_rows(
    rows: Sequence[Mapping[str, str]],
    comparisons: Sequence[Mapping[str, str]],
    metrics: Sequence[str],
    missing_policy: str,
    comparisons_are_explicit: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    index: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    variants_by_run: Dict[Tuple[str, str], set] = defaultdict(set)
    for row_number, row in enumerate(rows, start=2):
        common = _normalize_common(row, row_number)
        variant = str(row.get("variant", "")).strip()
        if not variant:
            raise ValueError(f"Missing variant in CSV row {row_number}.")
        key = tuple(common[field] for field in BASE_FIELDS) + (variant,)
        if key in index:
            raise ValueError(f"Duplicate entity/seed/variant identity: {key}")
        normalized = dict(common)
        normalized["variant"] = variant
        for metric in metrics:
            normalized[metric] = _finite(row.get(metric), metric, f"CSV row {row_number}")
        index[key] = normalized
        variants_by_run[(common["run_name"], common["group"])].add(variant)

    paired: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    for run_key in sorted(variants_by_run):
        run_name, group = run_key
        available = variants_by_run[run_key]
        run_identities = {
            key[:5]
            for key in index
            if key[0] == run_name and key[1] == group
        }
        for comparison in comparisons:
            name, lhs, rhs = (comparison[field] for field in ("name", "lhs", "rhs"))
            if lhs not in available or rhs not in available:
                diagnostics.append(
                    {
                        "run_name": run_name,
                        "group": group,
                        "comparison": name,
                        "lhs": lhs,
                        "rhs": rhs,
                        "status": "unavailable_variants",
                        "n_complete_pairs": 0,
                        "n_missing_lhs": "",
                        "n_missing_rhs": "",
                        "n_dropped_identities": "",
                        "detail": "available_variants=" + ";".join(sorted(available)),
                    }
                )
                if comparisons_are_explicit:
                    raise ValueError(
                        f"Explicit comparison {name!r} is unavailable in {run_key}: "
                        f"lhs={lhs in available}, rhs={rhs in available}."
                    )
                continue

            missing_lhs = [identity for identity in run_identities if identity + (lhs,) not in index]
            missing_rhs = [identity for identity in run_identities if identity + (rhs,) not in index]
            incomplete = set(missing_lhs) | set(missing_rhs)
            if incomplete and missing_policy == "error":
                raise ValueError(
                    f"Comparison {name!r} has incomplete entity/seed pairs in {run_key}: "
                    f"missing_lhs={len(missing_lhs)}, missing_rhs={len(missing_rhs)}. "
                    "Rerun the frozen grid or explicitly use --missing-policy drop."
                )

            complete = sorted(run_identities - incomplete)
            for identity in complete:
                lhs_row = index[identity + (lhs,)]
                rhs_row = index[identity + (rhs,)]
                output = dict(zip(BASE_FIELDS, identity))
                output.update({"comparison": name, "lhs": lhs, "rhs": rhs})
                for metric in metrics:
                    output[metric] = lhs_row[metric] - rhs_row[metric]
                paired.append(output)
            diagnostics.append(
                {
                    "run_name": run_name,
                    "group": group,
                    "comparison": name,
                    "lhs": lhs,
                    "rhs": rhs,
                    "status": "complete" if not incomplete else "dropped_incomplete_pairs",
                    "n_complete_pairs": len(complete),
                    "n_missing_lhs": len(missing_lhs),
                    "n_missing_rhs": len(missing_rhs),
                    "n_dropped_identities": len(incomplete),
                    "detail": "paired as lhs-rhs; no value-based comparison selection",
                }
            )
    if not paired:
        raise ValueError("No complete, available comparisons were found in the entity table.")
    return paired, diagnostics


def _normalize_paired_rows(
    rows: Sequence[Mapping[str, str]], metrics: Sequence[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    normalized_rows: List[Dict[str, Any]] = []
    seen = set()
    comparison_sides: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    counts: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)
    for row_number, row in enumerate(rows, start=2):
        common = _normalize_common(row, row_number)
        comparison = str(row.get("comparison", "")).strip()
        lhs = str(row.get("lhs", "")).strip()
        rhs = str(row.get("rhs", "")).strip()
        if not comparison or not lhs or not rhs:
            raise ValueError(f"Missing comparison/lhs/rhs in CSV row {row_number}.")
        key = tuple(common[field] for field in BASE_FIELDS) + (comparison,)
        if key in seen:
            raise ValueError(f"Duplicate paired entity/seed identity: {key}")
        seen.add(key)
        run_comparison = (common["run_name"], common["group"], comparison)
        previous = comparison_sides.setdefault(run_comparison, (lhs, rhs))
        if previous != (lhs, rhs):
            raise ValueError(
                f"Comparison {run_comparison} mixes lhs/rhs definitions: {previous} and {(lhs, rhs)}."
            )
        output = dict(common)
        output.update({"comparison": comparison, "lhs": lhs, "rhs": rhs})
        for metric in metrics:
            output[metric] = _finite(row.get(metric), metric, f"CSV row {row_number}")
        normalized_rows.append(output)
        counts[(common["run_name"], common["group"], comparison, lhs, rhs)] += 1

    diagnostics = [
        {
            "run_name": key[0],
            "group": key[1],
            "comparison": key[2],
            "lhs": key[3],
            "rhs": key[4],
            "status": "paired_input",
            "n_complete_pairs": count,
            "n_missing_lhs": "unknown_from_paired_input",
            "n_missing_rhs": "unknown_from_paired_input",
            "n_dropped_identities": "unknown_from_paired_input",
            "detail": "Pair completeness must be established by summarize_factorial.py/run_plan.json.",
        }
        for key, count in sorted(counts.items())
    ]
    return normalized_rows, diagnostics


def _validate_entity_gate_provenance(
    headers: Sequence[str], rows: Sequence[Mapping[str, str]], source: Path
) -> None:
    _require_fields(headers, ("plan_hash", "config_hash"), source)
    plan_hashes: Dict[Tuple[str, str], set] = defaultdict(set)
    for row_number, row in enumerate(rows, start=2):
        run_name = str(row.get("run_name", "")).strip()
        group = str(row.get("group", "")).strip()
        plan_hash = str(row.get("plan_hash", "")).strip()
        config_hash = str(row.get("config_hash", "")).strip()
        if not plan_hash or not config_hash:
            raise ValueError(
                f"Formal gate input has empty plan_hash/config_hash in CSV row {row_number}."
            )
        plan_hashes[(run_name, group)].add(plan_hash)
    mixed = {key: values for key, values in plan_hashes.items() if len(values) != 1}
    if mixed:
        raise ValueError(
            "Formal gate input mixes frozen run plans: "
            + ", ".join(f"{key}={sorted(values)}" for key, values in sorted(mixed.items()))
        )


def _nested_values(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    nested: Dict[str, Dict[str, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        key = (row["family"], row["entity"], int(row["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate family/entity/seed in one comparison: {key}")
        seen.add(key)
        nested[row["family"]][row["entity"]].append((int(row["seed"]), float(row[metric])))
    return {
        family: {
            entity: sorted(seed_values)
            for entity, seed_values in sorted(entities.items())
        }
        for family, entities in sorted(nested.items())
    }


def _hierarchical_mean(nested: Mapping[str, Mapping[str, Sequence[Tuple[int, float]]]]) -> float:
    family_means = []
    for entities in nested.values():
        entity_means = [math.fsum(value for _, value in values) / len(values) for values in entities.values()]
        family_means.append(math.fsum(entity_means) / len(entity_means))
    return math.fsum(family_means) / len(family_means)


def _stream_seed(base_seed: int, *parts: str) -> int:
    payload = "\0".join([str(base_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _bootstrap_distribution(
    nested: Mapping[str, Mapping[str, Sequence[Tuple[int, float]]]],
    n_bootstrap: int,
    stream_seed: int,
    family_mode: str,
) -> np.ndarray:
    rng = np.random.default_rng(stream_seed)
    families = list(nested)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for bootstrap_index in range(n_bootstrap):
        sampled_family_means = []
        if family_mode == "resample":
            family_indices = rng.integers(0, len(families), size=len(families))
        elif family_mode == "fixed":
            family_indices = np.arange(len(families))
        else:
            raise ValueError(f"Unsupported family bootstrap mode: {family_mode}")
        for family_index in family_indices:
            entities = list(nested[families[int(family_index)]].values())
            sampled_entity_means = []
            for entity_index in rng.integers(0, len(entities), size=len(entities)):
                values = np.asarray(
                    [value for _, value in entities[int(entity_index)]], dtype=np.float64
                )
                seed_indices = rng.integers(0, len(values), size=len(values))
                sampled_entity_means.append(float(values[seed_indices].mean()))
            sampled_family_means.append(float(np.mean(sampled_entity_means)))
        samples[bootstrap_index] = float(np.mean(sampled_family_means))
    return samples


def _sampling_diagnostics(
    nested: Mapping[str, Mapping[str, Sequence[Tuple[int, float]]]],
    dropped_pairs: int,
    family_mode: str,
) -> Dict[str, Any]:
    entities_per_family = [len(entities) for entities in nested.values()]
    seed_sets = [
        tuple(seed for seed, _ in values)
        for entities in nested.values()
        for values in entities.values()
    ]
    seeds_per_entity = [len(values) for entities in nested.values() for values in entities.values()]
    issues = []
    if family_mode == "resample" and len(nested) < 3:
        issues.append("fewer_than_3_families_for_resampled_family_ci")
    if family_mode == "fixed" and len(nested) < 2:
        issues.append("fewer_than_2_fixed_family_strata")
    if sum(entities_per_family) < 3:
        issues.append("fewer_than_3_entities")
    if min(seeds_per_entity) < 2:
        issues.append("fewer_than_2_seeds_for_an_entity")
    if any(seed_set != seed_sets[0] for seed_set in seed_sets[1:]):
        issues.append("unbalanced_seed_sets")
    if dropped_pairs:
        issues.append(f"dropped_{dropped_pairs}_incomplete_pairs")
    return {
        "n_families": len(nested),
        "n_entities": sum(entities_per_family),
        "n_pairs": sum(seeds_per_entity),
        "min_entities_per_family": min(entities_per_family),
        "max_entities_per_family": max(entities_per_family),
        "min_seeds_per_entity": min(seeds_per_entity),
        "max_seeds_per_entity": max(seeds_per_entity),
        "family_mode": family_mode,
        "balanced_seed_grid": not any(issue == "unbalanced_seed_sets" for issue in issues),
        "reliability": "limited" if issues else "standard",
        "warnings": ";".join(issues),
    }


def _bootstrap_summaries(
    paired_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    n_bootstrap: int,
    base_seed: int,
    ci_level: float,
    diagnostics: Sequence[Mapping[str, Any]],
    family_mode: str = "resample",
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        key = (row["run_name"], row["group"], row["comparison"], row["lhs"], row["rhs"])
        grouped[key].append(row)
    dropped = {
        (row["run_name"], row["group"], row["comparison"]): int(row["n_dropped_identities"])
        for row in diagnostics
        if isinstance(row.get("n_dropped_identities"), int)
    }
    alpha = (1.0 - ci_level) / 2.0
    output = []
    for key, rows in sorted(grouped.items()):
        run_name, group, comparison, lhs, rhs = key
        for metric in metrics:
            nested = _nested_values(rows, metric)
            stream_seed = _stream_seed(base_seed, run_name, group, comparison, metric)
            samples = _bootstrap_distribution(
                nested, n_bootstrap, stream_seed, family_mode
            )
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            sampling = _sampling_diagnostics(
                nested, dropped.get((run_name, group, comparison), 0), family_mode
            )
            output.append(
                {
                    "run_name": run_name,
                    "group": group,
                    "comparison": comparison,
                    "lhs": lhs,
                    "rhs": rhs,
                    "metric": metric,
                    "mean": _hierarchical_mean(nested),
                    "paired_row_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "bootstrap_mean": float(samples.mean()),
                    "bootstrap_std": float(samples.std(ddof=1)) if n_bootstrap > 1 else 0.0,
                    "ci_level": ci_level,
                    "ci_lower": float(np.quantile(samples, alpha, method="linear")),
                    "ci_upper": float(np.quantile(samples, 1.0 - alpha, method="linear")),
                    "n_bootstrap": n_bootstrap,
                    "base_seed": base_seed,
                    "stream_seed": stream_seed,
                    **sampling,
                }
            )
    return output


def _family_means(rows: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, float]:
    nested = _nested_values(rows, metric)
    return {
        family: _hierarchical_mean({family: entities})
        for family, entities in nested.items()
    }


def _gate_row(
    run_name: str,
    group: str,
    profile: str,
    comparison: str,
    criterion: str,
    metric: str,
    scope: str,
    value: Any,
    operator: str,
    threshold: Any,
    status: str,
    detail: str,
) -> Dict[str, Any]:
    return {
        "run_name": run_name,
        "group": group,
        "profile": profile,
        "comparison": comparison,
        "criterion": criterion,
        "metric": metric,
        "scope": scope,
        "observed_value": value,
        "operator": operator,
        "threshold": threshold,
        "status": status,
        "detail": detail,
    }


def _status(value: float, operator: str, threshold: float) -> str:
    if operator == ">=":
        return "pass" if value >= threshold else "fail"
    if operator == ">":
        return "pass" if value > threshold else "fail"
    raise ValueError(f"Unsupported gate operator: {operator}")


def _gate_preconditions(
    summary: Optional[Mapping[str, Any]],
    profile: str,
    missing_policy: str,
    minimum_gate_bootstrap: int,
) -> Tuple[bool, str]:
    issues = []
    if summary is None:
        issues.append("no_auc_pr_or_vus_pr_summary")
        return False, ";".join(issues)
    if missing_policy != "error":
        issues.append("missing_policy_must_be_error")
    if summary.get("balanced_seed_grid") is not True:
        issues.append("seed_grid_is_not_balanced")
    if summary.get("reliability") != "standard":
        issues.append("bootstrap_reliability_is_not_standard")
    if int(summary.get("n_bootstrap", 0)) < minimum_gate_bootstrap:
        issues.append(
            f"n_bootstrap_below_predeclared_minimum_{minimum_gate_bootstrap}"
        )
    expected_family_mode = "fixed" if profile == "p4" else "resample"
    if summary.get("family_mode") != expected_family_mode:
        issues.append(f"{profile}_requires_family_mode_{expected_family_mode}")

    required_seeds = 3 if profile == "p2" else 5
    required_family_count = 4 if profile == "p2" else 2
    if int(summary.get("n_families", 0)) < required_family_count:
        issues.append(f"fewer_than_{required_family_count}_required_families")
    if int(summary.get("min_seeds_per_entity", 0)) < required_seeds:
        issues.append(f"fewer_than_{required_seeds}_paired_seeds_per_entity")
    if int(summary.get("n_entities", 0)) < required_family_count:
        issues.append(f"fewer_than_{required_family_count}_required_entities")
    return not issues, ";".join(issues) if issues else "all_gate_preconditions_met"


def _gate_diagnostics(
    paired_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    profile: str,
    primary_comparison: str,
    required_families: Sequence[str],
    missing_policy: str = "error",
    minimum_gate_bootstrap: int = 10000,
) -> List[Dict[str, Any]]:
    grouped_rows: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        grouped_rows[(row["run_name"], row["group"], row["comparison"])].append(row)
    summary_index = {
        (row["run_name"], row["group"], row["comparison"], row["metric"]): row
        for row in summaries
    }
    primary_keys = sorted(key for key in grouped_rows if key[2] == primary_comparison)
    if not primary_keys:
        raise ValueError(f"Primary comparison {primary_comparison!r} is absent from the input.")

    output: List[Dict[str, Any]] = []
    for run_name, group, comparison in primary_keys:
        rows = grouped_rows[(run_name, group, comparison)]
        auc_summary = summary_index.get((run_name, group, comparison, "auc_pr"))
        vus_summary = summary_index.get((run_name, group, comparison, "VUS_PR"))
        reference = auc_summary or vus_summary
        adequate, insufficiency = _gate_preconditions(
            reference, profile, missing_policy, minimum_gate_bootstrap
        )
        output.append(
            _gate_row(
                run_name,
                group,
                profile,
                comparison,
                "gate_preconditions",
                "",
                "provenance_and_sampling",
                "",
                "all_required",
                "",
                "eligible" if adequate else "insufficient_data",
                insufficiency,
            )
        )

        if profile == "p2":
            if auc_summary is None:
                output.append(_gate_row(run_name, group, profile, comparison, "predeclared_pair_auc_pr_delta", "auc_pr", "family_macro", "", ">=", 0.01, "not_evaluated", "auc_pr was not requested"))
            else:
                family_auc = _family_means(rows, "auc_pr")
                hard_status = _status(float(auc_summary["mean"]), ">=", 0.01) if adequate else "insufficient_data"
                pair_detail = (
                    "Pair-specific criterion only: the comparator is caller-predeclared; "
                    "this is not the experiment plan's strongest-alternative test."
                )
                output.append(_gate_row(run_name, group, profile, comparison, "predeclared_pair_auc_pr_delta", "auc_pr", "family_macro", auc_summary["mean"], ">=", 0.01, hard_status, pair_detail if adequate else insufficiency + ";" + pair_detail))
                positive_count = sum(value > 0 for value in family_auc.values())
                output.append(_gate_row(run_name, group, profile, comparison, "positive_family_count", "auc_pr", "family", positive_count, ">=", 3, _status(float(positive_count), ">=", 3.0) if adequate else "insufficient_data", f"n_families={len(family_auc)}" if adequate else insufficiency))
                worst_family = min(family_auc.values())
                output.append(_gate_row(run_name, group, profile, comparison, "worst_family_delta", "auc_pr", "family", worst_family, ">=", -0.02, _status(worst_family, ">=", -0.02) if adequate else "insufficient_data", "Minimum of predeclared family means" if adequate else insufficiency))
                entity_means = []
                entity_groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
                for row in rows:
                    entity_groups[(row["family"], row["entity"])].append(float(row["auc_pr"]))
                for values in entity_groups.values():
                    entity_means.append(math.fsum(values) / len(values))
                seed_groups: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
                for row in rows:
                    seed_groups[int(row["seed"])].append(row)
                seed_means = [_hierarchical_mean(_nested_values(seed_rows, "auc_pr")) for seed_rows in seed_groups.values()]
                detail = (
                    f"positive_entity_means={sum(value > 0 for value in entity_means)}/{len(entity_means)};"
                    f"positive_seed_macros={sum(value > 0 for value in seed_means)}/{len(seed_means)};"
                    "reported only because the plan gives no numeric concentration cutoff"
                )
                output.append(_gate_row(run_name, group, profile, comparison, "single_seed_or_entity_concentration", "auc_pr", "diagnostic", "", "report_only", "", "reported", detail))
            if vus_summary is None:
                output.append(_gate_row(run_name, group, profile, comparison, "macro_vus_pr_nonnegative", "VUS_PR", "family_macro", "", ">=", 0.0, "not_evaluated", "VUS_PR was not requested"))
            else:
                output.append(_gate_row(run_name, group, profile, comparison, "macro_vus_pr_nonnegative", "VUS_PR", "family_macro", vus_summary["mean"], ">=", 0.0, _status(float(vus_summary["mean"]), ">=", 0.0) if adequate else "insufficient_data", "" if adequate else insufficiency))

        if profile == "p4":
            if auc_summary is None:
                output.append(_gate_row(run_name, group, profile, comparison, "auc_pr_ci_lower_positive", "auc_pr", "hierarchical_ci", "", ">", 0.0, "not_evaluated", "auc_pr was not requested"))
            else:
                output.append(_gate_row(run_name, group, profile, comparison, "auc_pr_ci_lower_positive", "auc_pr", "hierarchical_ci", auc_summary["ci_lower"], ">", 0.0, _status(float(auc_summary["ci_lower"]), ">", 0.0) if adequate else "insufficient_data", "" if adequate else insufficiency))
                family_auc = _family_means(rows, "auc_pr")
                for family in required_families:
                    if family not in family_auc:
                        output.append(_gate_row(run_name, group, profile, comparison, "required_family_delta", "auc_pr", family, "", ">=", -0.01, "insufficient_data", f"Required family {family!r} is absent"))
                    else:
                        output.append(_gate_row(run_name, group, profile, comparison, "required_family_delta", "auc_pr", family, family_auc[family], ">=", -0.01, _status(family_auc[family], ">=", -0.01) if adequate else "insufficient_data", "" if adequate else insufficiency))
                ambiguous = float(auc_summary["ci_lower"]) <= 0.0 <= float(auc_summary["ci_upper"]) and abs(float(auc_summary["mean"])) < 0.005
                output.append(_gate_row(run_name, group, profile, comparison, "small_ambiguous_effect_stop_condition", "auc_pr", "family_macro", int(ambiguous), "report_only", "CI includes 0 and abs(mean) < 0.005", "flagged" if ambiguous else "clear", "This is a stop-claim diagnostic, not an automatic model decision."))
            if vus_summary is None:
                output.append(_gate_row(run_name, group, profile, comparison, "macro_vus_pr_nonnegative", "VUS_PR", "family_macro", "", ">=", 0.0, "not_evaluated", "VUS_PR was not requested"))
            else:
                output.append(_gate_row(run_name, group, profile, comparison, "macro_vus_pr_nonnegative", "VUS_PR", "family_macro", vus_summary["mean"], ">=", 0.0, _status(float(vus_summary["mean"]), ">=", 0.0) if adequate else "insufficient_data", "" if adequate else insufficiency))
            output.append(_gate_row(run_name, group, profile, comparison, "fixed_threshold_false_alarm_calibration", "normal_fpr", "calibration", "", "report_only", "alpha", "not_evaluated", "The score-metric input has no fixed-threshold normal-FPR column; evaluate this from a separately frozen calibration report."))
    return output


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute deterministic family -> entity -> seed hierarchical bootstrap "
            "intervals for predeclared PatternAD paired comparisons."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Defaults to INPUT_PARENT/bootstrap.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--comparison",
        action="append",
        help="Entity input only: repeat NAME:LHS:RHS. Defaults to manifest comparisons.",
    )
    parser.add_argument("--metrics", nargs="+", help="Defaults to all recognized score columns.")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument(
        "--family-mode",
        choices=("resample", "fixed"),
        default="resample",
        help=(
            "Resample families for development-population inference, or keep "
            "predeclared families as fixed strata while resampling entity/seed."
        ),
    )
    parser.add_argument(
        "--missing-policy",
        choices=("error", "drop"),
        default="error",
        help="Entity input only. Drop is explicit and recorded; error is the default.",
    )
    parser.add_argument("--gate-profile", choices=("p2", "p4"))
    parser.add_argument(
        "--minimum-gate-bootstrap",
        type=int,
        default=10000,
        help="Fail-closed minimum for any gate criterion to receive pass status.",
    )
    parser.add_argument(
        "--primary-comparison",
        help="Predeclared comparison for gate diagnostics; never selected from observed values.",
    )
    parser.add_argument(
        "--required-family",
        action="append",
        help="P4 family floor; defaults to HAI21 and SMD.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.n_bootstrap < 2:
        raise ValueError("--n-bootstrap must be at least 2.")
    if args.minimum_gate_bootstrap < 2:
        raise ValueError("--minimum-gate-bootstrap must be at least 2.")
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError("--ci-level must be strictly between 0 and 1.")
    if bool(args.gate_profile) != bool(args.primary_comparison):
        raise ValueError("--gate-profile and --primary-comparison must be provided together.")

    headers, raw_rows = _read_csv(input_path)
    metrics = _metric_names(headers, args.metrics)
    if {"comparison", "lhs", "rhs"}.issubset(headers):
        input_type = "paired_entity_seed_delta"
        _require_fields(headers, (*BASE_FIELDS, "comparison", "lhs", "rhs", *metrics), input_path)
        if args.comparison:
            raise ValueError("--comparison is only valid for entity_seed_score_metrics.csv input.")
        if args.gate_profile:
            raise ValueError(
                "Formal gate diagnostics require entity_seed_score_metrics.csv with "
                "plan_hash/config_hash provenance. A paired delta CSV cannot prove that "
                "uniformly missing planned identities were not omitted."
            )
        paired_rows, input_diagnostics = _normalize_paired_rows(raw_rows, metrics)
        comparison_source = "paired_input"
    elif "variant" in headers:
        input_type = "entity_seed_score_metrics"
        _require_fields(headers, (*BASE_FIELDS, "variant", *metrics), input_path)
        if args.gate_profile:
            _validate_entity_gate_provenance(headers, raw_rows, input_path)
        comparisons, explicit = _comparison_definitions(args.comparison, args.manifest.resolve())
        paired_rows, input_diagnostics = _paired_from_entity_rows(
            raw_rows, comparisons, metrics, args.missing_policy, explicit
        )
        comparison_source = "command_line" if explicit else str(args.manifest.resolve())
    else:
        raise ValueError(
            "Input is neither entity_seed_score_metrics.csv (variant column) nor "
            "paired_entity_seed_delta.csv (comparison/lhs/rhs columns)."
        )

    summaries = _bootstrap_summaries(
        paired_rows,
        metrics,
        args.n_bootstrap,
        args.seed,
        args.ci_level,
        input_diagnostics,
        args.family_mode,
    )
    gates: List[Dict[str, Any]] = []
    required_families = args.required_family or (["HAI21", "SMD"] if args.gate_profile == "p4" else [])
    if args.gate_profile:
        gates = _gate_diagnostics(
            paired_rows,
            summaries,
            args.gate_profile,
            args.primary_comparison,
            required_families,
            args.missing_policy,
            args.minimum_gate_bootstrap,
        )

    output_dir = (args.output_dir or (input_path.parent / "bootstrap")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "run_name", "group", "comparison", "lhs", "rhs", "metric", "mean",
        "paired_row_std", "bootstrap_mean", "bootstrap_std", "ci_level", "ci_lower",
        "ci_upper", "n_bootstrap", "base_seed", "stream_seed", "n_families",
        "n_entities", "n_pairs", "min_entities_per_family", "max_entities_per_family",
        "min_seeds_per_entity", "max_seeds_per_entity", "family_mode", "balanced_seed_grid",
        "reliability", "warnings",
    ]
    diagnostic_fields = [
        "run_name", "group", "comparison", "lhs", "rhs", "status",
        "n_complete_pairs", "n_missing_lhs", "n_missing_rhs", "n_dropped_identities", "detail",
    ]
    _write_csv(output_dir / "hierarchical_bootstrap.csv", summaries, summary_fields)
    _write_csv(output_dir / "input_diagnostics.csv", input_diagnostics, diagnostic_fields)
    if gates:
        _write_csv(
            output_dir / "gate_diagnostics.csv",
            gates,
            [
                "run_name", "group", "profile", "comparison", "criterion", "metric",
                "scope", "observed_value", "operator", "threshold", "status", "detail",
            ],
        )
    _write_json(
        output_dir / "bootstrap_metadata.json",
        {
            "schema_version": 1,
            "input": str(input_path),
            "input_sha256": _file_sha256(input_path),
            "input_type": input_type,
            "comparison_source": comparison_source,
            "metrics": metrics,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "ci_level": args.ci_level,
            "family_mode": args.family_mode,
            "missing_policy": args.missing_policy,
            "minimum_gate_bootstrap": args.minimum_gate_bootstrap,
            "gate_profile": args.gate_profile,
            "primary_comparison": args.primary_comparison,
            "required_families": required_families,
            "statistic": (
                "fixed equal-family strata of resampled entity/seed means"
                if args.family_mode == "fixed"
                else "resampled equal family mean of resampled entity/seed means"
            ),
            "selection_policy": "no value-based metric/comparison/threshold selection",
        },
    )
    limited = sum(row["reliability"] == "limited" for row in summaries)
    print(
        f"Wrote {len(summaries)} comparison/metric bootstrap rows to {output_dir}; "
        f"limited-reliability rows={limited}."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
