#!/usr/bin/env python3
"""Evaluate aligned scores on the PatternAD contextual synthetic suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "patternad" / "synthetic_suite.json"
DEFAULT_FACTORIAL = REPO_ROOT / "config" / "patternad" / "factorial_ablation.json"
MECHANISM_ORDER = (
    "same_deviation_different_context",
    "slow_drift_vs_abrupt_shift",
    "dependency_break",
    "context_ood",
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "patternad_synthetic"


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


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return normalized or "unnamed"


def conformal_upper_threshold(calibration_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample upper conformal cutoff using calibration scores only."""
    scores = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be non-empty and finite.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("target_fpr must be between zero and one.")
    if alpha == 0.0:
        return math.inf
    if alpha == 1.0:
        return -math.inf
    rank = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    if rank > scores.size:
        return math.inf
    return float(np.partition(scores, rank - 1)[rank - 1])


def _score_path(
    artifact_dir: Path, score_dir: Optional[Path], mechanism: str
) -> Path:
    directory = artifact_dir if score_dir is None else score_dir
    return directory / f"{mechanism}.npz"


def run_patternad_variant(
    config: Mapping[str, Any],
    artifact_dir: Path,
    factorial_manifest: Path,
    variant: str,
    seed: int,
    score_dir: Path,
    num_epochs: Optional[int] = None,
) -> Path:
    """Fit one frozen variant once and export aligned scores for all mechanisms."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ts_benchmark.baselines.PatternAD.PatternAD import PatternAD
    from ts_benchmark.utils.random_utils import fix_random_seed

    factorial = _load_json(factorial_manifest)
    if variant not in factorial.get("variants", {}):
        raise ValueError(f"Unknown factorial variant {variant!r}.")
    hyperparameters = dict(factorial["shared_hyperparameters"])
    hyperparameters.update(factorial["variants"][variant]["hyperparameters"])
    hyperparameters["train_mask_seed"] = int(seed)
    if num_epochs is not None:
        if num_epochs < 1:
            raise ValueError("--num-epochs must be positive.")
        hyperparameters["num_epochs"] = int(num_epochs)
    expected_window = int(config["evaluation"].get("score_window_length", 1))
    if int(hyperparameters.get("seq_len", expected_window)) != expected_window:
        raise ValueError(
            "Synthetic FPR guards were frozen for score_window_length="
            f"{expected_window}, but the selected variant uses seq_len="
            f"{hyperparameters.get('seq_len')}. Regenerate with a matching config."
        )

    train_length = int(config["train_length"])
    test_length = int(config["test_length"])
    evaluation = config["evaluation"]
    calibration_length = int(
        math.ceil(train_length * float(evaluation["calibration_fraction"]))
    )
    calibration_start = train_length - calibration_length
    fit_end = calibration_start - int(evaluation["calibration_gap"])
    first_path = artifact_dir / f"{MECHANISM_ORDER[0]}.npz"
    with np.load(first_path, allow_pickle=False) as payload:
        first_values = np.asarray(payload["values"], dtype=np.float64)
        first_labels = np.asarray(payload["labels"], dtype=np.uint8).reshape(-1)
    if np.any(first_labels[:train_length]):
        raise ValueError("The common PatternAD fit split is contaminated.")
    for mechanism in MECHANISM_ORDER[1:]:
        with np.load(
            artifact_dir / f"{mechanism}.npz", allow_pickle=False
        ) as payload:
            current_train = np.asarray(payload["values"], dtype=np.float64)[
                :train_length
            ]
        if not np.array_equal(current_train, first_values[:train_length]):
            raise ValueError("All mechanisms must share an identical clean train split.")
    index = pd.date_range("2024-01-01", periods=train_length + test_length, freq="min")
    columns = list(config["channel_names"])
    fit_data = pd.DataFrame(first_values[:fit_end], index=index[:fit_end], columns=columns)
    fit_text = pd.DataFrame(
        np.zeros((fit_end, 1), dtype=np.float32), index=index[:fit_end], columns=["text"]
    )
    fit_labels = pd.DataFrame(
        np.zeros((fit_end, 1), dtype=np.uint8), index=index[:fit_end], columns=["label"]
    )
    calibration_data = pd.DataFrame(
        first_values[calibration_start:train_length],
        index=index[calibration_start:train_length],
        columns=columns,
    )
    calibration_text = pd.DataFrame(
        np.zeros((calibration_length, 1), dtype=np.float32),
        index=index[calibration_start:train_length],
        columns=["text"],
    )

    fix_random_seed(seed)
    model = PatternAD(**hyperparameters)
    model.detect_multi_fit(fit_data, fit_text, fit_labels)
    calibration_score = np.asarray(
        model.detect_multi_score(calibration_data, calibration_text)[0], dtype=np.float64
    ).reshape(-1)
    calibration_components = model.get_last_score_components()
    if calibration_score.size != calibration_length:
        raise RuntimeError("PatternAD returned a misaligned calibration score.")
    if not calibration_components:
        raise RuntimeError("PatternAD did not return score decomposition components.")
    for name, values in calibration_components.items():
        if np.asarray(values).reshape(-1).size != calibration_length:
            raise RuntimeError(
                f"PatternAD calibration component {name!r} is misaligned."
            )

    score_dir.mkdir(parents=True, exist_ok=True)
    filler = float(np.median(calibration_score))
    for mechanism in MECHANISM_ORDER:
        with np.load(artifact_dir / f"{mechanism}.npz", allow_pickle=False) as payload:
            values = np.asarray(payload["values"], dtype=np.float64)
        test_data = pd.DataFrame(
            values[train_length:], index=index[train_length:], columns=columns
        )
        test_text = pd.DataFrame(
            np.zeros((test_length, 1), dtype=np.float32),
            index=index[train_length:],
            columns=["text"],
        )
        test_score = np.asarray(
            model.detect_multi_score(test_data, test_text)[0], dtype=np.float64
        ).reshape(-1)
        test_components = model.get_last_score_components()
        if test_score.size != test_length:
            raise RuntimeError(f"PatternAD returned a misaligned score for {mechanism}.")
        aligned = np.full(train_length + test_length, filler, dtype=np.float64)
        aligned[calibration_start:train_length] = calibration_score
        aligned[train_length:] = test_score
        aligned_arrays = {"score": aligned}
        if set(test_components) != set(calibration_components):
            raise RuntimeError("PatternAD score component keys changed between calls.")
        for name in sorted(calibration_components):
            calibration_values = np.asarray(
                calibration_components[name], dtype=np.float64
            ).reshape(-1)
            test_values = np.asarray(test_components[name], dtype=np.float64).reshape(-1)
            if test_values.size != test_length:
                raise RuntimeError(
                    f"PatternAD test component {name!r} is misaligned."
                )
            component_aligned = np.full(
                train_length + test_length,
                float(np.median(calibration_values)),
                dtype=np.float64,
            )
            component_aligned[calibration_start:train_length] = calibration_values
            component_aligned[train_length:] = test_values
            if not np.isfinite(component_aligned).all():
                raise RuntimeError(
                    f"PatternAD score component {name!r} contains non-finite values."
                )
            aligned_arrays[name] = component_aligned
        np.savez_compressed(score_dir / f"{mechanism}.npz", **aligned_arrays)
    run_metadata = {
        "schema_version": 1,
        "variant": variant,
        "seed": int(seed),
        "factorial_manifest": str(factorial_manifest.resolve()),
        "factorial_manifest_hash": _canonical_hash(factorial),
        "synthetic_config_hash": _canonical_hash(config),
        "fit_range": [0, fit_end],
        "calibration_range": [calibration_start, train_length],
        "test_range": [train_length, train_length + test_length],
        "hyperparameters": hyperparameters,
        "score_key": "score",
        "component_keys": sorted(calibration_components),
    }
    with (score_dir / "score_run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    return score_dir


def _load_score(path: Path, key: str, expected_length: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing aligned score file: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload.files:
            raise KeyError(
                f"Score file {path} has no key {key!r}; available={payload.files}."
            )
        score = np.asarray(payload[key], dtype=np.float64).reshape(-1)
    if score.size != expected_length:
        raise ValueError(
            f"Score length for {path.name} is {score.size}; expected {expected_length}. "
            "Scores must cover official train followed by test."
        )
    if not np.isfinite(score).all():
        raise ValueError(f"Score file {path} contains NaN or infinity.")
    return score


def _event_map(metadata: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    events = {event["event_id"]: event for event in metadata.get("events", [])}
    if len(events) != len(metadata.get("events", [])):
        raise ValueError(f"Duplicate event id in {metadata.get('mechanism')} metadata.")
    return events


def _ordering_rows(
    mechanism: str,
    scores: np.ndarray,
    metadata: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    events = _event_map(metadata)
    rows = []
    for ordering in metadata.get("orderings", []):
        higher = events[ordering["higher_event"]]
        lower = events[ordering["lower_event"]]
        higher_score = float(scores[int(higher["start"]) : int(higher["end"])].mean())
        lower_score = float(scores[int(lower["start"]) : int(lower["end"])].mean())
        rows.append(
            {
                "mechanism": mechanism,
                "ordering": ordering["name"],
                "hypothesis": ordering["hypothesis"],
                "higher_event": ordering["higher_event"],
                "lower_event": ordering["lower_event"],
                "higher_score": higher_score,
                "lower_score": lower_score,
                "margin": higher_score - lower_score,
                "correct": bool(higher_score > lower_score),
            }
        )
    return rows


def evaluate_scores(
    config: Mapping[str, Any],
    artifact_dir: Path,
    score_dir: Optional[Path],
    score_key: str,
    method_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_length = int(config["train_length"])
    total_length = train_length + int(config["test_length"])
    evaluation = config["evaluation"]
    calibration_length = int(math.ceil(train_length * float(evaluation["calibration_fraction"])))
    calibration_start = train_length - calibration_length
    gap = int(evaluation["calibration_gap"])
    fit_end = calibration_start - gap
    if calibration_length < 1 or fit_end < 1:
        raise ValueError("Invalid fit/gap/calibration split in synthetic suite config.")

    target_fpr = float(evaluation["target_fpr"])
    expected_config_hash = _canonical_hash(config)
    mechanism_rows: List[Dict[str, Any]] = []
    ordering_rows: List[Dict[str, Any]] = []
    thresholds = []
    for mechanism in MECHANISM_ORDER:
        artifact_path = artifact_dir / f"{mechanism}.npz"
        metadata_path = artifact_dir / f"{mechanism}.metadata.json"
        metadata = _load_json(metadata_path)
        if metadata.get("config_hash") != expected_config_hash:
            raise ValueError(
                f"Generated artifact {metadata_path} does not match the current config."
            )
        with np.load(artifact_path, allow_pickle=False) as artifact:
            values = np.asarray(artifact["values"], dtype=np.float64)
            clean_values = np.asarray(
                artifact["clean_values"], dtype=np.float64
            )
            labels = np.asarray(artifact["labels"], dtype=np.uint8).reshape(-1)
            regimes = np.asarray(artifact["regime"], dtype=np.int8).reshape(-1)
            fpr_eligible = np.asarray(
                artifact["fpr_eligible"], dtype=bool
            ).reshape(-1)
            split_index = int(np.asarray(artifact["split_index"]).item())
        if split_index != train_length:
            raise ValueError(f"Artifact {artifact_path} has the wrong split index.")
        if not (
            labels.size == regimes.size == fpr_eligible.size == total_length
            and values.shape == clean_values.shape
            and values.shape[0] == total_length
        ):
            raise ValueError(f"Artifact arrays have inconsistent lengths: {artifact_path}")
        if np.any(labels[:train_length]):
            raise ValueError(f"Official train split is contaminated in {artifact_path}.")

        score_path = _score_path(artifact_dir, score_dir, mechanism)
        scores = _load_score(score_path, score_key, total_length)
        calibration_scores = scores[calibration_start:train_length]
        threshold = conformal_upper_threshold(calibration_scores, target_fpr)
        thresholds.append(threshold)
        test_slice = slice(train_length, total_length)
        test_labels = labels[test_slice]
        test_scores = scores[test_slice]
        if np.unique(test_labels).size != 2:
            raise ValueError(f"{mechanism} test split must contain both classes.")
        average_precision = float(average_precision_score(test_labels, test_scores))
        test_prevalence = float(np.mean(test_labels))

        fprs: Dict[int, float] = {}
        counts: Dict[int, int] = {}
        for regime_id in (0, 1):
            eligible = (
                (regimes == regime_id)
                & (labels == 0)
                & fpr_eligible
                & (np.arange(total_length) >= train_length)
            )
            counts[regime_id] = int(eligible.sum())
            if counts[regime_id] == 0:
                raise ValueError(f"No eligible regime-{regime_id} normal test points.")
            fprs[regime_id] = float(np.mean(scores[eligible] > threshold))
        regime_fpr_gap = abs(fprs[0] - fprs[1])

        current_orderings = _ordering_rows(mechanism, scores, metadata)
        raw_control_score = np.mean((values - clean_values) ** 2, axis=1)
        raw_control_orderings = {
            row["ordering"]: row
            for row in _ordering_rows(mechanism, raw_control_score, metadata)
        }
        for row in current_orderings:
            raw_row = raw_control_orderings[row["ordering"]]
            row["raw_control_margin"] = raw_row["margin"]
            row["raw_control_tied"] = bool(
                abs(raw_row["margin"])
                <= float(evaluation["maximum_abs_raw_control_margin"])
            )
        ordering_rows.extend(current_orderings)
        ordering_rate = (
            float(np.mean([row["correct"] for row in current_orderings]))
            if current_orderings
            else None
        )
        mechanism_rows.append(
            {
                "method": method_name,
                "mechanism": mechanism,
                "average_precision": average_precision,
                "test_prevalence": test_prevalence,
                "ap_over_prevalence": average_precision - test_prevalence,
                "calibration_threshold": threshold,
                "calibration_count": calibration_length,
                "fit_end": fit_end,
                "regime_0_fpr": fprs[0],
                "regime_1_fpr": fprs[1],
                "regime_0_count": counts[0],
                "regime_1_count": counts[1],
                "regime_fpr_gap": regime_fpr_gap,
                "matched_ordering_rate": ordering_rate,
                "score_file": str(score_path),
                "score_sha256": _file_sha256(score_path),
                "score_key": score_key,
            }
        )

    ordering_rate = float(np.mean([row["correct"] for row in ordering_rows]))
    maximum_abs_raw_control_margin = float(
        max(abs(row["raw_control_margin"]) for row in ordering_rows)
    )
    maximum_fpr_gap = float(max(row["regime_fpr_gap"] for row in mechanism_rows))
    summary = {
        "schema_version": 1,
        "suite_id": config["suite_id"],
        "config_hash": expected_config_hash,
        "method": method_name,
        "score_key": score_key,
        "score_source": str(artifact_dir if score_dir is None else score_dir),
        "threshold_provenance": {
            "source": "official_train_calibration_tail_only",
            "calibration_start": calibration_start,
            "calibration_end": train_length,
            "fit_end": fit_end,
            "gap": gap,
            "target_fpr": target_fpr,
            "test_scores_used_for_threshold": False,
            "fpr_exclusion_guard": int(
                evaluation.get("score_window_length", 1)
            )
            - 1,
        },
        "macro_average_precision": float(
            np.mean([row["average_precision"] for row in mechanism_rows])
        ),
        "matched_ordering_rate": ordering_rate,
        "maximum_abs_raw_control_margin": maximum_abs_raw_control_margin,
        "maximum_regime_fpr_gap": maximum_fpr_gap,
        "gates": {
            "scope": (
                "Contract checks for matched ordering and normal-regime FPR only; "
                "they do not certify per-mechanism detectability. Inspect every AP "
                "and AP-minus-prevalence value. Context OOD is intentionally a "
                "negative control for a conditional-only score."
            ),
            "ordering": {
                "required": float(evaluation["minimum_ordering_rate"]),
                "observed": ordering_rate,
                "pass": bool(ordering_rate >= float(evaluation["minimum_ordering_rate"])),
            },
            "raw_magnitude_negative_control": {
                "required_maximum_abs_margin": float(
                    evaluation["maximum_abs_raw_control_margin"]
                ),
                "observed": maximum_abs_raw_control_margin,
                "pass": bool(
                    maximum_abs_raw_control_margin
                    <= float(evaluation["maximum_abs_raw_control_margin"])
                ),
            },
            "regime_fpr_gap": {
                "required_maximum": float(evaluation["maximum_regime_fpr_gap"]),
                "observed": maximum_fpr_gap,
                "pass": bool(
                    maximum_fpr_gap
                    <= float(evaluation["maximum_regime_fpr_gap"])
                ),
            },
            "mechanism_ap_over_prevalence": {
                mechanism: {
                    "required": float(required),
                    "observed": next(
                        row["ap_over_prevalence"]
                        for row in mechanism_rows
                        if row["mechanism"] == mechanism
                    ),
                    "pass": bool(
                        next(
                            row["ap_over_prevalence"]
                            for row in mechanism_rows
                            if row["mechanism"] == mechanism
                        )
                        >= float(required)
                    ),
                }
                for mechanism, required in evaluation.get(
                    "minimum_ap_over_prevalence", {}
                ).items()
            },
        },
        "mechanism_ap_diagnostics": {
            row["mechanism"]: {
                "average_precision": row["average_precision"],
                "test_prevalence": row["test_prevalence"],
                "ap_over_prevalence": row["ap_over_prevalence"],
                "interpretation": (
                    "negative_control_for_conditional_only_score"
                    if row["mechanism"] == "context_ood"
                    else "must_be_reported_not_hidden_by_contract_checks"
                ),
            }
            for row in mechanism_rows
        },
        "mechanisms": mechanism_rows,
        "matched_orderings": ordering_rows,
    }
    return summary, mechanism_rows, ordering_rows


def component_ordering_rows(
    config: Mapping[str, Any],
    artifact_dir: Path,
    score_dir: Optional[Path],
    primary_score_key: str,
) -> List[Dict[str, Any]]:
    if score_dir is None:
        return []
    total_length = int(config["train_length"]) + int(config["test_length"])
    rows: List[Dict[str, Any]] = []
    expected_keys: Optional[set] = None
    for mechanism in MECHANISM_ORDER:
        metadata = _load_json(artifact_dir / f"{mechanism}.metadata.json")
        if not metadata.get("orderings"):
            continue
        score_path = score_dir / f"{mechanism}.npz"
        with np.load(score_path, allow_pickle=False) as payload:
            component_keys = set(payload.files) - {primary_score_key}
            if expected_keys is None:
                expected_keys = component_keys
            elif component_keys != expected_keys:
                raise ValueError("Score component keys differ across mechanisms.")
            for component in sorted(component_keys):
                values = np.asarray(payload[component], dtype=np.float64).reshape(-1)
                if values.size != total_length or not np.isfinite(values).all():
                    raise ValueError(
                        f"Invalid component {component!r} in {score_path}."
                    )
                for row in _ordering_rows(mechanism, values, metadata):
                    row["component"] = component
                    rows.append(row)
    return rows


def write_evaluation(
    output_dir: Path,
    summary: Mapping[str, Any],
    mechanism_rows: Sequence[Mapping[str, Any]],
    ordering_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]] = (),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "contextual_evaluation.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    pd.DataFrame(mechanism_rows).to_csv(
        output_dir / "mechanism_metrics.csv", index=False
    )
    pd.DataFrame(ordering_rows).to_csv(
        output_dir / "matched_orderings.csv", index=False
    )
    component_path = output_dir / "score_component_orderings.csv"
    if component_rows:
        pd.DataFrame(component_rows).to_csv(component_path, index=False)
    elif component_path.exists():
        component_path.unlink()
    return summary_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Suite config; defaults to artifact resolved_config.json when present.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Generated suite directory; defaults to output.artifact_dir in config.",
    )
    parser.add_argument(
        "--score-dir",
        type=Path,
        help=(
            "Directory containing one <mechanism>.npz per mechanism. Omit to read "
            "a reference score embedded in generated artifacts."
        ),
    )
    parser.add_argument(
        "--score-key",
        help="NPZ key. External files default to config evaluation.score_key.",
    )
    parser.add_argument("--method-name", default="unnamed")
    parser.add_argument(
        "--patternad-variant",
        help="Fit and score one variant from factorial_ablation.json before evaluation.",
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--factorial-manifest", type=Path, default=DEFAULT_FACTORIAL)
    parser.add_argument(
        "--generated-score-dir",
        type=Path,
        help="Output for --patternad-variant scores; defaults below --output-dir.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Explicit smoke-only epoch override; omit for the frozen manifest value.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to an isolated generator/method/model-seed directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing evaluation for the exact same output identity.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_config = _load_json(
        args.config.resolve() if args.config is not None else DEFAULT_CONFIG
    )
    artifact_dir = (
        args.artifact_dir.resolve()
        if args.artifact_dir is not None
        else _resolve_repo_path(base_config["output"]["artifact_dir"])
    )
    resolved_config = artifact_dir / "resolved_config.json"
    config = (
        _load_json(resolved_config)
        if args.config is None and resolved_config.is_file()
        else base_config
    )
    if args.score_dir is not None and args.patternad_variant is not None:
        raise ValueError("Use either --score-dir or --patternad-variant, not both.")
    score_dir = args.score_dir.resolve() if args.score_dir is not None else None
    method_name = args.method_name
    if method_name == "unnamed":
        if args.patternad_variant is not None:
            method_name = f"{args.patternad_variant}_seed_{args.seed}"
        elif score_dir is None:
            method_name = "oracle_contract"
        else:
            method_name = "external_scores"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT
        / _safe_component(method_name)
        / f"generator_seed_{int(config['seed'])}"
        / (f"model_seed_{args.seed}" if args.patternad_variant is not None else "evaluation")
    )
    existing_summary = output_dir / "contextual_evaluation.json"
    if existing_summary.exists() and not args.overwrite:
        raise FileExistsError(
            f"Evaluation already exists: {existing_summary}. Use a new output "
            "identity or pass --overwrite explicitly."
        )
    if args.patternad_variant is not None:
        generated_score_dir = (
            args.generated_score_dir.resolve()
            if args.generated_score_dir is not None
            else output_dir / "scores"
        )
        score_dir = run_patternad_variant(
            config,
            artifact_dir,
            args.factorial_manifest.resolve(),
            args.patternad_variant,
            args.seed,
            generated_score_dir,
            args.num_epochs,
        )
    if args.score_key is not None:
        score_key = args.score_key
    elif score_dir is None:
        score_key = "oracle_context_score"
    else:
        score_key = str(config["evaluation"]["score_key"])
    summary, mechanism_rows, ordering_rows = evaluate_scores(
        config,
        artifact_dir,
        score_dir,
        score_key,
        method_name,
    )
    component_rows = component_ordering_rows(
        config, artifact_dir, score_dir, score_key
    )
    path = write_evaluation(
        output_dir,
        summary,
        mechanism_rows,
        ordering_rows,
        component_rows,
    )
    print(f"Wrote contextual evaluation: {path}")
    print(
        "macro_AP={:.6f} ordering_rate={:.3f} max_regime_FPR_gap={:.6f}".format(
            summary["macro_average_precision"],
            summary["matched_ordering_rate"],
            summary["maximum_regime_fpr_gap"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
