#!/usr/bin/env python3
"""One controlled gate for target-blind adaptive decomposition on CATCH."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ts_benchmark.baselines.apd_catch.APDCATCH import APDCATCH, NextPointDataset


SEEDS = (20261, 20262, 20263)
VARIANTS = ("causal_catch", "fixed", "adaptive")
ANOMALIES = ("spike", "level", "periodic", "relation")


def _mode_schedule(length: int, rng: np.random.Generator) -> np.ndarray:
    schedule = np.empty(length, dtype=np.int64)
    start = 0
    mode = int(rng.integers(0, 3))
    while start < length:
        end = min(length, start + int(rng.integers(220, 361)))
        schedule[start:end] = mode
        mode = int(rng.choice([candidate for candidate in range(3) if candidate != mode]))
        start = end
    return schedule


def generate_normal_series(
    length: int, rng: np.random.Generator, channels: int = 4
) -> np.ndarray:
    regimes = _mode_schedule(length, rng)
    target_level = np.asarray((0.0, 0.75, -0.55))[regimes]
    target_amplitude = np.asarray((0.85, 1.35, 1.05))[regimes]
    target_frequency = np.asarray((0.047, 0.079, 0.033))[regimes]
    target_noise = np.asarray((0.06, 0.10, 0.08))[regimes]

    level = np.empty(length)
    amplitude = np.empty(length)
    frequency = np.empty(length)
    level[0] = target_level[0]
    amplitude[0] = target_amplitude[0]
    frequency[0] = target_frequency[0]
    for index in range(1, length):
        level[index] = 0.94 * level[index - 1] + 0.06 * target_level[index]
        amplitude[index] = 0.94 * amplitude[index - 1] + 0.06 * target_amplitude[index]
        frequency[index] = 0.94 * frequency[index - 1] + 0.06 * target_frequency[index]

    phase = 2.0 * np.pi * np.cumsum(frequency)
    shared_noise = rng.normal(size=length)
    values = np.empty((length, channels), dtype=np.float64)
    for channel in range(channels):
        channel_scale = np.linspace(0.85, 1.18, channels)[channel]
        carrier = np.sin(phase + np.linspace(0.0, 0.85, channels)[channel])
        harmonic = 0.28 * np.sin(
            0.5 * phase + np.linspace(0.1, 0.7, channels)[channel]
        )
        values[:, channel] = channel_scale * (
            level + amplitude * (carrier + harmonic)
        )
        values[:, channel] += target_noise * (
            0.55 * shared_noise + 0.45 * rng.normal(size=length)
        )
    return values.astype(np.float32)


def inject_anomaly(values: np.ndarray, anomaly: str) -> Tuple[np.ndarray, np.ndarray]:
    result = values.copy()
    labels = np.zeros(len(values), dtype=np.int64)
    start = len(values) // 2
    if anomaly == "spike":
        for index, position in enumerate(np.arange(start, start + 336, 24)):
            channel = index % result.shape[1]
            result[position, channel] += 3.0 if index % 2 == 0 else -3.0
            labels[position] = 1
    elif anomaly == "level":
        end = start + 320
        result[start:end, :2] += np.linspace(0.0, 1.55, end - start)[:, None]
        labels[start:end] = 1
    elif anomaly == "periodic":
        end = start + 448
        time_index = np.arange(end - start)
        component = 0.46 * np.sin(2.0 * np.pi * 0.19 * time_index)
        result[start:end] += component[:, None] * np.linspace(
            0.8, 1.15, result.shape[1]
        )
        labels[start:end] = 1
    elif anomaly == "relation":
        end = start + 360
        original = result[start:end, 0].copy()
        blend = np.ones(end - start)
        blend[:56] = np.linspace(0.0, 1.0, 56)
        result[start:end, 0] = (1.0 - blend) * original + blend * (-original)
        labels[start:end] = 1
    else:
        raise ValueError(f"unsupported anomaly family {anomaly!r}")
    return result, labels


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / (np.arange(len(ranked)) + 1.0)
    return float(precision[ranked == 1].sum() / positives)


def threshold_at_fpr(scores: np.ndarray, fpr: float = 0.01) -> float:
    return float(np.quantile(scores, 1.0 - fpr))


def recall_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> float:
    return float(((scores > threshold) & (labels == 1)).sum() / labels.sum())


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def score_trace(model: APDCATCH, values: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(values)
    score, _ = model.detect_score(frame)
    return score[model.config.seq_len :]


@torch.no_grad()
def model_diagnostics(model: APDCATCH, values: np.ndarray) -> Dict[str, float]:
    dataset = NextPointDataset(values, model.config.seq_len)
    sample_count = min(len(dataset), 64)
    history = torch.stack([dataset[index][0] for index in range(sample_count)])
    model.model.eval()
    first = model.model(history.to(model.device))
    second = model.model(history.clone().to(model.device))
    return {
        "cutoff_mean": float(first["cutoff"].mean().cpu()),
        "cutoff_std": float(first["cutoff"].std(unbiased=False).cpu()),
        "partition_error": float(first["partition_error"].cpu()),
        "repeat_mean_max_diff": float(
            (first["mean"] - second["mean"]).abs().max().cpu()
        ),
        "repeat_scale_max_diff": float(
            (first["scale"] - second["scale"]).abs().max().cpu()
        ),
        "repeat_cutoff_max_diff": float(
            (first["cutoff"] - second["cutoff"]).abs().max().cpu()
        ),
    }


def controlled_hyperparameters(seed: int, smoke: bool) -> Dict[str, object]:
    if smoke:
        return {
            "seq_len": 64,
            "patch_size": 8,
            "patch_stride": 4,
            "cf_dim": 8,
            "d_model": 8,
            "d_ff": 16,
            "e_layers": 1,
            "n_heads": 2,
            "head_dim": 4,
            "dropout": 0.0,
            "head_dropout": 0.0,
            "batch_size": 128,
            "num_epochs": 1,
            "patience": 1,
            "mask_update_interval": 2,
            "seed": seed,
        }
    return {
        "seq_len": 128,
        "patch_size": 8,
        "patch_stride": 4,
        "cf_dim": 32,
        "d_model": 32,
        "d_ff": 64,
        "e_layers": 2,
        "n_heads": 4,
        "head_dim": 8,
        "dropout": 0.1,
        "head_dropout": 0.1,
        "batch_size": 256,
        "num_epochs": 8,
        "patience": 3,
        "mask_update_interval": 5,
        "seed": seed,
    }


def run_seed(
    seed: int,
    train_length: int,
    calibration_length: int,
    test_length: int,
    smoke: bool,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    train = generate_normal_series(train_length, rng)
    calibration = generate_normal_series(calibration_length, rng)
    normal_test = generate_normal_series(test_length, rng)
    anomaly_traces = {
        anomaly: inject_anomaly(generate_normal_series(test_length, rng), anomaly)
        for anomaly in ANOMALIES
    }

    variant_results: Dict[str, object] = {}
    parameter_counts = []
    for variant in VARIANTS:
        print(f"seed={seed} variant={variant}", flush=True)
        model = APDCATCH(variant=variant, **controlled_hyperparameters(seed, smoke))
        model.detect_fit(pd.DataFrame(train))
        parameter_counts.append(model.fit_summary.trainable_parameters)
        calibration_score = score_trace(model, calibration)
        threshold = threshold_at_fpr(calibration_score)
        normal_score = score_trace(model, normal_test)
        anomaly_results = {}
        for anomaly, (values, labels) in anomaly_traces.items():
            score = score_trace(model, values)
            aligned_labels = labels[model.config.seq_len :]
            anomaly_results[anomaly] = {
                "average_precision": average_precision(aligned_labels, score),
                "recall_at_1pct_fpr": recall_at_threshold(
                    aligned_labels, score, threshold
                ),
            }
        variant_results[variant] = {
            "fit": vars(model.fit_summary),
            "normal_fpr": float((normal_score > threshold).mean()),
            "diagnostics": model_diagnostics(model, calibration),
            "anomalies": anomaly_results,
        }

    if len(set(parameter_counts)) != 1:
        raise AssertionError(f"variant parameter counts differ: {parameter_counts}")
    return {
        "seed": seed,
        "trainable_parameters": parameter_counts[0],
        "variants": variant_results,
    }


def aggregate(seed_results: List[Mapping[str, object]]) -> Dict[str, object]:
    variants: Dict[str, object] = {}
    for variant in VARIANTS:
        variants[variant] = {
            "normal_fpr": mean_std(
                [result["variants"][variant]["normal_fpr"] for result in seed_results]
            ),
            "anomalies": {
                anomaly: {
                    metric: mean_std(
                        [
                            result["variants"][variant]["anomalies"][anomaly][
                                metric
                            ]
                            for result in seed_results
                        ]
                    )
                    for metric in ("average_precision", "recall_at_1pct_fpr")
                }
                for anomaly in ANOMALIES
            },
        }

    mean_ap = {
        variant: float(
            np.mean(
                [
                    variants[variant]["anomalies"][anomaly]["average_precision"][
                        "mean"
                    ]
                    for anomaly in ANOMALIES
                ]
            )
        )
        for variant in VARIANTS
    }
    family_wins = [
        anomaly
        for anomaly in ANOMALIES
        if variants["adaptive"]["anomalies"][anomaly]["average_precision"][
            "mean"
        ]
        > variants["causal_catch"]["anomalies"][anomaly]["average_precision"][
            "mean"
        ]
    ]
    invariant_pass = all(
        result["variants"][variant]["diagnostics"][name] == 0.0
        for result in seed_results
        for variant in VARIANTS
        for name in (
            "repeat_mean_max_diff",
            "repeat_scale_max_diff",
            "repeat_cutoff_max_diff",
        )
    )
    partition_pass = all(
        result["variants"]["adaptive"]["diagnostics"]["partition_error"]
        <= 1e-7
        for result in seed_results
    )
    gates = {
        "normal_fpr_pass": variants["adaptive"]["normal_fpr"]["mean"] <= 0.03,
        "adaptive_vs_causal_pass": mean_ap["adaptive"] > mean_ap["causal_catch"],
        "adaptive_vs_fixed_pass": mean_ap["adaptive"] > mean_ap["fixed"],
        "family_wins_vs_causal": family_wins,
        "at_least_two_family_wins_pass": len(family_wins) >= 2,
        "target_blind_repeat_pass": invariant_pass,
        "partition_pass": partition_pass,
        "mean_average_precision": mean_ap,
    }
    gates["overall_pass"] = all(
        gates[name]
        for name in (
            "normal_fpr_pass",
            "adaptive_vs_causal_pass",
            "adaptive_vs_fixed_pass",
            "at_least_two_family_wins_pass",
            "target_blind_repeat_pass",
            "partition_pass",
        )
    )
    return {"variants": variants, "gates": gates}


def write_csv(path: Path, summary: Mapping[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "variant",
                "anomaly",
                "average_precision_mean",
                "average_precision_std",
                "recall_at_1pct_fpr_mean",
                "recall_at_1pct_fpr_std",
                "normal_fpr_mean",
                "normal_fpr_std",
            )
        )
        for variant in VARIANTS:
            normal = summary["variants"][variant]["normal_fpr"]
            for anomaly in ANOMALIES:
                metrics = summary["variants"][variant]["anomalies"][anomaly]
                writer.writerow(
                    (
                        variant,
                        anomaly,
                        metrics["average_precision"]["mean"],
                        metrics["average_precision"]["std"],
                        metrics["recall_at_1pct_fpr"]["mean"],
                        metrics["recall_at_1pct_fpr"]["std"],
                        normal["mean"],
                        normal["std"],
                    )
                )


def print_summary(summary: Mapping[str, object]) -> None:
    print("APD-CATCH controlled mechanism gate")
    for variant in VARIANTS:
        normal = summary["variants"][variant]["normal_fpr"]
        print(f"{variant}: normal FPR={normal['mean']:.3f} +/- {normal['std']:.3f}")
        for anomaly in ANOMALIES:
            metrics = summary["variants"][variant]["anomalies"][anomaly]
            print(
                f"  {anomaly:8s} AP={metrics['average_precision']['mean']:.3f} "
                f"R={metrics['recall_at_1pct_fpr']['mean']:.3f}"
            )
    print(json.dumps(summary["gates"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/analysis/apd_catch_mechanism_gate"),
    )
    parser.add_argument("--train-length", type=int, default=4800)
    parser.add_argument("--calibration-length", type=int, default=1800)
    parser.add_argument("--test-length", type=int, default=3200)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    seeds = SEEDS
    if args.smoke:
        seeds = SEEDS[:1]
        args.train_length = 900
        args.calibration_length = 700
        args.test_length = 1000
    seed_results = [
        run_seed(
            seed,
            args.train_length,
            args.calibration_length,
            args.test_length,
            args.smoke,
        )
        for seed in seeds
    ]
    summary = aggregate(seed_results)
    artifact = {
        "experiment": "apd_catch_target_blind_adaptive_decomposition_gate_v1",
        "seeds": list(seeds),
        "variants": list(VARIANTS),
        "anomaly_families": list(ANOMALIES),
        "protocol": "past_to_next_point; disjoint_normal_calibration; 1pct_fpr",
        "seed_results": seed_results,
        "summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "summary.csv"
    json_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(csv_path, summary)
    print_summary(summary)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
