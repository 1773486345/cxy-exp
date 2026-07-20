"""Read-only applicability analysis for frozen CATCH and MSD-CATCH results.

This script never imports a benchmark runner, trains a model, or scores a model.
It reads frozen archives, raw data, metadata, and existing score/checkpoint files
to describe where the fixed three-scale decomposition was helpful or harmful.
"""

from __future__ import annotations

import csv
import io
import json
import math
import pickle
import tarfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional


TASKS = [
    "PSM",
    "Genesis",
    "GECCO",
    "CalIt2",
    "NYC",
    "MSL",
    *[f"ASD_dataset_{index}" for index in range(1, 13)],
    "CICIDS",
    "Creditcard",
    "SMAP",
    "SMD",
    "SWAT",
]
PAPER_ORDER = [
    "PSM",
    "Genesis",
    "GECCO",
    "CalIt2",
    "NYC",
    "MSL",
    "ASD",
    "CICIDS",
    "Creditcard",
    "SMAP",
    "SMD",
    "SWAT",
]
FORMAL_MSD_JSON = {"CalIt2", "NYC", "MSL"}
FORMAL_MSD_LOG = {"PSM", "Genesis", "GECCO"}
EPSILON = 1e-8


def numeric(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def finite(value) -> bool:
    return numeric(value) is not None


def config_integer(params: Optional[dict], key: str) -> Optional[int]:
    value = numeric(params.get(key)) if isinstance(params, dict) else None
    if value is None or value < 1 or not value.is_integer():
        return None
    return int(value)


def paper_dataset(task: str) -> str:
    return "ASD" if task.startswith("ASD_dataset_") else task


def read_tar_row(path: Path) -> dict:
    with tarfile.open(path, "r:gz") as archive:
        member = next(member for member in archive.getmembers() if member.isfile())
        return next(csv.DictReader(io.TextIOWrapper(archive.extractfile(member), encoding="utf-8")))


def choose_valid_archive(paths: Iterable[Path]) -> tuple[Optional[Path], Optional[dict]]:
    candidates = []
    for path in paths:
        row = read_tar_row(path)
        if not row.get("log_info") and finite(row.get("auc_pr")) and finite(row.get("auc_roc")):
            candidates.append((path, row))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0].stat().st_mtime)


def msd_json_row(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    score = payload["scores"]["total_score"]
    return {
        "auc_pr": score["auc_pr"],
        "auc_roc": score["auc_roc"],
        "model_params": payload.get("config"),
        "strategy_args": None,
        "payload": payload,
    }


def msd_log_row(path: Path) -> Optional[dict]:
    """Parse the terminal formal result emitted by the three first-screen runs."""
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        scores = payload.get("scores")
        total = scores.get("total_score") if isinstance(scores, dict) else None
        if isinstance(total, dict) and finite(total.get("auc_pr")) and finite(total.get("auc_roc")):
            return {
                "auc_pr": total["auc_pr"],
                "auc_roc": total["auc_roc"],
                "model_params": None,
                "strategy_args": None,
                "payload": payload,
            }
    return None


def metric_group(delta: Optional[float]) -> str:
    if delta is None:
        return "N/A"
    if delta > 0.01:
        return "gain"
    if delta < -0.01:
        return "loss"
    return "neutral"


def discover_metrics(repo: Path, catch_root: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    sources = []
    for task in TASKS:
        catch_path, catch_row = choose_valid_archive(
            (catch_root / "result" / "score" / "CATCH" / task).rglob("CATCH*.tar.gz")
        )
        if catch_row is None:
            catch_pr = catch_roc = None
            catch_params = None
            catch_source = None
        else:
            catch_pr = numeric(catch_row["auc_pr"])
            catch_roc = numeric(catch_row["auc_roc"])
            catch_params = json.loads(catch_row["model_params"])
            catch_source = str(catch_path)
            sources.append(
                {
                    "task": task,
                    "artifact": "CATCH detect_score archive",
                    "path": catch_source,
                    "status": "used",
                    "notes": "Frozen score archive",
                }
            )

        msd_payload = None
        if task in FORMAL_MSD_JSON:
            msd_path = repo / "result" / "msd_catch_total_screen" / f"{task}.json"
            if msd_path.exists():
                msd_row = msd_json_row(msd_path)
                msd_source = str(msd_path)
                msd_kind = "formal total-screen JSON"
            else:
                msd_row = None
                msd_source = None
                msd_kind = "N/A"
        elif task in FORMAL_MSD_LOG:
            msd_path = repo / "result" / "msd_catch_screen" / f"{task}.log"
            if msd_path.exists():
                msd_row = msd_log_row(msd_path)
                msd_source = str(msd_path) if msd_row is not None else None
                msd_kind = "formal total-screen log" if msd_row is not None else "N/A"
            else:
                msd_row = None
                msd_source = None
                msd_kind = "N/A"
        else:
            msd_path, msd_tar_row = choose_valid_archive(
                (repo / "result" / "score" / "by_dataset" / task / "MSDCATCH").rglob(
                    "MSDCATCH*.tar.gz"
                )
            )
            if msd_tar_row is None:
                msd_row = None
                msd_source = None
                msd_kind = "N/A"
            else:
                msd_row = {
                    "auc_pr": msd_tar_row["auc_pr"],
                    "auc_roc": msd_tar_row["auc_roc"],
                    "model_params": json.loads(msd_tar_row["model_params"]),
                    "strategy_args": json.loads(msd_tar_row["strategy_args"]),
                    "payload": None,
                }
                msd_source = str(msd_path)
                msd_kind = "formal by-dataset archive"

        if msd_row is None:
            msd_pr = msd_roc = delta_pr = delta_roc = None
            parameter_match = "N/A"
            primary_score = "N/A"
            notes = "No valid formal MSD archive found."
        else:
            msd_pr = numeric(msd_row["auc_pr"])
            msd_roc = numeric(msd_row["auc_roc"])
            delta_pr = msd_pr - catch_pr if msd_pr is not None and catch_pr is not None else None
            delta_roc = msd_roc - catch_roc if msd_roc is not None and catch_roc is not None else None
            msd_payload = msd_row["payload"]
            msd_params = msd_row["model_params"]
            parameter_match = (
                "N/A" if msd_params is None else str(catch_params == msd_params).lower()
            )
            primary_score = (
                msd_payload.get("primary_score", "total_score") if msd_payload else "total_score"
            )
            notes = ""
            sources.append(
                {
                    "task": task,
                    "artifact": "MSD formal result",
                    "path": msd_source,
                    "status": "used",
                    "notes": msd_kind,
                }
            )
        rows.append(
            {
                "task": task,
                "paper_dataset": paper_dataset(task),
                "formal_seq_len": config_integer(catch_params, "seq_len"),
                "formal_patch_size": config_integer(catch_params, "patch_size"),
                "catch_auc_pr": catch_pr,
                "msd_auc_pr": msd_pr,
                "delta_auc_pr": delta_pr,
                "pr_group": metric_group(delta_pr),
                "catch_auc_roc": catch_roc,
                "msd_auc_roc": msd_roc,
                "delta_auc_roc": delta_roc,
                "roc_group": metric_group(delta_roc),
                "catch_source": catch_source,
                "msd_source": msd_source,
                "msd_source_kind": msd_kind,
                "parameter_match": parameter_match,
                "primary_score": primary_score,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows), sources


def metadata_lookup(repo: Path) -> dict[str, dict]:
    metadata = pd.read_csv(repo / "dataset" / "anomaly_detect" / "DETECT_META.csv")
    return {Path(row.file_name).stem: row.to_dict() for _, row in metadata.iterrows()}


def read_train_and_labels(path: Path, train_length: int, test_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Stream the same contiguous blocks selected by process_data_df in the formal loader."""
    series_length = train_length + test_length
    total_rows = sum(len(chunk) for chunk in pd.read_csv(path, usecols=["data"], chunksize=200_000))
    if series_length < 1 or total_rows % series_length:
        raise ValueError(
            f"{path.name} has {total_rows} rows, incompatible with formal series length {series_length}"
        )
    block_count = total_rows // series_length
    if block_count < 2:
        raise ValueError(f"{path.name} does not contain feature and label blocks")

    feature_chunks: list[list[np.ndarray]] = [[] for _ in range(block_count - 1)]
    label_chunks: list[np.ndarray] = []
    row_offset = 0
    for chunk in pd.read_csv(path, usecols=["data"], chunksize=200_000):
        values = pd.to_numeric(chunk["data"], errors="coerce").to_numpy(dtype=np.float64)
        offset = 0
        while offset < len(values):
            absolute = row_offset + offset
            block = absolute // series_length
            within_block = absolute % series_length
            count = min(len(values) - offset, series_length - within_block)
            segment = values[offset : offset + count]
            if block < block_count - 1 and within_block < train_length:
                feature_chunks[block].append(segment[: min(count, train_length - within_block)])
            elif block == block_count - 1:
                label_start = max(train_length - within_block, 0)
                if label_start < count:
                    label_chunks.append(segment[label_start:])
            offset += count
        row_offset += len(values)

    features = []
    for values in feature_chunks:
        feature = np.concatenate(values)[:train_length]
        if len(feature) != train_length:
            raise ValueError(f"{path.name} has a feature block with {len(feature)} train points, expected {train_length}")
        median = np.nanmedian(feature)
        features.append(np.nan_to_num(feature, nan=median if math.isfinite(median) else 0.0))
    labels = np.concatenate(label_chunks)[:test_length] if label_chunks else np.empty(0)
    if len(labels) != test_length:
        raise ValueError(f"{path.name} has {len(labels)} formal test labels, expected {test_length}")
    return np.column_stack(features), (labels > 0).astype(np.int8)


def anomaly_segments(labels: np.ndarray) -> np.ndarray:
    starts = np.flatnonzero(np.diff(np.r_[0, labels, 0]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[0, labels, 0]) == -1)
    return ends - starts


def legal_odd_kernel(kernel: int, sequence_length: int) -> int:
    maximum = sequence_length if sequence_length % 2 else sequence_length - 1
    return max(1, min(max(1, kernel), maximum))


def moving_average(values: np.ndarray, kernel: int) -> np.ndarray:
    """Exact replicate-padded centered average used by frozen MSDCATCH_model.py."""
    padding = kernel // 2
    padded = np.pad(values, ((padding, padding), (0, 0)), mode="edge")
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(padded, axis=0)))
    return (cumulative[kernel:] - cumulative[:-kernel]) / kernel


def checkpoint_gate(path: Path) -> Optional[dict[str, torch.Tensor]]:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") or payload.get("model_state")
    if not isinstance(state, dict):
        return None
    expected = ["scale_gate.net.0.weight", "scale_gate.net.0.bias", "scale_gate.net.2.weight", "scale_gate.net.2.bias"]
    extracted = {}
    for name in expected:
        source = next((key for key in state if key.endswith(name)), None)
        if source is None:
            return None
        extracted[name] = state[source].detach().cpu()
    return extracted


def gate_weights(windows: np.ndarray, state: dict[str, torch.Tensor]) -> np.ndarray:
    values = torch.from_numpy(windows.astype(np.float32, copy=False))
    means = values.mean(dim=1)
    stds = values.std(dim=1, unbiased=False)
    differences = values.diff(dim=1).abs().mean(dim=1) if values.shape[1] > 1 else torch.zeros_like(means)
    features = torch.stack((means, stds, differences), dim=-1)
    with torch.no_grad():
        hidden = torch_functional.gelu(
            torch_functional.linear(features, state["scale_gate.net.0.weight"], state["scale_gate.net.0.bias"])
        )
        logits = torch_functional.linear(hidden, state["scale_gate.net.2.weight"], state["scale_gate.net.2.bias"])
        return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)


def describe_training_data(
    train: np.ndarray,
    labels: np.ndarray,
    formal_test_length: int,
    sequence_length: int,
    patch_size: int,
    gate_state: Optional[dict[str, torch.Tensor]],
) -> tuple[dict, dict]:
    channels = train.shape[1]
    means = train.mean(axis=0)
    stds = np.maximum(train.std(axis=0), EPSILON)
    standardized = (train - means) / stds
    window_count = len(standardized) // sequence_length
    usable = window_count * sequence_length
    standardized = standardized[:usable]
    raw = train[:usable]
    global_correlation = None
    if channels > 1:
        global_correlation = np.nan_to_num(np.corrcoef(standardized, rowvar=False))

    drift_mean_sum = drift_std_sum = drift_count = 0.0
    correlation_sum = correlation_count = 0.0
    low_sum = np.zeros(channels)
    entropy_sum = np.zeros(channels)
    concentration_sum = np.zeros(channels)
    spectral_count = 0
    previous_mean = previous_std = None
    raw_energy = trend_energy = residual_energy = 0.0
    identity_error = 0.0
    scale_sum = np.zeros(3)
    scale_entropy_sum = 0.0
    scale_count = 0
    dominant_count = np.zeros(3, dtype=np.int64)
    kernels = tuple(legal_odd_kernel(value, sequence_length) for value in (patch_size - 1, 2 * patch_size - 1, 4 * patch_size - 1))

    for start in range(0, usable, sequence_length * 32):
        stop = min(start + sequence_length * 32, usable)
        raw_block = raw[start:stop].reshape(-1, sequence_length, channels)
        standard_block = standardized[start:stop].reshape(-1, sequence_length, channels)
        learned_weights = gate_weights(standard_block, gate_state) if gate_state is not None else None
        for index, (raw_window, standard_window) in enumerate(zip(raw_block, standard_block)):
            window_mean = raw_window.mean(axis=0)
            window_std = raw_window.std(axis=0)
            if previous_mean is not None:
                drift_mean_sum += float(np.abs(window_mean - previous_mean).sum() / stds.sum())
                drift_std_sum += float(np.abs(window_std - previous_std).sum() / stds.sum())
                drift_count += 1
            previous_mean, previous_std = window_mean, window_std

            spectrum = np.abs(np.fft.rfft(standard_window, axis=0)) ** 2
            nonzero = spectrum[1:]
            if len(nonzero):
                total = np.maximum(nonzero.sum(axis=0), EPSILON)
                low_count = max(1, math.ceil(len(nonzero) * 0.1))
                power = nonzero / total
                low_sum += nonzero[:low_count].sum(axis=0) / total
                entropy_sum += -(power * np.log(np.maximum(power, EPSILON))).sum(axis=0) / math.log(len(nonzero))
                concentration_sum += np.sort(nonzero, axis=0)[-min(3, len(nonzero)):].sum(axis=0) / total
                spectral_count += 1

            if global_correlation is not None:
                window_correlation = np.nan_to_num(np.corrcoef(standard_window, rowvar=False))
                correlation_sum += float(np.linalg.norm(window_correlation - global_correlation, ord="fro"))
                correlation_count += 1

            candidates = np.stack([moving_average(standard_window, kernel) for kernel in kernels], axis=-1)
            if learned_weights is None:
                trend = candidates.mean(axis=-1)
            else:
                weights = learned_weights[index]
                trend = (candidates * weights[None, :, :]).sum(axis=-1)
                scale_sum += weights.sum(axis=0)
                entropy = -(weights * np.log(np.maximum(weights, EPSILON))).sum(axis=-1) / math.log(3)
                scale_entropy_sum += float(entropy.sum())
                scale_count += weights.shape[0]
                dominant_count += np.bincount(weights.argmax(axis=-1), minlength=3)
            residual = standard_window - trend
            raw_energy += float(np.square(standard_window).sum())
            trend_energy += float(np.square(trend).sum())
            residual_energy += float(np.square(residual).sum())
            identity_error = max(identity_error, float(np.abs(trend + residual - standard_window).max()))

    labels_complete = len(labels) == formal_test_length
    segments = anomaly_segments(labels) if labels_complete else None
    base = {
        "channel_count": channels,
        "train_length": len(train),
        "test_length": formal_test_length,
        "label_length_observed": len(labels),
        "label_coverage_complete": labels_complete,
        "anomaly_ratio": float(labels.mean()) if labels_complete else None,
        "anomaly_segment_count": len(segments) if labels_complete else None,
        "median_anomaly_segment_length": float(np.median(segments)) if labels_complete and len(segments) else None,
        "seq_len": sequence_length,
        "patch_size": patch_size,
        "train_windows": window_count,
        "mean_drift": drift_mean_sum / drift_count if drift_count else None,
        "variance_drift": drift_std_sum / drift_count if drift_count else None,
        "low_frequency_energy_ratio_mean": float((low_sum / spectral_count).mean()) if spectral_count else None,
        "low_frequency_energy_ratio_median": float(np.median(low_sum / spectral_count)) if spectral_count else None,
        "spectral_entropy": float((entropy_sum / spectral_count).mean()) if spectral_count else None,
        "periodicity_top3_ratio": float((concentration_sum / spectral_count).mean()) if spectral_count else None,
        "correlation_drift": correlation_sum / correlation_count if correlation_count else None,
    }
    decomp = {
        "trend_energy_over_raw": trend_energy / max(raw_energy, EPSILON),
        "residual_energy_over_raw": residual_energy / max(raw_energy, EPSILON),
        "trend_over_residual_energy": trend_energy / max(residual_energy, EPSILON),
        "decomposition_identity_max_error": identity_error,
        "kernels": json.dumps(kernels),
        "decomposition_mode": "checkpoint_scale_gate" if gate_state is not None else "deterministic_equal_weight_no_checkpoint",
        "scale_weight_mean": json.dumps((scale_sum / scale_count).tolist()) if scale_count else None,
        "scale_weight_entropy": scale_entropy_sum / scale_count if scale_count else None,
        "dominant_scale": int(dominant_count.argmax()) if scale_count else None,
        "dominant_scale_frequency": float(dominant_count.max() / scale_count) if scale_count else None,
    }
    return base, decomp


def quantile_diagnostics(labels: np.ndarray, scores: np.ndarray) -> dict:
    normal = scores[labels == 0]
    anomalous = scores[labels == 1]
    normal_iqr = np.subtract(*np.percentile(normal, [75, 25])) if len(normal) else np.nan
    def top_rate(fraction: float) -> float:
        count = max(1, math.ceil(len(scores) * fraction))
        top = np.argpartition(scores, -count)[-count:]
        return float(labels[top].mean())
    return {
        "normal_score_median": float(np.median(normal)) if len(normal) else None,
        "anomaly_score_median": float(np.median(anomalous)) if len(anomalous) else None,
        "normal_score_iqr": float(normal_iqr),
        "anomaly_score_iqr": float(np.subtract(*np.percentile(anomalous, [75, 25]))) if len(anomalous) else None,
        "median_separation_over_normal_iqr": float((np.median(anomalous) - np.median(normal)) / max(normal_iqr, EPSILON)) if len(anomalous) and len(normal) else None,
        "top_1pct_anomaly_fraction": top_rate(0.01),
        "top_5pct_anomaly_fraction": top_rate(0.05),
    }


def formal_msd_payload(repo: Path, task: str) -> Optional[dict]:
    json_path = repo / "result" / "msd_catch_total_screen" / f"{task}.json"
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    log_path = repo / "result" / "msd_catch_screen" / f"{task}.log"
    if task in FORMAL_MSD_LOG and log_path.exists():
        for line in reversed(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload.get("scores"), dict):
                return payload
    return None


def score_diagnostics(repo: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        source = repo / "result" / "msd_catch_total_screen" / f"{task}_scores.npz"
        metric = metrics.loc[metrics.task == task].iloc[0]
        payload = formal_msd_payload(repo, task)
        scores = payload.get("scores", {}) if payload else {}
        common = {
            "task": task,
            "paper_dataset": paper_dataset(task),
            "catch_msd_total_score_spearman": None,
            "catch_score_source": None,
            "msd_score_source": str(source) if source.exists() else None,
            "trend_auc_pr": scores.get("trend_score", {}).get("auc_pr"),
            "residual_auc_pr": scores.get("residual_score", {}).get("auc_pr"),
            "total_auc_pr": metric.msd_auc_pr,
        }
        if not source.exists():
            rows.append({**common, "model": "CATCH", "status": "N/A_no_continuous_score_archive"})
            rows.append(
                {
                    **common,
                    "model": "MSDCATCH",
                    "status": "component_metrics_only" if payload else "N/A_no_continuous_score_archive",
                }
            )
            continue
        if payload is None:
            raise ValueError(f"Missing formal MSD payload for continuous score archive: {task}")
        with np.load(source, allow_pickle=False) as archive:
            labels = archive["labels"].astype(np.int8)
            total = archive["total_score"].astype(np.float64)
        rows.append({**common, "model": "CATCH", "status": "N/A_no_continuous_score_archive"})
        rows.append(
            {
                **common,
                **quantile_diagnostics(labels, total),
                "model": "MSDCATCH",
                "status": "available",
                "trend_auc_pr": scores.get("trend_score", {}).get("auc_pr"),
                "residual_auc_pr": scores.get("residual_score", {}).get("auc_pr"),
                "total_auc_pr": scores.get("total_score", {}).get("auc_pr"),
            }
        )
    return pd.DataFrame(rows)


def paper_metrics(task_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "catch_auc_pr", "msd_auc_pr", "delta_auc_pr", "catch_auc_roc", "msd_auc_roc", "delta_auc_roc"
    ]
    rows = []
    for paper in PAPER_ORDER:
        subset = task_metrics[task_metrics.paper_dataset == paper]
        row = {"paper_dataset": paper, "task_count": len(subset)}
        for column in numeric_columns:
            row[column] = subset[column].dropna().mean() if subset[column].notna().any() else None
        row["pr_group"] = metric_group(row["delta_auc_pr"])
        row["roc_group"] = metric_group(row["delta_auc_roc"])
        row["formal_msd_task_count"] = int(subset.msd_auc_pr.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def spearman(values: pd.DataFrame, descriptor: str, target: str) -> tuple[Optional[float], int, Optional[str], Optional[float], bool]:
    subset = values[["task", descriptor, target]].dropna()
    if len(subset) < 3:
        return None, len(subset), None, None, False
    baseline = subset[descriptor].rank(method="average").corr(subset[target].rank(method="average"))
    shifts = []
    for index, row in subset.iterrows():
        remaining = subset.drop(index)
        value = remaining[descriptor].rank(method="average").corr(remaining[target].rank(method="average"))
        shifts.append((abs(value - baseline), row.task, value))
    shift, task, leave_one_out = max(shifts)
    return float(baseline), len(subset), task, float(shift), bool(np.sign(leave_one_out) != np.sign(baseline))


def correlations(task_values: pd.DataFrame, paper_values: pd.DataFrame) -> pd.DataFrame:
    descriptors = [
        "channel_count", "mean_drift", "variance_drift", "low_frequency_energy_ratio_mean",
        "spectral_entropy", "periodicity_top3_ratio", "correlation_drift", "trend_energy_over_raw",
        "residual_energy_over_raw", "trend_over_residual_energy",
    ]
    rows = []
    for level, values in (("task", task_values), ("paper", paper_values)):
        for descriptor in descriptors:
            for target in ("delta_auc_pr", "delta_auc_roc"):
                rho, count, influential, shift, sign_flip = spearman(values, descriptor, target)
                rows.append(
                    {
                        "level": level,
                        "descriptor": descriptor,
                        "target": target,
                        "spearman_rho": rho,
                        "n": count,
                        "most_influential_leave_one_out": influential,
                        "max_leave_one_out_shift": shift,
                        "leave_one_out_sign_flip": sign_flip,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_paper_descriptors(task_descriptors: pd.DataFrame, paper_metric_rows: pd.DataFrame) -> pd.DataFrame:
    numeric = task_descriptors.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for paper in PAPER_ORDER:
        subset = task_descriptors[task_descriptors.paper_dataset == paper]
        row = {"task": paper, "paper_dataset": paper}
        for column in numeric:
            row[column] = subset[column].dropna().mean() if subset[column].notna().any() else None
        metric = paper_metric_rows[paper_metric_rows.paper_dataset == paper].iloc[0]
        row["delta_auc_pr"] = metric.delta_auc_pr
        row["delta_auc_roc"] = metric.delta_auc_roc
        rows.append(row)
    return pd.DataFrame(rows)


def paper_reference(repo: Path, catch_root: Path) -> pd.DataFrame:
    documents = list(repo.glob("README*")) + list(catch_root.glob("README*"))
    documents += list(repo.rglob("*.md")) + list(catch_root.rglob("*.md"))
    candidates = []
    for path in documents:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "CATCH" in text and ("AUC-PR" in text or "AUC-ROC" in text):
            candidates.append(str(path))
    source = "No reliable structured original CATCH paper table found in repository."
    if candidates:
        source += " Candidate documents were not treated as a numeric paper table: " + "; ".join(sorted(set(candidates))[:3])
    return pd.DataFrame(
        {
            "dataset": PAPER_ORDER,
            "baseline": "N/A",
            "paper_pr_roc": "N/A",
            "catch": "N/A",
            "msd": "N/A",
            "source": source,
            "status": "Paper-reported reference not rerun under current commit; no reliable table parsed.",
        }
    )


def report(output: Path, task_metrics: pd.DataFrame, paper: pd.DataFrame, correlations_df: pd.DataFrame) -> None:
    available = task_metrics.dropna(subset=["delta_auc_pr"])
    gains = int((available.delta_auc_pr > 0.01).sum())
    losses_or_neutral = int((available.delta_auc_pr <= 0.01).sum())
    qualified = []
    for descriptor in correlations_df.descriptor.unique():
        task = correlations_df[(correlations_df.level == "task") & (correlations_df.descriptor == descriptor) & (correlations_df.target == "delta_auc_pr")].iloc[0]
        paper_row = correlations_df[(correlations_df.level == "paper") & (correlations_df.descriptor == descriptor) & (correlations_df.target == "delta_auc_pr")].iloc[0]
        if (
            task.spearman_rho is not None and paper_row.spearman_rho is not None
            and np.sign(task.spearman_rho) == np.sign(paper_row.spearman_rho)
            and abs(task.spearman_rho) >= 0.35 and abs(paper_row.spearman_rho) >= 0.35
            and not task.leave_one_out_sign_flip and not paper_row.leave_one_out_sign_flip
            and gains >= 3 and losses_or_neutral >= 3
        ):
            qualified.append(descriptor)
    if qualified:
        conclusion = "A: at least one descriptor meets the numerical cross-level screen: " + ", ".join(qualified)
    elif gains and losses_or_neutral:
        conclusion = "B: performance has mixed task groups, but no descriptor meets all cross-level hypothesis criteria."
    else:
        conclusion = "C: no repeated gain/loss grouping is available in the current formal records."
    lines = [
        "# Decomposition Applicability Report",
        "",
        "This report is read-only: it does not call a training, scoring, or benchmark entry point.",
        "",
        "## Formal Performance Coverage",
        f"- Formal MSD delta available for {len(available)}/{len(task_metrics)} execution tasks.",
        "- PSM, Genesis, and GECCO use the terminal formal total-screen JSON line in their MSD logs.",
        "- ASD paper-level values are equal-weight macro means of its 12 execution tasks.",
        "",
        "## Descriptor Method",
        "- Raw descriptors use normal training prefixes and non-overlapping formal seq_len windows.",
        "- Spectral quantities are per-channel training-window summaries; correlation drift is N/A for one channel.",
        "- Decomposition uses replicate-padded moving averages and verifies trend + residual = input.",
        "",
        "## Task-Level Performance",
        "```text",
        task_metrics[["task", "catch_auc_pr", "msd_auc_pr", "delta_auc_pr", "pr_group", "catch_auc_roc", "msd_auc_roc", "delta_auc_roc", "roc_group", "msd_source_kind"]].to_string(index=False),
        "```",
        "",
        "## Paper-Level Performance",
        "```text",
        paper[["paper_dataset", "task_count", "catch_auc_pr", "msd_auc_pr", "delta_auc_pr", "catch_auc_roc", "msd_auc_roc", "delta_auc_roc"]].to_string(index=False),
        "```",
        "",
        "## Score Diagnostics",
        "- CATCH continuous scores are not stored in the formal archives, so CATCH-versus-MSD Spearman score correlation is N/A without a rerun.",
        "- MSD quantile diagnostics are available only where a frozen `*_scores.npz` exists; component AUCs from formal logs are still retained.",
        "",
        "## Descriptor Correlations",
        "```text",
        correlations_df[["level", "descriptor", "target", "spearman_rho", "n", "most_influential_leave_one_out", "max_leave_one_out_shift", "leave_one_out_sign_flip"]].to_string(index=False),
        "```",
        "",
        "## Paper Baseline Reference",
        "- No reliable structured original CATCH paper table was present locally; paper-reported values remain N/A rather than being substituted by current reruns.",
        "",
        "## Conclusion",
        f"- {conclusion}",
        "- This is descriptive evidence from frozen runs, not a model-selection rule or a new implementation proposal.",
    ]
    (output / "DECOMPOSITION_APPLICABILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    catch_root = repo.parent / "CATCH-master"
    output = repo / "result" / "decomposition_applicability"
    output.mkdir(parents=True, exist_ok=True)

    task_metrics, sources = discover_metrics(repo, catch_root)
    metadata = metadata_lookup(repo)
    descriptor_rows = []
    decomposition_rows = []

    for task in TASKS:
        metric = task_metrics.loc[task_metrics.task == task].iloc[0]
        meta = metadata.get(task)
        raw_path = repo / "dataset" / "anomaly_detect" / "data" / f"{task}.csv"
        sources.append(
            {
                "task": task,
                "artifact": "raw long-format dataset",
                "path": str(raw_path),
                "status": "used" if raw_path.exists() else "missing",
                "notes": "Normal training prefix only for descriptors",
            }
        )
        sources.append(
            {
                "task": task,
                "artifact": "DETECT_META entry",
                "path": str(repo / "dataset" / "anomaly_detect" / "DETECT_META.csv"),
                "status": "used" if meta else "missing",
                "notes": "Recorded train_lens and test_lens",
            }
        )
        bhd_json = repo / "result" / "bhd_msd_catch_screen" / f"{task}.json"
        bhd_archives = list((repo / "result" / "score" / "by_dataset" / task).rglob("*BHD*.tar.gz"))
        bhd_path = bhd_json if bhd_json.exists() else (bhd_archives[0] if bhd_archives else None)
        sources.append(
            {
                "task": task,
                "artifact": "BHD frozen full-line result context",
                "path": str(bhd_path) if bhd_path else None,
                "status": "available_not_compared" if bhd_path else "missing",
                "notes": "Full-line BHD evidence retained as background; not used in CATCH-versus-MSD deltas",
            }
        )
        if meta is None or not raw_path.exists() or not finite(metric.formal_seq_len) or not finite(metric.formal_patch_size):
            descriptor_rows.append(
                {
                    "task": task,
                    "paper_dataset": paper_dataset(task),
                    "descriptor_status": "N/A_missing_raw_metadata_or_formal_config",
                }
            )
            decomposition_rows.append(
                {
                    "task": task,
                    "paper_dataset": paper_dataset(task),
                    "decomposition_status": "N/A_missing_raw_metadata_or_formal_config",
                }
            )
            continue

        train_length = config_integer(meta, "train_lens")
        test_length = config_integer(meta, "test_lens")
        if train_length is None or test_length is None:
            raise ValueError(f"Invalid DETECT_META lengths for {task}")
        checkpoint_path = repo / "result" / "msd_catch_total_screen" / f"{task}.pt"
        gate_state = checkpoint_gate(checkpoint_path)
        sources.append(
            {
                "task": task,
                "artifact": "MSD scale-gate checkpoint",
                "path": str(checkpoint_path) if checkpoint_path.exists() else None,
                "status": "used" if gate_state is not None else "not_available",
                "notes": "Weights inspected only; no model forward, scoring, or training",
            }
        )
        print(f"[descriptor] {task}", flush=True)
        train, labels = read_train_and_labels(raw_path, train_length, test_length)
        base, decomp = describe_training_data(
            train=train,
            labels=labels,
            formal_test_length=test_length,
            sequence_length=int(metric.formal_seq_len),
            patch_size=int(metric.formal_patch_size),
            gate_state=gate_state,
        )
        descriptor_rows.append(
            {
                "task": task,
                "paper_dataset": paper_dataset(task),
                "descriptor_status": "available",
                **base,
            }
        )
        decomposition_rows.append(
            {
                "task": task,
                "paper_dataset": paper_dataset(task),
                "decomposition_status": "available",
                **decomp,
            }
        )

    descriptors = pd.DataFrame(descriptor_rows)
    decomposition = pd.DataFrame(decomposition_rows)
    diagnostics = score_diagnostics(repo, task_metrics)
    paper = paper_metrics(task_metrics)
    task_descriptor_values = descriptors.merge(decomposition, on=["task", "paper_dataset"], how="outer")
    task_values = task_metrics.merge(task_descriptor_values, on=["task", "paper_dataset"], how="left")
    paper_values = aggregate_paper_descriptors(task_descriptor_values, paper)
    correlations_df = correlations(task_values, paper_values)
    baseline_reference = paper_reference(repo, catch_root)

    task_metrics.to_csv(output / "task_level_metrics.csv", index=False)
    paper.to_csv(output / "paper_level_metrics.csv", index=False)
    descriptors.to_csv(output / "dataset_descriptors.csv", index=False)
    decomposition.to_csv(output / "decomposition_descriptors.csv", index=False)
    diagnostics.to_csv(output / "score_distribution_diagnostics.csv", index=False)
    correlations_df.to_csv(output / "descriptor_delta_correlations.csv", index=False)
    baseline_reference.to_csv(output / "paper_baseline_reference.csv", index=False)
    pd.DataFrame(sources).to_csv(output / "analysis_sources.csv", index=False)
    report(output, task_metrics, paper, correlations_df)
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
