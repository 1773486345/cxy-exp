"""Small fixed-step profiler for a fair CATCH/MSD/BHD comparison task."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.bhd_msd_catch.BHDMSDCATCH import BHDMSDCATCH
from ts_benchmark.baselines.bhd_msd_catch.models.BHDMSDCATCH_model import BHDMSDCATCHModel
from ts_benchmark.baselines.catch.CATCH import CATCH
from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel
from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_criterion
from ts_benchmark.baselines.msd_catch.MSDCATCH import MSDCATCH
from ts_benchmark.baselines.msd_catch.models.MSDCATCH_model import MSDCATCHModel
from ts_benchmark.baselines.rsa_msd_catch.RSAMSDCATCH import _set_seed
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split
from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource


MODEL_FACTORIES = {
    "catch": (CATCH, CATCHModel),
    "msd": (MSDCATCH, MSDCATCHModel),
    "bhd": (BHDMSDCATCH, BHDMSDCATCHModel),
}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _repeating_batches(loader):
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("profiling loader is empty")


def _scaled_splits(detector, train: pd.DataFrame, test: pd.DataFrame):
    train_value, _ = train_val_split(train, 0.8, None)
    detector.scaler.fit(train_value.values)
    train_value = pd.DataFrame(
        detector.scaler.transform(train_value.values),
        columns=train_value.columns,
        index=train_value.index,
    )
    test_value = pd.DataFrame(
        detector.scaler.transform(test.values), columns=test.columns, index=test.index
    )
    return train_value, test_value


def _prepare(model_name: str, params: dict, train: pd.DataFrame, test: pd.DataFrame):
    detector_class, model_class = MODEL_FACTORIES[model_name]
    detector = detector_class(**params)
    detector.detect_hyper_param_tune(train)
    detector.config.task_name = "anomaly_detection"
    detector.config.c_in = train.shape[1]
    if model_name == "catch":
        detector.model = model_class(detector.config).to(detector.device)
    else:
        detector.model = model_class(detector.config).to(detector.device)
    train_value, test_value = _scaled_splits(detector, train, test)
    train_loader = anomaly_detection_data_provider(
        train_value,
        batch_size=detector.config.batch_size,
        win_size=detector.config.seq_len,
        step=1,
        mode="train",
    )
    score_loader = anomaly_detection_data_provider(
        test_value,
        batch_size=detector.config.batch_size,
        win_size=detector.config.seq_len,
        step=1,
        mode="thre",
    )
    main_parameters = [
        parameter
        for name, parameter in detector.model.named_parameters()
        if "mask_generator" not in name
    ]
    optimizer = torch.optim.Adam(main_parameters, lr=detector.config.lr)
    parameters = sum(parameter.numel() for parameter in detector.model.parameters() if parameter.requires_grad)
    return detector, train_loader, score_loader, optimizer, parameters


def _train_loss(model_name: str, detector, batch: torch.Tensor) -> torch.Tensor:
    if model_name != "catch":
        loss, _ = detector._loss(batch)
        return loss
    output, complex_output, dcloss = detector.model(batch)
    reconstruction = detector.criterion(output, batch)
    normalized = detector.model.revin_layer(batch, "transform")
    auxiliary = detector.auxi_loss(complex_output, normalized)
    return reconstruction + detector.config.dc_lambda * dcloss + detector.config.auxi_lambda * auxiliary


@torch.no_grad()
def _score_step(model_name: str, detector, batch: torch.Tensor) -> None:
    if model_name != "catch":
        detector._score_batch(batch)
        return
    output, _, _ = detector.model(batch)
    temporal = (batch - output).square().mean(dim=-1)
    frequency = frequency_criterion(detector.config)(batch, output).mean(dim=-1)
    _ = temporal + detector.config.score_lambda * frequency


def _measure_train(model_name: str, detector, loader, optimizer, warmup_steps: int, steps: int) -> float:
    iterator = _repeating_batches(loader)
    for _ in range(warmup_steps):
        batch, _ = next(iterator)
        optimizer.zero_grad()
        loss = _train_loss(model_name, detector, batch.float().to(detector.device))
        loss.backward()
        optimizer.step()
    _sync()
    start = time.perf_counter()
    for _ in range(steps):
        batch, _ = next(iterator)
        optimizer.zero_grad()
        loss = _train_loss(model_name, detector, batch.float().to(detector.device))
        loss.backward()
        optimizer.step()
    _sync()
    return (time.perf_counter() - start) / steps


def _measure_score(model_name: str, detector, loader, warmup_steps: int, steps: int) -> float:
    iterator = _repeating_batches(loader)
    for _ in range(warmup_steps):
        batch, _ = next(iterator)
        _score_step(model_name, detector, batch.float().to(detector.device))
    _sync()
    start = time.perf_counter()
    for _ in range(steps):
        batch, _ = next(iterator)
        _score_step(model_name, detector, batch.float().to(detector.device))
    _sync()
    return (time.perf_counter() - start) / steps


def profile(args) -> dict:
    _set_seed(args.seed)
    source = LocalAnomalyDetectDataSource()
    series_name = args.dataset + ".csv"
    source.load_series_list([series_name])
    data = source.dataset.get_series(series_name).reset_index(drop=True)
    train_length = int(source.dataset.get_series_meta_info(series_name)["train_lens"].item())
    features = data.columns != "label"
    train = data.iloc[:train_length].loc[:, features]
    test = data.iloc[train_length:].loc[:, features]
    params = json.loads(args.params)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    detector, train_loader, score_loader, optimizer, parameters = _prepare(
        args.model, params, train, test
    )
    if not len(train_loader) or not len(score_loader):
        raise ValueError("profiling requires non-empty training and score loaders")
    train_seconds = _measure_train(
        args.model, detector, train_loader, optimizer, args.warmup_steps, args.steps
    )
    score_seconds = _measure_score(
        args.model, detector, score_loader, args.warmup_steps, args.steps)
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "params": params,
        "parameters": parameters,
        "train_batch_seconds": train_seconds,
        "score_batch_seconds": score_seconds,
        "epoch_batches": len(train_loader),
        "score_batches": len(score_loader),
        "estimated_fit_seconds": train_seconds * len(train_loader) * detector.config.num_epochs,
        "estimated_score_seconds": score_seconds * len(score_loader),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "profile_steps": args.steps,
        "warmup_steps": args.warmup_steps,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _load_real_dataset(dataset: str):
    source = LocalAnomalyDetectDataSource()
    series_name = dataset + ".csv"
    source.load_series_list([series_name])
    data = source.dataset.get_series(series_name).reset_index(drop=True)
    train_length = int(source.dataset.get_series_meta_info(series_name)["train_lens"].item())
    features = data.columns != "label"
    train = data.iloc[:train_length].loc[:, features]
    test = data.iloc[train_length:].loc[:, features]
    labels = data.iloc[train_length:]["label"].to_numpy(dtype=int)
    return train, test, labels


def run_full(args) -> dict:
    """Run one fair real-data task and save its checkpoint and continuous scores."""
    _set_seed(args.seed)
    params = json.loads(args.params)
    output_dir = Path(args.result_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.dataset}.json"
    score_path = output_dir / f"{args.dataset}_scores.npz"
    checkpoint_path = output_dir / f"{args.dataset}.pt"
    if any(path.exists() for path in (json_path, score_path, checkpoint_path)):
        raise FileExistsError(f"refusing to overwrite an existing result for {args.model}/{args.dataset}")

    train, test, labels = _load_real_dataset(args.dataset)
    detector_class, _ = MODEL_FACTORIES[args.model]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    detector = detector_class(**params)
    fit_start = time.perf_counter()
    if args.model == "catch":
        detector.detect_fit(train, test)
    else:
        detector.detect_fit(train)
    fit_seconds = time.perf_counter() - fit_start

    score_start = time.perf_counter()
    total_score, _ = detector.detect_score(test)
    score_seconds = time.perf_counter() - score_start
    total_score = np.asarray(total_score, dtype=np.float64).reshape(-1)
    scored_length = len(total_score)
    if scored_length > len(labels):
        raise ValueError("model produced more scores than test labels")
    score_arrays = {"total_score": total_score}
    for name, values in getattr(detector, "last_scores", {}).items():
        values = np.asarray(values)
        if values.ndim == 1 and len(values) == scored_length:
            score_arrays[name] = values
    for name, values in getattr(detector, "last_diagnostics", {}).items():
        values = np.asarray(values)
        if values.shape[0] == scored_length:
            score_arrays[f"diagnostic_{name}"] = values
    if scored_length < len(labels):
        score_arrays = {
            name: np.pad(values, (0, len(labels) - scored_length), constant_values=0.0)
            if values.ndim == 1
            else np.pad(values, ((0, len(labels) - scored_length),) + ((0, 0),) * (values.ndim - 1), constant_values=0.0)
            for name, values in score_arrays.items()
        }
    primary = score_arrays["total_score"]
    if not np.isfinite(primary).all():
        raise FloatingPointError("total_score contains NaN or Inf")

    parameters = sum(parameter.numel() for parameter in detector.model.parameters() if parameter.requires_grad)
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "primary_score": "total_score",
        "config": params,
        "resolved_config": dict(detector.config.__dict__),
        "parameters": parameters,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "scored_length_before_padding": scored_length,
        "score_keys": sorted(score_arrays),
        "metrics": {
            "auc_pr": float(average_precision_score(labels, primary)),
            "auc_roc": float(roc_auc_score(labels, primary)),
        },
        "data": {
            "variables": int(train.shape[1]),
            "train_length": int(len(train)),
            "test_length": int(len(test)),
            "anomaly_rate": float(labels.mean()),
            "anomaly_segments": int(np.sum(np.diff(np.r_[0, labels, 0]) == 1)),
        },
    }
    torch.save(
        {
            "model_state": detector.model.state_dict(),
            "config": dict(detector.config.__dict__),
            "scaler_mean": detector.scaler.mean_,
            "scaler_scale": detector.scaler.scale_,
            "seed": args.seed,
        },
        checkpoint_path,
    )
    np.savez_compressed(score_path, labels=labels, **score_arrays)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_FACTORIES), required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output")
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--result-dir")
    args = parser.parse_args()
    if args.full_run:
        if not args.result_dir:
            parser.error("--result-dir is required with --full-run")
        run_full(args)
    else:
        profile(args)


if __name__ == "__main__":
    main()
