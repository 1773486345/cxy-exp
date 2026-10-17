#!/usr/bin/env python
"""Run a small, explicit-split CATCH-common task.

This adapter imports the protected CATCH implementation but does not call its
benchmark entry point (which owns a different internal split).  All training,
validation, calibration and scoring coordinates are supplied explicitly and
are persisted as execution evidence for the read-only common evaluator.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.catch.CATCH import TransformerConfig
from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel
from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_loss, frequency_criterion
from ts_benchmark.baselines.mnrf_view_probe.evaluation import strict_integer_indices, validate_common_splits


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    temporary = Path(path).with_name(Path(path).name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _windows(values, endpoints, seq_len):
    endpoints = strict_integer_indices(endpoints, "window endpoints", upper_bound=len(values))
    if len(endpoints) == 0 or np.any(endpoints < seq_len - 1):
        raise ValueError("explicit split has no complete windows")
    return np.stack([values[int(endpoint) - seq_len + 1:int(endpoint) + 1] for endpoint in endpoints]).astype(np.float32)


def _window_rows(endpoints, seq_len, length, name):
    endpoints = strict_integer_indices(endpoints, f"{name} endpoints", upper_bound=length)
    if len(endpoints) == 0 or np.any(endpoints < seq_len - 1):
        raise ValueError(f"{name} has an incomplete window")
    rows = np.unique(np.concatenate([
        np.arange(int(endpoint) - seq_len + 1, int(endpoint) + 1, dtype=np.int64)
        for endpoint in endpoints
    ]))
    return rows


def _validate_window_boundaries(frame, endpoints_by_name, seq_len, test_start):
    """Reject windows that overlap splits, entities, or unexplained time gaps."""
    length = len(frame)
    rows = {name: _window_rows(endpoints, seq_len, length, name)
            for name, endpoints in endpoints_by_name.items()}
    names = tuple(rows)
    for left, name_left in enumerate(names):
        for name_right in names[left + 1:]:
            if np.intersect1d(rows[name_left], rows[name_right]).size:
                raise ValueError(f"{name_left}/{name_right} windows overlap; provide purged split endpoints")
    if np.any(rows["test"] < int(test_start)):
        raise ValueError("test windows cross the train/test boundary; test endpoints must start at test_start + seq_len - 1")
    timestamp_column = next((column for column in frame.columns
                             if str(column).lower() in {"timestamp", "date", "time", "datetime"}), None)
    if timestamp_column is not None:
        timestamps = pd.to_datetime(frame[timestamp_column], errors="raise", utc=True).view("int64").to_numpy()
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing for CATCH-common windows")
        train_rows = rows["train"]
        train_deltas = np.diff(timestamps[train_rows])
        train_deltas = train_deltas[train_deltas > 0]
        if not len(train_deltas):
            raise ValueError("cannot infer a positive training sampling interval")
        interval = float(np.median(train_deltas))
        for name, selected in rows.items():
            if len(selected) > 1:
                deltas = np.diff(timestamps[selected])
                if np.any(np.abs(deltas / interval - 1.0) > 0.05):
                    raise ValueError(f"{name} window crosses a timestamp gap")
    boundary_column = next((column for column in frame.columns
                            if str(column).lower() in {"entity_id", "session_id", "record_id", "scenario_id"}), None)
    if boundary_column is not None:
        boundary = frame[boundary_column].to_numpy()
        for name, endpoints in endpoints_by_name.items():
            for endpoint in endpoints:
                start = int(endpoint) - seq_len + 1
                if np.any(boundary[start:int(endpoint) + 1] != boundary[start]):
                    raise ValueError(f"{name} window crosses entity boundary")
    return rows


def _score_batch(model, batch, frequency_metric, score_lambda):
    outputs = model(batch)[0]
    temporal = (batch - outputs).square().mean(dim=-1)
    frequency = frequency_metric(batch, outputs).mean(dim=-1)
    return outputs, temporal, frequency, temporal + float(score_lambda) * frequency


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    for name in ("checkpoints", "results", "evidence"):
        (output / name).mkdir()
    frame = pd.read_csv(args.data_csv)
    label_column = next((column for column in frame.columns if str(column).lower() == "label"), None)
    if label_column is None:
        raise ValueError("CSV must contain a label column")
    excluded = {label_column, "timestamp", "date", "time", "datetime", "index",
                "entity_id", "session_id", "record_id", "scenario_id"}
    feature_columns = [column for column in frame.columns if column not in excluded]
    values = frame[feature_columns].to_numpy(dtype=np.float32)
    labels = frame[label_column].to_numpy(dtype=np.int64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("data features must be finite with at least two variables")
    train_indices = np.load(args.train_indices, allow_pickle=False)
    validation_indices = np.load(args.validation_indices, allow_pickle=False)
    calibration_indices = np.load(args.calibration_indices, allow_pickle=False)
    split_record = validate_common_splits(train_indices, validation_indices, calibration_indices, len(values))
    for name, indices in (("train", train_indices), ("validation", validation_indices), ("calibration", calibration_indices)):
        if len(indices) == 0:
            raise ValueError(f"{name} split is empty")
    train_endpoints = strict_integer_indices(train_indices, "train endpoints", upper_bound=len(values))
    validation_endpoints = strict_integer_indices(validation_indices, "validation endpoints", upper_bound=len(values))
    calibration_endpoints = strict_integer_indices(calibration_indices, "calibration endpoints", upper_bound=len(values))
    test_indices = np.arange(int(args.test_start) + args.seq_len - 1, len(values), dtype=np.int64)
    if len(test_indices) == 0:
        raise ValueError("test interval has no complete windows")
    endpoint_sets = {"train": train_endpoints, "validation": validation_endpoints,
                     "calibration": calibration_endpoints, "test": test_indices}
    fit_rows = _validate_window_boundaries(frame, endpoint_sets, args.seq_len, args.test_start)["train"]
    scaler = StandardScaler().fit(values[fit_rows])
    normalized = scaler.transform(values).astype(np.float32)
    fit_rows_path = output / "evidence" / "train_fit_rows.npy"
    np.save(fit_rows_path, fit_rows)
    np.savez(output / "normalization.npz", mean=scaler.mean_, scale=scaler.scale_,
             fit_rows_sha256=_sha256(fit_rows_path), fit_rows_count=np.asarray(len(fit_rows)))
    config = TransformerConfig(seq_len=args.seq_len, patch_size=args.patch_size,
                               patch_stride=args.patch_stride, d_model=args.d_model,
                               cf_dim=args.cf_dim, d_ff=args.d_ff, e_layers=args.e_layers,
                               n_heads=args.n_heads, head_dim=args.head_dim, dropout=args.dropout,
                               lr=args.learning_rate, num_epochs=args.epochs,
                               batch_size=args.batch_size, patience=args.patience,
                               score_lambda=args.score_lambda, Mlr=args.mask_learning_rate,
                               inference_patch_size=args.inference_patch_size,
                               inference_patch_stride=args.inference_patch_stride,
                               pct_start=args.pct_start)
    config.c_in = values.shape[1]
    config.enc_in = values.shape[1]
    config.dec_in = values.shape[1]
    config.c_out = values.shape[1]
    config.task_name = args.task_name
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = CATCHModel(config).to(device)
    criterion = torch.nn.MSELoss()
    aux_loss = frequency_loss(config)
    train_windows = torch.from_numpy(_windows(normalized, train_endpoints, args.seq_len))
    validation_windows = torch.from_numpy(_windows(normalized, validation_endpoints, args.seq_len))
    main_params = [parameter for name, parameter in model.named_parameters()
                   if "mask_generator" not in name]
    optimizer = torch.optim.Adam(main_params, lr=args.learning_rate)
    mask_optimizer = torch.optim.Adam(model.mask_generator.parameters(), lr=args.mask_learning_rate)
    train_loader = DataLoader(TensorDataset(train_windows), batch_size=args.batch_size, shuffle=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, epochs=args.epochs,
        steps_per_epoch=max(len(train_loader), 1), pct_start=args.pct_start)
    mask_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        mask_optimizer, max_lr=args.mask_learning_rate, epochs=args.epochs,
        steps_per_epoch=max(len(train_loader), 1), pct_start=args.pct_start)
    frequency_metric = frequency_criterion(config)
    best = float("inf")
    best_state = None
    patience_count = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            mask_optimizer.zero_grad(set_to_none=True)
            reconstruction, complex_output, dc_loss = model(batch)
            normalized_batch = model.revin_layer(batch, "transform")
            loss = criterion(reconstruction, batch) + config.dc_lambda * dc_loss + config.auxi_lambda * aux_loss(complex_output, normalized_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite CATCH-common training loss")
            loss.backward()
            optimizer.step()
            mask_optimizer.step()
            scheduler.step()
            mask_scheduler.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(torch.stack([
                criterion(model(batch.to(device))[0], batch.to(device)).detach().cpu()
                for (batch,) in DataLoader(TensorDataset(validation_windows), batch_size=args.batch_size, shuffle=False)
            ]).mean())
        if validation_loss < best:
            best = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        print(json.dumps({"epoch": epoch + 1, "train_nll": float(np.mean(losses)),
                          "validation_nll": validation_loss, "best_validation_nll": best,
                          "patience": patience_count}, sort_keys=True), flush=True)
        if patience_count >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("CATCH-common produced no validation checkpoint")
    model.load_state_dict(best_state)
    checkpoint = output / "checkpoints" / "catch_common.pt"
    torch.save({"task_name": args.task_name, "protocol": "CATCH-common", "state_dict": best_state,
                "score_protocol": {"temporal": "pointwise_mse", "frequency": "CATCH-frequency_criterion",
                                   "score_lambda": float(args.score_lambda),
                                   "inference_patch_size": int(args.inference_patch_size),
                                   "inference_patch_stride": int(args.inference_patch_stride)},
                "normalization_sha256": _sha256(output / "normalization.npz"),
                "train_fit_rows_sha256": _sha256(fit_rows_path)}, checkpoint)
    model.eval()
    calibration_windows = torch.from_numpy(_windows(normalized, calibration_endpoints, args.seq_len))
    with torch.no_grad():
        calibration_scores = torch.cat([
            _score_batch(model, batch.to(device), frequency_metric, args.score_lambda)[3][:, -1].detach().cpu()
            for (batch,) in DataLoader(TensorDataset(calibration_windows), batch_size=args.batch_size, shuffle=False)
        ]).numpy()
    threshold = float(np.quantile(calibration_scores, args.calibration_quantile))
    np.savez(output / "results" / "calibration_scores.npz", scores=calibration_scores,
             time_indices=calibration_endpoints, threshold=np.asarray(threshold))
    test_windows = torch.from_numpy(_windows(normalized, test_indices, args.seq_len))
    with torch.no_grad():
        scores = torch.cat([
            _score_batch(model, batch.to(device), frequency_metric, args.score_lambda)[3][:, -1].detach().cpu()
            for (batch,) in DataLoader(TensorDataset(test_windows), batch_size=args.batch_size, shuffle=False)
        ]).numpy()
    predictions = (scores > threshold).astype(np.int64)
    np.savez_compressed(output / "results" / "aligned_scores.npz", scores=scores,
                        predictions=predictions, time_indices=test_indices,
                        threshold=np.asarray(threshold), actual=labels[test_indices])
    data_identity = hashlib.sha256(values.tobytes()).hexdigest()
    run_identity = hashlib.sha256(json.dumps({"task_name": args.task_name, "data_identity": data_identity,
                                               "splits": split_record, "seq_len": args.seq_len}, sort_keys=True).encode()).hexdigest()
    common = {"protocol": "CATCH-common", "status": "complete", "task_name": args.task_name,
              "run_identity": run_identity, "data_identity": data_identity,
              "split_identity": split_record, "checkpoint_path": str(checkpoint),
              "checkpoint_sha256": _sha256(checkpoint),
              "normalization_path": str((output / "normalization.npz").resolve()),
              "normalization_sha256": _sha256(output / "normalization.npz"),
              "train_fit_rows_path": str(fit_rows_path.resolve()),
              "train_fit_rows_sha256": _sha256(fit_rows_path),
              "score_protocol": {"temporal": "pointwise_mse", "frequency": "CATCH-frequency_criterion",
                                 "score_lambda": float(args.score_lambda),
                                 "inference_patch_size": int(args.inference_patch_size),
                                 "inference_patch_stride": int(args.inference_patch_stride)},
              "train_indices_sha256": split_record["train"]["sha256"],
              "validation_indices_sha256": split_record["validation"]["sha256"],
              "calibration_indices_sha256": split_record["calibration"]["sha256"]}
    _atomic_json(output / "evidence" / "training.json", common)
    calibration_evidence = dict(common, threshold=threshold,
                                calibration_quantile=args.calibration_quantile,
                                calibration_scores_path=str((output / "results" / "calibration_scores.npz").resolve()),
                                calibration_scores_sha256=_sha256(output / "results" / "calibration_scores.npz"))
    _atomic_json(output / "evidence" / "calibration.json", calibration_evidence)
    _atomic_json(output / "results" / "metrics.json", {"status": "complete", "task_name": args.task_name,
                                                          "protocol": "CATCH-common", "threshold": threshold,
                                                          "test_count": int(len(scores))})
    np.save(output / "evidence" / "train_indices.npy", train_endpoints)
    np.save(output / "evidence" / "validation_indices.npy", validation_endpoints)
    np.save(output / "evidence" / "calibration_indices.npy", calibration_endpoints)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--train-indices", required=True)
    parser.add_argument("--validation-indices", required=True)
    parser.add_argument("--calibration-indices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--test-start", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--cf-dim", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--calibration-quantile", type=float, default=0.995)
    parser.add_argument("--score-lambda", type=float, default=0.05)
    parser.add_argument("--mask-learning-rate", type=float, default=1e-5)
    parser.add_argument("--inference-patch-size", type=int, default=32)
    parser.add_argument("--inference-patch-stride", type=int, default=1)
    parser.add_argument("--pct-start", type=float, default=0.3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
