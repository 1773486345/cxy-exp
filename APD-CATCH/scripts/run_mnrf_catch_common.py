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
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.catch.CATCH import TransformerConfig
from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel
from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_loss, frequency_criterion
from ts_benchmark.baselines.catch.utils.tools import adjust_learning_rate
from ts_benchmark.baselines.mnrf_view_probe.evaluation import (
    calibrated_quantile, strict_integer_indices, validate_common_splits,
)
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
from ts_benchmark.baselines.mnrf_view_probe.probe import _all_boundary_ids, _chunked_array_hash, _hash_strings
from ts_benchmark.baselines.mnrf_view_probe.common_contract import (
    CONTRACT_SCHEMA, endpoints_for_ranges, expected_contract_endpoints,
    missing_contract_fields, time_semantics_compatible,
)
from scripts.mnrf_common_data import load_task_frame


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_identity(values):
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True).encode())
    flat = array.reshape(-1)
    for start in range(0, flat.size, 65536):
        digest.update(np.ascontiguousarray(flat[start:start + 65536]).tobytes())
    return digest.hexdigest()


def _string_identity(values):
    digest = hashlib.sha256()
    for value in np.asarray(values, dtype=object).reshape(-1):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _atomic_json(path, payload):
    temporary = Path(path).with_name(Path(path).name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class _WindowDataset(Dataset):
    """Read only the current window when a DataLoader requests a sample."""

    def __init__(self, values, endpoints, seq_len):
        self.values = values
        self.endpoints = strict_integer_indices(endpoints, "window endpoints", upper_bound=len(values))
        self.seq_len = int(seq_len)
        if len(self.endpoints) == 0 or np.any(self.endpoints < self.seq_len - 1):
            raise ValueError("explicit split has no complete windows")

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, index):
        endpoint = int(self.endpoints[index])
        window = np.asarray(self.values[endpoint - self.seq_len + 1:endpoint + 1], dtype=np.float32)
        return (torch.from_numpy(window),)


def _windows(values, endpoints, seq_len):
    return _WindowDataset(values, endpoints, seq_len)


def _window_intervals(endpoints, seq_len, length, name):
    endpoints = strict_integer_indices(endpoints, f"{name} endpoints", upper_bound=length)
    if len(endpoints) == 0 or np.any(endpoints < seq_len - 1):
        raise ValueError(f"{name} has an incomplete window")
    intervals = np.column_stack((endpoints - int(seq_len) + 1, endpoints + 1))
    return intervals[np.argsort(intervals[:, 0], kind="stable")]


def _endpoints_for_ranges(segment_ranges, lower, upper, seq_len):
    """Derive the complete, non-subsampled endpoint sequence from frozen segments."""
    return endpoints_for_ranges(segment_ranges, lower, upper, seq_len)


def _intervals_overlap(left, right):
    i = j = 0
    while i < len(left) and j < len(right):
        if max(int(left[i, 0]), int(right[j, 0])) < min(int(left[i, 1]), int(right[j, 1])):
            return True
        if left[i, 1] <= right[j, 1]:
            i += 1
        else:
            j += 1
    return False


def _indices_overlap_intervals(indices, intervals):
    ordered = np.sort(np.asarray(indices, dtype=np.int64))
    for start, stop in intervals:
        position = np.searchsorted(ordered, start, side="left")
        if position < len(ordered) and ordered[position] < stop:
            return True
    return False


def _boundary_keys(frame):
    columns = [column for column in frame.columns
               if str(column).lower() in {"entity_id", "session_id", "record_id", "scenario_id"}]
    if not columns:
        return np.zeros(len(frame), dtype=np.int64), columns
    composite = frame[columns].astype(object).where(frame[columns].notna(), "<NA>").astype(str)
    keys = composite.agg("\x1f".join, axis=1).to_numpy(dtype=object)
    return keys, columns


def _training_intervals_by_boundary(frame, endpoints, seq_len, timestamps, expected_interval=None):
    """Return one frozen training cadence from raw adjacent rows.

    Cadence is a property of the frozen training protocol, not of each
    overlapping window or of each entity seen later in test.  Computing it
    from raw adjacent rows also avoids retaining O(number_of_windows * L)
    temporary difference arrays.
    """
    keys, _ = _boundary_keys(frame)
    if expected_interval is not None:
        cadence = float(expected_interval)
        if not np.isfinite(cadence) or cadence <= 0:
            raise ValueError("frozen temporal rule must contain a positive expected interval")
        return cadence
    endpoints = strict_integer_indices(endpoints, "training endpoints", upper_bound=len(frame))
    if len(endpoints) == 0:
        return None
    raw_start = max(0, int(endpoints.min()) - int(seq_len) + 1)
    raw_stop = min(len(frame), int(endpoints.max()) + 1)
    deltas = []
    for left in range(raw_start, raw_stop - 1):
        if keys[left] != keys[left + 1]:
            continue
        delta = int(timestamps[left + 1]) - int(timestamps[left])
        if delta <= 0:
            raise ValueError("training timestamps contain duplicate or reversed rows")
        deltas.append(delta)
    if not deltas:
        return None
    return float(np.median(np.asarray(deltas, dtype=np.int64)))


def _validate_window_boundaries(frame, endpoints_by_name, seq_len, test_start, expected_interval=None):
    """Reject windows that overlap splits, entities, or unexplained time gaps."""
    length = len(frame)
    intervals = {name: _window_intervals(endpoints, seq_len, length, name)
            for name, endpoints in endpoints_by_name.items()}
    names = tuple(intervals)
    for left, name_left in enumerate(names):
        for name_right in names[left + 1:]:
            if _intervals_overlap(intervals[name_left], intervals[name_right]):
                raise ValueError(f"{name_left}/{name_right} windows overlap; provide purged split endpoints")
    if np.any(intervals["test"][:, 0] < int(test_start)):
        raise ValueError("test windows cross the train/test boundary; test endpoints must start at test_start + seq_len - 1")
    boundary_columns = [column for column in frame.columns
                        if str(column).lower() in {"entity_id", "session_id", "record_id", "scenario_id"}]
    for boundary_column in boundary_columns:
        boundary = frame[boundary_column].to_numpy()
        for name, endpoints in endpoints_by_name.items():
            for endpoint in endpoints:
                start = int(endpoint) - seq_len + 1
                if np.any(boundary[start:int(endpoint) + 1] != boundary[start]):
                    raise ValueError(f"{name} window crosses {boundary_column} boundary")
    timestamp_column = next((column for column in frame.columns
                             if str(column).lower() in {"timestamp", "date", "time", "datetime"}), None)
    has_timestamp = timestamp_column is not None or isinstance(frame.index, pd.DatetimeIndex)
    if has_timestamp:
        timestamps = np.asarray(MNRFLiteV2._time_index(frame).values_ns, dtype=np.int64)
        if np.any(timestamps == np.iinfo(np.int64).min):
            raise ValueError("timestamps contain NaT values")
        boundary_keys, _ = _boundary_keys(frame)
        cuts = (np.flatnonzero(boundary_keys[1:] != boundary_keys[:-1]) + 1).astype(int)
        bounds = np.r_[0, cuts, len(timestamps)]
        for left, right in zip(bounds[:-1], bounds[1:]):
            if right - left > 1 and np.any(np.diff(timestamps[left:right]) <= 0):
                raise ValueError("timestamps duplicate or reverse within an entity/session segment")
        cadence = _training_intervals_by_boundary(
            frame, endpoints_by_name["train"], seq_len, timestamps,
            expected_interval=expected_interval,
        )
        if cadence is None:
            raise ValueError("cannot infer a positive training sampling interval")
        # Check each actual window, never the union of disjoint windows.  A
        # legal purge or separate entity therefore cannot look like a gap.
        for name, endpoints in endpoints_by_name.items():
            for endpoint in endpoints:
                start = int(endpoint) - seq_len + 1
                key = boundary_keys[start]
                if np.any(boundary_keys[start:int(endpoint) + 1] != key):
                    continue  # The boundary-specific error below is more precise.
                deltas = np.diff(timestamps[start:int(endpoint) + 1])
                if np.any(deltas <= 0):
                    raise ValueError(f"{name} window has duplicate or reversed timestamps")
                if np.any(np.abs(deltas / cadence - 1.0) > 0.05):
                    raise ValueError(f"{name} window crosses a timestamp gap")
    return intervals


def _filter_test_endpoints(frame, endpoints, seq_len, train_endpoints, expected_interval=None):
    """Keep only complete test windows; gaps create cold-start exclusions."""
    endpoints = np.asarray(endpoints, dtype=np.int64)
    timestamp_column = next((column for column in frame.columns
                             if str(column).lower() in {"timestamp", "date", "time", "datetime"}), None)
    timestamps = None
    cadence = None
    if timestamp_column is not None or isinstance(frame.index, pd.DatetimeIndex):
        timestamps = np.asarray(MNRFLiteV2._time_index(frame).values_ns, dtype=np.int64)
        cadence = _training_intervals_by_boundary(
            frame, train_endpoints, seq_len, timestamps,
            expected_interval=expected_interval,
        )
        if cadence is None:
            raise ValueError("cannot infer a positive training sampling interval")
        boundary_keys, _ = _boundary_keys(frame)
    boundary_columns = [column for column in frame.columns
                        if str(column).lower() in {"entity_id", "session_id", "record_id", "scenario_id"}]
    valid = []
    for endpoint in endpoints:
        start = int(endpoint) - seq_len + 1
        if any(np.any(frame[column].to_numpy()[start:int(endpoint) + 1] != frame[column].to_numpy()[start])
               for column in boundary_columns):
            continue
        if timestamps is not None:
            deltas = np.diff(timestamps[start:int(endpoint) + 1])
            if np.any(deltas <= 0):
                continue
            if np.any(np.abs(deltas / cadence - 1.0) > 0.05):
                continue
        valid.append(int(endpoint))
    return np.asarray(valid, dtype=np.int64)


def _score_batch(model, batch, frequency_metric, score_lambda):
    outputs = model(batch)[0]
    temporal = (batch - outputs).square().mean(dim=-1)
    frequency = frequency_metric(batch, outputs).mean(dim=-1)
    return outputs, temporal, frequency, temporal + float(score_lambda) * frequency


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    frame, label_column = load_task_frame(args.data_csv)
    values, feature_columns = MNRFLiteV2._features_from_frame(frame)
    labels = frame[label_column].to_numpy(dtype=np.int64)
    time_index = MNRFLiteV2._time_index(frame)
    canonical_time = np.asarray(time_index.values_ns, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("data features must be finite with at least two variables")
    contract = None
    contract_path = getattr(args, "common_contract", "")
    if contract_path:
        contract_path = str(Path(contract_path).resolve())
        contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
        if contract.get("schema") != CONTRACT_SCHEMA or contract.get("status") != "complete":
            raise ValueError("common contract is incomplete or unsupported")
        missing_contract = missing_contract_fields(contract)
        if missing_contract:
            raise ValueError("common contract is missing required fields: " + ", ".join(missing_contract))
        source_identity = contract["probe_source_identity"]
        if contract.get("data_csv_path") and Path(contract["data_csv_path"]).resolve() != Path(args.data_csv).resolve():
            raise ValueError("task CSV path differs from the frozen common contract")
        if contract.get("data_csv_sha256") and contract["data_csv_sha256"] != _sha256(args.data_csv):
            raise ValueError("task CSV bytes differ from the frozen common contract")
        probe_manifest_path = Path(contract["probe_run"]) / "probe_manifest.json"
        if (not probe_manifest_path.is_file()
                or contract.get("probe_manifest_sha256") != _sha256(probe_manifest_path)):
            raise ValueError("common contract Probe manifest identity differs")
        probe_manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
        if probe_manifest.get("source_identity") != source_identity:
            raise ValueError("common contract Probe source identity differs from its manifest")
        if source_identity.get("task_name") != contract.get("task_name"):
            raise ValueError("common contract task identity differs")
        if source_identity.get("temporal_rule") != contract.get("temporal_rule"):
            raise ValueError("common contract temporal rule is not sourced from Probe")
        expected_len = int(contract.get("catch_seq_len", args.seq_len))
        if expected_len != int(args.seq_len):
            raise ValueError(f"CATCH seq_len={args.seq_len} differs from common contract catch_seq_len={expected_len}")
        catch_records = contract.get("catch_splits", {})
        for name in ("train", "validation", "calibration"):
            record = catch_records.get(name)
            if not isinstance(record, dict) or not Path(record.get("path", "")).is_file():
                raise ValueError(f"common contract is missing CATCH {name} endpoints")
            if Path(record["path"]).resolve() != Path(getattr(args, f"{name}_indices")).resolve():
                raise ValueError(f"{name} endpoint path is not the one derived by the common contract")
            if not record.get("sha256") or _sha256(record["path"]) != record["sha256"]:
                raise ValueError(f"common contract {name} endpoint hash differs")
        test_record = contract.get("catch_test")
        if not isinstance(test_record, dict) or not Path(test_record.get("path", "")).is_file():
            raise ValueError("common contract is missing CATCH test endpoints")
        if not test_record.get("sha256") or _sha256(test_record["path"]) != test_record["sha256"]:
            raise ValueError("common contract test endpoint hash differs")
        contract_test_indices = np.load(test_record["path"], allow_pickle=False)
        expected_by_name = {}
        for name in ("train", "validation", "calibration"):
            expected_by_name[name] = expected_contract_endpoints(contract, name, args.seq_len)
            actual = np.load(catch_records[name]["path"], allow_pickle=False)
            if not np.array_equal(strict_integer_indices(actual, f"contract {name} endpoints"), expected_by_name[name]):
                raise ValueError(f"common contract {name} endpoints are not the exact frozen derivation")
        expected_test = expected_contract_endpoints(contract, "test", args.seq_len)
        if not np.array_equal(strict_integer_indices(contract_test_indices, "contract test endpoints"), expected_test):
            raise ValueError("common contract test endpoints are not the exact frozen derivation")
        train_rows_record = contract.get("train_rows")
        if (not isinstance(train_rows_record, dict) or not train_rows_record.get("path")
                or not train_rows_record.get("sha256") or not Path(train_rows_record["path"]).is_file()
                or _sha256(train_rows_record["path"]) != train_rows_record["sha256"]):
            raise ValueError("common contract raw training rows binding is missing or changed")
        if Path(args.train_rows).resolve() != Path(train_rows_record["path"]).resolve():
            raise ValueError("raw training rows path is not the one frozen by the common contract")
        fit_rows_candidate = np.load(train_rows_record["path"], allow_pickle=False)
        expected_rows = np.arange(*map(int, contract.get("raw_training_range", ())), dtype=np.int64)
        if not np.array_equal(strict_integer_indices(fit_rows_candidate, "contract raw training rows"), expected_rows):
            raise ValueError("common contract raw training rows differ from Probe frozen training rows")
        labels_record = contract.get("labels")
        if (not isinstance(labels_record, dict) or not labels_record.get("path")
                or not labels_record.get("sha256") or not Path(labels_record["path"]).is_file()
                or _sha256(labels_record["path"]) != labels_record["sha256"]):
            raise ValueError("common contract labels binding is missing or changed")
        frozen_labels = np.load(labels_record["path"], allow_pickle=False)
        if (frozen_labels.ndim != 1 or frozen_labels.dtype.kind not in "iu"
                or not np.array_equal(frozen_labels.astype(np.int64), labels)):
            raise ValueError("loaded full-coordinate labels differ from the common contract")
    else:
        contract_test_indices = None
    test_start = int(contract.get("train_length")) if contract is not None else int(args.test_start)
    if contract is not None and int(contract.get("input_length", -1)) != len(values):
        raise ValueError("common contract input length differs")
    if contract is not None and getattr(args, "test_start", None) is not None and int(args.test_start) != test_start:
        raise ValueError("test_start differs from the common contract")
    if test_start < 0:
        raise ValueError("test_start is required unless common_contract supplies train_length")
    if contract is not None:
        if list(map(str, feature_columns)) != list(map(str, contract.get("variable_names", []))):
            raise ValueError("common contract variable order differs from the CATCH data")
        if _chunked_array_hash(values) != contract.get("data_identity"):
            raise ValueError("common contract feature data identity differs")
        time_index = MNRFLiteV2._time_index(frame)
        if not time_semantics_compatible(
                contract.get("time_kind"), contract.get("source_timezone"),
                time_index.kind, time_index.source_timezone):
            raise ValueError("common contract time kind/timezone differs")
        if _chunked_array_hash(time_index.values_ns) != contract.get("time_identity"):
            raise ValueError("common contract time identity differs")
        boundary_ids, boundary_columns = _all_boundary_ids(frame)
        if boundary_columns != contract.get("boundary_columns", []):
            raise ValueError("common contract boundary columns differ")
        boundary_hash = None if boundary_ids is None else _hash_strings(boundary_ids)
        if boundary_hash != contract.get("boundary_identity"):
            raise ValueError("common contract boundary identity differs")
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
    test_indices = (strict_integer_indices(contract_test_indices, "contract test endpoints", upper_bound=len(values))
                    if contract_test_indices is not None else
                    np.arange(test_start + args.seq_len - 1, len(values), dtype=np.int64))
    if contract_test_indices is None:
        test_indices = _filter_test_endpoints(frame, test_indices, args.seq_len, train_indices)
    if len(test_indices) == 0:
        raise ValueError("test interval has no complete windows")
    endpoint_sets = {"train": train_endpoints, "validation": validation_endpoints,
                     "calibration": calibration_endpoints, "test": test_indices}
    expected_interval = None
    if contract is not None:
        expected_interval = contract.get("temporal_rule", {}).get("expected_interval_ns")
    boundary_rows = _validate_window_boundaries(
        frame, endpoint_sets, args.seq_len, test_start,
        expected_interval=expected_interval,
    )
    if args.train_rows:
        fit_rows = strict_integer_indices(np.load(args.train_rows, allow_pickle=False), "raw training rows", upper_bound=len(values))
        if len(fit_rows) == 0:
            raise ValueError("raw training rows are empty")
        if np.any(fit_rows >= test_start):
            raise ValueError("raw training rows must be before test_start")
        forbidden = np.concatenate([boundary_rows[name] for name in ("validation", "calibration", "test")])
        if _indices_overlap_intervals(fit_rows, forbidden):
            raise ValueError("raw training rows overlap validation, calibration, or test windows")
    else:
        raise ValueError("--train-rows is required to distinguish raw training rows from window endpoints")
    scaler = StandardScaler().fit(values[fit_rows])
    output.mkdir(parents=True)
    for name in ("checkpoints", "results", "evidence"):
        (output / name).mkdir()
    normalized = scaler.transform(values).astype(np.float32)
    fit_rows_path = output / "evidence" / "train_fit_rows.npy"
    np.save(fit_rows_path, fit_rows)
    np.savez(output / "normalization.npz", mean=scaler.mean_, scale=scaler.scale_,
             fit_rows_sha256=_sha256(fit_rows_path), fit_rows_count=np.asarray(len(fit_rows)))
    model_config = getattr(args, "model_config", None)
    if model_config is None:
        config = TransformerConfig(seq_len=args.seq_len, patch_size=args.patch_size,
                                   patch_stride=args.patch_stride, d_model=args.d_model,
                                   cf_dim=args.cf_dim, d_ff=args.d_ff, e_layers=args.e_layers,
                                   n_heads=args.n_heads, head_dim=args.head_dim, dropout=args.dropout,
                                   lr=args.learning_rate, num_epochs=args.epochs,
                                   batch_size=args.batch_size, patience=args.patience,
                                   score_lambda=args.score_lambda, Mlr=args.mask_learning_rate,
                                   auxi_lambda=getattr(args, "auxi_lambda", 0.005),
                                   dc_lambda=getattr(args, "dc_lambda", 0.005),
                                   temperature=getattr(args, "temperature", 0.07),
                                   head_dropout=getattr(args, "head_dropout", 0.1),
                                   anomaly_ratio=getattr(args, "anomaly_ratio", [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25]),
                                   use_multi_scale=getattr(args, "use_multi_scale", "False"),
                                   small_kernel_merged=getattr(args, "small_kernel_merged", "False"),
                                   itr=getattr(args, "itr", 1),
                                   inference_patch_size=args.inference_patch_size,
                                   inference_patch_stride=args.inference_patch_stride,
                                   pct_start=args.pct_start)
    else:
        config = TransformerConfig(**dict(model_config))
        if (int(config.seq_len) != int(args.seq_len)
                or int(config.batch_size) != int(args.batch_size)
                or int(config.num_epochs) != int(args.epochs)
                or float(config.lr) != float(args.learning_rate)
                or float(config.Mlr) != float(args.mask_learning_rate)):
            raise ValueError("runtime training arguments differ from archived CATCH model config")
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
    train_windows = _windows(normalized, train_endpoints, args.seq_len)
    validation_windows = _windows(normalized, validation_endpoints, args.seq_len)
    main_params = [parameter for name, parameter in model.named_parameters()
                   if "mask_generator" not in name]
    optimizer = torch.optim.Adam(main_params, lr=args.learning_rate)
    mask_optimizer = torch.optim.Adam(model.mask_generator.parameters(), lr=args.mask_learning_rate)
    train_loader = DataLoader(train_windows, batch_size=args.batch_size, shuffle=False)
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
    mask_update_interval = max(1, min(int(len(train_loader) / 10), 100))
    mask_optimizer.zero_grad()
    training_started = __import__("time").perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, complex_output, dc_loss = model(batch)
            normalized_batch = model.revin_layer(batch, "transform")
            loss = criterion(reconstruction, batch) + config.dc_lambda * dc_loss + config.auxi_lambda * aux_loss(complex_output, normalized_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite CATCH-common training loss")
            # Preserve the original CATCH schedule: mask gradients accumulate
            # and are stepped only every 10% of an epoch (up to 100 batches),
            # while the main optimizer updates every batch.  OneCycle's
            # epoch-level learning-rate adjustment is applied below.
            if ((len(losses) + 1) % mask_update_interval) == 0:
                mask_optimizer.step()
                mask_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(torch.stack([
                criterion(model(batch.to(device))[0], batch.to(device)).detach().cpu()
                for (batch,) in DataLoader(validation_windows, batch_size=args.batch_size, shuffle=False)
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
        adjust_learning_rate(optimizer, scheduler, epoch + 1, config, printout=False)
        adjust_learning_rate(mask_optimizer, mask_scheduler, epoch + 1, config, printout=False)
    training_seconds = __import__("time").perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("CATCH-common produced no validation checkpoint")
    model.load_state_dict(best_state)
    checkpoint = output / "checkpoints" / "catch_common.pt"
    torch.save({"task_name": args.task_name, "protocol": "CATCH-common", "state_dict": best_state,
                "model_config": dict(vars(config)),
                "score_protocol": {"temporal": "pointwise_mse", "frequency": "CATCH-frequency_criterion",
                                   "score_lambda": float(args.score_lambda),
                                   "inference_patch_size": int(args.inference_patch_size),
                                   "inference_patch_stride": int(args.inference_patch_stride)},
                "normalization_sha256": _sha256(output / "normalization.npz"),
                "train_fit_rows_sha256": _sha256(fit_rows_path)}, checkpoint)
    model.eval()
    calibration_windows = _windows(normalized, calibration_endpoints, args.seq_len)
    inference_started = __import__("time").perf_counter()
    with torch.no_grad():
        calibration_scores = torch.cat([
            _score_batch(model, batch.to(device), frequency_metric, args.score_lambda)[3][:, -1].detach().cpu()
            for (batch,) in DataLoader(calibration_windows, batch_size=args.batch_size, shuffle=False)
        ]).numpy()
    threshold = calibrated_quantile(calibration_scores, args.calibration_quantile)
    np.savez(output / "results" / "calibration_scores.npz", scores=calibration_scores,
             time_indices=calibration_endpoints, threshold=np.asarray(threshold))
    test_windows = _windows(normalized, test_indices, args.seq_len)
    with torch.no_grad():
        scores = torch.cat([
            _score_batch(model, batch.to(device), frequency_metric, args.score_lambda)[3][:, -1].detach().cpu()
            for (batch,) in DataLoader(test_windows, batch_size=args.batch_size, shuffle=False)
        ]).numpy()
    predictions = (scores > threshold).astype(np.int64)
    np.savez_compressed(output / "results" / "aligned_scores.npz", scores=scores,
                        predictions=predictions, time_indices=test_indices,
                        time_values_ns=canonical_time[test_indices],
                        threshold=np.asarray(threshold), actual=labels[test_indices])
    inference_seconds = __import__("time").perf_counter() - inference_started
    score_path = output / "results" / "aligned_scores.npz"
    test_actual = labels[test_indices]
    production_binding = {
        "test_scores_path": str(score_path.resolve()),
        "test_scores_sha256": _sha256(score_path),
        "test_indices_sha256": hashlib.sha256(np.ascontiguousarray(test_indices).tobytes()).hexdigest(),
        "test_predictions_sha256": hashlib.sha256(np.ascontiguousarray(predictions).tobytes()).hexdigest(),
        "test_actual_sha256": hashlib.sha256(np.ascontiguousarray(test_actual).tobytes()).hexdigest(),
        "test_count": int(len(test_indices)),
        "test_threshold": float(threshold),
    }
    data_identity = _array_identity(values)
    boundary_columns = [column for column in frame.columns
                        if str(column).lower() in {"entity_id", "session_id", "record_id", "scenario_id"}]
    time_identity = _array_identity(canonical_time)
    time_kind = time_index.kind
    boundary_values, boundary_columns = _boundary_keys(frame)
    boundary_identity = _string_identity(boundary_values) if boundary_columns else None
    variable_identity = hashlib.sha256(json.dumps(feature_columns).encode("utf-8")).hexdigest()
    run_identity = hashlib.sha256(json.dumps({"task_name": args.task_name, "data_identity": data_identity,
                                               "splits": split_record, "seq_len": args.seq_len}, sort_keys=True).encode()).hexdigest()
    common = {"protocol": "CATCH-common", "status": "complete", "task_name": args.task_name,
              "run_identity": run_identity, "data_identity": data_identity,
              "time_identity": time_identity, "variable_identity": variable_identity,
              "time_kind": time_kind, "seq_len": int(args.seq_len),
              "test_start": test_start,
              "feature_columns": list(feature_columns),
              "boundary_values_identity": boundary_identity,
              "model_config": dict(vars(config)),
              "boundary_columns": [str(column) for column in boundary_columns],
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
              "calibration_indices_sha256": split_record["calibration"]["sha256"],
              **production_binding,
              "common_contract_path": contract_path or None,
              "common_contract_sha256": _sha256(contract_path) if contract_path else None}
    _atomic_json(output / "evidence" / "training.json", common)
    calibration_evidence = dict(common, threshold=threshold,
                                calibration_quantile=args.calibration_quantile,
                                calibration_scores_path=str((output / "results" / "calibration_scores.npz").resolve()),
                                calibration_scores_sha256=_sha256(output / "results" / "calibration_scores.npz"))
    _atomic_json(output / "evidence" / "calibration.json", calibration_evidence)
    _atomic_json(output / "results" / "metrics.json", {"status": "complete", "task_name": args.task_name,
                                                          "protocol": "CATCH-common", "threshold": threshold,
                                                          "train_rows": int(len(fit_rows)),
                                                          "train_windows": int(len(train_endpoints)),
                                                          "validation_windows": int(len(validation_endpoints)),
                                                          "calibration_windows": int(len(calibration_endpoints)),
                                                          "test_count": int(len(scores)),
                                                          "training_time_seconds": float(training_seconds),
                                                          "calibration_and_inference_time_seconds": float(inference_seconds)})
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
    parser.add_argument("--train-rows", required=True,
                        help="raw training row coordinates used to fit the scaler")
    parser.add_argument("--output", required=True)
    parser.add_argument("--common-contract", default="",
                        help="validated Probe common_contract.json; derives CATCH endpoints read-only")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--test-start", type=int, default=None,
                        help="official test start; read from --common-contract when supplied")
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
    parser.add_argument("--auxi-lambda", type=float, default=0.005)
    parser.add_argument("--dc-lambda", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--anomaly-ratio", type=float, default=1.0)
    parser.add_argument("--use-multi-scale", default="False")
    parser.add_argument("--small-kernel-merged", default="False")
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--auxi-lambda", type=float, default=0.005)
    parser.add_argument("--dc-lambda", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--anomaly-ratio", type=float, default=1.0)
    parser.add_argument("--use-multi-scale", default="False")
    parser.add_argument("--small-kernel-merged", default="False")
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--inference-patch-size", type=int, default=32)
    parser.add_argument("--inference-patch-stride", type=int, default=1)
    parser.add_argument("--pct-start", type=float, default=0.3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
