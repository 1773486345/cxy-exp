"""Benchmark adapter for the two-stage MNRF-Lite V2 detector."""

import json
import os
import platform
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .config import MNRFConfig
from .training import MNRFTrainer, purged_ranges, seed_all


class MNRFLiteV2:
    model_name = "MNRFLiteV2"

    def __init__(self, **kwargs):
        self.config = MNRFConfig.from_kwargs(**kwargs)
        self.device = self._select_device(self.config.device)
        self.trainer: Optional[MNRFTrainer] = None
        self.variable_names = None
        self.output_dir = Path(self.config.output_dir) if self.config.output_dir else None
        self._fitted = False
        self.training_metadata = {}

    @staticmethod
    def required_hyper_params():
        return {}

    @staticmethod
    def _select_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        selected = torch.device(device)
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested device {device}, but CUDA is unavailable")
        if selected.type == "cuda":
            try:
                torch.empty(0, device=selected)
            except RuntimeError as error:
                raise RuntimeError(f"requested device {device}, but CUDA cannot be initialized for this process: {error}") from error
        return selected

    def _make_output_dirs(self):
        if self.output_dir is None:
            self.output_dir = Path("result/score/MNRF-Lite-V2/adhoc/run-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for name in ("checkpoints", "cache", "results", "logs"):
            (self.output_dir / name).mkdir()

    @staticmethod
    def _features_from_frame(frame):
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        excluded = {"label", "timestamp", "date", "time", "datetime", "index"}
        feature_columns = [column for column in frame.columns if str(column).lower() not in excluded]
        if len(feature_columns) < 2:
            raise ValueError(f"MNRF-Lite V2 requires at least two variables, got {len(feature_columns)}")
        values = frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("time series contains NaN or infinite values")
        return values, feature_columns

    def detect_hyper_param_tune(self, train_data):
        values, columns = self._features_from_frame(train_data)
        if len(values) < self.config.seq_len:
            raise ValueError(f"training series length {len(values)} is shorter than seq_len={self.config.seq_len}")
        self.variable_names = tuple(map(str, columns))
        return len(columns)

    def detect_fit(self, train_data, train_label=None):
        fit_start = time.perf_counter()
        self.config.validate()
        seed_all(self.config.seed)
        values, columns = self._features_from_frame(train_data)
        if len(columns) < 2:
            raise ValueError("MNRF-Lite V2 requires D >= 2")
        if len(values) < self.config.seq_len:
            raise ValueError("training segment is shorter than the configured window")
        self.variable_names = tuple(map(str, columns))
        ranges = purged_ranges(len(values), self.config.seq_len)
        self._make_output_dirs()
        self.trainer = MNRFTrainer(self.config, len(columns), self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        input_train_start, input_train_end = ranges["train"]
        self.trainer.fixed_input.fit(values[input_train_start:input_train_end])
        standardized = self.trainer.fixed_input.transform(values)
        stage1_start = time.perf_counter()
        stage1_best = self.trainer.stage1(
            standardized,
            ranges["train"],
            ranges["validation"],
            self.output_dir / "checkpoints",
        )
        stage1_seconds = time.perf_counter() - stage1_start
        cache_dir = self.output_dir / "cache"
        train_path, train_count, train_indices = self.trainer.extract_cache(standardized, ranges["train"], "train", cache_dir)
        validation_path, validation_count, validation_indices = self.trainer.extract_cache(standardized, ranges["validation"], "validation", cache_dir)
        calibration_path, calibration_count, calibration_indices = self.trainer.extract_cache(standardized, ranges["calibration"], "calibration", cache_dir)
        train_features = np.load(train_path, mmap_mode="r")
        validation_features = np.load(validation_path, mmap_mode="r")
        calibration_features = np.load(calibration_path, mmap_mode="r")
        self.trainer.fixed_feature.fit(train_features, chunk_size=self.config.cache_chunk_size)
        feature_mean, feature_scale = self.trainer.fixed_feature.mean_, self.trainer.fixed_feature.scale_
        standardized_train = (train_features - feature_mean) / feature_scale
        standardized_validation = (validation_features - feature_mean) / feature_scale
        standardized_calibration = (calibration_features - feature_mean) / feature_scale
        stage2_start = time.perf_counter()
        stage2_best = self.trainer.stage2(
            standardized_train,
            standardized_validation,
            self.output_dir / "checkpoints",
        )
        stage2_seconds = time.perf_counter() - stage2_start
        self.trainer.calibrate(standardized_validation, standardized_calibration)
        labels = None if train_label is None else np.asarray(train_label).reshape(-1)
        label_audit = {"available": labels is not None, "count": 0, "positive_count": None, "used_for_filtering": False}
        if labels is not None:
            label_audit.update({
                "count": int(len(labels)),
                "positive_count": int(np.sum(labels == 1)),
                "other_value_count": int(np.sum(~np.isin(labels, (0, 1)))),
            })
        self.training_metadata = {
            "config": self.config.to_dict(),
            "device": str(self.device),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "variable_names": list(self.variable_names),
            "train_label_audit": label_audit,
            "split_ranges": {
                name: {"start": int(start), "end_exclusive": int(end), "points": int(end - start)}
                for name, (start, end) in (
                    ("train", ranges["train"]),
                    ("validation", ranges["validation"]),
                    ("calibration", ranges["calibration"]),
                )
            },
            "purge_gap_points": ranges["gap"],
            "window_counts": {
                "train": train_count,
                "validation": validation_count,
                "calibration": calibration_count,
            },
            "stage1_best_validation_macro_nll": stage1_best,
            "stage1_seconds": stage1_seconds,
            "stage2_best_validation_nll": stage2_best,
            "stage2_seconds": stage2_seconds,
            "total_fit_seconds": time.perf_counter() - fit_start,
            "feature_fingerprint": self.trainer.fingerprint(),
            "parameter_counts": {
                "encoder": sum(parameter.numel() for parameter in self.trainer.encoder.parameters()),
                "fusion": sum(parameter.numel() for parameter in self.trainer.flow.parameters()),
                "deployment_total": sum(parameter.numel() for parameter in self.trainer.encoder.parameters())
                + sum(parameter.numel() for parameter in self.trainer.flow.parameters()),
            },
            "threshold": self.trainer.threshold,
            "calibration_quantile": self.config.calibration_quantile,
            "feature_cache_paths": {
                "train": str(train_path), "validation": str(validation_path), "calibration": str(calibration_path)
            },
            "feature_cache_window_end_indices": {
                "train": str(train_indices), "validation": str(validation_indices), "calibration": str(calibration_indices)
            },
            "feature_cache_window_counts": {
                "train": train_count, "validation": validation_count, "calibration": calibration_count
            },
        }
        if self.device.type == "cuda":
            self.training_metadata["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
            self.training_metadata["cuda_peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
        with open(self.output_dir / "cache_manifest.json", "w", encoding="utf-8") as handle:
            json.dump({
                "variable_names": list(self.variable_names),
                "architecture_config": {key: self.config.to_dict()[key] for key in (
                    "seq_len", "pattern_len", "hidden_dim", "view_dim"
                )},
                "input_mean": self.trainer.fixed_input.mean_.tolist(),
                "input_scale": self.trainer.fixed_input.scale_.tolist(),
                "encoder_fingerprint": self.training_metadata["feature_fingerprint"],
                "feature_cache_paths": self.training_metadata["feature_cache_paths"],
                "window_end_indices": self.training_metadata["feature_cache_window_end_indices"],
            }, handle, indent=2)
        self._save_artifacts()
        self._fitted = True

    def _save_artifacts(self):
        assert self.trainer is not None
        root = self.output_dir
        np.savez(
            root / "input_statistics.npz",
            mean=self.trainer.fixed_input.mean_,
            scale=self.trainer.fixed_input.scale_,
            feature_mean=self.trainer.fixed_feature.mean_,
            feature_scale=self.trainer.fixed_feature.scale_,
            calibration_center=self.trainer.calibration_center,
            calibration_scale=self.trainer.calibration_scale,
            noise_vectors=self.trainer.noise_vectors.cpu().numpy(),
            threshold=np.asarray(self.trainer.threshold, dtype=np.float64),
            variable_names=np.asarray(self.variable_names),
        )
        with open(root / "training_log.json", "w", encoding="utf-8") as handle:
            json.dump(self.trainer.history, handle, indent=2)
        with open(root / "run_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(self.training_metadata, handle, indent=2, default=str)
        artifact = {
            "config": self.config.to_dict(),
            "variable_names": self.variable_names,
            "encoder_state": self.trainer.encoder.state_dict(),
            "flow_state": self.trainer.flow.state_dict(),
            "input_mean": self.trainer.fixed_input.mean_,
            "input_scale": self.trainer.fixed_input.scale_,
            "feature_mean": self.trainer.fixed_feature.mean_,
            "feature_scale": self.trainer.fixed_feature.scale_,
            "noise_vectors": self.trainer.noise_vectors.cpu(),
            "calibration_center": self.trainer.calibration_center,
            "calibration_scale": self.trainer.calibration_scale,
            "threshold": self.trainer.threshold,
            "training_metadata": self.training_metadata,
        }
        torch.save(artifact, root / "deployment.pt")

    def load_checkpoint(self, path):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        requested_device = self.config.device
        config = MNRFConfig.from_kwargs(**artifact["config"])
        config.device = requested_device
        if self.config.output_dir:
            config.output_dir = self.config.output_dir
        self.config = config
        self.device = self._select_device(config.device)
        names = tuple(artifact["variable_names"])
        self.variable_names = names
        self.trainer = MNRFTrainer(config, len(names), self.device)
        self.trainer.encoder.load_state_dict(artifact["encoder_state"])
        self.trainer.flow.load_state_dict(artifact["flow_state"])
        for parameter in self.trainer.encoder.parameters():
            parameter.requires_grad_(False)
        self.trainer.encoder.eval()
        self.trainer.flow.eval()
        self.trainer.fixed_input.mean_ = np.asarray(artifact["input_mean"], dtype=np.float32)
        self.trainer.fixed_input.scale_ = np.asarray(artifact["input_scale"], dtype=np.float32)
        self.trainer.fixed_feature.mean_ = np.asarray(artifact["feature_mean"], dtype=np.float32)
        self.trainer.fixed_feature.scale_ = np.asarray(artifact["feature_scale"], dtype=np.float32)
        self.trainer.noise_vectors = artifact["noise_vectors"].to(self.device)
        self.trainer.calibration_center = np.asarray(artifact["calibration_center"], dtype=np.float32)
        self.trainer.calibration_scale = np.asarray(artifact["calibration_scale"], dtype=np.float32)
        self.trainer.threshold = float(artifact["threshold"])
        self.training_metadata = artifact.get("training_metadata", {})
        self._fitted = True
        return self

    @torch.no_grad()
    def score_with_metadata(self, test_data):
        if not self._fitted or self.trainer is None:
            raise RuntimeError("detect_fit or load_checkpoint must be called before scoring")
        values, columns = self._features_from_frame(test_data)
        names = tuple(map(str, columns))
        if names != self.variable_names:
            raise ValueError("test variable names/order do not match training variables")
        normalized = self.trainer.fixed_input.transform(values)
        total = len(normalized)
        valid_mask = np.zeros(total, dtype=bool)
        if total < self.config.seq_len:
            return {
                "scores": np.empty(0, dtype=np.float32),
                "predictions": np.empty(0, dtype=np.int64),
                "time_indices": np.empty(0, dtype=np.int64),
                "valid_mask": valid_mask,
                "per_variable_nll": np.empty((0, len(columns)), dtype=np.float32),
                "per_variable_calibrated": np.empty((0, len(columns)), dtype=np.float32),
                "threshold": self.trainer.threshold,
            }
        per_variable = []
        chunk = max(1, self.config.cache_chunk_size)
        count = total - self.config.seq_len + 1
        for start in range(0, count, chunk):
            stop = min(count, start + chunk)
            windows = np.stack([
                normalized[i:i + self.config.seq_len]
                for i in range(start, stop)
            ])
            raw = self.trainer.encoder.encode_views(torch.as_tensor(windows, device=self.device))["features"].cpu().numpy()
            fixed = self.trainer.fixed_feature.transform(raw)
            per_variable.append(self.trainer.per_variable_scores(fixed))
        nll = np.concatenate(per_variable, axis=0)
        calibrated = (nll - self.trainer.calibration_center) / self.trainer.calibration_scale
        scores = np.max(calibrated, axis=1).astype(np.float32)
        predictions = (scores > self.trainer.threshold).astype(np.int64)
        time_indices = np.arange(self.config.seq_len - 1, total, dtype=np.int64)
        valid_mask[time_indices] = True
        return {
            "scores": scores,
            "predictions": predictions,
            "time_indices": time_indices,
            "valid_mask": valid_mask,
            "per_variable_nll": nll.astype(np.float32),
            "per_variable_calibrated": calibrated.astype(np.float32),
            "threshold": self.trainer.threshold,
        }

    def detect_score(self, test_data):
        result = self.score_with_metadata(test_data)
        return result["scores"], result["predictions"]


def load_model(path, device="auto"):
    model = MNRFLiteV2(device=device)
    return model.load_checkpoint(path)
