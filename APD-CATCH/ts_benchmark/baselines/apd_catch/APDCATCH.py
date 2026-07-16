"""Benchmark adapter for anomaly-preserving CATCH."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .models.APDCATCH_model import APDCATCHModel, gaussian_nll


DEFAULT_HYPER_PARAMS = {
    "variant": "adaptive",
    "seq_len": 192,
    "patch_size": 16,
    "patch_stride": 8,
    "cf_dim": 64,
    "d_model": 128,
    "d_ff": 256,
    "e_layers": 3,
    "n_heads": 2,
    "head_dim": 64,
    "dropout": 0.2,
    "head_dropout": 0.1,
    "regular_lambda": 0.5,
    "temperature": 0.07,
    "cutoff_min": 0.03,
    "cutoff_max": 0.75,
    "fixed_cutoff": 0.20,
    "cutoff_temperature": 0.03,
    "cutoff_regularization": 0.001,
    "minimum_scale": 0.05,
    "maximum_scale": 3.0,
    "scale_floor_quantile": 0.05,
    "scale_floor_global_ratio": 0.01,
    "scale_floor_samples": 2048,
    "dc_lambda": 0.005,
    "lr": 0.001,
    "Mlr": 0.0001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "num_epochs": 10,
    "patience": 3,
    "validation_ratio": 0.2,
    "calibration_fpr": 0.01,
    "score_aggregation": "mean",
    "mask_update_interval": 10,
    "max_grad_norm": 1.0,
    "seed": 20261,
}


class APDCATCHConfig:
    def __init__(self, **kwargs):
        unknown = set(kwargs) - set(DEFAULT_HYPER_PARAMS)
        if unknown:
            raise ValueError(f"Unknown APD-CATCH hyperparameters: {sorted(unknown)}")
        for name, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, name, kwargs.get(name, value))
        if self.score_aggregation not in {"mean", "max"}:
            raise ValueError("score_aggregation must be 'mean' or 'max'")
        if not 0 < self.validation_ratio < 0.5:
            raise ValueError("validation_ratio must be in (0, 0.5)")
        if not 0 < self.calibration_fpr < 0.5:
            raise ValueError("calibration_fpr must be in (0, 0.5)")
        if not 0 < self.scale_floor_quantile < 0.5:
            raise ValueError("scale_floor_quantile must be in (0, 0.5)")
        if not 0 < self.scale_floor_global_ratio <= 1:
            raise ValueError("scale_floor_global_ratio must be in (0, 1]")
        if self.scale_floor_samples < 1:
            raise ValueError("scale_floor_samples must be positive")


class NextPointDataset(Dataset):
    def __init__(self, values: np.ndarray, history_length: int):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"values must be two-dimensional, got {values.shape}")
        if len(values) <= history_length:
            raise ValueError(
                f"sequence length {len(values)} must exceed history length {history_length}"
            )
        self.values = torch.from_numpy(values)
        self.history_length = history_length

    def __len__(self) -> int:
        return len(self.values) - self.history_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = index + self.history_length
        return self.values[index:end], self.values[end]


@dataclass
class FitSummary:
    epochs: int
    best_validation_nll: float
    calibration_threshold: float
    trainable_parameters: int


class APDCATCH:
    """CATCH frequency-channel backbone with target-blind adaptive decomposition."""

    def __init__(self, **kwargs):
        self.config = APDCATCHConfig(**kwargs)
        self.model_name = f"APDCATCH_{self.config.variant}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.fit_summary = None
        self.calibration_threshold = None

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self) -> str:
        return self.model_name

    def _seed_everything(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _build_model(self, n_vars: int) -> None:
        self.config.c_in = n_vars
        self.model = APDCATCHModel(self.config).to(self.device)

    def _training_scale_floor(self, values: np.ndarray) -> np.ndarray:
        """Derive a target-blind lower scale bound from training histories only."""
        history_length = self.config.seq_len
        endpoints = np.linspace(
            history_length,
            len(values) - 1,
            num=min(self.config.scale_floor_samples, len(values) - history_length),
            dtype=np.int64,
        )
        window_scales = []
        for endpoint in endpoints:
            history = values[endpoint - history_length : endpoint]
            median = np.median(history, axis=0)
            window_scales.append(1.4826 * np.median(np.abs(history - median), axis=0))
        lower_local_scale = np.quantile(
            np.asarray(window_scales), self.config.scale_floor_quantile, axis=0
        )
        global_median = np.median(values, axis=0)
        global_scale = 1.4826 * np.median(np.abs(values - global_median), axis=0)
        floor = np.maximum(
            lower_local_scale, self.config.scale_floor_global_ratio * global_scale
        )
        return np.maximum(floor, np.finfo(np.float32).eps).astype(np.float32)

    def _split_train_validation(
        self, train_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        values = train_data.to_numpy(dtype=np.float32, copy=True)
        border = int(len(values) * (1.0 - self.config.validation_ratio))
        if border <= self.config.seq_len or len(values) - border < 2:
            raise ValueError(
                "training sequence is too short for the configured history and validation split"
            )
        train_values = values[:border]
        validation_values = values[border - self.config.seq_len :]
        return train_values, validation_values

    def _loader(
        self,
        values: np.ndarray,
        shuffle: bool,
        seed_offset: int = 0,
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + seed_offset)
        return DataLoader(
            NextPointDataset(values, self.config.seq_len),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            generator=generator,
            drop_last=False,
        )

    def _cutoff_boundary_loss(self, cutoff: torch.Tensor) -> torch.Tensor:
        position = (cutoff - self.config.cutoff_min) / (
            self.config.cutoff_max - self.config.cutoff_min
        )
        return (
            torch.relu(0.05 - position).square()
            + torch.relu(position - 0.95).square()
        ).mean()

    def _batch_loss(
        self, history: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        output = self.model(history)
        nll = gaussian_nll(target, output["mean"], output["scale"]).mean()
        cutoff_loss = self._cutoff_boundary_loss(output["cutoff"])
        loss = nll + self.config.dc_lambda * output["channel_loss"]
        if self.config.variant == "adaptive":
            loss = loss + self.config.cutoff_regularization * cutoff_loss
        return loss, {
            "nll": float(nll.detach().cpu()),
            "channel_loss": float(output["channel_loss"].detach().cpu()),
            "cutoff_loss": float(cutoff_loss.detach().cpu()),
        }

    @torch.no_grad()
    def _validation_nll(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        for history, target in loader:
            history = history.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            output = self.model(history)
            nll = gaussian_nll(target, output["mean"], output["scale"])
            total += float(nll.sum().cpu())
            count += nll.numel()
        return total / max(count, 1)

    def detect_fit(self, train_data: pd.DataFrame, train_label=None) -> None:
        del train_label
        self._seed_everything()
        train_values, validation_values = self._split_train_validation(train_data)
        self._build_model(train_values.shape[1])
        self.model.set_scale_floor(self._training_scale_floor(train_values))
        train_loader = self._loader(train_values, shuffle=True)
        validation_loader = self._loader(validation_values, shuffle=False, seed_offset=1)

        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        print(
            f"APD-CATCH variant={self.config.variant} device={self.device} "
            f"train_windows={len(train_loader.dataset)} "
            f"validation_windows={len(validation_loader.dataset)} "
            f"parameters={trainable_parameters}",
            flush=True,
        )

        mask_parameters = list(self.model.mask_generator.parameters())
        mask_parameter_ids = {id(parameter) for parameter in mask_parameters}
        main_parameters = [
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in mask_parameter_ids
        ]
        main_optimizer = torch.optim.AdamW(
            main_parameters,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        mask_optimizer = torch.optim.Adam(mask_parameters, lr=self.config.Mlr)

        best_state = None
        best_validation = float("inf")
        stale_epochs = 0
        completed_epochs = 0
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            self.model.train()
            mask_optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            epoch_batches = 0
            for batch_index, (history, target) in enumerate(train_loader):
                history = history.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                main_optimizer.zero_grad(set_to_none=True)
                loss, _ = self._batch_loss(history, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch={epoch}, batch={batch_index}"
                    )
                loss.backward()
                epoch_loss += float(loss.detach().cpu())
                epoch_batches += 1
                torch.nn.utils.clip_grad_norm_(
                    main_parameters, self.config.max_grad_norm
                )
                main_optimizer.step()

                update_mask = (
                    (batch_index + 1) % self.config.mask_update_interval == 0
                    or batch_index + 1 == len(train_loader)
                )
                if update_mask:
                    torch.nn.utils.clip_grad_norm_(
                        mask_parameters, self.config.max_grad_norm
                    )
                    mask_optimizer.step()
                    mask_optimizer.zero_grad(set_to_none=True)

            completed_epochs = epoch + 1
            validation_nll = self._validation_nll(validation_loader)
            improved = validation_nll < best_validation - 1e-6
            print(
                f"epoch={completed_epochs}/{self.config.num_epochs} "
                f"train_loss={epoch_loss / max(epoch_batches, 1):.6f} "
                f"validation_nll={validation_nll:.6f} "
                f"improved={improved} seconds={time.time() - epoch_start:.1f}",
                flush=True,
            )
            if improved:
                best_validation = validation_nll
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        if best_state is None:
            raise RuntimeError("training did not produce a finite validation checkpoint")
        self.model.load_state_dict(best_state)
        validation_scores = self._score_values(validation_values, include_prefix=False)
        self.calibration_threshold = float(
            np.quantile(validation_scores, 1.0 - self.config.calibration_fpr)
        )
        self.fit_summary = FitSummary(
            epochs=completed_epochs,
            best_validation_nll=float(best_validation),
            calibration_threshold=self.calibration_threshold,
            trainable_parameters=trainable_parameters,
        )
        print(
            f"fit_complete epochs={completed_epochs} "
            f"best_validation_nll={best_validation:.6f} "
            f"calibration_threshold={self.calibration_threshold:.6f}",
            flush=True,
        )

    def _aggregate_score(self, channel_score: torch.Tensor) -> torch.Tensor:
        if self.config.score_aggregation == "max":
            return channel_score.max(dim=-1).values
        return channel_score.mean(dim=-1)

    @torch.no_grad()
    def _score_values(
        self, values: np.ndarray, include_prefix: bool, diagnostics: bool = False
    ):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit first.")
        values = np.asarray(values, dtype=np.float32)
        loader = self._loader(values, shuffle=False, seed_offset=2)
        self.model.eval()
        batches = []
        diagnostic_batches = {
            "channel_nll": [],
            "prediction_mean": [],
            "prediction_scale": [],
            "cutoff": [],
        }
        for history, target in loader:
            history = history.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            output = self.model(history)
            channel_score = gaussian_nll(
                target, output["mean"], output["scale"]
            )
            batches.append(self._aggregate_score(channel_score).cpu().numpy())
            if diagnostics:
                diagnostic_batches["channel_nll"].append(channel_score.cpu().numpy())
                diagnostic_batches["prediction_mean"].append(output["mean"].cpu().numpy())
                diagnostic_batches["prediction_scale"].append(output["scale"].cpu().numpy())
                diagnostic_batches["cutoff"].append(output["cutoff"].cpu().numpy())
        scores = np.concatenate(batches).astype(np.float64, copy=False)
        if include_prefix:
            aligned = np.zeros(len(values), dtype=np.float64)
            aligned[self.config.seq_len :] = scores
            scores = aligned
        if not diagnostics:
            return scores
        return scores, {
            name: np.concatenate(parts).astype(np.float32, copy=False)
            for name, parts in diagnostic_batches.items()
        }

    def detect_score(self, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        scores = self._score_values(
            test.to_numpy(dtype=np.float32, copy=True), include_prefix=True
        )
        return scores, scores

    def score_with_diagnostics(self, test: pd.DataFrame):
        """Score a test stream once and retain target-blind explanatory outputs."""
        return self._score_values(
            test.to_numpy(dtype=np.float32, copy=True),
            include_prefix=True,
            diagnostics=True,
        )

    def detect_label(self, test: pd.DataFrame):
        if self.calibration_threshold is None:
            raise ValueError("Model not trained. Call detect_fit first.")
        scores, _ = self.detect_score(test)
        labels = (scores > self.calibration_threshold).astype(np.int64)
        labels[: self.config.seq_len] = 0
        key = f"normal_calibration_fpr_{self.config.calibration_fpr:g}"
        return {key: labels}, scores

    @torch.no_grad()
    def target_blind_invariance(self, history: torch.Tensor) -> Dict[str, float]:
        """Return deterministic repeat differences for all target-independent outputs."""
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit first.")
        self.model.eval()
        history = history.to(self.device)
        first = self.model(history)
        second = self.model(history.clone())
        return {
            name: float((first[name] - second[name]).abs().max().cpu())
            for name in ("mean", "scale", "cutoff", "partition_error")
        }
