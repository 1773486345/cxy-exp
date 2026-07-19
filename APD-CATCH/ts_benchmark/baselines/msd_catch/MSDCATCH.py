"""Benchmark adapter for Multi-Scale Decomposition CATCH (MSD-CATCH)."""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler

from ts_benchmark.baselines.catch.CATCH import DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS
from ts_benchmark.baselines.catch.utils.fre_rec_loss import (
    frequency_criterion,
    frequency_loss,
)
from ts_benchmark.baselines.catch.utils.tools import EarlyStopping, adjust_learning_rate
from ts_benchmark.baselines.msd_catch.models.MSDCATCH_model import MSDCATCHModel
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split


DEFAULT_MSD_CATCH_HYPER_PARAMS = {
    **DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS,
    "lambda_trend": 0.5,
    "lambda_residual": 0.5,
    "scale_gate_hidden": 16,
}


class MSDCATCHConfig:
    def __init__(self, **kwargs) -> None:
        for key, value in DEFAULT_MSD_CATCH_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self) -> int:
        return self.seq_len

    @property
    def learning_rate(self) -> float:
        return self.lr


class MSDCATCH:
    """Two independent CATCH branches over an adaptive trend/residual split."""

    def __init__(self, **kwargs) -> None:
        self.config = MSDCATCHConfig(**kwargs)
        self.model_name = "MSD-CATCH"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.criterion = nn.MSELoss()
        self.elementwise_criterion = nn.MSELoss(reduction="none")
        self.auxi_loss = frequency_loss(self.config)
        self.seq_len = self.config.seq_len
        self.model = None
        self.last_scores: Dict[str, np.ndarray] = {}
        self.reference_score_stats: Dict[str, Tuple[float, float]] = {}
        self.reference_delta_thresholds: Dict[str, float] = {}
        self.last_bonus_masks: Dict[str, np.ndarray] = {}
        self.trainable_parameters = 0
        self.training_seconds = 0.0

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self) -> str:
        return self.model_name

    def detect_hyper_param_tune(self, train_data: pd.DataFrame) -> None:
        self.config.c_in = train_data.shape[1]
        self.config.enc_in = train_data.shape[1]
        self.config.dec_in = train_data.shape[1]
        self.config.c_out = train_data.shape[1]
        self.config.label_len = 48
        self.config.task_name = "anomaly_detection"

    def _branch_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        complex_prediction: torch.Tensor,
        dcloss: torch.Tensor,
        branch,
    ) -> torch.Tensor:
        reconstruction = self.criterion(prediction, target)
        normalized_target = branch.revin_layer(target, "transform")
        auxiliary = self.auxi_loss(complex_prediction, normalized_target)
        return reconstruction + self.config.dc_lambda * dcloss + self.config.auxi_lambda * auxiliary

    def _loss(self, input_batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        outputs = self.model(input_batch)
        total_loss = self.criterion(outputs["x_hat"], input_batch)
        trend_loss = self._branch_loss(
            outputs["trend_hat"],
            outputs["trend"],
            outputs["trend_complex"],
            outputs["trend_dcloss"],
            self.model.trend_branch,
        )
        residual_loss = self._branch_loss(
            outputs["residual_hat"],
            outputs["residual"],
            outputs["residual_complex"],
            outputs["residual_dcloss"],
            self.model.residual_branch,
        )
        loss = (
            total_loss
            + self.config.lambda_trend * trend_loss
            + self.config.lambda_residual * residual_loss
        )
        diagnostics = {
            "total": float(total_loss.detach().cpu()),
            "trend": float(trend_loss.detach().cpu()),
            "residual": float(residual_loss.detach().cpu()),
        }
        return loss, diagnostics

    @torch.no_grad()
    def detect_validate(self, valid_data_loader) -> float:
        self.model.eval()
        losses = []
        for input_batch, _ in valid_data_loader:
            input_batch = input_batch.float().to(self.device)
            loss, _ = self._loss(input_batch)
            losses.append(float(loss.detach().cpu()))
        self.model.train()
        return float(np.mean(losses)) if losses else float("inf")

    def detect_fit(self, train_data: pd.DataFrame, train_label=None) -> None:
        del train_label
        self.detect_hyper_param_tune(train_data)
        self.model = MSDCATCHModel(self.config).to(self.device)
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)
        train_data_value = np.ascontiguousarray(
            self.scaler.transform(train_data_value.values), dtype=np.float32
        )
        valid_data = np.ascontiguousarray(
            self.scaler.transform(valid_data.values), dtype=np.float32
        )
        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="train",
        )
        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="val",
        )
        self.reference_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="thre",
        )
        self.trainable_parameters = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        print(f"Total trainable parameters: {self.trainable_parameters}")

        main_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if "mask_generator" not in name
        ]
        mask_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if "mask_generator" in name
        ]
        self.optimizer = torch.optim.Adam(main_parameters, lr=self.config.lr)
        self.optimizerM = torch.optim.Adam(mask_parameters, lr=self.config.Mlr)
        train_steps = len(self.train_data_loader)
        if train_steps < 1:
            raise ValueError("training data is shorter than seq_len")
        scheduler = lr_scheduler.OneCycleLR(
            self.optimizer,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.lr,
        )
        schedulerM = lr_scheduler.OneCycleLR(
            self.optimizerM,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.Mlr,
        )
        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        start = time.time()
        mask_update_interval = max(1, min(train_steps // 10, 100))

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_start = time.time()
            losses = []
            for step, (input_batch, _) in enumerate(self.train_data_loader, start=1):
                self.optimizer.zero_grad()
                input_batch = input_batch.float().to(self.device)
                loss, diagnostics = self._loss(input_batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("MSD-CATCH training loss became non-finite")
                loss.backward()
                self.optimizer.step()
                if step % mask_update_interval == 0 or step == train_steps:
                    self.optimizerM.step()
                    self.optimizerM.zero_grad()
                losses.append(float(loss.detach().cpu()))
                if step % 100 == 0:
                    print(
                        "\titers: {} epoch: {} | total: {:.7f} trend: {:.7f} residual: {:.7f}".format(
                            step,
                            epoch + 1,
                            diagnostics["total"],
                            diagnostics["trend"],
                            diagnostics["residual"],
                        )
                    )
            valid_loss = self.detect_validate(self.valid_data_loader)
            print(
                "Epoch: {} cost time: {:.2f}s | Train Loss: {:.7f} Vali Loss: {:.7f}".format(
                    epoch + 1,
                    time.time() - epoch_start,
                    float(np.mean(losses)),
                    valid_loss,
                )
            )
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, scheduler, epoch + 1, self.config)
            adjust_learning_rate(self.optimizerM, schedulerM, epoch + 1, self.config, printout=False)

        self.training_seconds = time.time() - start
        self.model.load_state_dict(self.early_stopping.check_point)
        reference_raw_scores = self._raw_scores(self.reference_data_loader)
        self.reference_score_stats = self._fit_score_statistics(reference_raw_scores)
        self.reference_delta_thresholds = self._fit_delta_thresholds(reference_raw_scores)

    @torch.no_grad()
    def _score_batch(self, input_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.model(input_batch)
        criterion = frequency_criterion(self.config)

        def catch_score(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
            temporal = self.elementwise_criterion(target, prediction).mean(dim=-1)
            frequency = criterion(target, prediction).mean(dim=-1)
            return temporal + self.config.score_lambda * frequency

        return {
            "total_score": catch_score(input_batch, outputs["x_hat"]),
            "trend_score": catch_score(outputs["trend"], outputs["trend_hat"]),
            "residual_score": catch_score(outputs["residual"], outputs["residual_hat"]),
        }

    @torch.no_grad()
    def _raw_scores(self, data_loader) -> Dict[str, np.ndarray]:
        self.model.eval()
        chunks = {"total_score": [], "trend_score": [], "residual_score": []}
        for input_batch, _ in data_loader:
            input_batch = input_batch.float().to(self.device)
            batch_scores = self._score_batch(input_batch)
            for name, values in batch_scores.items():
                if not torch.isfinite(values).all():
                    raise FloatingPointError(f"{name} contains NaN or Inf")
                chunks[name].append(values.detach().cpu().numpy().reshape(-1))
        return {
            name: np.concatenate(values) if values else np.empty(0, dtype=np.float32)
            for name, values in chunks.items()
        }

    def _fit_score_statistics(
        self, raw_scores: Dict[str, np.ndarray]
    ) -> Dict[str, Tuple[float, float]]:
        statistics = {}
        for name, values in raw_scores.items():
            if not len(values):
                raise ValueError("normal reference data is shorter than seq_len")
            mean = float(np.mean(values))
            std = max(float(np.std(values)), np.finfo(np.float32).eps)
            statistics[name] = (mean, std)
        return statistics

    def _normalized_scores(self, raw_scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.reference_score_stats:
            raise ValueError("score normalization statistics are unavailable; call detect_fit first")
        normalized = {}
        for name in ("total_score", "trend_score", "residual_score"):
            mean, std = self.reference_score_stats[name]
            normalized[name] = (raw_scores[name] - mean) / std
        return normalized

    def _fit_delta_thresholds(self, raw_scores: Dict[str, np.ndarray]) -> Dict[str, float]:
        normalized = self._normalized_scores(raw_scores)
        return {
            "trend_delta_threshold": float(
                np.quantile(normalized["trend_score"] - normalized["total_score"], 0.95)
            ),
            "residual_delta_threshold": float(
                np.quantile(
                    normalized["residual_score"] - normalized["total_score"], 0.95
                )
            ),
        }

    def _fuse_scores(self, raw_scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        normalized = self._normalized_scores(raw_scores)
        if not self.reference_delta_thresholds:
            raise ValueError("branch delta thresholds are unavailable; call detect_fit first")
        total_z = normalized["total_score"]
        trend_z = normalized["trend_score"]
        residual_z = normalized["residual_score"]
        fixed_fusion = (total_z + trend_z + residual_z) / 3.0
        trend_bonus = np.maximum(
            trend_z - total_z - self.reference_delta_thresholds["trend_delta_threshold"],
            0.0,
        )
        residual_bonus = np.maximum(
            residual_z
            - total_z
            - self.reference_delta_thresholds["residual_delta_threshold"],
            0.0,
        )
        anchored_fusion = total_z + 0.25 * trend_bonus + 0.25 * residual_bonus
        self.last_bonus_masks = {
            "trend_bonus_nonzero": trend_bonus > 0.0,
            "residual_bonus_nonzero": residual_bonus > 0.0,
        }
        result = {
            **raw_scores,
            "fixed_fusion_score": fixed_fusion,
            "anchored_fusion_score": anchored_fusion,
            # Compatibility alias for callers of the initial fixed-fusion adapter.
            "fusion_score": fixed_fusion,
        }
        lengths = {len(values) for values in result.values()}
        if len(lengths) != 1 or not np.isfinite(anchored_fusion).all():
            raise FloatingPointError("MSD-CATCH score vectors are inconsistent or non-finite")
        return result

    def detect_score(self, test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        test = np.ascontiguousarray(
            self.scaler.transform(test.values), dtype=np.float32
        )
        test_loader = anomaly_detection_data_provider(
            test,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="thre",
        )
        self.last_scores = self._fuse_scores(self._raw_scores(test_loader))
        # The frozen MSD-CATCH evaluation uses the reconstructed whole-series
        # score as its only primary anomaly score. Other scores are diagnostics.
        return self.last_scores["total_score"], self.last_scores["total_score"]

    def detect_label(self, test: pd.DataFrame):
        test_total_score, _ = self.detect_score(test)
        reference_total_score = self._raw_scores(self.reference_data_loader)["total_score"]
        ratios = self.config.anomaly_ratio
        if not isinstance(ratios, list):
            ratios = [ratios]
        predictions = {
            ratio: (test_total_score > np.percentile(reference_total_score, 100 - ratio)).astype(int)
            for ratio in ratios
        }
        return predictions, test_total_score
