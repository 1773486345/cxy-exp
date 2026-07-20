"""Benchmark adapter for Raw-Anchored Multi-Scale Decomposition CATCH."""

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
from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_criterion, frequency_loss
from ts_benchmark.baselines.catch.utils.tools import EarlyStopping, adjust_learning_rate
from ts_benchmark.baselines.ra_msd_catch.models.RAMSDCATCH_model import RAMSDCATCHModel
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split


DEFAULT_RA_MSD_CATCH_HYPER_PARAMS = {
    **DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS,
    "scale_gate_hidden": 16,
}


class RAMSDCATCHConfig:
    def __init__(self, **kwargs) -> None:
        for key, value in DEFAULT_RA_MSD_CATCH_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self) -> int:
        return self.seq_len

    @property
    def learning_rate(self) -> float:
        return self.lr


class RAMSDCATCH:
    """CATCH with bounded decomposition-view residuals in raw token space."""

    def __init__(self, **kwargs) -> None:
        self.config = RAMSDCATCHConfig(**kwargs)
        self.model_name = "RA-MSD-CATCH"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.criterion = nn.MSELoss()
        self.elementwise_criterion = nn.MSELoss(reduction="none")
        self.auxi_loss = frequency_loss(self.config)
        self.seq_len = self.config.seq_len
        self.model = None
        self.last_scores: Dict[str, np.ndarray] = {}
        self.last_diagnostics: Dict[str, float] = {}
        self.trainable_parameters = 0
        self.training_seconds = 0.0

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self) -> str:
        return self.model_name

    def detect_hyper_param_tune(self, train_data: pd.DataFrame) -> None:
        channels = train_data.shape[1]
        self.config.c_in = channels
        self.config.enc_in = channels
        self.config.dec_in = channels
        self.config.c_out = channels
        self.config.label_len = 48
        self.config.task_name = "anomaly_detection"

    def _catch_loss(self, input_batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        outputs = self.model(input_batch)
        reconstruction = self.criterion(outputs["x_hat"], input_batch)
        normalized_target = self.model.raw_encoder.revin_layer(input_batch, "transform")
        auxiliary = self.auxi_loss(outputs["output_complex"], normalized_target)
        loss = reconstruction + self.config.dc_lambda * outputs["raw_dcloss"] + self.config.auxi_lambda * auxiliary
        return loss, {
            "reconstruction": float(reconstruction.detach().cpu()),
            "frequency": float(auxiliary.detach().cpu()),
            "channel": float(outputs["raw_dcloss"].detach().cpu()),
        }

    @torch.no_grad()
    def detect_validate(self, valid_data_loader) -> float:
        self.model.eval()
        losses = []
        for input_batch, _ in valid_data_loader:
            input_batch = input_batch.float().to(self.device)
            outputs = self.model(input_batch)
            loss = self.criterion(outputs["x_hat"], input_batch)
            losses.append(float(loss.detach().cpu()))
        self.model.train()
        return float(np.mean(losses)) if losses else float("inf")

    def _training_step(
        self,
        input_batch: torch.Tensor,
        step: int,
        train_steps: int,
        mask_update_interval: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.optimizer.zero_grad()
        should_step_mask = step % mask_update_interval == 0 or step == train_steps
        if should_step_mask:
            self.optimizerM.step()
            self.optimizerM.zero_grad()

        loss, diagnostics = self._catch_loss(input_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("RA-MSD-CATCH training loss became non-finite")
        loss.backward()
        self.optimizer.step()
        return loss, diagnostics

    def detect_fit(self, train_data: pd.DataFrame, train_label=None) -> None:
        del train_label
        self.detect_hyper_param_tune(train_data)
        self.model = RAMSDCATCHModel(self.config).to(self.device)
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
        self.trainable_parameters = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        print(f"Total trainable parameters: {self.trainable_parameters}")
        print(f"Module parameters: {self.model.module_parameter_counts()}")

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
                input_batch = input_batch.float().to(self.device)
                loss, diagnostics = self._training_step(
                    input_batch,
                    step,
                    train_steps,
                    mask_update_interval,
                )
                losses.append(float(loss.detach().cpu()))
                if step % 100 == 0:
                    print(
                        "\titers: {} epoch: {} | reconstruction: {:.7f} frequency: {:.7f} channel: {:.7f}".format(
                            step,
                            epoch + 1,
                            diagnostics["reconstruction"],
                            diagnostics["frequency"],
                            diagnostics["channel"],
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

    @torch.no_grad()
    def _score_batch(self, input_batch: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_batch)
        frequency = frequency_criterion(self.config)(input_batch, outputs["x_hat"]).mean(dim=-1)
        temporal = self.elementwise_criterion(input_batch, outputs["x_hat"]).mean(dim=-1)
        return temporal + self.config.score_lambda * frequency

    @torch.no_grad()
    def _collect_scores(self, data_loader) -> np.ndarray:
        self.model.eval()
        score_chunks = []
        for input_batch, _ in data_loader:
            scores = self._score_batch(input_batch.float().to(self.device))
            if not torch.isfinite(scores).all():
                raise FloatingPointError("RA-MSD-CATCH total_score contains NaN or Inf")
            score_chunks.append(scores.detach().cpu().numpy().reshape(-1))
        return np.concatenate(score_chunks) if score_chunks else np.empty(0, dtype=np.float32)

    def detect_score(self, test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        test_data = np.ascontiguousarray(self.scaler.transform(test.values), dtype=np.float32)
        test_loader = anomaly_detection_data_provider(
            test_data,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="thre",
        )
        total_score = self._collect_scores(test_loader)
        self.last_scores = {"total_score": total_score}
        self.last_diagnostics = {
            "alpha_trend": float(self.model.alpha_trend.detach().cpu()),
            "alpha_residual": float(self.model.alpha_residual.detach().cpu()),
        }
        return total_score, total_score
