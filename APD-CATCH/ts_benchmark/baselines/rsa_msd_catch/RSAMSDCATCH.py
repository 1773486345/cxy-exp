"""Benchmark adapter for RSA-MSD-CATCH."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler

from ts_benchmark.baselines.catch.CATCH import DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS
from ts_benchmark.baselines.catch.utils.fre_rec_loss import (
    frequency_criterion,
    frequency_loss,
)
from ts_benchmark.baselines.catch.utils.tools import EarlyStopping, adjust_learning_rate
from ts_benchmark.baselines.rsa_msd_catch.models.RSAMSDCATCH_model import RSAMSDCATCHModel
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split


DEFAULT_RSA_MSD_CATCH_HYPER_PARAMS = {
    **DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS,
    "scale_gate_hidden": 16,
}


class RSAMSDCATCHConfig:
    def __init__(self, **kwargs) -> None:
        for key, value in DEFAULT_RSA_MSD_CATCH_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self) -> int:
        return self.seq_len

    @property
    def learning_rate(self) -> float:
        return self.lr


class RSAMSDCATCH:
    """Shared-backbone, raw-preserving multi-scale CATCH detector."""

    def __init__(self, **kwargs) -> None:
        self.config = RSAMSDCATCHConfig(**kwargs)
        self.model_name = "RSA-MSD-CATCH"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.criterion = nn.MSELoss()
        self.elementwise_criterion = nn.MSELoss(reduction="none")
        self.auxi_loss = frequency_loss(self.config)
        self.seq_len = self.config.seq_len
        self.model = None
        self.last_scores: Dict[str, np.ndarray] = {}
        self.last_diagnostics: Dict[str, np.ndarray] = {}
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

    def _catch_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        complex_prediction: torch.Tensor,
        dcloss: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction = self.criterion(prediction, target)
        normalized_target = self.model.shared_catch_backbone.normalize_for_loss(target)
        auxiliary = self.auxi_loss(complex_prediction, normalized_target)
        return reconstruction + self.config.dc_lambda * dcloss + self.config.auxi_lambda * auxiliary

    def _loss(self, input_batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        outputs = self.model(input_batch)
        combined_complex = outputs["trend_complex"] + outputs["residual_complex"]
        combined_dcloss = 0.5 * (outputs["trend_dcloss"] + outputs["residual_dcloss"])
        final_loss = self._catch_loss(
            outputs["x_hat"], input_batch, combined_complex, combined_dcloss
        )
        trend_loss = self._catch_loss(
            outputs["trend_hat"],
            outputs["trend"],
            outputs["trend_complex"],
            outputs["trend_dcloss"],
        )
        residual_loss = self._catch_loss(
            outputs["residual_hat"],
            outputs["residual"],
            outputs["residual_complex"],
            outputs["residual_dcloss"],
        )
        loss = final_loss + 0.25 * trend_loss + 0.25 * residual_loss
        return loss, {
            "final": float(final_loss.detach().cpu()),
            "trend": float(trend_loss.detach().cpu()),
            "residual": float(residual_loss.detach().cpu()),
        }

    @torch.no_grad()
    def detect_validate(self, valid_data_loader) -> float:
        self.model.eval()
        losses = []
        for input_batch, _ in valid_data_loader:
            loss, _ = self._loss(input_batch.float().to(self.device))
            losses.append(float(loss.detach().cpu()))
        self.model.train()
        return float(np.mean(losses)) if losses else float("inf")

    def detect_fit(self, train_data: pd.DataFrame, train_label=None) -> None:
        del train_label
        self.detect_hyper_param_tune(train_data)
        self.model = RSAMSDCATCHModel(self.config).to(self.device)
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)
        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )
        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
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
                self.optimizer.zero_grad()
                input_batch = input_batch.float().to(self.device)
                loss, diagnostics = self._loss(input_batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("RSA-MSD-CATCH training loss became non-finite")
                loss.backward()
                self.optimizer.step()
                if step % mask_update_interval == 0 or step == train_steps:
                    self.optimizerM.step()
                    self.optimizerM.zero_grad()
                losses.append(float(loss.detach().cpu()))
                if step % 100 == 0:
                    print(
                        "\titers: {} epoch: {} | final: {:.7f} trend: {:.7f} residual: {:.7f}".format(
                            step,
                            epoch + 1,
                            diagnostics["final"],
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

    @torch.no_grad()
    def _score_batch(
        self, input_batch: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        outputs = self.model(input_batch)
        criterion = frequency_criterion(self.config)

        def catch_score(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
            temporal = self.elementwise_criterion(target, prediction).mean(dim=-1)
            frequency = criterion(target, prediction).mean(dim=-1)
            return temporal + self.config.score_lambda * frequency

        scores = {
            "total_score": catch_score(input_batch, outputs["x_hat"]),
            "decomp_score": catch_score(input_batch, outputs["decomp_hat"]),
            "trend_score": catch_score(outputs["trend"], outputs["trend_hat"]),
            "residual_score": catch_score(outputs["residual"], outputs["residual_hat"]),
            "raw_correction_score": (outputs["raw_gate"] * outputs["raw_correction"])
            .square()
            .mean(dim=-1),
        }
        time_steps = input_batch.shape[1]
        diagnostics = {
            "raw_gate": outputs["raw_gate"],
            "scale_entropy": outputs["scale_entropy"].unsqueeze(1).expand(-1, time_steps, -1),
            "scale_weights": outputs["scale_weights"].unsqueeze(1).expand(-1, time_steps, -1, -1),
        }
        return scores, diagnostics

    @torch.no_grad()
    def _collect_scores(self, data_loader) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        self.model.eval()
        score_chunks = {
            name: []
            for name in (
                "total_score",
                "decomp_score",
                "trend_score",
                "residual_score",
                "raw_correction_score",
            )
        }
        diagnostic_chunks = {"raw_gate": [], "scale_entropy": [], "scale_weights": []}
        for input_batch, _ in data_loader:
            input_batch = input_batch.float().to(self.device)
            scores, diagnostics = self._score_batch(input_batch)
            for name, values in scores.items():
                if not torch.isfinite(values).all():
                    raise FloatingPointError(f"{name} contains NaN or Inf")
                score_chunks[name].append(values.detach().cpu().numpy().reshape(-1))
            for name, values in diagnostics.items():
                if not torch.isfinite(values).all():
                    raise FloatingPointError(f"{name} contains NaN or Inf")
                if name == "scale_weights":
                    diagnostic_chunks[name].append(
                        values.detach().cpu().numpy().reshape(-1, values.shape[-2], values.shape[-1])
                    )
                else:
                    diagnostic_chunks[name].append(
                        values.detach().cpu().numpy().reshape(-1, values.shape[-1])
                    )
        scores = {
            name: np.concatenate(values) if values else np.empty(0, dtype=np.float32)
            for name, values in score_chunks.items()
        }
        diagnostics = {
            name: np.concatenate(values) if values else np.empty(0, dtype=np.float32)
            for name, values in diagnostic_chunks.items()
        }
        lengths = {len(values) for values in scores.values()}
        if len(lengths) != 1 or not all(np.isfinite(values).all() for values in diagnostics.values()):
            raise FloatingPointError("RSA-MSD-CATCH score diagnostics are inconsistent or non-finite")
        return scores, diagnostics

    def detect_score(self, test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        test_loader = anomaly_detection_data_provider(
            test,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="thre",
        )
        self.last_scores, self.last_diagnostics = self._collect_scores(test_loader)
        return self.last_scores["total_score"], self.last_scores["total_score"]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_rsa_msd_catch_screen(
    dataset_name: str,
    params: Dict,
    output_dir: str | Path,
    seed: int = 2021,
) -> Dict:
    """Run one fixed-config real-data screen and persist model and diagnostics."""
    from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource

    _set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source = LocalAnomalyDetectDataSource()
    series_name = f"{dataset_name}.csv"
    source.load_series_list([series_name])
    data = source.dataset.get_series(series_name).reset_index(drop=True)
    train_length = int(source.dataset.get_series_meta_info(series_name)["train_lens"].item())
    features = data.columns != "label"
    train = data.iloc[:train_length].loc[:, features]
    test = data.iloc[train_length:].loc[:, features]
    labels = data.iloc[train_length:]["label"].to_numpy(dtype=int)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    detector = RSAMSDCATCH(**params)
    detector.detect_fit(train)
    score_start = time.time()
    total_score, _ = detector.detect_score(test)
    score_seconds = time.time() - score_start
    scores = detector.last_scores
    diagnostics = detector.last_diagnostics
    scored_length = len(total_score)
    if scored_length > len(labels):
        raise ValueError("RSA-MSD-CATCH produced more scores than test labels")
    # Match UnFixedDetectScore: non-overlapping threshold windows leave a short
    # tail which the benchmark pads with zero anomaly score before evaluation.
    if scored_length < len(labels):
        scores = {
            name: np.pad(values, (0, len(labels) - scored_length), constant_values=0.0)
            for name, values in scores.items()
        }
    metrics = {
        name: {
            "auc_pr": float(average_precision_score(labels, values)),
            "auc_roc": float(roc_auc_score(labels, values)),
        }
        for name, values in scores.items()
    }
    raw_gate = diagnostics["raw_gate"]
    gate_labels = labels[:scored_length]
    normal = gate_labels == 0
    anomaly = gate_labels == 1
    raw_gate_summary = {
        "mean": float(raw_gate.mean()),
        "std": float(raw_gate.std()),
        "normal_mean": float(raw_gate[normal].mean()) if normal.any() else 0.0,
        "anomaly_mean": float(raw_gate[anomaly].mean()) if anomaly.any() else 0.0,
        "per_variable_mean": raw_gate.mean(axis=0).tolist(),
    }
    scale_entropy = diagnostics["scale_entropy"]
    parameter_counts = detector.model.module_parameter_counts()
    result = {
        "dataset": dataset_name,
        "seed": seed,
        "primary_score": "total_score",
        "parameters": detector.trainable_parameters,
        "module_parameters": parameter_counts,
        "fit_seconds": detector.training_seconds,
        "score_seconds": score_seconds,
        "scored_length_before_padding": scored_length,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "data": {
            "variables": int(train.shape[1]),
            "train_length": int(len(train)),
            "test_length": int(len(test)),
            "anomaly_rate": float(labels.mean()),
            "anomaly_segments": int(np.sum(np.diff(np.r_[0, labels, 0]) == 1)),
        },
        "scores": metrics,
        "raw_gate": raw_gate_summary,
        "scale_entropy": {
            "mean": float(scale_entropy.mean()),
            "per_variable_mean": scale_entropy.mean(axis=0).tolist(),
        },
        "config": params,
    }
    torch.save(
        {
            "model_state": detector.model.state_dict(),
            "config": dict(detector.config.__dict__),
            "scaler_mean": detector.scaler.mean_,
            "scaler_scale": detector.scaler.scale_,
            "module_parameters": parameter_counts,
            "seed": seed,
        },
        output_path / f"{dataset_name}.pt",
    )
    np.savez_compressed(
        output_path / f"{dataset_name}_scores.npz",
        labels=labels,
        **scores,
        **diagnostics,
    )
    with (output_path / f"{dataset_name}.json").open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result
