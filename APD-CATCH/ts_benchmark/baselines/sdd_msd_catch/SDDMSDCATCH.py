"""Benchmark adapter for Shared-Encoder Dual-Decoder MSD-CATCH."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import lr_scheduler

from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_loss
from ts_benchmark.baselines.catch.utils.tools import EarlyStopping, adjust_learning_rate
from ts_benchmark.baselines.rsa_msd_catch.RSAMSDCATCH import (
    DEFAULT_RSA_MSD_CATCH_HYPER_PARAMS,
    RSAMSDCATCH,
    RSAMSDCATCHConfig,
    _set_seed,
)
from ts_benchmark.baselines.sdd_msd_catch.models.SDDMSDCATCH_model import SDDMSDCATCHModel
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split


class SDDMSDCATCHConfig(RSAMSDCATCHConfig):
    def __init__(self, **kwargs) -> None:
        for key, value in DEFAULT_RSA_MSD_CATCH_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class SDDMSDCATCH(RSAMSDCATCH):
    """RSA-compatible adapter with a shared encoder and dual low-rank decoders."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = SDDMSDCATCHConfig(**kwargs)
        self.model_name = "SDD-MSD-CATCH"
        self.auxi_loss = frequency_loss(self.config)

    def _catch_loss(self, prediction, target, complex_prediction, dcloss):
        reconstruction = self.criterion(prediction, target)
        normalized_target = self.model.shared_encoder.normalize_for_loss(target)
        auxiliary = self.auxi_loss(complex_prediction, normalized_target)
        return reconstruction + self.config.dc_lambda * dcloss + self.config.auxi_lambda * auxiliary

    def detect_fit(self, train_data: pd.DataFrame, train_label=None) -> None:
        del train_label
        self.detect_hyper_param_tune(train_data)
        self.model = SDDMSDCATCHModel(self.config).to(self.device)
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
            train_data_value, self.config.batch_size, self.config.seq_len, 1, "train"
        )
        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data, self.config.batch_size, self.config.seq_len, 1, "val"
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
                    raise FloatingPointError("SDD-MSD-CATCH training loss became non-finite")
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
                    epoch + 1, time.time() - epoch_start, float(np.mean(losses)), valid_loss
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


def run_sdd_msd_catch_screen(
    dataset_name: str, params: Dict, output_dir: str | Path, seed: int = 2021
) -> Dict:
    """Run one fixed-config real-data SDD screen and save the standard diagnostics."""
    from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource

    _set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source = LocalAnomalyDetectDataSource()
    series_name = dataset_name + ".csv"
    source.load_series_list([series_name])
    data = source.dataset.get_series(series_name).reset_index(drop=True)
    train_length = int(source.dataset.get_series_meta_info(series_name)["train_lens"].item())
    features = data.columns != "label"
    train = data.iloc[:train_length].loc[:, features]
    test = data.iloc[train_length:].loc[:, features]
    labels = data.iloc[train_length:]["label"].to_numpy(dtype=int)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    detector = SDDMSDCATCH(**params)
    detector.detect_fit(train)
    score_start = time.time()
    total_score, _ = detector.detect_score(test)
    score_seconds = time.time() - score_start
    scores = detector.last_scores
    diagnostics = detector.last_diagnostics
    scored_length = len(total_score)
    if scored_length > len(labels):
        raise ValueError("SDD-MSD-CATCH produced more scores than test labels")
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
        "raw_gate": {
            "mean": float(raw_gate.mean()),
            "std": float(raw_gate.std()),
            "normal_mean": float(raw_gate[normal].mean()) if normal.any() else 0.0,
            "anomaly_mean": float(raw_gate[anomaly].mean()) if anomaly.any() else 0.0,
            "per_variable_mean": raw_gate.mean(axis=0).tolist(),
        },
        "scale_entropy": {
            "mean": float(diagnostics["scale_entropy"].mean()),
            "per_variable_mean": diagnostics["scale_entropy"].mean(axis=0).tolist(),
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
        output_path / (dataset_name + ".pt"),
    )
    np.savez_compressed(
        output_path / (dataset_name + "_scores.npz"),
        labels=labels,
        **scores,
        **diagnostics,
    )
    with (output_path / (dataset_name + ".json")).open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result
