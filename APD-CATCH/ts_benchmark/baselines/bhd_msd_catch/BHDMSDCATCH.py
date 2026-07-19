"""Benchmark adapter for Blockwise-Head Decomposition MSD-CATCH."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import lr_scheduler

from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_criterion, frequency_loss
from ts_benchmark.baselines.catch.utils.tools import EarlyStopping, adjust_learning_rate
from ts_benchmark.baselines.rsa_msd_catch.RSAMSDCATCH import _set_seed
from ts_benchmark.baselines.sdd_msd_catch.SDDMSDCATCH import (
    SDDMSDCATCH,
    SDDMSDCATCHConfig,
)
from ts_benchmark.baselines.bhd_msd_catch.models.BHDMSDCATCH_model import (
    BHDMSDCATCHModel,
    BHDNonFiniteTensorError,
    tensor_diagnostic_stats,
)
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split


class BHDMSDCATCHConfig(SDDMSDCATCHConfig):
    """The BHD configuration is intentionally identical to SDD's configuration."""


class BHDMSDCATCH(SDDMSDCATCH):
    """SDD-compatible detector with branch-specific blockwise reconstruction."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = BHDMSDCATCHConfig(**kwargs)
        self.model_name = "BHD-MSD-CATCH"
        self.auxi_loss = frequency_loss(self.config)
        self._debug_nonfinite = os.environ.get("BHD_MSD_CATCH_DEBUG_NONFINITE") == "1"
        self._runtime_context: Dict[str, Any] = {}
        self._last_loss_snapshot: Dict[str, Dict[str, float]] = {}

    def _loss_with_components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        complex_prediction: torch.Tensor,
        dcloss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        time_loss = self.criterion(prediction, target)
        normalized_target = self.model.shared_encoder.normalize_for_loss(target)
        frequency_loss_value = self.auxi_loss(complex_prediction, normalized_target)
        loss = time_loss + self.config.dc_lambda * dcloss + self.config.auxi_lambda * frequency_loss_value
        return loss, {
            "time_loss": time_loss,
            "frequency_loss": frequency_loss_value,
            "channel_loss": dcloss,
            "loss": loss,
        }

    def _raise_first_nonfinite_loss(
        self, outputs: Dict[str, torch.Tensor], components: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        ordered_tensors = (
            ("trend_dcloss", "trend", outputs["trend_dcloss"]),
            ("residual_dcloss", "residual", outputs["residual_dcloss"]),
            ("trend_reconstruction", "trend", outputs["trend_hat"]),
            ("residual_reconstruction", "residual", outputs["residual_hat"]),
            ("final_reconstruction", "final", outputs["x_hat"]),
            *(
                (f"{branch}_{name}", branch, value)
                for branch, branch_components in components.items()
                for name, value in branch_components.items()
            ),
        )
        for name, branch, value in ordered_tensors:
            if not torch.isfinite(value).all():
                loss_stats = {
                    loss_branch: {
                        component_name: tensor_diagnostic_stats(component_value)
                        for component_name, component_value in loss_components.items()
                    }
                    for loss_branch, loss_components in components.items()
                }
                raise BHDNonFiniteTensorError(
                    {
                        "branch": branch,
                        "tensor_name": name,
                        "tensor_stats": tensor_diagnostic_stats(value),
                        "losses": loss_stats,
                        "context": dict(self._runtime_context),
                    }
                )

    def _loss(self, input_batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        outputs = self.model(input_batch)
        combined_complex = outputs["trend_complex"] + outputs["residual_complex"]
        combined_dcloss = 0.5 * (outputs["trend_dcloss"] + outputs["residual_dcloss"])
        final_loss, final_components = self._loss_with_components(
            outputs["x_hat"], input_batch, combined_complex, combined_dcloss
        )
        trend_loss, trend_components = self._loss_with_components(
            outputs["trend_hat"],
            outputs["trend"],
            outputs["trend_complex"],
            outputs["trend_dcloss"],
        )
        residual_loss, residual_components = self._loss_with_components(
            outputs["residual_hat"],
            outputs["residual"],
            outputs["residual_complex"],
            outputs["residual_dcloss"],
        )
        loss = final_loss + 0.25 * trend_loss + 0.25 * residual_loss
        components = {
            "final": final_components,
            "trend": trend_components,
            "residual": residual_components,
            "total": {"loss": loss},
        }
        self._last_loss_snapshot = {
            branch: {
                name: float(value.detach().cpu()) if torch.isfinite(value).all() else None
                for name, value in branch_components.items()
            }
            for branch, branch_components in components.items()
        }
        if any(
            not torch.isfinite(value).all()
            for branch_components in components.values()
            for value in branch_components.values()
        ):
            self._raise_first_nonfinite_loss(outputs, components)
        return loss, {
            "final": float(final_loss.detach().cpu()),
            "trend": float(trend_loss.detach().cpu()),
            "residual": float(residual_loss.detach().cpu()),
        }

    @staticmethod
    def _git_commit() -> str | None:
        root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    def _save_nonfinite_diagnostic(self, diagnostic: Dict[str, Any]) -> None:
        is_swat = (
            self.config.seq_len == 2048
            and self.config.patch_size == 256
            and self.config.patch_stride == 64
        )
        if not is_swat:
            return
        output_dir = Path("result/score/by_dataset/SWAT/BHDMSDCATCH")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "nonfinite_diagnostic.json"
        if path.exists():
            path = output_dir / f"nonfinite_diagnostic.{time.time_ns()}.json"
        payload = {
            "git_commit": self._git_commit(),
            "dataset": "SWAT",
            "epoch": diagnostic.get("context", {}).get("epoch"),
            "global_step": diagnostic.get("context", {}).get("global_step"),
            "batch_index": diagnostic.get("context", {}).get("batch_index"),
            "branch": diagnostic.get("branch"),
            "tensor_name": diagnostic.get("tensor_name"),
            "tensor_stats": diagnostic.get("tensor_stats"),
            "contrastive_tensors": diagnostic.get("contrastive_tensors"),
            "losses": diagnostic.get("losses", self._last_loss_snapshot),
            "gradient": diagnostic.get("gradient"),
            "optimizer": diagnostic.get("optimizer"),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"BHD-MSD-CATCH non-finite diagnostic saved to {path}")

    def _raise_with_snapshot(self, diagnostic: Dict[str, Any]) -> None:
        self._save_nonfinite_diagnostic(diagnostic)
        raise FloatingPointError(
            "BHD-MSD-CATCH training became non-finite at "
            f"{diagnostic.get('tensor_name')} (branch={diagnostic.get('branch')}, "
            f"epoch={diagnostic.get('context', {}).get('epoch')}, "
            f"step={diagnostic.get('context', {}).get('global_step')})"
        )

    def _gradient_diagnostic(self) -> Dict[str, Any] | None:
        max_finite_gradient = 0.0
        squared_norm = 0.0
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if not torch.isfinite(gradient).all():
                return {
                    "first_nonfinite_parameter": name,
                    "tensor_stats": tensor_diagnostic_stats(gradient),
                    "global_gradient_norm": None,
                    "max_finite_gradient": max_finite_gradient,
                }
            max_finite_gradient = max(max_finite_gradient, float(gradient.abs().max().cpu()))
            squared_norm += float(gradient.square().sum().cpu())
        return {
            "first_nonfinite_parameter": None,
            "global_gradient_norm": squared_norm**0.5,
            "max_finite_gradient": max_finite_gradient,
        }

    def _optimizer_diagnostic(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any] | None:
        parameter_names = {id(parameter): name for name, parameter in self.model.named_parameters()}
        for name, parameter in self.model.named_parameters():
            if not torch.isfinite(parameter).all():
                return {
                    "first_nonfinite_parameter": name,
                    "tensor_stats": tensor_diagnostic_stats(parameter),
                }
        for parameter, state in optimizer.state.items():
            for state_name, state_value in state.items():
                if torch.is_tensor(state_value) and not torch.isfinite(state_value).all():
                    return {
                        "first_nonfinite_optimizer_state": state_name,
                        "parameter": parameter_names.get(id(parameter), "unknown"),
                        "tensor_stats": tensor_diagnostic_stats(state_value),
                    }
        return None

    def detect_fit(self, train_data: pd.DataFrame, train_label=None) -> None:
        del train_label
        self.detect_hyper_param_tune(train_data)
        self.model = BHDMSDCATCHModel(self.config).to(self.device)
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)
        train_data_value = np.ascontiguousarray(
            self.scaler.transform(train_data_value.values), dtype=np.float32
        )
        valid_data = np.ascontiguousarray(
            self.scaler.transform(valid_data.values), dtype=np.float32
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
                self._runtime_context = {
                    "epoch": epoch + 1,
                    "global_step": epoch * train_steps + step,
                    "batch_index": step,
                }
                self.model.set_diagnostic_context(self._runtime_context)
                try:
                    loss, diagnostics = self._loss(input_batch)
                except BHDNonFiniteTensorError as error:
                    self._raise_with_snapshot(error.diagnostic)
                if not torch.isfinite(loss):
                    self._raise_with_snapshot(
                        {
                            "branch": "total",
                            "tensor_name": "total_loss",
                            "tensor_stats": tensor_diagnostic_stats(loss),
                            "losses": self._last_loss_snapshot,
                            "context": dict(self._runtime_context),
                        }
                    )
                loss.backward()
                if self._debug_nonfinite:
                    gradient = self._gradient_diagnostic()
                    if gradient and gradient["first_nonfinite_parameter"] is not None:
                        self._raise_with_snapshot(
                            {
                                "branch": "backward",
                                "tensor_name": "gradient",
                                "tensor_stats": gradient["tensor_stats"],
                                "gradient": gradient,
                                "losses": self._last_loss_snapshot,
                                "context": dict(self._runtime_context),
                            }
                        )
                self.optimizer.step()
                if self._debug_nonfinite:
                    optimizer = self._optimizer_diagnostic(self.optimizer)
                    if optimizer is not None:
                        self._raise_with_snapshot(
                            {
                                "branch": "optimizer",
                                "tensor_name": "optimizer_state",
                                "tensor_stats": optimizer["tensor_stats"],
                                "optimizer": optimizer,
                                "losses": self._last_loss_snapshot,
                                "context": dict(self._runtime_context),
                            }
                        )
                if step % mask_update_interval == 0 or step == train_steps:
                    self.optimizerM.step()
                    self.optimizerM.zero_grad()
                    if self._debug_nonfinite:
                        optimizer = self._optimizer_diagnostic(self.optimizerM)
                        if optimizer is not None:
                            self._raise_with_snapshot(
                                {
                                    "branch": "mask_optimizer",
                                    "tensor_name": "optimizer_state",
                                    "tensor_stats": optimizer["tensor_stats"],
                                    "optimizer": optimizer,
                                    "losses": self._last_loss_snapshot,
                                    "context": dict(self._runtime_context),
                                }
                            )
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
            "trend_hat": outputs["trend_hat"],
            "residual_hat": outputs["residual_hat"],
            "decomp_hat": outputs["decomp_hat"],
            "raw_correction": outputs["raw_correction"],
            "raw_gate": outputs["raw_gate"],
            "scale_entropy": outputs["scale_entropy"].unsqueeze(1).expand(-1, time_steps, -1),
            "scale_weights": outputs["scale_weights"].unsqueeze(1).expand(-1, time_steps, -1, -1),
        }
        return scores, diagnostics

    @torch.no_grad()
    def _collect_scores(self, data_loader) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        self.model.eval()
        score_names = (
            "total_score",
            "decomp_score",
            "trend_score",
            "residual_score",
            "raw_correction_score",
        )
        diagnostic_names = (
            "trend_hat",
            "residual_hat",
            "decomp_hat",
            "raw_correction",
            "raw_gate",
            "scale_entropy",
            "scale_weights",
        )
        score_chunks = {name: [] for name in score_names}
        diagnostic_chunks = {name: [] for name in diagnostic_names}
        for input_batch, _ in data_loader:
            scores, diagnostics = self._score_batch(input_batch.float().to(self.device))
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
            raise FloatingPointError("BHD-MSD-CATCH score diagnostics are inconsistent or non-finite")
        return scores, diagnostics


def run_bhd_msd_catch_screen(
    dataset_name: str, params: Dict, output_dir: str | Path, seed: int = 2021
) -> Dict:
    """Run one fixed-config real-data BHD screen and save standard diagnostics."""
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
    detector = BHDMSDCATCH(**params)
    detector.detect_fit(train)
    score_start = time.time()
    total_score, _ = detector.detect_score(test)
    score_seconds = time.time() - score_start
    scores = detector.last_scores
    diagnostics = detector.last_diagnostics
    scored_length = len(total_score)
    if scored_length > len(labels):
        raise ValueError("BHD-MSD-CATCH produced more scores than test labels")
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
