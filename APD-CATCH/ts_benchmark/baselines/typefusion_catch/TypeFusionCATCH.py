"""Benchmark adapter for TypeFusion-CATCH v1."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.typefusion_catch.config import (
    DEFAULT_TYPEFUSION_HYPER_PARAMS,
    TypeFusionConfig,
)
from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split


_STAGE_ORDER = ("branch_pretrain", "fusion_train", "joint_finetune")
LOGGER = logging.getLogger(__name__)


class TypeFusionCATCH:
    """CATCH-compatible anomaly-score adapter without score calibration."""

    def __init__(self, **kwargs) -> None:
        self.config = TypeFusionConfig.from_kwargs(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TypeFusionCATCHModel] = None
        self.best_state = None
        self.stage_best_states: Dict[str, Dict[str, torch.Tensor]] = {}
        self.stage_start_states: Dict[str, Dict[str, torch.Tensor]] = {}
        self.stage_validation_losses: Dict[str, float] = {}
        self.stage_optimizer_lrs: Dict[str, float] = {}
        self.stage_optimizer_steps: Dict[str, int] = {}
        self.training_budget_summary: Dict[str, object] = {}
        self._scaler_fitted = False
        self.model_name = "TypeFusion-CATCH"

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self) -> str:
        return self.model_name

    def detect_hyper_param_tune(self, train_data: pd.DataFrame) -> None:
        self.config.c_in = int(train_data.shape[1])
        self.config.validate()

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _scaled_frame(scaler: StandardScaler, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            scaler.transform(frame.values), columns=frame.columns, index=frame.index
        )

    @staticmethod
    def _clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(dict(state_dict))

    def load_stage_checkpoint(
        self,
        state_dict: Mapping[str, torch.Tensor],
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        """Provide a complete prior-stage checkpoint for debug single-stage runs."""

        self.best_state = self._clone_state_dict(state_dict)
        if scaler is not None:
            if not hasattr(scaler, "mean_"):
                raise ValueError("The supplied scaler must already be fitted")
            self.scaler = copy.deepcopy(scaler)
            self._scaler_fitted = True

    def _prepare_loaders(self, train_data: pd.DataFrame, fit_scaler: bool):
        train_frame, valid_frame = train_val_split(train_data, 0.8, None)
        if fit_scaler:
            self.scaler.fit(train_frame.values)
            self._scaler_fitted = True
        elif not self._scaler_fitted:
            raise ValueError(
                "single_stage fusion_train/joint_finetune requires the prior fitted scaler"
            )
        train_frame = self._scaled_frame(self.scaler, train_frame)
        valid_frame = self._scaled_frame(self.scaler, valid_frame)
        train_loader = anomaly_detection_data_provider(
            train_frame, self.config.batch_size, self.config.seq_len, mode="train"
        )
        valid_loader = anomaly_detection_data_provider(
            valid_frame, self.config.batch_size, self.config.seq_len, mode="val"
        )
        return train_loader, valid_loader

    @staticmethod
    def _profile_timing_enabled() -> bool:
        return os.environ.get("TYPEFUSION_PROFILE_TIMING") == "1"

    def _synchronize_for_profile(self) -> None:
        if self._profile_timing_enabled() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _cuda_memory_mib(self) -> Tuple[Optional[float], Optional[float]]:
        if self.device.type != "cuda":
            return None, None
        return (
            torch.cuda.memory_allocated(self.device) / (1024 * 1024),
            torch.cuda.memory_reserved(self.device) / (1024 * 1024),
        )

    def _validation_loss(
        self,
        loader,
        compute_joint: bool,
    ) -> float:
        assert self.model is not None
        training_stage = getattr(self, "_active_training_stage", "unknown")
        validation_started = time.perf_counter()
        if self._profile_timing_enabled():
            LOGGER.info(
                "TypeFusion-CATCH validation start: stage=%s batches=%d",
                training_stage,
                len(loader),
            )
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch, _ in loader:
                output = self.model(batch.float().to(self.device), compute_joint=compute_joint)
                losses.append(float(output["losses"]["total"].detach().cpu()))
        self.model.train()
        validation_loss = float(np.mean(losses)) if losses else float("inf")
        if self._profile_timing_enabled():
            LOGGER.info(
                "TypeFusion-CATCH validation end: stage=%s batches=%d seconds=%.3f loss=%.8f",
                training_stage,
                len(loader),
                time.perf_counter() - validation_started,
                validation_loss,
            )
        return validation_loss

    @staticmethod
    def _print_interval_losses(
        training_stage: str,
        optimizer_steps: int,
        losses: Mapping[str, torch.Tensor],
    ) -> None:
        """Print only the losses belonging to the current training objective."""

        def value(name: str) -> float:
            return float(losses[name].detach().cpu())

        print(
            f"    iters: {optimizer_steps}, stage: {training_stage} | "
            f"total loss: {value('total'):.7f}",
            flush=True,
        )
        if training_stage == "branch_pretrain":
            print(
                f"    state loss: {value('state'):.7f} | "
                f"evolution loss: {value('evolution'):.7f}",
                flush=True,
            )
            print(
                f"    pattern time loss: {value('pattern_time'):.7f} | "
                f"pattern freq loss: {value('pattern_freq'):.7f}",
                flush=True,
            )
            print(f"    relation loss: {value('relation'):.7f}", flush=True)
            return
        if training_stage == "fusion_train":
            print(
                f"    joint loss: {value('joint'):.7f} | "
                f"branch mask loss: {value('branch_mask'):.7f}",
                flush=True,
            )
            return
        print(
            f"    state loss: {value('state'):.7f} | "
            f"evolution loss: {value('evolution'):.7f}",
            flush=True,
        )
        print(
            f"    pattern time loss: {value('pattern_time'):.7f} | "
            f"pattern freq loss: {value('pattern_freq'):.7f}",
            flush=True,
        )
        print(
            f"    relation loss: {value('relation'):.7f} | "
            f"joint loss: {value('joint'):.7f} | "
            f"branch mask loss: {value('branch_mask'):.7f}",
            flush=True,
        )

    def _stage_learning_rate(self, training_stage: str) -> float:
        if training_stage == "joint_finetune":
            return self.config.lr * self.config.joint_finetune_lr_scale
        return self.config.lr

    def _build_training_budget(self, train_loader) -> Dict[str, Optional[int]]:
        reference_total_steps = self.config.catch_train_epochs * len(train_loader)
        if self.config.training_budget_mode == "equal_total_steps":
            if reference_total_steps < len(_STAGE_ORDER):
                raise ValueError(
                    "equal_total_steps requires at least 3 reference optimizer steps "
                    "so every stage receives one step"
                )
            branch_steps = reference_total_steps // 3
            fusion_steps = reference_total_steps // 3
            joint_steps = reference_total_steps - branch_steps - fusion_steps
            allocation: Dict[str, Optional[int]] = {
                "branch_pretrain": branch_steps,
                "fusion_train": fusion_steps,
                "joint_finetune": joint_steps,
            }
        else:
            allocation = {stage: None for stage in _STAGE_ORDER}
            branch_steps = self.config.branch_pretrain_epochs * len(train_loader)
            fusion_steps = self.config.fusion_train_epochs * len(train_loader)
            joint_steps = self.config.joint_finetune_epochs * len(train_loader)

        self.training_budget_summary = {
            "mode": self.config.training_budget_mode,
            "reference_catch_epochs": self.config.catch_train_epochs,
            "reference_total_steps": reference_total_steps,
            "branch_pretrain_steps": branch_steps,
            "fusion_train_steps": fusion_steps,
            "joint_finetune_steps": joint_steps,
            "actual_branch_pretrain_steps": 0,
            "actual_fusion_train_steps": 0,
            "actual_joint_finetune_steps": 0,
            "actual_total_steps": 0,
        }
        return allocation

    def _refresh_actual_budget_summary(self) -> None:
        actual_total = 0
        for stage in _STAGE_ORDER:
            actual = self.stage_optimizer_steps.get(stage, 0)
            self.training_budget_summary[f"actual_{stage}_steps"] = actual
            actual_total += actual
        self.training_budget_summary["actual_total_steps"] = actual_total

    def _write_result_audit_metadata(self) -> None:
        """Persist existing stage accounting only when a task script opts in."""

        config_path = os.environ.get("TYPEFUSION_RUN_CONFIG_PATH")
        if not config_path:
            return
        path = Path(config_path)
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["training_budget_summary"] = self.training_budget_summary
            payload["stage_optimizer_steps"] = self.stage_optimizer_steps
            payload["stage_validation_losses"] = self.stage_validation_losses
            payload["completed_stages"] = [
                stage for stage in _STAGE_ORDER if stage in self.stage_best_states
            ]
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # Result accounting must never alter model fitting or evaluation.
            return

    def _train_stage(
        self,
        training_stage: str,
        train_loader,
        valid_loader,
        max_optimizer_steps: Optional[int] = None,
    ) -> None:
        assert self.model is not None
        if len(train_loader) == 0:
            raise ValueError("Training loader has no windows for the configured seq_len")
        self.model.set_training_stage(training_stage)
        self.stage_start_states[training_stage] = self._clone_state_dict(self.model.state_dict())
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError(f"{training_stage} has no trainable parameters")
        stage_lr = self._stage_learning_rate(training_stage)
        optimizer = torch.optim.Adam(parameters, lr=stage_lr)
        self.stage_optimizer_lrs[training_stage] = stage_lr
        compute_joint = training_stage != "branch_pretrain"
        planned_steps = (
            max_optimizer_steps
            if max_optimizer_steps is not None
            else self.config.epochs_for_stage(training_stage) * len(train_loader)
        )
        log_interval = min(100, max(1, planned_steps // 10))
        stage_started = time.perf_counter()
        self._active_training_stage = training_stage
        print(
            f">>>>>>> Stage: {training_stage} | Planned Steps: {planned_steps} | LR: {stage_lr:.10g}",
            flush=True,
        )
        stage_best = self._clone_state_dict(self.model.state_dict())
        best_loss = float("inf")
        stale_epochs = 0
        optimizer_steps = 0
        completed_epochs = 0
        interval_started = time.perf_counter()
        interval_steps = 0
        while max_optimizer_steps is None or optimizer_steps < max_optimizer_steps:
            if max_optimizer_steps is None and completed_epochs >= self.config.epochs_for_stage(training_stage):
                break
            completed_epochs += 1
            self.model.train()
            epoch_started = time.perf_counter()
            epoch_loss_total = 0.0
            epoch_steps = 0
            print(f">>>>>>> Stage: {training_stage} | Epoch: {completed_epochs}", flush=True)
            iterator = iter(train_loader)
            while True:
                data_wait_started = time.perf_counter()
                try:
                    batch, _ = next(iterator)
                except StopIteration:
                    break
                data_wait_seconds = time.perf_counter() - data_wait_started
                optimizer.zero_grad(set_to_none=True)
                self._synchronize_for_profile()
                transfer_started = time.perf_counter()
                batch = batch.float().to(self.device)
                self._synchronize_for_profile()
                host_to_device_seconds = time.perf_counter() - transfer_started
                self._synchronize_for_profile()
                forward_started = time.perf_counter()
                output = self.model(batch, compute_joint=compute_joint)
                self._synchronize_for_profile()
                forward_seconds = time.perf_counter() - forward_started
                loss = output["losses"]["total"]
                self._synchronize_for_profile()
                backward_started = time.perf_counter()
                loss.backward()
                self._synchronize_for_profile()
                backward_seconds = time.perf_counter() - backward_started
                self._synchronize_for_profile()
                optimizer_started = time.perf_counter()
                optimizer.step()
                self._synchronize_for_profile()
                optimizer_seconds = time.perf_counter() - optimizer_started
                optimizer_steps += 1
                interval_steps += 1
                epoch_steps += 1
                epoch_loss_total += float(loss.detach().cpu())
                if optimizer_steps == 1:
                    if self._profile_timing_enabled():
                        allocated_mib, reserved_mib = self._cuda_memory_mib()
                        LOGGER.info(
                            "TypeFusion-CATCH first batch: stage=%s data_wait_seconds=%.6f "
                            "host_to_device_seconds=%.6f forward_seconds=%.6f backward_seconds=%.6f "
                            "optimizer_seconds=%.6f total_loss=%.8f batch_shape=%s "
                            "cuda_allocated_mib=%s cuda_reserved_mib=%s",
                            training_stage,
                            data_wait_seconds,
                            host_to_device_seconds,
                            forward_seconds,
                            backward_seconds,
                            optimizer_seconds,
                            float(loss.detach().cpu()),
                            tuple(batch.shape),
                            "n/a" if allocated_mib is None else f"{allocated_mib:.2f}",
                            "n/a" if reserved_mib is None else f"{reserved_mib:.2f}",
                        )
                    else:
                        print(
                            f"First batch: stage={training_stage} shape={tuple(batch.shape)} "
                            f"loss={float(loss.detach().cpu()):.7f}",
                            flush=True,
                        )
                if optimizer_steps % log_interval == 0 or optimizer_steps == planned_steps:
                    average_step_seconds = (time.perf_counter() - interval_started) / interval_steps
                    remaining_steps = max(0, planned_steps - optimizer_steps)
                    self._print_interval_losses(
                        training_stage,
                        optimizer_steps,
                        output["losses"],
                    )
                    print(
                        f"    speed: {average_step_seconds:.4f}s/iter; "
                        f"left time: {remaining_steps * average_step_seconds:.4f}s",
                        flush=True,
                    )
                    interval_started = time.perf_counter()
                    interval_steps = 0
                if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
                    break
            train_loss = epoch_loss_total / epoch_steps if epoch_steps else float("inf")
            print(
                f"Epoch: {completed_epochs} cost time: {time.perf_counter() - epoch_started:.4f}s",
                flush=True,
            )
            validation_loss = self._validation_loss(valid_loader, compute_joint=compute_joint)
            print(
                f"Stage: {training_stage}, Epoch: {completed_epochs}, Steps: {optimizer_steps} | "
                f"Train Loss: {train_loss:.7f} Vali Loss: {validation_loss:.7f}",
                flush=True,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                stale_epochs = 0
                stage_best = self._clone_state_dict(self.model.state_dict())
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break
        self.model.load_state_dict(stage_best)
        self.stage_best_states[training_stage] = self._clone_state_dict(stage_best)
        self.stage_validation_losses[training_stage] = best_loss
        self.stage_optimizer_steps[training_stage] = optimizer_steps
        print(
            f">>>>>>> Stage {training_stage} finished\n"
            f"Steps: {optimizer_steps}/{planned_steps}\n"
            f"Best Vali Loss: {best_loss:.7f}\n"
            f"Cost Time: {time.perf_counter() - stage_started:.4f}s",
            flush=True,
        )

    def detect_fit(
        self,
        train_data: pd.DataFrame,
        test_data: Optional[pd.DataFrame] = None,
        previous_checkpoint: Optional[Mapping[str, torch.Tensor]] = None,
        previous_scaler: Optional[StandardScaler] = None,
    ) -> None:
        """Fit the default complete stage chain or one explicitly resumed stage.

        ``three_stage`` builds one model instance and runs branch pretraining,
        fusion training and joint fine-tuning consecutively.  ``single_stage``
        is a debugging mode: Fusion/Finetune require a prior full checkpoint
        and its fitted scaler, so frozen random branches are never permitted.
        """

        self._seed_everything(self.config.seed)
        self.detect_hyper_param_tune(train_data)
        LOGGER.info(
            "TypeFusion-CATCH detect_fit start: device=%s train_rows=%d channel_count=%d "
            "seq_len=%d batch_size=%d",
            self.device,
            len(train_data),
            train_data.shape[1],
            self.config.seq_len,
            self.config.batch_size,
        )
        if previous_scaler is not None:
            if not hasattr(previous_scaler, "mean_"):
                raise ValueError("previous_scaler must already be fitted")
            self.scaler = copy.deepcopy(previous_scaler)
            self._scaler_fitted = True

        if self.config.fit_mode == "three_stage":
            self.scaler = StandardScaler()
            self._scaler_fitted = False
            self.stage_best_states = {}
            self.stage_start_states = {}
            self.stage_validation_losses = {}
            self.stage_optimizer_lrs = {}
            self.stage_optimizer_steps = {}
            self.model = TypeFusionCATCHModel(self.config).to(self.device)
            loader_started = time.perf_counter()
            train_loader, valid_loader = self._prepare_loaders(train_data, fit_scaler=True)
            LOGGER.info(
                "TypeFusion-CATCH loaders ready: train_batches=%d validation_batches=%d "
                "loader_prepare_seconds=%.3f",
                len(train_loader),
                len(valid_loader),
                time.perf_counter() - loader_started,
            )
            stage_budget = self._build_training_budget(train_loader)
            for training_stage in _STAGE_ORDER:
                self._train_stage(
                    training_stage,
                    train_loader,
                    valid_loader,
                    max_optimizer_steps=stage_budget[training_stage],
                )
            self._refresh_actual_budget_summary()
            self.best_state = self._clone_state_dict(self.stage_best_states["joint_finetune"])
            self.model.load_state_dict(self.best_state)
            self._write_result_audit_metadata()
            print(
                ">>>>>>> TypeFusion-CATCH training completed\n"
                f"Total Optimizer Steps: {self.training_budget_summary['actual_total_steps']}\n"
                f"Reference CATCH Steps: {self.training_budget_summary['reference_total_steps']}",
                flush=True,
            )
            return

        if self.config.training_budget_mode != "debug_stage_epochs":
            raise ValueError(
                "single_stage is a debug workflow and requires training_budget_mode='debug_stage_epochs'"
            )
        training_stage = self.config.training_stage
        checkpoint = previous_checkpoint if previous_checkpoint is not None else self.best_state
        if training_stage == "branch_pretrain":
            self.scaler = StandardScaler()
            self._scaler_fitted = False
            self.stage_best_states = {}
            self.stage_start_states = {}
            self.stage_validation_losses = {}
            self.stage_optimizer_lrs = {}
            self.stage_optimizer_steps = {}
            self.model = TypeFusionCATCHModel(self.config).to(self.device)
            loader_started = time.perf_counter()
            train_loader, valid_loader = self._prepare_loaders(train_data, fit_scaler=True)
        else:
            if checkpoint is None:
                raise ValueError(
                    "single_stage fusion_train/joint_finetune requires an explicit prior checkpoint"
                )
            if not self._scaler_fitted:
                raise ValueError(
                    "single_stage fusion_train/joint_finetune requires the prior fitted scaler"
                )
            if self.model is None:
                self.model = TypeFusionCATCHModel(self.config).to(self.device)
            self.model.load_state_dict(checkpoint)
            loader_started = time.perf_counter()
            train_loader, valid_loader = self._prepare_loaders(train_data, fit_scaler=False)

        LOGGER.info(
            "TypeFusion-CATCH loaders ready: train_batches=%d validation_batches=%d "
            "loader_prepare_seconds=%.3f",
            len(train_loader),
            len(valid_loader),
            time.perf_counter() - loader_started,
        )

        self._build_training_budget(train_loader)
        self._train_stage(training_stage, train_loader, valid_loader)
        self._refresh_actual_budget_summary()
        self.best_state = self._clone_state_dict(self.stage_best_states[training_stage])
        self.model.load_state_dict(self.best_state)
        self._write_result_audit_metadata()

    def detect_score(self, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return only the joint normal reconstruction's pointwise error."""

        if self.model is None or self.best_state is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        scaled_test = self._scaled_frame(self.scaler, test)
        loader = anomaly_detection_data_provider(
            scaled_test, self.config.batch_size, self.config.seq_len, mode="thre"
        )
        self.model.load_state_dict(self.best_state)
        self.model.to(self.device).eval()
        point_scores = []
        with torch.no_grad():
            for batch, _ in loader:
                output = self.model(batch.float().to(self.device), compute_joint=True)
                # Same window/point flattening contract as CATCH, but no
                # frequency side score or post-hoc branch-score operation.
                point_scores.append(output["total_score"].mean(dim=-1).cpu().numpy())
        scores = np.concatenate(point_scores, axis=0).reshape(-1) if point_scores else np.empty(0)
        return scores, scores


MODEL_DEFAULTS = DEFAULT_TYPEFUSION_HYPER_PARAMS
