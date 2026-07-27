"""Benchmark adapter for TypeFusion-CATCH v1."""

from __future__ import annotations

import copy
import random
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
            torch.cuda.manual_seed_all(seed)

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

    def _validation_loss(self, loader) -> float:
        assert self.model is not None
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch, _ in loader:
                output = self.model(batch.float().to(self.device))
                losses.append(float(output["losses"]["total"].detach().cpu()))
        self.model.train()
        return float(np.mean(losses)) if losses else float("inf")

    def _train_stage(self, training_stage: str, train_loader, valid_loader) -> None:
        assert self.model is not None
        self.model.set_training_stage(training_stage)
        self.stage_start_states[training_stage] = self._clone_state_dict(self.model.state_dict())
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError(f"{training_stage} has no trainable parameters")
        optimizer = torch.optim.Adam(parameters, lr=self.config.lr)
        stage_best = self._clone_state_dict(self.model.state_dict())
        best_loss = float("inf")
        stale_epochs = 0
        for _ in range(self.config.epochs_for_stage(training_stage)):
            self.model.train()
            for batch, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                output = self.model(batch.float().to(self.device))
                loss = output["losses"]["total"]
                loss.backward()
                optimizer.step()
            validation_loss = self._validation_loss(valid_loader)
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
            self.model = TypeFusionCATCHModel(self.config).to(self.device)
            train_loader, valid_loader = self._prepare_loaders(train_data, fit_scaler=True)
            for training_stage in ("branch_pretrain", "fusion_train", "joint_finetune"):
                self._train_stage(training_stage, train_loader, valid_loader)
            self.best_state = self._clone_state_dict(self.stage_best_states["joint_finetune"])
            self.model.load_state_dict(self.best_state)
            return

        training_stage = self.config.training_stage
        checkpoint = previous_checkpoint if previous_checkpoint is not None else self.best_state
        if training_stage == "branch_pretrain":
            self.scaler = StandardScaler()
            self._scaler_fitted = False
            self.stage_best_states = {}
            self.stage_start_states = {}
            self.stage_validation_losses = {}
            self.model = TypeFusionCATCHModel(self.config).to(self.device)
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
            train_loader, valid_loader = self._prepare_loaders(train_data, fit_scaler=False)

        self._train_stage(training_stage, train_loader, valid_loader)
        self.best_state = self._clone_state_dict(self.stage_best_states[training_stage])
        self.model.load_state_dict(self.best_state)

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
                output = self.model(batch.float().to(self.device))
                # Same window/point flattening contract as CATCH, but no
                # frequency side score or post-hoc branch-score operation.
                point_scores.append(output["total_score"].mean(dim=-1).cpu().numpy())
        scores = np.concatenate(point_scores, axis=0).reshape(-1) if point_scores else np.empty(0)
        return scores, scores


MODEL_DEFAULTS = DEFAULT_TYPEFUSION_HYPER_PARAMS
