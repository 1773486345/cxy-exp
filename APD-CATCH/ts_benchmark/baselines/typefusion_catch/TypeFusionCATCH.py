"""Benchmark adapter for TypeFusion-CATCH v1."""

from __future__ import annotations

import copy
import random
from typing import Optional, Tuple

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

    def detect_fit(self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame] = None) -> None:
        """Train exactly the selected stage; orchestration across stages is explicit."""

        self._seed_everything(self.config.seed)
        self.detect_hyper_param_tune(train_data)
        train_frame, valid_frame = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_frame.values)
        train_frame = self._scaled_frame(self.scaler, train_frame)
        valid_frame = self._scaled_frame(self.scaler, valid_frame)
        self.model = TypeFusionCATCHModel(self.config).to(self.device)
        train_loader = anomaly_detection_data_provider(
            train_frame, self.config.batch_size, self.config.seq_len, mode="train"
        )
        valid_loader = anomaly_detection_data_provider(
            valid_frame, self.config.batch_size, self.config.seq_len, mode="val"
        )
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("Selected training stage has no trainable parameters")
        optimizer = torch.optim.Adam(parameters, lr=self.config.lr)
        best_loss = float("inf")
        stale_epochs = 0
        for _ in range(self.config.num_epochs):
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
                self.best_state = copy.deepcopy(self.model.state_dict())
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break
        if self.best_state is None:
            self.best_state = copy.deepcopy(self.model.state_dict())

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
