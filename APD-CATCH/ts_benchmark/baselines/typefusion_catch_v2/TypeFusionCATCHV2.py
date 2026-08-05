"""Benchmark adapter for the single-stage TypeFusion-CATCH v2 model."""

from __future__ import annotations

import random
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ts_benchmark.baselines.utils import SegLoader, anomaly_detection_data_provider, train_val_split

from .config import TypeFusionCATCHV2Config
from .typefusion_catch_v2 import TypeFusionCATCHV2Model


class TypeFusionCATCHV2:
    def __init__(self, **kwargs) -> None:
        self.config = TypeFusionCATCHV2Config.from_kwargs(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TypeFusionCATCHV2Model] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.optimizer_steps = 0
        self.best_validation_loss: Optional[float] = None
        self.training_metadata: Dict[str, object] = {}
        self.model_name = "TypeFusionCATCHV2"
        self._scaler_fitted = False
        self._fitted = False

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self) -> str:
        return self.model_name

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
    def _clone_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in state.items()}

    def detect_hyper_param_tune(self, train_data: pd.DataFrame) -> None:
        self.config.c_in = int(train_data.shape[1])
        self.config.validate()

    @staticmethod
    def _scaled_frame(scaler: StandardScaler, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(scaler.transform(frame.values).astype(np.float32), index=frame.index, columns=frame.columns)

    def _prepare_loaders(self, train_data: pd.DataFrame):
        train_frame, valid_frame = train_val_split(train_data, 0.8, None)
        if not self._scaler_fitted:
            self.scaler.fit(train_frame.values)
            self._scaler_fitted = True
        train_scaled = self._scaled_frame(self.scaler, train_frame)
        valid_scaled = self._scaled_frame(self.scaler, valid_frame)
        train_loader = anomaly_detection_data_provider(train_scaled, self.config.batch_size, self.config.seq_len, mode="train")
        # ``anomaly_detection_data_provider(..., mode="val")`` shuffles by
        # design.  Validation interventions are indexed and deterministic, so
        # their window order must also be deterministic.
        valid_dataset = SegLoader(valid_scaled.values, self.config.seq_len, 1, mode="val")
        valid_loader = DataLoader(valid_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0, drop_last=False)
        return train_loader, valid_loader

    def _validation_loss(self, loader) -> float:
        assert self.model is not None
        self.model.eval()
        losses = []
        global_index = 0
        with torch.no_grad():
            for batch, _ in loader:
                batch = batch.float().contiguous().to(self.device)
                intervention = self.model.intervention_generator.generate(batch, validation=True, sample_indices=range(global_index, global_index + batch.size(0)))
                result = self.model(batch, intervention=intervention)
                losses.append(float(result["losses"]["total"].detach().cpu()))
                global_index += batch.size(0)
        self.model.train()
        return float(np.mean(losses)) if losses else float("inf")

    def detect_fit(self, train_data: pd.DataFrame, train_label=None) -> None:
        self._seed_everything(self.config.seed)
        self.detect_hyper_param_tune(train_data)
        train_loader, valid_loader = self._prepare_loaders(train_data)
        self.model = TypeFusionCATCHV2Model(self.config).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        best_loss = float("inf")
        best_state = self._clone_state(self.model.state_dict())
        patience_left = self.config.patience
        for epoch in range(self.config.num_epochs):
            self.model.train()
            for batch, _ in train_loader:
                batch = batch.float().contiguous().to(self.device)
                intervention = self.model.intervention_generator.generate(batch)
                result = self.model(batch, intervention=intervention)
                loss = result["losses"]["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss: total")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self.optimizer_steps += 1
            validation_loss = self._validation_loss(valid_loader)
            print(f">>>>>>> TypeFusion-CATCH v2 | Epoch: {epoch + 1} | Train Loss: {float(loss.detach().cpu()):.7f} Vali Loss: {validation_loss:.7f}", flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = self._clone_state(self.model.state_dict())
                patience_left = self.config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        self.model.load_state_dict(best_state, strict=True)
        self.best_state = best_state
        self.best_validation_loss = best_loss
        self.training_metadata = {
            "completed_phases": ["single_stage"],
            "optimizer_steps": self.optimizer_steps,
            "best_validation_loss": self.best_validation_loss,
            "num_epochs": self.config.num_epochs,
            "loss_version": self.config.loss_version,
            "seed": self.config.seed,
        }
        self._fitted = True

    def detect_score(self, test_data: pd.DataFrame):
        if not self._fitted or self.model is None or not self._scaler_fitted:
            raise ValueError("Model not trained. Call detect_fit() first.")
        test_scaled = self._scaled_frame(self.scaler, test_data)
        loader = anomaly_detection_data_provider(test_scaled, self.config.batch_size, self.config.seq_len, mode="thre")
        self.model.eval()
        scores = []
        with torch.no_grad():
            for batch, _ in loader:
                batch = batch.float().contiguous().to(self.device)
                result = self.model(batch, compute_loss=False)
                score = result["joint_score"]
                if not torch.isfinite(score).all():
                    raise FloatingPointError("non-finite score: joint_score")
                scores.append(score.detach().cpu().numpy())
        values = np.concatenate(scores, axis=0).reshape(-1) if scores else np.empty(0, dtype=np.float32)
        return values, values
