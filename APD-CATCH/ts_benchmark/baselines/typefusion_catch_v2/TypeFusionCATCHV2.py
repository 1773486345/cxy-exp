"""CATCH-compatible adapter for TypeFusion-CATCH v2."""

from __future__ import annotations

import copy
import random
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.catch.CATCH import CATCH
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split

from .config import TypeFusionCATCHV2Config
from .typefusion_catch_v2 import TypeFusionCATCHV2Model


class TypeFusionCATCHV2:
    """Two-phase v2 adapter: original CATCH anchor followed by type tasks."""

    def __init__(self, **kwargs) -> None:
        self.config = TypeFusionCATCHV2Config.from_kwargs(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.anchor_adapter: Optional[CATCH] = None
        self.model: Optional[TypeFusionCATCHV2Model] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.anchor_best_validation_loss: Optional[float] = None
        self.type_best_validation_loss: Optional[float] = None
        self.anchor_optimizer_steps = 0
        self.type_optimizer_steps = 0
        self.completed_phases = []
        self.training_metadata: Dict[str, object] = {}
        self.model_name = "TypeFusionCATCHV2"
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
    def _copy_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in state.items()}

    def detect_hyper_param_tune(self, train_data: pd.DataFrame) -> None:
        self.config.c_in = int(train_data.shape[1])
        self.config.validate()

    def _catch_kwargs(self) -> Dict[str, object]:
        names = (
            "lr", "Mlr", "e_layers", "n_heads", "cf_dim", "d_ff", "d_model", "head_dim",
            "dropout", "head_dropout", "auxi_lambda", "score_lambda", "regular_lambda",
            "temperature", "patch_stride", "patch_size", "inference_patch_stride",
            "inference_patch_size", "dc_lambda", "catch_train_epochs", "batch_size",
            "patience", "seq_len", "affine", "subtract_last", "revin", "anomaly_ratio",
            "itr", "small_kernel_merged", "use_multi_scale", "pct_start",
        )
        result = {name: getattr(self.config, name) for name in names}
        result["num_epochs"] = self.config.catch_train_epochs
        result["c_in"] = self.config.c_in
        result["enc_in"] = self.config.c_in
        result["dec_in"] = self.config.c_in
        result["c_out"] = self.config.c_in
        result["individual"] = 0
        return result

    def _fit_anchor(self, train_data: pd.DataFrame) -> None:
        if self.config.anchor_adapter is not None:
            self.anchor_adapter = self.config.anchor_adapter
            if getattr(self.anchor_adapter, "model", None) is None:
                self.anchor_adapter.detect_fit(train_data, train_data)
        elif self.config.anchor_state_dict is not None or self.config.skip_anchor_fit:
            # Debug-only path for checkpoint parity tests. Formal defaults always
            # call the original adapter below.
            self.anchor_adapter = CATCH(**self._catch_kwargs())
            train_frame, _ = train_val_split(train_data, 0.8, None)
            self.anchor_adapter.scaler.fit(train_frame.values)
            self.anchor_adapter.detect_hyper_param_tune(train_data)
            self.anchor_adapter.model = __import__(
                "ts_benchmark.baselines.catch.models.CATCH_model", fromlist=["CATCHModel"]
            ).CATCHModel(self.anchor_adapter.config)
            self.anchor_adapter.model.to(self.device)
            self.anchor_adapter.early_stopping = type("Checkpoint", (), {})()
            self.anchor_adapter.early_stopping.check_point = self.config.anchor_state_dict or self.anchor_adapter.model.state_dict()
        else:
            self.anchor_adapter = CATCH(**self._catch_kwargs())
            # The benchmark's second argument is a compatibility label frame;
            # CATCH's implementation does not read it during fitting.
            self.anchor_adapter.detect_fit(train_data, train_data)
        self.scaler = copy.deepcopy(self.anchor_adapter.scaler)
        self.anchor_best_validation_loss = getattr(self.anchor_adapter, "early_stopping", None)
        if self.anchor_best_validation_loss is not None:
            self.anchor_best_validation_loss = getattr(self.anchor_best_validation_loss, "val_loss_min", None)

    def _build_model(self) -> None:
        assert self.anchor_adapter is not None
        self.model = TypeFusionCATCHV2Model(self.config, anchor_model=self.anchor_adapter.model).to(self.device)
        checkpoint = getattr(getattr(self.anchor_adapter, "early_stopping", None), "check_point", None)
        if checkpoint is None:
            checkpoint = self.config.anchor_state_dict
        if checkpoint is not None:
            self.model.load_anchor_state_dict(checkpoint, strict=True)
        self.model.freeze_anchor()

    def _scaled_loader_pair(self, train_data: pd.DataFrame):
        train_frame, valid_frame = train_val_split(train_data, 0.8, None)
        train_scaled = pd.DataFrame(self.scaler.transform(train_frame.values), index=train_frame.index, columns=train_frame.columns)
        valid_scaled = pd.DataFrame(self.scaler.transform(valid_frame.values), index=valid_frame.index, columns=valid_frame.columns)
        train_loader = anomaly_detection_data_provider(train_scaled, self.config.batch_size, self.config.seq_len, mode="train")
        valid_loader = anomaly_detection_data_provider(valid_scaled, self.config.batch_size, self.config.seq_len, mode="val")
        return train_loader, valid_loader

    def _validate_type(self, loader) -> float:
        assert self.model is not None
        self.model.eval()
        losses = []
        offset = 0
        with torch.no_grad():
            for batch, _ in loader:
                batch = batch.float().to(self.device)
                intervention = self.model.intervention_generator.generate(
                    batch, validation=True, sample_indices=range(offset, offset + batch.size(0))
                )
                output = self.model(batch, intervention=intervention)
                losses.append(float(output["losses"]["total"].detach().cpu()))
                offset += batch.size(0)
        self.model.train()
        return float(np.mean(losses)) if losses else float("inf")

    def _train_type(self, train_loader, valid_loader) -> None:
        assert self.model is not None
        self.model.set_phase_b_trainable()
        optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.config.lr)
        best_loss = float("inf")
        best_state = self._copy_state(self.model.state_dict())
        patience_left = self.config.patience
        for epoch in range(self.config.type_train_epochs):
            self.model.train()
            for batch, _ in train_loader:
                batch = batch.float().to(self.device)
                intervention = self.model.intervention_generator.generate(batch)
                output = self.model(batch, intervention=intervention)
                loss = output["losses"]["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("TypeFusionCATCHV2 Phase B loss is non-finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 10.0)
                optimizer.step()
                self.type_optimizer_steps += 1
            validation_loss = self._validate_type(valid_loader)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = self._copy_state(self.model.state_dict())
                patience_left = self.config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        self.model.load_state_dict(best_state, strict=True)
        self.best_state = best_state
        self.type_best_validation_loss = best_loss

    def detect_fit(self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame] = None) -> None:
        self._seed_everything(self.config.seed)
        self.detect_hyper_param_tune(train_data)
        self._fit_anchor(train_data)
        self._build_model()
        self.completed_phases = ["phase_a"]
        train_loader, valid_loader = self._scaled_loader_pair(train_data)
        self.anchor_optimizer_steps = self.config.catch_train_epochs * len(train_loader)
        self._train_type(train_loader, valid_loader)
        self.completed_phases.append("phase_b")
        self.training_metadata = {
            "completed_phases": list(self.completed_phases),
            "anchor_optimizer_steps": int(self.anchor_optimizer_steps),
            "type_optimizer_steps": int(self.type_optimizer_steps),
            "anchor_best_validation_loss": self.anchor_best_validation_loss,
            "type_best_validation_loss": self.type_best_validation_loss,
            "loss_version": self.config.loss_version,
            "seed": self.config.seed,
        }
        self._fitted = True

    def detect_score(self, test: pd.DataFrame):
        if not self._fitted or self.model is None:
            raise ValueError("Model not trained. Call detect_fit() first.")
        scaled = pd.DataFrame(self.scaler.transform(test.values), index=test.index, columns=test.columns)
        loader = anomaly_detection_data_provider(scaled, self.config.batch_size, self.config.seq_len, mode="thre")
        self.model.eval()
        scores = []
        with torch.no_grad():
            for batch, _ in loader:
                output = self.model(batch.float().to(self.device), compute_loss=False)
                score = output["joint_score"]
                if not torch.isfinite(score).all():
                    raise FloatingPointError("TypeFusionCATCHV2 produced NaN/Inf scores")
                scores.append(score.detach().cpu().numpy())
        values = np.concatenate(scores, axis=0).reshape(-1) if scores else np.empty(0, dtype=np.float32)
        return values, values
