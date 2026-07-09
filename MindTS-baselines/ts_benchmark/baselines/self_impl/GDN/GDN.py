from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import iqr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Subset


GDN_REPO = Path(__file__).resolve().parents[5] / "baseline_repos" / "GDN"
if GDN_REPO.exists() and str(GDN_REPO) not in sys.path:
    sys.path.insert(0, str(GDN_REPO))

from models.GDN import GDN as OfficialGDNModel  # noqa: E402
from util.env import set_device  # noqa: E402


DEFAULT_ANOMALY_RATIOS = [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25]


def _threshold_by_ratio(scores: np.ndarray, ratio: float) -> np.ndarray:
    ratio = float(np.clip(ratio, 0.0, 100.0))
    if scores.size == 0 or ratio <= 0:
        return np.zeros_like(scores, dtype=int)
    threshold = np.percentile(scores, 100.0 - ratio)
    return (scores > threshold).astype(int)


def _as_numeric_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    for column in ("date", "label"):
        if column in frame.columns:
            frame = frame.drop(columns=[column])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all")
    if frame.shape[1] == 0:
        raise ValueError("GDN requires at least one numeric feature column.")
    return frame


def _full_edge_index(node_num: int) -> torch.Tensor:
    rows, cols = [], []
    for src in range(node_num):
        for dst in range(node_num):
            if src != dst:
                rows.append(src)
                cols.append(dst)
    if not rows:
        rows, cols = [0], [0]
    return torch.tensor([rows, cols], dtype=torch.long)


class _GDNWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        edge_index: torch.Tensor,
        slide_win: int,
        slide_stride: int,
        train_mode: bool,
    ):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.edge_index = edge_index.long()
        self.slide_win = int(slide_win)
        stride = int(slide_stride) if train_mode else 1
        self.indices = list(range(self.slide_win, len(values), stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = self.indices[idx]
        x = self.values[end - self.slide_win : end].T.contiguous()
        y = self.values[end].contiguous()
        label = torch.tensor(0.0, dtype=torch.float32)
        return x, y, label, self.edge_index


@dataclass
class GDNConfig:
    anomaly_ratio: Iterable[float] = tuple(DEFAULT_ANOMALY_RATIOS)
    batch_size: int = 32
    num_epochs: int = 30
    slide_win: int = 5
    slide_stride: int = 1
    dim: int = 64
    topk: int = 5
    out_layer_num: int = 1
    out_layer_inter_dim: int = 128
    val_ratio: float = 0.2
    decay: float = 0.0
    lr: float = 0.001
    patience: int = 15
    score_topk: int = 1
    random_state: int = 2021


class GDN:
    def __init__(self, **kwargs):
        config_values = GDNConfig().__dict__
        config_values.update(kwargs)
        self.config = GDNConfig(**config_values)
        self.model_name = "GDN"
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_device(str(self.device))
        self.model: Optional[OfficialGDNModel] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.feature_columns_: Optional[pd.Index] = None
        self.train_scores_: Optional[np.ndarray] = None
        self.val_prediction_: Optional[np.ndarray] = None
        self.val_target_: Optional[np.ndarray] = None

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def _fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        frame = _as_numeric_frame(data)
        self.feature_columns_ = frame.columns
        values = self.imputer.fit_transform(frame.values)
        return self.scaler.fit_transform(values).astype(np.float32, copy=False)

    def _transform(self, data: pd.DataFrame) -> np.ndarray:
        frame = _as_numeric_frame(data)
        if self.feature_columns_ is not None:
            frame = frame.reindex(columns=self.feature_columns_)
        values = self.imputer.transform(frame.values)
        return self.scaler.transform(values).astype(np.float32, copy=False)

    def _make_dataset(self, values: np.ndarray, train_mode: bool) -> _GDNWindowDataset:
        if self.edge_index is None:
            self.edge_index = _full_edge_index(values.shape[1])
        return _GDNWindowDataset(
            values=values,
            edge_index=self.edge_index,
            slide_win=int(self.config.slide_win),
            slide_stride=int(self.config.slide_stride),
            train_mode=train_mode,
        )

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(self.config.batch_size),
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def _split_train_val(self, dataset: Dataset) -> Tuple[Subset, Subset]:
        dataset_len = len(dataset)
        val_len = max(1, int(dataset_len * float(self.config.val_ratio)))
        train_len = max(1, dataset_len - val_len)
        indices = torch.arange(dataset_len)
        train_subset = Subset(dataset, indices[:train_len])
        val_subset = Subset(dataset, indices[train_len:])
        if len(val_subset) == 0:
            val_subset = Subset(dataset, indices[:train_len])
        return train_subset, val_subset

    def _predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("GDN has not been fitted.")
        predictions, targets = [], []
        self.model.eval()
        with torch.no_grad():
            for x, y, _labels, edge_index in loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                edge_index = edge_index.float().to(self.device)
                predicted = self.model(x, edge_index).float()
                predictions.append(predicted.detach().cpu().numpy())
                targets.append(y.detach().cpu().numpy())
        return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)

    def _score_from_prediction(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        pad_front: bool,
    ) -> np.ndarray:
        delta = np.abs(prediction.astype(np.float64) - target.astype(np.float64))
        feature_scores = []
        for feature_idx in range(delta.shape[1]):
            feature_delta = delta[:, feature_idx]
            mid = np.median(feature_delta)
            spread = iqr(feature_delta)
            err_scores = (feature_delta - mid) / (abs(spread) + 1e-2)
            smoothed = np.zeros_like(err_scores)
            for idx in range(3, len(err_scores)):
                smoothed[idx] = np.mean(err_scores[idx - 3 : idx + 1])
            feature_scores.append(smoothed)

        full_scores = np.vstack(feature_scores)
        score_topk = min(int(self.config.score_topk), full_scores.shape[0])
        topk_indices = np.argpartition(full_scores, -score_topk, axis=0)[-score_topk:]
        scores = np.sum(np.take_along_axis(full_scores, topk_indices, axis=0), axis=0)
        if pad_front:
            scores = np.pad(scores, (int(self.config.slide_win), 0), mode="constant")
        return scores

    def detect_fit(
        self, train_data: pd.DataFrame, train_label: Optional[pd.DataFrame] = None
    ) -> None:
        torch.manual_seed(int(self.config.random_state))
        np.random.seed(int(self.config.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config.random_state))

        train_values = self._fit_transform(train_data)
        node_num = train_values.shape[1]
        self.edge_index = _full_edge_index(node_num)
        model_topk = min(int(self.config.topk), node_num)
        self.model = OfficialGDNModel(
            [self.edge_index],
            node_num,
            dim=int(self.config.dim),
            out_layer_inter_dim=int(self.config.out_layer_inter_dim),
            input_dim=int(self.config.slide_win),
            out_layer_num=int(self.config.out_layer_num),
            topk=model_topk,
        ).to(self.device)

        train_dataset = self._make_dataset(train_values, train_mode=True)
        fit_subset, val_subset = self._split_train_val(train_dataset)
        train_loader = self._loader(fit_subset, shuffle=True)
        val_loader = self._loader(val_subset, shuffle=False)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.decay),
        )

        best_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        stop_count = 0
        for epoch_idx in range(int(self.config.num_epochs)):
            self.model.train()
            losses = []
            for x, y, _labels, edge_index in train_loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                edge_index = edge_index.float().to(self.device)
                optimizer.zero_grad()
                predicted = self.model(x, edge_index).float()
                loss = F.mse_loss(predicted, y, reduction="mean")
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().item())

            val_pred, val_true = self._predict(val_loader)
            val_loss = float(np.mean((val_pred - val_true) ** 2))
            if epoch_idx == 0 or (epoch_idx + 1) % 5 == 0 or epoch_idx + 1 == int(self.config.num_epochs):
                print(
                    f"epoch ({epoch_idx + 1} / {int(self.config.num_epochs)}) "
                    f"(Loss:{np.mean(losses):.8f}, Val:{val_loss:.8f})",
                    flush=True,
                )

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                stop_count = 0
            else:
                stop_count += 1
            if stop_count >= int(self.config.patience):
                break

        self.model.load_state_dict(best_state)
        self.val_prediction_, self.val_target_ = self._predict(val_loader)
        train_score_dataset = self._make_dataset(train_values, train_mode=False)
        train_pred, train_true = self._predict(self._loader(train_score_dataset, shuffle=False))
        self.train_scores_ = self._score_from_prediction(
            train_pred,
            train_true,
            pad_front=False,
        )

    def _raw_test_scores(self, test_data: pd.DataFrame) -> np.ndarray:
        test_values = self._transform(test_data)
        test_dataset = self._make_dataset(test_values, train_mode=False)
        prediction, target = self._predict(self._loader(test_dataset, shuffle=False))
        return self._score_from_prediction(prediction, target, pad_front=False)

    def detect_score(self, test_data: pd.DataFrame):
        raw_scores = self._raw_test_scores(test_data)
        scores = np.pad(raw_scores, (int(self.config.slide_win), 0), mode="constant")
        return scores, scores

    def detect_label(self, test_data: pd.DataFrame):
        raw_scores = self._raw_test_scores(test_data)
        if self.train_scores_ is None:
            raise RuntimeError("GDN train scores are unavailable.")
        combined = np.concatenate([self.train_scores_, raw_scores], axis=0)
        anomaly_ratio = self.config.anomaly_ratio
        if isinstance(anomaly_ratio, (list, tuple, set)):
            ratios = anomaly_ratio
        else:
            ratios = [anomaly_ratio]
        preds = {}
        for ratio in ratios:
            raw_pred = _threshold_by_ratio(combined, float(ratio))[-len(raw_scores) :]
            preds[float(ratio)] = np.pad(
                raw_pred,
                (int(self.config.slide_win), 0),
                mode="constant",
            )
        padded_scores = np.pad(raw_scores, (int(self.config.slide_win), 0), mode="constant")
        return preds, padded_scores

    def __repr__(self) -> str:
        return self.model_name
