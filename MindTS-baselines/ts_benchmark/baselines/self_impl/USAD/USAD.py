from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


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
        raise ValueError("USAD requires at least one numeric feature column.")
    return frame


def _make_windows(values: np.ndarray, window_size: int) -> np.ndarray:
    if values.shape[0] < window_size:
        pad_len = window_size - values.shape[0]
        values = np.pad(values, ((0, pad_len), (0, 0)), mode="edge")
    starts = np.arange(values.shape[0] - window_size + 1)[:, None]
    offsets = np.arange(window_size)[None, :]
    windows = values[starts + offsets]
    return windows.reshape(windows.shape[0], -1).astype(np.float32, copy=False)


class Encoder(nn.Module):
    def __init__(self, in_size: int, latent_size: int):
        super().__init__()
        hidden1 = max(1, in_size // 2)
        hidden2 = max(1, in_size // 4)
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden1),
            nn.ReLU(True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(True),
            nn.Linear(hidden2, latent_size),
            nn.ReLU(True),
        )

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return self.net(window)


class Decoder(nn.Module):
    def __init__(self, latent_size: int, out_size: int):
        super().__init__()
        hidden2 = max(1, out_size // 4)
        hidden1 = max(1, out_size // 2)
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden2),
            nn.ReLU(True),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(True),
            nn.Linear(hidden1, out_size),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class UsadModel(nn.Module):
    def __init__(self, window_size: int, latent_size: int):
        super().__init__()
        self.encoder = Encoder(window_size, latent_size)
        self.decoder1 = Decoder(latent_size, window_size)
        self.decoder2 = Decoder(latent_size, window_size)

    def training_step(self, batch: torch.Tensor, epoch: int):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        inv_epoch = 1.0 / float(epoch)
        loss1 = inv_epoch * torch.mean((batch - w1) ** 2) + (
            1.0 - inv_epoch
        ) * torch.mean((batch - w3) ** 2)
        loss2 = inv_epoch * torch.mean((batch - w2) ** 2) - (
            1.0 - inv_epoch
        ) * torch.mean((batch - w3) ** 2)
        return loss1, loss2

    def score(self, batch: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        w1 = self.decoder1(self.encoder(batch))
        w2 = self.decoder2(self.encoder(w1))
        return alpha * torch.mean((batch - w1) ** 2, dim=1) + beta * torch.mean(
            (batch - w2) ** 2, dim=1
        )


@dataclass
class USADConfig:
    anomaly_ratio: Iterable[float] = tuple(DEFAULT_ANOMALY_RATIOS)
    batch_size: int = 128
    n_window: int = 12
    num_epochs: int = 100
    hidden_size: int = 100
    lr: float = 0.001
    alpha: float = 0.5
    beta: float = 0.5
    valid_ratio: float = 0.2
    random_state: int = 2021


class USAD:
    def __init__(self, **kwargs):
        config_values = USADConfig().__dict__
        config_values.update(kwargs)
        self.config = USADConfig(**config_values)
        self.model_name = "USAD"
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[UsadModel] = None
        self.train_energy_: Optional[np.ndarray] = None
        self.feature_columns_: Optional[pd.Index] = None

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

    def _loader(self, windows: np.ndarray, shuffle: bool) -> DataLoader:
        tensor = torch.from_numpy(windows)
        return DataLoader(
            TensorDataset(tensor),
            batch_size=int(self.config.batch_size),
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def _score_windows(self, windows: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("USAD has not been fitted.")
        loader = self._loader(windows, shuffle=False)
        scores = []
        self.model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                score = self.model.score(batch, self.config.alpha, self.config.beta)
                scores.append(score.detach().cpu().numpy())
        return np.concatenate(scores, axis=0) if scores else np.empty(0, dtype=float)

    def detect_fit(
        self, train_data: pd.DataFrame, train_label: Optional[pd.DataFrame] = None
    ) -> None:
        torch.manual_seed(int(self.config.random_state))
        np.random.seed(int(self.config.random_state))

        train_values = self._fit_transform(train_data)
        windows = _make_windows(train_values, int(self.config.n_window))
        split = int(np.floor((1.0 - float(self.config.valid_ratio)) * len(windows)))
        split = min(max(split, 1), len(windows))
        train_windows = windows[:split]
        valid_windows = windows[split:] if split < len(windows) else windows[:split]

        flat_window_size = train_windows.shape[1]
        latent_size = int(self.config.n_window) * int(self.config.hidden_size)
        self.model = UsadModel(flat_window_size, latent_size).to(self.device)
        optimizer1 = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder1.parameters()),
            lr=float(self.config.lr),
        )
        optimizer2 = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder2.parameters()),
            lr=float(self.config.lr),
        )

        train_loader = self._loader(train_windows, shuffle=False)
        valid_loader = self._loader(valid_windows, shuffle=False)
        for epoch_idx in range(int(self.config.num_epochs)):
            epoch = epoch_idx + 1
            self.model.train()
            for (batch,) in train_loader:
                batch = batch.to(self.device)

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss1, _ = self.model.training_step(batch, epoch)
                loss1.backward()
                optimizer1.step()

                optimizer1.zero_grad()
                optimizer2.zero_grad()

                _, loss2 = self.model.training_step(batch, epoch)
                loss2.backward()
                optimizer2.step()

                optimizer1.zero_grad()
                optimizer2.zero_grad()

            if epoch == 1 or epoch == int(self.config.num_epochs) or epoch % 10 == 0:
                val_losses = []
                self.model.eval()
                with torch.no_grad():
                    for (batch,) in valid_loader:
                        batch = batch.to(self.device)
                        loss1, loss2 = self.model.training_step(batch, epoch)
                        val_losses.append((loss1 + loss2).detach().cpu().item())
                print(f"Epoch [{epoch}], val_loss: {np.mean(val_losses):.6f}")

        self.train_energy_ = self._score_windows(windows)

    def detect_score(self, test_data: pd.DataFrame):
        test_values = self._transform(test_data)
        test_windows = _make_windows(test_values, int(self.config.n_window))
        test_energy = self._score_windows(test_windows)
        return test_energy, test_energy

    def detect_label(self, test_data: pd.DataFrame):
        test_energy, _ = self.detect_score(test_data)
        if self.train_energy_ is None:
            raise RuntimeError("USAD train scores are unavailable.")
        combined_energy = np.concatenate([self.train_energy_, test_energy], axis=0)
        anomaly_ratio = self.config.anomaly_ratio
        if isinstance(anomaly_ratio, (list, tuple, set)):
            ratios = anomaly_ratio
        else:
            ratios = [anomaly_ratio]
        preds = {
            float(ratio): _threshold_by_ratio(combined_energy, float(ratio))[
                -len(test_energy) :
            ]
            for ratio in ratios
        }
        return preds, test_energy

    def __repr__(self) -> str:
        return self.model_name
