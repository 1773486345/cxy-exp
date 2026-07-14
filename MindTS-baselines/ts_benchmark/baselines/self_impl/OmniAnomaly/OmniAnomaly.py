from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


DEFAULT_ANOMALY_RATIOS = [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNNER = PROJECT_ROOT / "scripts" / "baselines" / "omni_anomaly_runner.py"
OMNI_PYTHON = (
    Path(__file__).resolve().parents[5]
    / ".env"
    / "envs"
    / "omni_tf1"
    / "bin"
    / "python"
)


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
        raise ValueError("OmniAnomaly requires at least one numeric feature column.")
    return frame


@dataclass
class OmniAnomalyConfig:
    anomaly_ratio: Iterable[float] = tuple(DEFAULT_ANOMALY_RATIOS)
    window_length: int = 100
    z_dim: int = 3
    rnn_hidden: int = 500
    dense_dim: int = 500
    nf_layers: int = 20
    max_epoch: int = 10
    batch_size: int = 50
    test_batch_size: int = 50
    test_n_z: int = 1
    valid_portion: float = 0.3
    initial_lr: float = 0.001
    lr_anneal_factor: float = 0.5
    lr_anneal_epoch_freq: int = 40
    std_epsilon: float = 1e-4
    gradient_clip_norm: float = 10.0
    valid_step_freq: int = 100
    posterior_flow_type: str = "nf"
    random_state: int = 2021
    timeout: int = 60000


class OmniAnomaly:
    def __init__(self, **kwargs):
        config_values = OmniAnomalyConfig().__dict__
        config_values.update(kwargs)
        self.config = OmniAnomalyConfig(**config_values)
        self.model_name = "OmniAnomaly"
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = MinMaxScaler()
        self.feature_columns_: Optional[pd.Index] = None
        self.train_values_: Optional[np.ndarray] = None
        self.train_scores_: Optional[np.ndarray] = None
        self.test_scores_: Optional[np.ndarray] = None
        self.cached_test_shape_: Optional[tuple] = None

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

    def detect_fit(
        self, train_data: pd.DataFrame, train_label: Optional[pd.DataFrame] = None
    ) -> None:
        self.train_values_ = self._fit_transform(train_data)

    def _run_scores(self, test_data: pd.DataFrame) -> np.ndarray:
        if self.train_values_ is None:
            raise RuntimeError("OmniAnomaly has not been fitted.")
        test_values = self._transform(test_data)
        if self.cached_test_shape_ == test_values.shape and self.test_scores_ is not None:
            return self.test_scores_
        if not OMNI_PYTHON.exists():
            raise RuntimeError(f"OmniAnomaly Python env not found: {OMNI_PYTHON}")
        if len(self.train_values_) < int(self.config.window_length):
            raise ValueError("Training series is shorter than OmniAnomaly window_length.")
        if len(test_values) < int(self.config.window_length):
            raise ValueError("Test series is shorter than OmniAnomaly window_length.")

        work_root = PROJECT_ROOT / "result" / "label" / "_baseline_logs" / "omni_runs"
        work_root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="run_", dir=str(work_root)))
        input_path = work_dir / "input.npz"
        output_path = work_dir / "scores.npz"
        np.savez(input_path, train=self.train_values_, test=test_values)
        cmd = [
            str(OMNI_PYTHON),
            "-u",
            str(RUNNER),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--seed",
            str(int(self.config.random_state)),
            "--window-length",
            str(int(self.config.window_length)),
            "--z-dim",
            str(int(self.config.z_dim)),
            "--rnn-hidden",
            str(int(self.config.rnn_hidden)),
            "--dense-dim",
            str(int(self.config.dense_dim)),
            "--nf-layers",
            str(int(self.config.nf_layers)),
            "--max-epoch",
            str(int(self.config.max_epoch)),
            "--batch-size",
            str(int(self.config.batch_size)),
            "--test-batch-size",
            str(int(self.config.test_batch_size)),
            "--test-n-z",
            str(int(self.config.test_n_z)),
            "--valid-portion",
            str(float(self.config.valid_portion)),
            "--initial-lr",
            str(float(self.config.initial_lr)),
            "--lr-anneal-factor",
            str(float(self.config.lr_anneal_factor)),
            "--lr-anneal-epoch-freq",
            str(int(self.config.lr_anneal_epoch_freq)),
            "--std-epsilon",
            str(float(self.config.std_epsilon)),
            "--gradient-clip-norm",
            str(float(self.config.gradient_clip_norm)),
            "--valid-step-freq",
            str(int(self.config.valid_step_freq)),
            "--posterior-flow-type",
            str(self.config.posterior_flow_type),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
            timeout=int(self.config.timeout),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"OmniAnomaly runner failed with code {completed.returncode}")
        scores = np.load(output_path)
        self.train_scores_ = np.asarray(scores["train_score"], dtype=float)
        raw_test_scores = np.asarray(scores["test_score"], dtype=float)
        pad_front = int(self.config.window_length) - 1
        self.test_scores_ = np.pad(raw_test_scores, (pad_front, 0), mode="constant")
        self.cached_test_shape_ = test_values.shape
        shutil.rmtree(work_dir, ignore_errors=True)
        return self.test_scores_

    def detect_score(self, test_data: pd.DataFrame):
        scores = self._run_scores(test_data)
        return scores, scores

    def detect_label(self, test_data: pd.DataFrame):
        scores = self._run_scores(test_data)
        if self.train_scores_ is None:
            raise RuntimeError("OmniAnomaly train scores are unavailable.")
        raw_test_scores = scores[int(self.config.window_length) - 1 :]
        combined = np.concatenate([self.train_scores_, raw_test_scores], axis=0)
        anomaly_ratio = self.config.anomaly_ratio
        if isinstance(anomaly_ratio, (list, tuple, set)):
            ratios = anomaly_ratio
        else:
            ratios = [anomaly_ratio]
        preds = {}
        for ratio in ratios:
            raw_pred = _threshold_by_ratio(combined, float(ratio))[-len(raw_test_scores) :]
            preds[float(ratio)] = np.pad(
                raw_pred,
                (int(self.config.window_length) - 1, 0),
                mode="constant",
            )
        return preds, scores

    def __repr__(self) -> str:
        return self.model_name
