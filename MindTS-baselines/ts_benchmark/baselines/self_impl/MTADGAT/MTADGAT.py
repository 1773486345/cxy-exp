from __future__ import annotations

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
RUNNER = PROJECT_ROOT / "scripts" / "baselines" / "mtad_gat_runner.py"
MTAD_GAT_PYTHON = (
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
        raise ValueError("MTAD-GAT requires at least one numeric feature column.")
    return frame


@dataclass
class MTADGATConfig:
    anomaly_ratio: Iterable[float] = tuple(DEFAULT_ANOMALY_RATIOS)
    window_length: int = 100
    run_mode: str = "FORECASTING"
    num_train_steps: int = 100
    batch_size: int = 128
    gru_hidden: int = 64
    fc_hidden: int = 64
    vae_latent: int = 18
    conv1d_filter_width: int = 7
    learning_rate: float = 5e-6
    clip_gradients: float = 0.1
    dropout_prob: float = 0.0
    gamma: float = 0.8
    save_checkpoints_steps: int = 100
    log_step_count_steps: int = 20
    keep_checkpoint_max: int = 2
    shuffle_buffer_size: int = 29000
    dataset_reader_buffer_size: int = 1048576
    random_state: int = 20260612
    timeout: int = 60000


class MTADGAT:
    def __init__(self, **kwargs):
        config_values = MTADGATConfig().__dict__
        config_values.update(kwargs)
        self.config = MTADGATConfig(**config_values)
        self.model_name = "MTADGAT"
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

    def detect_fit(self, train_data: pd.DataFrame, train_label: Optional[pd.DataFrame] = None) -> None:
        self.train_values_ = self._fit_transform(train_data)

    def _run_scores(self, test_data: pd.DataFrame) -> np.ndarray:
        if self.train_values_ is None:
            raise RuntimeError("MTAD-GAT has not been fitted.")
        test_values = self._transform(test_data)
        if self.cached_test_shape_ == test_values.shape and self.test_scores_ is not None:
            return self.test_scores_
        if not MTAD_GAT_PYTHON.exists():
            raise RuntimeError(f"MTAD-GAT Python env not found: {MTAD_GAT_PYTHON}")
        pad_front = int(self.config.window_length) + 1
        if len(self.train_values_) <= pad_front:
            raise ValueError("Training series is shorter than MTAD-GAT window_length + 1.")
        if len(test_values) <= pad_front:
            raise ValueError("Test series is shorter than MTAD-GAT window_length + 1.")

        work_root = PROJECT_ROOT / "result" / "label" / "_baseline_logs" / "mtad_gat_runs"
        work_root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="run_", dir=str(work_root)))
        input_path = work_dir / "input.npz"
        output_path = work_dir / "scores.npz"
        np.savez(
            input_path,
            train=self.train_values_,
            test=test_values,
            label=np.zeros([len(test_values)], dtype=np.int64),
        )

        cmd = [
            str(MTAD_GAT_PYTHON),
            str(RUNNER),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--work-dir",
            str(work_dir / "mtad_gat"),
            "--seed",
            str(int(self.config.random_state)),
            "--window-size",
            str(int(self.config.window_length)),
            "--run-mode",
            str(self.config.run_mode),
            "--num-train-steps",
            str(int(self.config.num_train_steps)),
            "--batch-size",
            str(int(self.config.batch_size)),
            "--gru-hidden",
            str(int(self.config.gru_hidden)),
            "--fc-hidden",
            str(int(self.config.fc_hidden)),
            "--vae-latent",
            str(int(self.config.vae_latent)),
            "--conv1d-filter-width",
            str(int(self.config.conv1d_filter_width)),
            "--learning-rate",
            str(float(self.config.learning_rate)),
            "--clip-gradients",
            str(float(self.config.clip_gradients)),
            "--dropout-prob",
            str(float(self.config.dropout_prob)),
            "--gamma",
            str(float(self.config.gamma)),
            "--save-checkpoints-steps",
            str(int(self.config.save_checkpoints_steps)),
            "--log-step-count-steps",
            str(int(self.config.log_step_count_steps)),
            "--keep-checkpoint-max",
            str(int(self.config.keep_checkpoint_max)),
            "--shuffle-buffer-size",
            str(int(self.config.shuffle_buffer_size)),
            "--dataset-reader-buffer-size",
            str(int(self.config.dataset_reader_buffer_size)),
            "--timeout",
            str(int(self.config.timeout)),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(self.config.timeout),
            check=False,
        )
        print(completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(f"MTAD-GAT runner failed with code {completed.returncode}")
        scores = np.load(output_path)
        self.train_scores_ = np.asarray(scores["train_score"], dtype=float)
        raw_test_scores = np.asarray(scores["test_score"], dtype=float)
        self.test_scores_ = np.pad(raw_test_scores, (pad_front, 0), mode="constant")
        self.cached_test_shape_ = test_values.shape
        return self.test_scores_

    def detect_score(self, test_data: pd.DataFrame):
        scores = self._run_scores(test_data)
        return scores, scores

    def detect_label(self, test_data: pd.DataFrame):
        scores = self._run_scores(test_data)
        if self.train_scores_ is None:
            raise RuntimeError("MTAD-GAT train scores are unavailable.")
        pad_front = int(self.config.window_length) + 1
        raw_test_scores = scores[pad_front:]
        combined = np.concatenate([self.train_scores_, raw_test_scores], axis=0)
        anomaly_ratio = self.config.anomaly_ratio
        if isinstance(anomaly_ratio, (list, tuple, set)):
            ratios = anomaly_ratio
        else:
            ratios = [anomaly_ratio]
        preds = {}
        for ratio in ratios:
            raw_pred = _threshold_by_ratio(combined, float(ratio))[-len(raw_test_scores) :]
            preds[float(ratio)] = np.pad(raw_pred, (pad_front, 0), mode="constant")
        return preds, scores

    def __repr__(self) -> str:
        return self.model_name
