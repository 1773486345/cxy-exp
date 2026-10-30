"""Read anomaly-detection task files through the benchmark's canonical loader."""

from pathlib import Path

import numpy as np
import pandas as pd

from ts_benchmark.data.utils import read_data


def load_task_frame(path):
    path = Path(path).resolve()
    header = pd.read_csv(path, nrows=0).columns
    if {"date", "data", "cols"}.issubset({str(value).lower() for value in header}):
        frame = read_data(str(path))
    else:
        frame = pd.read_csv(path)
        timestamp = next((column for column in frame.columns
                          if str(column).lower() in {"timestamp", "date", "time", "datetime"}), None)
        if timestamp is not None:
            parsed = pd.to_datetime(frame.pop(timestamp), errors="raise")
            frame.index = pd.DatetimeIndex(parsed)
    if not hasattr(frame, "columns") or not hasattr(frame, "index"):
        raise ValueError("benchmark loader did not return a tabular time series")
    label_columns = [column for column in frame.columns if str(column).lower() == "label"]
    if len(label_columns) != 1:
        raise ValueError("loaded task must contain exactly one label column")
    label_column = label_columns[0]
    labels = np.asarray(frame[label_column])
    if labels.ndim != 1 or labels.dtype.kind not in "iuf" or labels.dtype.kind == "b":
        raise ValueError("labels must be a one-dimensional real numeric vector")
    if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only finite exact 0/1 values")
    frame = frame.copy()
    frame[label_column] = labels.astype(np.int8, copy=False)
    return frame, label_column
