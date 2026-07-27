"""Small archive fixtures for result-registration unit tests only."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


FORMAL_TYPEFUSION_OUTPUT_NAME = "TypeFusionCATCH"


def write_archive(
    path: Path,
    model_name: str,
    data_name: str,
    params: Mapping[str, Any],
    *,
    seed: int = 2021,
    auc_roc: float = 0.8,
    auc_pr: float = 0.7,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "strategy_args": json.dumps({"strategy_name": "unfixed_detect_score", "seed": seed}),
                "model_params": json.dumps(dict(params), sort_keys=True),
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
                "R_AUC_ROC": 0.75,
                "R_AUC_PR": 0.65,
                "VUS_ROC": 0.73,
                "VUS_PR": 0.63,
                "file_name": data_name,
                "fit_time": 1.2,
                "inference_time": 0.3,
            }
        ]
    )
    payload = frame.to_csv(index=False).encode("utf-8")
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("result.csv")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def write_report(
    path: Path,
    model_name: str,
    params: Mapping[str, Any],
    *,
    seed: int = 2021,
    auc_roc: float = 0.8,
) -> None:
    column = model_name + ";" + json.dumps(dict(params), sort_keys=True)
    pd.DataFrame(
        [
            {
                "strategy_args": json.dumps({"strategy_name": "unfixed_detect_score", "seed": seed}),
                "metric_name": "auc_roc",
                column: auc_roc,
            }
        ]
    ).to_csv(path, index=False)
