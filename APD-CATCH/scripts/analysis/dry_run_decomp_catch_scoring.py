"""CPU-only dry-run for fixed CATCH decomposition scoring; writes no result files."""

from __future__ import annotations

import copy
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ts_benchmark.baselines.catch.CATCH import CATCH
from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel
from ts_benchmark.baselines.decomp_catch.scoring import CATCHDecompositionScorer


def _frame(length: int, offset: float = 0.0) -> pd.DataFrame:
    values = np.arange(length * 2, dtype=np.float32).reshape(length, 2) / 10.0
    return pd.DataFrame(values + offset, columns=["channel_0", "channel_1"])


def _detector() -> CATCH:
    detector = CATCH(
        seq_len=8,
        patch_size=4,
        patch_stride=4,
        inference_patch_size=4,
        inference_patch_stride=4,
        cf_dim=4,
        d_model=4,
        d_ff=8,
        e_layers=1,
        n_heads=1,
        head_dim=4,
        dropout=0.0,
        head_dropout=0.0,
        batch_size=1,
        affine=False,
    )
    detector.device = torch.device("cpu")
    train = _frame(24, offset=-1.0)
    detector.detect_hyper_param_tune(train)
    detector.config.task_name = "anomaly_detection"
    detector.config.c_in = 2
    detector.model = CATCHModel(detector.config).eval()
    detector.scaler.fit(train.values)
    detector.early_stopping = SimpleNamespace(
        check_point=copy.deepcopy(detector.model.state_dict())
    )
    return detector


def main() -> None:
    scoring_seed = 20260717
    detector = _detector()
    scorer = CATCHDecompositionScorer(detector)
    reference = _frame(24, offset=-0.5)
    test = _frame(18, offset=0.25)
    stats = scorer.fit_normalization_stats(reference, "validation", scoring_seed)
    parameter_count_before = sum(parameter.numel() for parameter in detector.model.parameters())

    torch.manual_seed(scoring_seed)
    with redirect_stdout(io.StringIO()):
        original_score, _ = detector.detect_score(test)
    result = scorer.score_dataframe(test, stats, scoring_seed)
    parameter_count_after = sum(parameter.numel() for parameter in detector.model.parameters())
    max_equivalence_error = float(
        np.max(np.abs(result["original_score"] - original_score))
    )

    for name in (
        "original_score",
        "time_score",
        "slow_score",
        "fast_score",
        "fusion_score",
    ):
        print(f"{name} shape: {result[name].shape}")
    print(f"time_index range: {result['time_index'][0]}..{result['time_index'][-1]}")
    print(f"dropped_tail_length: {result['dropped_tail_length']}")
    print(f"moving-average window: {result['window_size']}")
    print(f"model parameter count before/after: {parameter_count_before}/{parameter_count_after}")
    print(f"maximum original-score equivalence error: {max_equivalence_error:.8g}")


if __name__ == "__main__":
    main()
