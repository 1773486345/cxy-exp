import subprocess
from pathlib import Path

import numpy as np
import torch

from ts_benchmark.baselines.msd_catch.MSDCATCH import MSDCATCH
from ts_benchmark.baselines.msd_catch.models.MSDCATCH_model import MSDCATCHModel


ROOT = Path(__file__).resolve().parents[1]


def _tiny_config():
    return {
        "seq_len": 32,
        "patch_size": 8,
        "patch_stride": 8,
        "inference_patch_size": 8,
        "inference_patch_stride": 1,
        "cf_dim": 4,
        "d_model": 4,
        "d_ff": 8,
        "e_layers": 1,
        "n_heads": 1,
        "head_dim": 4,
        "batch_size": 2,
        "dropout": 0.0,
        "head_dropout": 0.0,
        "c_in": 3,
    }


def test_msd_catch_decomposition_forward_backward_and_scores():
    torch.manual_seed(7)
    config = MSDCATCH(**_tiny_config()).config
    model = MSDCATCHModel(config)
    x = torch.randn(3, config.seq_len, config.c_in)
    outputs = model(x)

    assert outputs["trend"].shape == x.shape
    assert outputs["residual"].shape == x.shape
    assert outputs["scale_weights"].shape == (x.shape[0], x.shape[2], 3)
    assert torch.allclose(outputs["scale_weights"].sum(dim=-1), torch.ones_like(x[:, 0, :]))
    assert torch.allclose(outputs["trend"] + outputs["residual"], x)
    assert torch.allclose(outputs["trend_hat"] + outputs["residual_hat"], outputs["x_hat"])
    assert all(kernel % 2 == 1 and kernel <= config.seq_len for kernel in outputs["kernels"])

    trend_params = {id(parameter) for parameter in model.trend_branch.parameters()}
    residual_params = {id(parameter) for parameter in model.residual_branch.parameters()}
    assert trend_params.isdisjoint(residual_params)

    loss = (
        (outputs["x_hat"] - x).square().mean()
        + outputs["trend_dcloss"]
        + outputs["residual_dcloss"]
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert torch.isfinite(loss)

    detector = MSDCATCH(**_tiny_config())
    detector.model = model
    detector.reference_score_stats = {
        name: (0.0, 1.0)
        for name in ("total_score", "trend_score", "residual_score")
    }
    detector.reference_delta_thresholds = {
        "trend_delta_threshold": 0.0,
        "residual_delta_threshold": 0.0,
    }
    scores = detector._fuse_scores(
        {
            name: values.detach().cpu().numpy().reshape(-1)
            for name, values in detector._score_batch(x).items()
        }
    )
    assert set(scores) == {
        "total_score",
        "trend_score",
        "residual_score",
        "fixed_fusion_score",
        "anchored_fusion_score",
        "fusion_score",
    }
    assert len({len(values) for values in scores.values()}) == 1
    assert all(torch.isfinite(torch.as_tensor(values)).all() for values in scores.values())
    assert np.all(scores["anchored_fusion_score"] >= scores["total_score"])


def test_original_catch_directory_is_unmodified():
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", "ts_benchmark/baselines/catch"],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0
