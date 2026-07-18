import hashlib
import subprocess
from pathlib import Path

import torch

from ts_benchmark.baselines.rsa_msd_catch.RSAMSDCATCH import RSAMSDCATCH
from ts_benchmark.baselines.rsa_msd_catch.models.RSAMSDCATCH_model import (
    RSAMSDCATCHModel,
    SharedCATCHBackbone,
)


ROOT = Path(__file__).resolve().parents[1]


def _tiny_config():
    return {
        "seq_len": 32,
        "patch_size": 8,
        "patch_stride": 8,
        "inference_patch_size": 8,
        "inference_patch_stride": 1,
        "cf_dim": 4,
        "d_model": 8,
        "d_ff": 16,
        "e_layers": 1,
        "n_heads": 1,
        "head_dim": 4,
        "batch_size": 2,
        "dropout": 0.0,
        "head_dropout": 0.0,
        "c_in": 3,
    }


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path.rglob("*.py")):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def test_rsa_msd_catch_forward_backward_and_invariants():
    torch.manual_seed(11)
    detector = RSAMSDCATCH(**_tiny_config())
    model = RSAMSDCATCHModel(detector.config)
    x = torch.randn(3, detector.config.seq_len, detector.config.c_in)
    calls = []
    handle = model.shared_catch_backbone.frequency_transformer.register_forward_hook(
        lambda *_: calls.append(1)
    )
    outputs = model(x)
    handle.remove()

    assert torch.allclose(outputs["trend"] + outputs["residual"], x)
    assert len(calls) == 2
    assert isinstance(model.shared_catch_backbone, SharedCATCHBackbone)
    trend_params = {id(parameter) for parameter in model.trend_adapter.parameters()}
    residual_params = {id(parameter) for parameter in model.residual_adapter.parameters()}
    assert trend_params.isdisjoint(residual_params)
    assert torch.count_nonzero(model.trend_adapter.up.weight) == 0
    assert torch.count_nonzero(model.residual_adapter.up.weight) == 0
    assert torch.count_nonzero(model.raw_adapter.output_projection.weight) == 0
    assert not any(isinstance(module, SharedCATCHBackbone) for module in model.raw_adapter.modules())
    expected_gate = 0.5 * torch.sigmoid(torch.tensor(-2.0))
    assert torch.all(outputs["raw_gate"] >= 0)
    assert torch.all(outputs["raw_gate"] <= 0.5)
    assert torch.allclose(outputs["raw_gate"].mean(), expected_gate, atol=1e-6)
    assert outputs["x_hat"].shape == x.shape
    assert torch.isfinite(outputs["x_hat"]).all()
    assert torch.isfinite(outputs["raw_correction"]).all()

    loss = (
        (outputs["x_hat"] - x).square().mean()
        + outputs["trend_dcloss"]
        + outputs["residual_dcloss"]
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
    counts = model.module_parameter_counts()
    assert counts["total"] == sum(
        value for name, value in counts.items() if name not in {"total", "other"}
    ) + counts["other"]


def test_rsa_model_does_not_modify_existing_baselines():
    msd_path = ROOT / "ts_benchmark" / "baselines" / "msd_catch"
    before = _tree_digest(msd_path)
    model = RSAMSDCATCHModel(RSAMSDCATCH(**_tiny_config()).config)
    _ = model(torch.randn(1, 32, 3))
    assert _tree_digest(msd_path) == before
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", "ts_benchmark/baselines/catch"],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0
