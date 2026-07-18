import hashlib
import subprocess
from pathlib import Path

import torch

from ts_benchmark.baselines.catch.models.CATCH_model import Flatten_Head
from ts_benchmark.baselines.rsa_msd_catch.models.RSAMSDCATCH_model import RawGateNetwork, RawStructureAdapter
from ts_benchmark.baselines.sdd_msd_catch.SDDMSDCATCH import SDDMSDCATCH
from ts_benchmark.baselines.sdd_msd_catch.models.SDDMSDCATCH_model import (
    FactorizedLinear,
    SDDMSDCATCHModel,
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


def test_sdd_shared_encoder_dual_decoder_smoke():
    torch.manual_seed(19)
    detector = SDDMSDCATCH(**_tiny_config())
    model = SDDMSDCATCHModel(detector.config)
    x = torch.randn(3, detector.config.seq_len, detector.config.c_in)
    events = []
    handles = [
        model.shared_encoder.register_forward_hook(lambda *_: events.append("encoder")),
        model.low_rank_exchange.register_forward_hook(lambda *_: events.append("exchange")),
        model.trend_adapter.register_forward_hook(lambda *_: events.append("trend_adapter")),
        model.residual_adapter.register_forward_hook(lambda *_: events.append("residual_adapter")),
        model.trend_decoder.register_forward_hook(lambda *_: events.append("trend_decoder")),
        model.residual_decoder.register_forward_hook(lambda *_: events.append("residual_decoder")),
    ]
    outputs = model(x)
    for handle in handles:
        handle.remove()

    assert torch.allclose(outputs["trend"] + outputs["residual"], x)
    assert events[:2] == ["encoder", "encoder"]
    assert events.index("exchange") < events.index("trend_adapter") < events.index("trend_decoder")
    assert events.index("exchange") < events.index("residual_adapter") < events.index("residual_decoder")
    trend_decoder_params = {id(parameter) for parameter in model.trend_decoder.parameters()}
    residual_decoder_params = {id(parameter) for parameter in model.residual_decoder.parameters()}
    assert trend_decoder_params.isdisjoint(residual_decoder_params)
    assert not any(isinstance(module, Flatten_Head) for module in model.modules())
    assert all(module.effective_rank <= 64 for module in model.modules() if isinstance(module, FactorizedLinear))
    assert isinstance(model.raw_adapter, RawStructureAdapter)
    assert isinstance(model.raw_gate_network, RawGateNetwork)
    assert torch.count_nonzero(model.raw_adapter.output_projection.weight) == 0
    assert torch.count_nonzero(model.raw_adapter.output_projection.bias) == 0
    assert torch.count_nonzero(model.raw_gate_network.output.weight) == 0
    assert torch.allclose(
        model.raw_gate_network.output.bias,
        torch.full_like(model.raw_gate_network.output.bias, -2.0),
    )
    assert torch.all(outputs["raw_gate"] >= 0)
    assert torch.all(outputs["raw_gate"] <= 0.5)
    expected_gate = 0.5 * torch.sigmoid(torch.tensor(-2.0))
    assert torch.allclose(outputs["raw_gate"].mean(), expected_gate, atol=1e-6)
    assert outputs["x_hat"].shape == x.shape
    assert torch.isfinite(outputs["x_hat"]).all()

    loss = (
        (outputs["x_hat"] - x).square().mean()
        + outputs["trend_dcloss"]
        + outputs["residual_dcloss"]
    )
    loss.backward()
    assert torch.isfinite(loss)
    detector.model = model
    scores, diagnostics = detector._score_batch(x)
    assert scores["total_score"].numel() == x.shape[0] * x.shape[1]
    assert diagnostics["raw_gate"].shape == x.shape
    assert all(torch.isfinite(value).all() for value in scores.values())
    assert model.module_parameter_counts()["total"] == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def test_sdd_does_not_modify_existing_baselines():
    paths = [
        ROOT / "ts_benchmark" / "baselines" / "msd_catch",
        ROOT / "ts_benchmark" / "baselines" / "rsa_msd_catch",
    ]
    before = [_tree_digest(path) for path in paths]
    _ = SDDMSDCATCHModel(SDDMSDCATCH(**_tiny_config()).config)(torch.randn(1, 32, 3))
    assert [_tree_digest(path) for path in paths] == before
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", "ts_benchmark/baselines/catch"],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0
