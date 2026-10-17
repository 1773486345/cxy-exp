import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
from ts_benchmark.baselines.mnrf_lite_v2.fusion import AlternatingAffineFlow
from ts_benchmark.baselines.mnrf_lite_v2.backbone import CausalResidualTCN
from ts_benchmark.baselines.mnrf_lite_v2.training import (
    GaussianAuxiliaryDecoder,
    MNRFTrainer,
    purged_ranges,
    seed_all,
)
from ts_benchmark.baselines.mnrf_lite_v2.views import FourViewEncoder


@pytest.fixture(scope="session", autouse=True)
def fixed_cpu_threads():
    torch.set_num_threads(1)
    yield


def test_config_defaults_and_purged_ranges():
    config = MNRFConfig()
    assert (config.seq_len, config.pattern_len, config.hidden_dim) == (128, 32, 32)
    ranges = purged_ranges(1534, 128)
    assert ranges["gap"] == 127
    assert ranges["window_counts"] == {"train": 897, "validation": 1, "calibration": 1}
    assert ranges["validation"][0] - ranges["train"][1] == 127
    assert ranges["calibration"][0] - ranges["validation"][1] == 127
    with pytest.raises(ValueError, match="cannot produce"):
        purged_ranges(1533, 128)


def test_four_view_order_and_causal_target_information():
    seed_all(2021)
    encoder = FourViewEncoder(variables=2)
    values = torch.randn(3, 40, 2)
    output = encoder.encode_views(values)
    assert output["features"].shape == (3, 2, 25)
    assert torch.equal(output["features"][..., 0], output["state"][..., 0])
    changed = values.clone()
    changed[:, -1, 0] += 1000
    target = encoder.encode_views(changed)
    assert torch.allclose(output["evolution"][:, 0], target["evolution"][:, 0])
    changed = values.clone()
    changed[:, :, 0] += 1000
    target = encoder.encode_views(changed)
    assert torch.allclose(output["relation"][:, 0], target["relation"][:, 0], atol=1e-6, rtol=1e-6)


def test_tcn_outputs_do_not_depend_on_future_and_nonmultiple_batch():
    seed_all(9)
    tcn = CausalResidualTCN()
    values = torch.randn(3, 40, 2, 32)
    output = tcn(values)
    changed = values.clone()
    changed[:, 21:] += 100
    changed_output = tcn(changed)
    assert output.shape == (3, 40, 2, 32)
    assert torch.allclose(output[:, :21], changed_output[:, :21], atol=1e-6, rtol=1e-6)


def test_pattern_masked_content_cannot_leak():
    seed_all(10)
    encoder = FourViewEncoder(variables=2)
    values = torch.randn(2, 40, 2)
    starts = torch.tensor([[2, 12], [20, 7]])
    valid = torch.ones_like(values, dtype=torch.bool)
    z, target, positions = encoder.encode_pretrain(values, valid, "pattern", starts)
    changed = values.clone()
    for n in range(2):
        for d in range(2):
            start = int(starts[n, d])
            changed[n, 8 + start:8 + start + 8, d] += 1000
    z_changed, target_changed, positions_changed = encoder.encode_pretrain(changed, valid, "pattern", starts)
    assert torch.equal(z, z_changed)
    assert not torch.equal(target, target_changed)
    assert torch.equal(positions, positions_changed)
    assert torch.all((positions >= 0) & (positions <= 1))


def test_flow_inverse_logdet_and_numerical_jacobian():
    seed_all(21)
    flow = AlternatingAffineFlow()
    for block in flow.blocks:
        for layer in block.outputs.values():
            torch.nn.init.normal_(layer.weight, std=0.015)
            torch.nn.init.normal_(layer.bias, std=0.01)
    value = torch.randn(1, 1, 25, requires_grad=True)
    identity = torch.randn(1, 32)
    output = flow(value, identity)
    restored, inverse_logdet = flow.inverse(output["z"], identity)
    assert torch.allclose(restored, value, atol=2e-5, rtol=2e-5)
    assert torch.allclose(inverse_logdet, -output["logdet"], atol=2e-5, rtol=2e-5)
    jacobian = torch.autograd.functional.jacobian(
        lambda item: flow._couple(item, identity)[0].reshape(-1), value
    ).reshape(25, 25)
    numerical = torch.linalg.slogdet(jacobian).logabsdet
    assert torch.allclose(numerical, output["logdet"].sum(), atol=2e-4, rtol=2e-4)
    expected_nll = 0.5 * output["z"].square().sum(-1) + 25 / 2 * np.log(2 * np.pi) - output["logdet"]
    assert torch.allclose(output["nll"], expected_nll)


def test_first_coupling_updates_mechanism_b_only():
    flow = AlternatingAffineFlow()
    for layer in flow.blocks[0].outputs.values():
        torch.nn.init.normal_(layer.weight, std=0.02)
        torch.nn.init.normal_(layer.bias, std=0.03)
    values = torch.randn(3, 2, 25)
    identity = flow._identity(torch.randn(2, 32), 3, 2)
    variable = flow.variable_maps[0](identity)
    parameters = flow.blocks[0].condition(values, variable, flow.fusion_identity, flow.fusion_identity)
    updated = values.clone()
    for name in flow.blocks[0].target_names:
        start, end = {"E": (1, 9), "P": (9, 17)}[name]
        raw = parameters[name]
        scale, shift = torch.tanh(raw[..., :end-start]), raw[..., end-start:]
        updated[..., start:end] = values[..., start:end] * torch.exp(scale) + shift
    assert torch.equal(updated[..., 0], values[..., 0])
    assert torch.equal(updated[..., 17:25], values[..., 17:25])
    assert not torch.equal(updated[..., 1:17], values[..., 1:17])


def test_stage_one_gradients_and_stage_two_freezing():
    seed_all(4)
    config = MNRFConfig(seq_len=32, batch_size=8, stage1_epochs=1, stage2_epochs=1)
    trainer = MNRFTrainer(config, variables=2, device=torch.device("cpu"))
    windows = torch.randn(4, 32, 2)
    loss = trainer._pretrain_batch(windows, "evolution")
    loss.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainer.encoder.parameters())
    assert all(parameter.grad is None for parameter in trainer.flow.parameters())
    for parameter in trainer.encoder.parameters():
        parameter.grad = None
    before = {name: value.detach().clone() for name, value in trainer.encoder.state_dict().items()}
    features = np.random.default_rng(2).normal(size=(12, 2, 25)).astype(np.float32)
    trainer.fixed_feature.fit(features[:8])
    with tempfile.TemporaryDirectory() as directory:
        trainer.stage2(features[:8], features[8:], directory)
    assert all(torch.equal(before[name], value) for name, value in trainer.encoder.state_dict().items())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in trainer.encoder.parameters())


def test_small_fit_score_alignment_checkpoint_roundtrip():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(430, 3)), columns=("x", "y", "z"))
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/run"
        config = dict(seq_len=32, pattern_len=32, batch_size=32, stage1_epochs=1,
                      stage2_epochs=1, patience=1, cache_chunk_size=128,
                      device="cpu", output_dir=path)
        model = MNRFLiteV2(**config)
        model.detect_fit(frame)
        scored = model.score_with_metadata(frame.iloc[:64])
        assert scored["scores"].shape == (33,)
        assert np.array_equal(scored["time_indices"], np.arange(31, 64))
        assert not scored["valid_mask"][:31].any()
        assert scored["valid_mask"].sum() == 33
        assert np.isfinite(scored["scores"]).all()
        deployed = MNRFLiteV2(device="cpu").load_checkpoint(f"{path}/deployment.pt")
        rescored = deployed.score_with_metadata(frame.iloc[:64])
        assert np.allclose(scored["scores"], rescored["scores"], atol=1e-6, rtol=1e-6)
        model.config.cache_chunk_size = 7
        rechunked = model.score_with_metadata(frame.iloc[:64])
        assert np.allclose(scored["scores"], rechunked["scores"], atol=1e-5, rtol=1e-5)
        with pytest.raises(ValueError, match="variable names/order"):
            deployed.detect_score(frame.iloc[:64].loc[:, ["z", "y", "x"]])
        resumed = MNRFLiteV2(**{**config, "resume_from": path})
        resumed.detect_fit(frame)
        resumed_scores = resumed.score_with_metadata(frame.iloc[:64])
        assert np.allclose(scored["scores"], resumed_scores["scores"], atol=1e-6, rtol=1e-6)
        changed = frame.copy()
        changed.iloc[0, 0] += 1
        with pytest.raises(ValueError, match="resume rejected"):
            MNRFLiteV2(**{**config, "resume_from": path}).detect_fit(changed)


def test_benchmark_model_import():
    from ts_benchmark.models import get_models

    models = get_models({"models": [{"model_name": "mnrf_lite_v2.MNRFLiteV2", "model_hyper_params": {"device": "cpu"}}]})
    assert models[0].model_name == "MNRFLiteV2"
    assert models[0]().config.device == "cpu"

