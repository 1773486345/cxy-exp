import copy
import math

import numpy as np
import pytest
import torch

from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
from ts_benchmark.baselines.mnrf_lite_v2.fusion import AlternatingAffineFlow
from ts_benchmark.baselines.mnrf_lite_v2.layout import FeatureLayout
from ts_benchmark.baselines.mnrf_lite_v2.training import GaussianAuxiliaryDecoder, MNRFTrainer
from ts_benchmark.baselines.mnrf_lite_v2.views import FourViewEncoder


LAYOUTS = ((8, 8, 8), (4, 4, 4), (3, 5, 7), (8, 16, 32))


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("e_dim,p_dim,r_dim", LAYOUTS)
def test_layout_encoder_auxiliary_and_flow_roundtrip(e_dim, p_dim, r_dim):
    torch.manual_seed(71)
    config = MNRFConfig.from_kwargs(evolution_dim=e_dim, pattern_dim=p_dim, relation_dim=r_dim)
    layout = config.layout
    encoder = FourViewEncoder(variables=2, layout=layout)
    values = torch.randn(2, 40, 2)
    output = encoder.encode_views(values)
    assert {key: output[key].shape[-1] for key in ("state", "evolution", "pattern", "relation")} == {
        "state": 1, "evolution": e_dim, "pattern": p_dim, "relation": r_dim,
    }
    assert output["features"].shape == (2, 2, 1 + e_dim + p_dim + r_dim)
    assert layout.to_dict()["slices"] == {
        "S": [0, 1], "E": [1, 1 + e_dim],
        "P": [1 + e_dim, 1 + e_dim + p_dim],
        "R": [1 + e_dim + p_dim, layout.total_dim],
    }

    auxiliary = GaussianAuxiliaryDecoder(2, layout=layout)
    identity = encoder.point.variable_identity
    for task, task_index in (("evolution", 0), ("pattern", 1), ("relation", 2)):
        z, target, relative = encoder.encode_pretrain(values, task=task)
        loss = auxiliary.loss(z, target, relative, identity, task_index)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in auxiliary.parameters())

    flow = AlternatingAffineFlow(layout=layout)
    for block in flow.blocks:
        for layer in block.outputs.values():
            torch.nn.init.normal_(layer.weight, std=0.003)
            torch.nn.init.normal_(layer.bias, std=0.002)
    point = torch.randn(1, 2, layout.total_dim)
    result = flow(point, identity)
    restored, inverse_logdet = flow.inverse(result["z"], identity)
    assert torch.allclose(restored, point, atol=2e-5, rtol=2e-5)
    assert torch.allclose(inverse_logdet, -result["logdet"], atol=2e-5, rtol=2e-5)
    expected = (0.5 * result["z"].square().sum(-1)
                + layout.total_dim / 2 * math.log(2 * math.pi) - result["logdet"])
    assert torch.allclose(result["nll"], expected)


def test_default_layout_matches_legacy_view_dim_semantics():
    assert MNRFConfig().layout.to_dict() == FeatureLayout(8, 8, 8).to_dict()
    assert MNRFConfig(view_dim=4).layout.to_dict() == FeatureLayout(4, 4, 4).to_dict()
    config = MNRFConfig.from_kwargs(view_dim=4, evolution_dim=3)
    assert config.layout.to_dict() == FeatureLayout(3, 4, 4).to_dict()
    torch.manual_seed(5)
    legacy = FourViewEncoder(variables=2, view_dim=8)
    torch.manual_seed(5)
    explicit = FourViewEncoder(variables=2, layout=FeatureLayout(8, 8, 8))
    assert all(torch.equal(legacy.state_dict()[key], explicit.state_dict()[key])
               for key in legacy.state_dict())
    sample = torch.randn(1, 40, 2)
    assert torch.equal(legacy.encode_views(sample)["features"], explicit.encode_views(sample)["features"])


@pytest.mark.parametrize("e_dim,p_dim,r_dim", LAYOUTS)
def test_stage2_freezes_encoder_and_uses_layout_feature_width(tmp_path, e_dim, p_dim, r_dim):
    config = MNRFConfig(seq_len=32, batch_size=2, stage1_epochs=1, stage2_epochs=1,
                        evolution_dim=e_dim, pattern_dim=p_dim, relation_dim=r_dim)
    trainer = MNRFTrainer(config, variables=2, device=torch.device("cpu"))
    assert trainer.feature_dim == 1 + e_dim + p_dim + r_dim
    for parameter in trainer.encoder.parameters():
        parameter.requires_grad_(False)
    encoder_before = copy.deepcopy(trainer.encoder.state_dict())
    features = np.random.default_rng(4).normal(size=(4, 2, trainer.feature_dim)).astype(np.float32)
    trainer.fixed_feature.fit(features[:2])
    trainer.stage2(features[:2], features[2:], tmp_path)
    assert all(torch.equal(encoder_before[key], value) for key, value in trainer.encoder.state_dict().items())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in trainer.encoder.parameters())


def test_layout_rejects_nonpositive_and_inconsistent_serialization():
    with pytest.raises(ValueError, match="positive integer"):
        FeatureLayout(0, 8, 8)
    broken = FeatureLayout(3, 5, 7).to_dict()
    broken["F"] += 1
    with pytest.raises(ValueError, match="F mismatch|layout is inconsistent"):
        FeatureLayout.from_dict(broken)
