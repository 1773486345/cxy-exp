import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.catch.CATCH import CATCH
from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel, Flatten_Head
from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_criterion, frequency_loss
from ts_benchmark.baselines.ra_msd_catch.RAMSDCATCH import RAMSDCATCH
from ts_benchmark.baselines.ra_msd_catch.models.RAMSDCATCH_model import (
    CATCHReconstructionHead,
    RAMSDCATCHModel,
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


def _catch_latent(model, x):
    values = model.revin_layer(x, "norm").permute(0, 2, 1)
    values = torch.fft.fft(values)
    real = values.real.unfold(-1, model.patch_size, model.patch_stride).permute(0, 2, 1, 3)
    imag = values.imag.unfold(-1, model.patch_size, model.patch_stride).permute(0, 2, 1, 3)
    batch_size, patch_num, channels, _ = real.shape
    real = real.reshape(batch_size * patch_num, channels, model.patch_size)
    imag = imag.reshape(batch_size * patch_num, channels, model.patch_size)
    patches = torch.cat((real, imag), dim=-1)
    return model.frequency_transformer(patches, model.mask_generator(patches))


def _errors(actual, expected):
    delta = (actual - expected).abs()
    return float(delta.max()), float(delta.mean())


def _assert_equivalent(actual, expected):
    maximum, mean = _errors(actual, expected)
    assert maximum <= 1e-6, (maximum, mean)
    assert mean <= 1e-7, (maximum, mean)


def test_ra_structure_and_bounded_auxiliary_modulation():
    torch.manual_seed(17)
    model = RAMSDCATCHModel(RAMSDCATCH(**_tiny_config()).config)
    x = torch.randn(2, 32, 3)
    outputs = model(x)

    assert torch.allclose(outputs["trend"] + outputs["residual"], x)
    assert outputs["h_raw"].shape == outputs["h_trend"].shape == outputs["h_residual"].shape
    assert outputs["h_raw"].dtype == outputs["h_trend"].dtype == outputs["h_residual"].dtype
    assert outputs["h_raw"].device == outputs["h_trend"].device == outputs["h_residual"].device
    assert torch.equal(outputs["h_fused"], outputs["h_raw"])
    assert torch.isfinite(outputs["x_hat"]).all()
    assert torch.isfinite(outputs["raw_dcloss"])
    assert model.alpha_trend_param.item() == model.alpha_residual_param.item() == 0.0
    assert abs(float(model.alpha_trend)) <= 0.25
    assert abs(float(model.alpha_residual)) <= 0.25
    with torch.no_grad():
        model.alpha_trend_param.fill_(100.0)
        model.alpha_residual_param.fill_(-100.0)
    assert 0.249 < float(model.alpha_trend) <= 0.25
    assert -0.25 <= float(model.alpha_residual) < -0.249

    assert isinstance(model.raw_reconstruction_head, CATCHReconstructionHead)
    assert sum(isinstance(module, CATCHReconstructionHead) for module in model.modules()) == 1
    assert sum(isinstance(module, Flatten_Head) for module in model.modules()) == 2
    assert not any(isinstance(module, Flatten_Head) for module in model.trend_encoder.modules())
    assert not any(isinstance(module, Flatten_Head) for module in model.residual_encoder.modules())
    assert {id(parameter) for parameter in model.trend_encoder.parameters()}.isdisjoint(
        {id(parameter) for parameter in model.residual_encoder.parameters()}
    )
    assert {id(parameter) for parameter in model.raw_encoder.parameters()}.isdisjoint(
        {id(parameter) for parameter in model.trend_encoder.parameters()}
    )
    for raw_parameter, trend_parameter, residual_parameter in zip(
        model.raw_encoder.parameters(),
        model.trend_encoder.parameters(),
        model.residual_encoder.parameters(),
    ):
        assert torch.equal(raw_parameter, trend_parameter)
        assert torch.equal(raw_parameter, residual_parameter)
    assert {id(parameter) for parameter in model.raw_reconstruction_head.head_f1.parameters()}.isdisjoint(
        {id(parameter) for parameter in model.raw_reconstruction_head.head_f2.parameters()}
    )
    assert model.module_parameter_counts()["total"] == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_ra_initial_catch_path_equivalence_for_latent_output_loss_and_score():
    torch.manual_seed(23)
    config = CATCH(**_tiny_config()).config
    catch = CATCHModel(config).eval()
    ra = RAMSDCATCHModel(config).eval()
    ra.load_from_catch_model(catch)
    x = torch.randn(2, config.seq_len, config.c_in)

    state = torch.get_rng_state()
    torch.set_rng_state(state)
    catch_latent, catch_dcloss = _catch_latent(catch, x)
    torch.set_rng_state(state)
    ra_outputs = ra(x)
    _assert_equivalent(ra_outputs["h_raw"], catch_latent)
    _assert_equivalent(ra_outputs["raw_dcloss"], catch_dcloss)

    state = torch.get_rng_state()
    torch.set_rng_state(state)
    catch_output, catch_complex, catch_dcloss = catch(x)
    torch.set_rng_state(state)
    ra_outputs = ra(x)
    _assert_equivalent(ra_outputs["x_hat"], catch_output)
    _assert_equivalent(ra_outputs["output_complex"], catch_complex)
    _assert_equivalent(ra_outputs["raw_dcloss"], catch_dcloss)

    criterion = nn.MSELoss()
    auxi_loss = frequency_loss(config)
    state = torch.get_rng_state()
    torch.set_rng_state(state)
    catch_output, catch_complex, catch_dcloss = catch(x)
    catch_loss = (
        criterion(catch_output, x)
        + config.dc_lambda * catch_dcloss
        + config.auxi_lambda * auxi_loss(catch_complex, catch.revin_layer(x, "transform"))
    )
    torch.set_rng_state(state)
    detector = RAMSDCATCH(**_tiny_config())
    detector.model = ra
    ra_loss, _ = detector._catch_loss(x)
    _assert_equivalent(ra_loss, catch_loss)

    score_criterion = frequency_criterion(config)
    catch_score = (
        nn.MSELoss(reduction="none")(x, catch_output).mean(dim=-1)
        + config.score_lambda * score_criterion(x, catch_output).mean(dim=-1)
    )
    torch.set_rng_state(state)
    ra_score = detector._score_batch(x)
    _assert_equivalent(ra_score, catch_score)
    assert torch.isfinite(ra_score).all()

    detector.scaler = StandardScaler().fit(x.reshape(-1, config.c_in).numpy())
    detector.model = ra.eval()
    total_score, reference_score = detector.detect_score(
        pd.DataFrame(np.random.default_rng(5).normal(size=(64, config.c_in)))
    )
    assert set(detector.last_scores) == {"total_score"}
    assert np.isfinite(total_score).all()
    np.testing.assert_array_equal(total_score, reference_score)


def test_ra_gradients_are_finite_at_zero_and_reach_auxiliaries_after_modulation():
    torch.manual_seed(31)
    model = RAMSDCATCHModel(RAMSDCATCH(**_tiny_config()).config).train()
    x = torch.randn(2, 32, 3)
    outputs = model(x)
    initial_loss = (outputs["x_hat"] - x).square().mean() + outputs["raw_dcloss"]
    initial_loss.backward()
    assert model.alpha_trend_param.grad is not None
    assert model.alpha_residual_param.grad is not None
    assert torch.isfinite(model.alpha_trend_param.grad)
    assert torch.isfinite(model.alpha_residual_param.grad)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.raw_encoder.parameters()
    )

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.alpha_trend_param.fill_(0.1)
        model.alpha_residual_param.fill_(-0.1)
    outputs = model(x)
    modulated_loss = (outputs["x_hat"] - x).square().mean() + outputs["raw_dcloss"]
    modulated_loss.backward()
    for module in (
        model.trend_adapter,
        model.residual_adapter,
        model.trend_encoder,
        model.residual_encoder,
    ):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
    assert torch.isfinite(modulated_loss)


def test_ra_validation_uses_only_final_reconstruction_mse():
    class ValidationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.raw_dcloss = torch.tensor(0.0)
            self.frequency = torch.tensor(0.0)

        def forward(self, input_batch):
            return {
                "x_hat": input_batch + 0.5,
                "raw_dcloss": self.raw_dcloss,
                "output_complex": self.frequency,
            }

    detector = RAMSDCATCH(**_tiny_config())
    detector.device = torch.device("cpu")
    detector.model = ValidationModel()
    detector._catch_loss = lambda _: (_ for _ in ()).throw(AssertionError("validation used training loss"))
    batch = torch.zeros(2, 32, 3)

    first_loss = detector.detect_validate([(batch, None)])
    detector.model.raw_dcloss = torch.tensor(1e9)
    detector.model.frequency = torch.tensor(1e9)
    second_loss = detector.detect_validate([(batch, None)])

    assert first_loss == second_loss == 0.25
    assert detector.model.training


def test_ra_mask_optimizer_steps_before_current_forward():
    class RecordingOptimizer:
        def __init__(self, name, events):
            self.name = name
            self.events = events

        def step(self):
            self.events.append(f"{self.name}.step")

        def zero_grad(self):
            self.events.append(f"{self.name}.zero_grad")

    detector = RAMSDCATCH(**_tiny_config())
    events = []
    detector.optimizer = RecordingOptimizer("main", events)
    detector.optimizerM = RecordingOptimizer("mask", events)
    loss = torch.tensor(1.0, requires_grad=True)
    loss.register_hook(lambda _: events.append("loss.backward"))

    def catch_loss(_):
        events.append("forward")
        return loss, {"reconstruction": 0.0}

    detector._catch_loss = catch_loss

    detector._training_step(torch.zeros(2, 32, 3), step=1, train_steps=1, mask_update_interval=1)

    assert events == [
        "main.zero_grad",
        "mask.step",
        "mask.zero_grad",
        "forward",
        "loss.backward",
        "main.step",
    ]


def test_ra_real_mask_update_precedes_next_forward_without_autograd_mismatch():
    torch.manual_seed(41)
    config = _tiny_config()
    config.update({"c_in": 2, "seq_len": 32, "patch_size": 16, "patch_stride": 16})
    detector = RAMSDCATCH(**config)
    detector.device = torch.device("cpu")
    detector.model = RAMSDCATCHModel(detector.config).to(detector.device).train()
    main_parameters = [
        parameter
        for name, parameter in detector.model.named_parameters()
        if "mask_generator" not in name
    ]
    mask_parameters = [
        parameter
        for name, parameter in detector.model.named_parameters()
        if "mask_generator" in name
    ]
    detector.optimizer = torch.optim.Adam(main_parameters, lr=1e-3)
    detector.optimizerM = torch.optim.Adam(mask_parameters, lr=1e-3)
    first_batch = torch.randn(2, 32, 2)
    second_batch = torch.randn(2, 32, 2)

    first_loss, _ = detector._training_step(
        first_batch,
        step=1,
        train_steps=2,
        mask_update_interval=2,
    )
    first_gradients = [parameter.grad for parameter in mask_parameters if parameter.grad is not None]
    assert torch.isfinite(first_loss)
    assert first_gradients
    assert all(torch.isfinite(gradient).all() for gradient in first_gradients)
    assert any(torch.count_nonzero(gradient) for gradient in first_gradients)

    masks_before_second_step = [parameter.detach().clone() for parameter in mask_parameters]
    original_forward = detector.model.forward
    mask_step_happened_before_forward = []

    def forward_after_mask_step(input_batch):
        mask_step_happened_before_forward.append(
            any(
                not torch.equal(parameter.detach(), before)
                for parameter, before in zip(mask_parameters, masks_before_second_step)
            )
        )
        return original_forward(input_batch)

    detector.model.forward = forward_after_mask_step
    second_loss, _ = detector._training_step(
        second_batch,
        step=2,
        train_steps=2,
        mask_update_interval=2,
    )
    second_gradients = [parameter.grad for parameter in mask_parameters if parameter.grad is not None]

    assert mask_step_happened_before_forward == [True]
    assert torch.isfinite(second_loss)
    assert second_gradients
    assert all(torch.isfinite(gradient).all() for gradient in second_gradients)
    assert any(torch.count_nonzero(gradient) for gradient in second_gradients)


def test_ra_does_not_modify_frozen_baselines():
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "--",
            "ts_benchmark/baselines/catch",
            "ts_benchmark/baselines/msd_catch",
            "ts_benchmark/baselines/bhd_msd_catch",
        ],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0
