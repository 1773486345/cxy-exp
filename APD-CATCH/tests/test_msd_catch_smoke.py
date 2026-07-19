import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import RandomSampler, SequentialSampler

from ts_benchmark.baselines.msd_catch.MSDCATCH import MSDCATCH
from ts_benchmark.baselines.msd_catch.models.MSDCATCH_model import MSDCATCHModel
from ts_benchmark.baselines.utils import anomaly_detection_data_provider


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


def test_msd_numpy_scaled_loaders_match_dataframe_loaders():
    config = _tiny_config()
    train = pd.DataFrame(
        np.arange(96 * config["c_in"], dtype=np.float64).reshape(96, config["c_in"]),
        columns=[f"feature_{index}" for index in range(config["c_in"])],
    )
    valid = pd.DataFrame(
        np.arange(96 * config["c_in"], 160 * config["c_in"], dtype=np.float64).reshape(
            64, config["c_in"]
        ),
        columns=train.columns,
    )
    test = pd.DataFrame(
        np.arange(160 * config["c_in"], 256 * config["c_in"], dtype=np.float64).reshape(
            96, config["c_in"]
        ),
        columns=train.columns,
    )
    scaler = StandardScaler().fit(train.values)
    legacy_train = pd.DataFrame(scaler.transform(train.values), columns=train.columns, index=train.index)
    legacy_valid = pd.DataFrame(scaler.transform(valid.values), columns=valid.columns, index=valid.index)
    legacy_test = pd.DataFrame(scaler.transform(test.values), columns=test.columns, index=test.index)
    optimized_train = np.ascontiguousarray(scaler.transform(train.values), dtype=np.float32)
    optimized_valid = np.ascontiguousarray(scaler.transform(valid.values), dtype=np.float32)
    optimized_test = np.ascontiguousarray(scaler.transform(test.values), dtype=np.float32)

    loader_pairs = (
        (
            anomaly_detection_data_provider(legacy_train, config["batch_size"], config["seq_len"], 1, "train"),
            anomaly_detection_data_provider(optimized_train, config["batch_size"], config["seq_len"], 1, "train"),
            RandomSampler,
        ),
        (
            anomaly_detection_data_provider(legacy_valid, config["batch_size"], config["seq_len"], 1, "val"),
            anomaly_detection_data_provider(optimized_valid, config["batch_size"], config["seq_len"], 1, "val"),
            RandomSampler,
        ),
        (
            anomaly_detection_data_provider(legacy_train, config["batch_size"], config["seq_len"], 1, "thre"),
            anomaly_detection_data_provider(optimized_train, config["batch_size"], config["seq_len"], 1, "thre"),
            SequentialSampler,
        ),
        (
            anomaly_detection_data_provider(legacy_test, config["batch_size"], config["seq_len"], 1, "thre"),
            anomaly_detection_data_provider(optimized_test, config["batch_size"], config["seq_len"], 1, "thre"),
            SequentialSampler,
        ),
    )
    for legacy_loader, optimized_loader, sampler_type in loader_pairs:
        assert isinstance(legacy_loader.sampler, sampler_type)
        assert isinstance(optimized_loader.sampler, sampler_type)
        assert len(legacy_loader.dataset) == len(optimized_loader.dataset)
        for index in (0, len(legacy_loader.dataset) // 2, len(legacy_loader.dataset) - 1):
            legacy_window, _ = legacy_loader.dataset[index]
            optimized_window, _ = optimized_loader.dataset[index]
            assert legacy_window.dtype == optimized_window.dtype == np.float32
            np.testing.assert_array_equal(legacy_window, optimized_window)
    assert all(values.dtype == np.float32 and values.flags.c_contiguous for values in (
        optimized_train,
        optimized_valid,
        optimized_test,
    ))

    legacy_train_loader, optimized_train_loader, _ = loader_pairs[0]
    torch.manual_seed(41)
    legacy_batch, _ = next(iter(legacy_train_loader))
    torch.manual_seed(41)
    optimized_batch, _ = next(iter(optimized_train_loader))
    assert legacy_batch.dtype == optimized_batch.dtype == torch.float32
    assert legacy_batch.shape == optimized_batch.shape
    torch.testing.assert_close(legacy_batch, optimized_batch, rtol=0, atol=0)

    torch.manual_seed(43)
    detector = MSDCATCH(**config)
    detector.device = torch.device("cpu")
    detector.model = MSDCATCHModel(detector.config).train()
    detector.scaler = scaler
    detector.reference_score_stats = {
        name: (0.0, 1.0) for name in ("total_score", "trend_score", "residual_score")
    }
    detector.reference_delta_thresholds = {
        "trend_delta_threshold": 0.0,
        "residual_delta_threshold": 0.0,
    }

    torch.manual_seed(47)
    legacy_outputs = detector.model(legacy_batch)
    torch.manual_seed(47)
    optimized_outputs = detector.model(optimized_batch)
    for name, legacy_value in legacy_outputs.items():
        optimized_value = optimized_outputs[name]
        if torch.is_tensor(legacy_value):
            torch.testing.assert_close(legacy_value, optimized_value, rtol=0, atol=0)
        else:
            assert legacy_value == optimized_value

    def loss_components(outputs, input_batch):
        components = {"total_time_loss": detector.criterion(outputs["x_hat"], input_batch)}
        branch_losses = {}
        for branch_name in ("trend", "residual"):
            branch = getattr(detector.model, f"{branch_name}_branch")
            target = outputs[branch_name]
            time_loss = detector.criterion(outputs[f"{branch_name}_hat"], target)
            frequency_loss = detector.auxi_loss(
                outputs[f"{branch_name}_complex"], branch.revin_layer(target, "transform")
            )
            channel_loss = outputs[f"{branch_name}_dcloss"]
            branch_loss = (
                time_loss
                + detector.config.dc_lambda * channel_loss
                + detector.config.auxi_lambda * frequency_loss
            )
            components.update(
                {
                    f"{branch_name}_time_loss": time_loss,
                    f"{branch_name}_frequency_loss": frequency_loss,
                    f"{branch_name}_channel_loss": channel_loss,
                    f"{branch_name}_loss": branch_loss,
                }
            )
            branch_losses[branch_name] = branch_loss
        components["total_loss"] = (
            components["total_time_loss"]
            + detector.config.lambda_trend * branch_losses["trend"]
            + detector.config.lambda_residual * branch_losses["residual"]
        )
        return components

    legacy_components = loss_components(legacy_outputs, legacy_batch)
    optimized_components = loss_components(optimized_outputs, optimized_batch)
    for name, legacy_value in legacy_components.items():
        torch.testing.assert_close(legacy_value, optimized_components[name], rtol=0, atol=0)

    torch.manual_seed(53)
    legacy_loss, legacy_diagnostics = detector._loss(legacy_batch)
    torch.manual_seed(53)
    optimized_loss, optimized_diagnostics = detector._loss(optimized_batch)
    torch.testing.assert_close(legacy_loss, optimized_loss, rtol=0, atol=0)
    assert legacy_diagnostics == optimized_diagnostics

    detector.model.eval()
    legacy_test_loader, optimized_test_loader, _ = loader_pairs[-1]
    torch.manual_seed(61)
    legacy_scores = detector._fuse_scores(detector._raw_scores(legacy_test_loader))
    torch.manual_seed(61)
    optimized_total_score, _ = detector.detect_score(test)
    assert len(optimized_total_score) == len(legacy_scores["total_score"])
    for name, legacy_score in legacy_scores.items():
        np.testing.assert_allclose(detector.last_scores[name], legacy_score, rtol=1e-6, atol=1e-6)
    assert len(optimized_test_loader.dataset) == len(legacy_test_loader.dataset)
