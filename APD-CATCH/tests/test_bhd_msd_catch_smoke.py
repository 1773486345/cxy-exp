import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import RandomSampler, SequentialSampler

from ts_benchmark.baselines.catch.models.CATCH_model import Flatten_Head
from ts_benchmark.baselines.rsa_msd_catch.models.RSAMSDCATCH_model import (
    RawGateNetwork,
    RawStructureAdapter,
)
from ts_benchmark.baselines.sdd_msd_catch.models.SDDMSDCATCH_model import (
    FactorizedFlattenHead,
    FactorizedLinear,
)
from ts_benchmark.baselines.bhd_msd_catch.BHDMSDCATCH import BHDMSDCATCH
from ts_benchmark.baselines.bhd_msd_catch.models.BHDMSDCATCH_model import (
    BHDMSDCATCHModel,
    BHDNonFiniteTensorError,
    BlockwiseDecoder,
    StableDynamicalContrastiveLoss,
)
from ts_benchmark.baselines.utils import anomaly_detection_data_provider


ROOT = Path(__file__).resolve().parents[1]


def _tiny_config():
    return {
        "seq_len": 32,
        "patch_size": 8,
        "patch_stride": 4,
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


def test_bhd_blockwise_decoder_smoke():
    torch.manual_seed(31)
    detector = BHDMSDCATCH(**_tiny_config())
    model = BHDMSDCATCHModel(detector.config)
    x = torch.randn(3, detector.config.seq_len, detector.config.c_in)
    events = []
    handles = [
        model.shared_encoder.register_forward_hook(lambda *_: events.append("encoder")),
        model.low_rank_exchange.register_forward_hook(lambda *_: events.append("exchange")),
        model.trend_adapter.register_forward_hook(lambda *_: events.append("trend_adapter")),
        model.residual_adapter.register_forward_hook(lambda *_: events.append("residual_adapter")),
        model.trend_block_decoder.register_forward_hook(lambda *_: events.append("trend_decoder")),
        model.residual_block_decoder.register_forward_hook(lambda *_: events.append("residual_decoder")),
    ]
    outputs = model(x)
    for handle in handles:
        handle.remove()

    assert torch.allclose(outputs["trend"] + outputs["residual"], x)
    assert events[:2] == ["encoder", "encoder"]
    assert events.index("exchange") < events.index("trend_adapter") < events.index("trend_decoder")
    assert events.index("exchange") < events.index("residual_adapter") < events.index("residual_decoder")
    trend_params = {id(parameter) for parameter in model.trend_block_decoder.parameters()}
    residual_params = {id(parameter) for parameter in model.residual_block_decoder.parameters()}
    assert trend_params.isdisjoint(residual_params)
    assert not any(isinstance(module, (Flatten_Head, FactorizedLinear, FactorizedFlattenHead)) for module in model.modules())
    assert isinstance(model.trend_block_decoder, BlockwiseDecoder)
    assert model.trend_block_decoder.last_input_shape == (
        x.shape[0],
        x.shape[-1],
        model.trend_block_decoder.patch_num,
        model.feature_dim,
    )
    expected_patch_shape = (
        x.shape[0],
        x.shape[-1],
        model.trend_block_decoder.patch_num,
        detector.config.patch_size,
    )
    assert outputs["trend_patch_real"].shape == expected_patch_shape
    assert outputs["trend_patch_imag"].shape == expected_patch_shape
    assert outputs["residual_patch_real"].shape == expected_patch_shape
    assert outputs["residual_patch_imag"].shape == expected_patch_shape
    assert outputs["trend_overlap_count"].shape[-1] == detector.config.seq_len
    assert torch.all(outputs["trend_overlap_count"] > 0)
    assert torch.all(outputs["residual_overlap_count"] > 0)
    assert outputs["trend_complex"].shape == outputs["residual_complex"].shape
    assert outputs["trend_hat"].shape == x.shape
    assert outputs["residual_hat"].shape == x.shape
    assert outputs["x_hat"].shape == x.shape
    assert isinstance(model.raw_adapter, RawStructureAdapter)
    assert isinstance(model.raw_gate_network, RawGateNetwork)
    assert torch.count_nonzero(model.raw_adapter.output_projection.weight) == 0
    assert torch.count_nonzero(model.raw_gate_network.output.weight) == 0
    assert torch.allclose(
        model.raw_gate_network.output.bias,
        torch.full_like(model.raw_gate_network.output.bias, -2.0),
    )
    assert torch.all(outputs["raw_gate"] >= 0)
    assert torch.all(outputs["raw_gate"] <= 0.5)
    expected_gate = 0.5 * torch.sigmoid(torch.tensor(-2.0))
    assert torch.allclose(outputs["raw_gate"].mean(), expected_gate, atol=1e-6)
    assert torch.isfinite(outputs["x_hat"]).all()

    loss = (
        (outputs["x_hat"] - x).square().mean()
        + outputs["trend_dcloss"]
        + outputs["residual_dcloss"]
    )
    loss.backward()
    assert torch.isfinite(loss)
    detector.model = model
    scores, diagnostics = detector._score_batch(x, collect_diagnostics=True)
    assert scores["total_score"].numel() == x.shape[0] * x.shape[1]
    assert diagnostics["trend_hat"].shape == x.shape
    assert all(torch.isfinite(value).all() for value in scores.values())
    assert model.module_parameter_counts()["total"] == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def test_bhd_does_not_modify_unrelated_baselines():
    paths = [
        ROOT / "ts_benchmark" / "baselines" / name
        for name in ("catch", "rsa_msd_catch", "sdd_msd_catch")
    ]
    before = [_tree_digest(path) for path in paths]
    _ = BHDMSDCATCHModel(BHDMSDCATCH(**_tiny_config()).config)(torch.randn(1, 32, 3))
    assert [_tree_digest(path) for path in paths] == before
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "--",
            "ts_benchmark/baselines/catch",
            "ts_benchmark/baselines/rsa_msd_catch",
            "ts_benchmark/baselines/sdd_msd_catch",
        ],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0


def _original_contrastive_loss(scores, mask, norm_matrix, temperature, k):
    batch_size = scores.shape[0]
    n_vars = scores.shape[-1]
    cosine = (scores / norm_matrix).mean(1)
    pos_scores = torch.exp(cosine / temperature) * mask
    all_scores = torch.exp(cosine / temperature)
    clustering_loss = -torch.log(pos_scores.sum(dim=-1) / all_scores.sum(dim=-1))
    eye = torch.eye(n_vars, device=mask.device).unsqueeze(0).repeat(batch_size, 1, 1)
    regular_loss = torch.norm(
        eye.reshape(batch_size, -1) - mask.reshape(batch_size, -1), p=1, dim=-1
    ) / (n_vars * (n_vars - 1))
    return (clustering_loss.mean(1) + k * regular_loss).mean()


def test_bhd_contrastive_loss_preserves_formula_and_zero_norm_protection():
    loss_module = StableDynamicalContrastiveLoss(temperature=0.1, k=0.3)
    mask = torch.eye(3).expand(2, -1, -1)

    torch.manual_seed(37)
    normal_scores = (0.05 * torch.randn(2, 1, 3, 3)).requires_grad_()
    legacy_scores = normal_scores.detach().clone().requires_grad_()
    normal_norm = torch.ones_like(normal_scores)
    stable_loss = loss_module(normal_scores, mask, normal_norm)
    legacy_loss = _original_contrastive_loss(legacy_scores, mask, normal_norm, 0.1, 0.3)
    stable_loss.backward()
    legacy_loss.backward()
    torch.testing.assert_close(stable_loss, legacy_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(normal_scores.grad, legacy_scores.grad, rtol=1e-5, atol=1e-7)

    for norm_matrix in (torch.zeros_like(normal_scores), torch.full_like(normal_scores, 1e-20)):
        scores = torch.zeros_like(normal_scores, requires_grad=True)
        loss = loss_module(scores, mask, norm_matrix)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(scores.grad).all()

    for value in (100.0, -100.0):
        scores = torch.full_like(normal_scores, value, requires_grad=True)
        legacy_loss = _original_contrastive_loss(scores, mask, torch.ones_like(scores), 0.1, 0.3)
        assert not torch.isfinite(legacy_loss)
        stable_loss = loss_module(scores, mask, torch.ones_like(scores))
        assert not torch.isfinite(stable_loss)


def test_bhd_contrastive_loss_reports_first_nonfinite_tensor():
    loss_module = StableDynamicalContrastiveLoss(temperature=0.1, k=0.3)
    loss_module.set_diagnostic_context(
        {
            "branch": "residual",
            "layer": 0,
            "epoch": 2,
            "global_step": 11,
            "batch_index": 3,
            "debug_nonfinite": True,
        }
    )
    scores = torch.zeros(2, 1, 3, 3)
    scores[0, 0, 0, 0] = float("nan")
    norm_matrix = torch.ones_like(scores)
    mask = torch.eye(3).expand(2, -1, -1)
    try:
        loss_module(scores, mask, norm_matrix)
    except BHDNonFiniteTensorError as error:
        assert error.diagnostic["branch"] == "residual"
        assert error.diagnostic["tensor_name"] == "scores"
        assert error.diagnostic["context"]["global_step"] == 11
    else:
        raise AssertionError("expected a BHDNonFiniteTensorError")


def test_bhd_model_preserves_contrastive_layer_context():
    model = BHDMSDCATCHModel(BHDMSDCATCH(**_tiny_config()).config)
    model.set_diagnostic_context({"epoch": 3, "global_step": 17, "batch_index": 5})
    assert [loss._diagnostic_context["layer"] for loss in model._contrastive_losses] == [0]
    assert all(loss._diagnostic_context["global_step"] == 17 for loss in model._contrastive_losses)


def test_bhd_numpy_scaled_loader_matches_dataframe_loader():
    config = _tiny_config()
    raw = pd.DataFrame(
        np.arange(96 * config["c_in"], dtype=np.float64).reshape(96, config["c_in"]),
        columns=[f"feature_{index}" for index in range(config["c_in"])],
    )
    scaler = StandardScaler().fit(raw.values)
    legacy_data = pd.DataFrame(
        scaler.transform(raw.values), columns=raw.columns, index=raw.index
    )
    optimized_data = np.ascontiguousarray(scaler.transform(raw.values), dtype=np.float32)
    legacy_loader = anomaly_detection_data_provider(
        legacy_data, config["batch_size"], config["seq_len"], 1, "train"
    )
    optimized_loader = anomaly_detection_data_provider(
        optimized_data, config["batch_size"], config["seq_len"], 1, "train"
    )

    assert optimized_data.dtype == np.float32
    assert optimized_data.flags.c_contiguous
    assert isinstance(legacy_loader.sampler, RandomSampler)
    assert isinstance(optimized_loader.sampler, RandomSampler)
    assert len(legacy_loader.dataset) == len(optimized_loader.dataset)
    for index in (0, len(legacy_loader.dataset) // 2, len(legacy_loader.dataset) - 1):
        legacy_window, _ = legacy_loader.dataset[index]
        optimized_window, _ = optimized_loader.dataset[index]
        assert legacy_window.dtype == optimized_window.dtype == np.float32
        np.testing.assert_array_equal(legacy_window, optimized_window)

    torch.manual_seed(41)
    legacy_batch, _ = next(iter(legacy_loader))
    torch.manual_seed(41)
    optimized_batch, _ = next(iter(optimized_loader))
    assert legacy_batch.dtype == optimized_batch.dtype == torch.float32
    assert legacy_batch.shape == optimized_batch.shape
    torch.testing.assert_close(legacy_batch, optimized_batch, rtol=0, atol=0)

    torch.manual_seed(43)
    detector = BHDMSDCATCH(**config)
    detector.device = torch.device("cpu")
    detector.model = BHDMSDCATCHModel(detector.config).train()
    detector.scaler = scaler
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

    torch.manual_seed(53)
    default_loss, _ = detector._loss(legacy_batch)
    assert detector._last_loss_snapshot == {}
    detector._debug_nonfinite = True
    torch.manual_seed(53)
    legacy_loss, _ = detector._loss(legacy_batch)
    legacy_components = detector._last_loss_snapshot
    torch.manual_seed(53)
    optimized_loss, _ = detector._loss(optimized_batch)
    optimized_components = detector._last_loss_snapshot
    torch.testing.assert_close(legacy_loss, optimized_loss, rtol=0, atol=0)
    for branch, legacy_values in legacy_components.items():
        assert optimized_components[branch].keys() == legacy_values.keys()
        for name, legacy_value in legacy_values.items():
            np.testing.assert_allclose(optimized_components[branch][name], legacy_value, rtol=0, atol=0)

    legacy_score_loader = anomaly_detection_data_provider(
        legacy_data, config["batch_size"], config["seq_len"], 1, "thre"
    )
    optimized_score_loader = anomaly_detection_data_provider(
        optimized_data, config["batch_size"], config["seq_len"], 1, "thre"
    )
    assert isinstance(legacy_score_loader.sampler, SequentialSampler)
    assert isinstance(optimized_score_loader.sampler, SequentialSampler)
    assert len(legacy_score_loader.dataset) == len(optimized_score_loader.dataset)
    for index in (0, len(legacy_score_loader.dataset) // 2, len(legacy_score_loader.dataset) - 1):
        legacy_window, _ = legacy_score_loader.dataset[index]
        optimized_window, _ = optimized_score_loader.dataset[index]
        assert legacy_window.dtype == optimized_window.dtype == np.float32
        np.testing.assert_array_equal(legacy_window, optimized_window)

    detector.model.eval()
    torch.manual_seed(61)
    diagnostic_scores, diagnostic_values = detector._collect_scores(
        legacy_score_loader, collect_diagnostics=True
    )
    torch.manual_seed(61)
    default_scores, default_diagnostics = detector._collect_scores(legacy_score_loader)
    assert set(default_scores) == {"total_score"}
    assert default_diagnostics == {}
    assert set(diagnostic_scores) == {
        "total_score",
        "decomp_score",
        "trend_score",
        "residual_score",
        "raw_correction_score",
    }
    assert set(diagnostic_values) == {
        "trend_hat",
        "residual_hat",
        "decomp_hat",
        "raw_correction",
        "raw_gate",
        "scale_entropy",
        "scale_weights",
    }
    np.testing.assert_allclose(
        default_scores["total_score"], diagnostic_scores["total_score"], rtol=1e-6, atol=1e-6
    )
    torch.manual_seed(61)
    optimized_total_score, _ = detector.detect_score(raw)
    assert set(detector.last_scores) == {"total_score"}
    assert detector.last_diagnostics == {}
    assert len(optimized_total_score) == len(diagnostic_scores["total_score"])
    for name, legacy_score in default_scores.items():
        np.testing.assert_allclose(detector.last_scores[name], legacy_score, rtol=1e-6, atol=1e-6)
    torch.manual_seed(61)
    detector.detect_score(raw, collect_diagnostics=True)
    assert set(detector.last_diagnostics) == set(diagnostic_values)
    np.testing.assert_allclose(
        detector.last_scores["total_score"], diagnostic_scores["total_score"], rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(default_loss, legacy_loss, rtol=0, atol=0)


def test_bhd_debug_scans_metadata_and_label_scores_are_explicit():
    probe = BHDMSDCATCH(**_tiny_config())
    calls = {"gradient": 0, "optimizer": 0}

    def gradient_scan():
        calls["gradient"] += 1
        return {"first_nonfinite_parameter": None}

    def optimizer_scan(_):
        calls["optimizer"] += 1
        return None

    probe._gradient_diagnostic = gradient_scan
    probe._optimizer_diagnostic = optimizer_scan
    probe._debug_nonfinite = False
    probe._run_debug_after_backward()
    probe._run_debug_after_optimizer_step(object(), "optimizer")
    assert calls == {"gradient": 0, "optimizer": 0}
    probe._debug_nonfinite = True
    probe._run_debug_after_backward()
    probe._run_debug_after_optimizer_step(object(), "optimizer")
    assert calls == {"gradient": 1, "optimizer": 1}

    with tempfile.TemporaryDirectory() as directory:
        params = _tiny_config()
        first = BHDMSDCATCH(
            **params,
            dataset_name="first_dataset",
            diagnostic_output_dir=str(Path(directory) / "first"),
        )
        second = BHDMSDCATCH(
            **params,
            dataset_name="second_dataset",
            diagnostic_output_dir=str(Path(directory) / "second"),
        )
        first.device = second.device = torch.device("cpu")
        torch.manual_seed(71)
        first.model = BHDMSDCATCHModel(first.config).eval()
        torch.manual_seed(71)
        second.model = BHDMSDCATCHModel(second.config).eval()
        x = torch.randn(2, params["seq_len"], params["c_in"])
        torch.manual_seed(73)
        first_outputs = first.model(x)
        torch.manual_seed(73)
        second_outputs = second.model(x)
        torch.testing.assert_close(first_outputs["x_hat"], second_outputs["x_hat"], rtol=0, atol=0)
        torch.manual_seed(79)
        first_loss, _ = first._loss(x)
        torch.manual_seed(79)
        second_loss, _ = second._loss(x)
        torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)

        diagnostic = {
            "branch": "total",
            "tensor_name": "total_loss",
            "tensor_stats": {"finite_fraction": 0.0},
            "context": {"epoch": 1, "global_step": 2, "batch_index": 2},
        }
        first._save_nonfinite_diagnostic(diagnostic)
        second._save_nonfinite_diagnostic(diagnostic)
        first_path = Path(directory) / "first" / "nonfinite_diagnostic.json"
        second_path = Path(directory) / "second" / "nonfinite_diagnostic.json"
        assert first_path.exists() and second_path.exists()
        assert json.loads(first_path.read_text())["dataset"] == "first_dataset"
        assert json.loads(second_path.read_text())["dataset"] == "second_dataset"

    label_detector = BHDMSDCATCH(**_tiny_config())
    label_detector.config.anomaly_ratio = [50]
    label_detector.reference_data_loader = object()
    test_total_score = np.array([0.24, 0.26], dtype=np.float32)
    reference_total_score = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    label_detector.detect_score = lambda _: (test_total_score, test_total_score)
    label_detector._collect_scores = lambda *_args, **_kwargs: ({"total_score": reference_total_score}, {})
    predictions, returned_score = label_detector.detect_label(pd.DataFrame(np.zeros((2, 3))))
    np.testing.assert_array_equal(predictions[50], np.array([0, 1]))
    np.testing.assert_array_equal(returned_score, test_total_score)
