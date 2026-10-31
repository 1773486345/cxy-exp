import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_mnrf_frozen_decoder_diagnostic import (
    _endpoint_labels,
    _split_predictions,
    _score_from_matrix,
    decoder_nll,
    load_decoder,
)
from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
from ts_benchmark.baselines.mnrf_lite_v2.training import MNRFTrainer


def _trainer(tmp_path, evolution_dim=8):
    import torch

    config = MNRFConfig(seq_len=32, pattern_len=32, batch_size=2,
                        evolution_dim=evolution_dim, pattern_dim=8,
                        relation_dim=8, device="cpu", output_dir=str(tmp_path))
    return MNRFTrainer(config, variables=2, device=torch.device("cpu")), config


@pytest.mark.parametrize("task,task_index", [("E", 0), ("R", 2)])
def test_decoder_formula_and_chunked_evaluation_are_identical(tmp_path, task, task_index):
    import torch

    trainer, _ = _trainer(tmp_path)
    latent = np.arange(3 * 2 * 8, dtype=np.float32).reshape(3, 2, 8) / 11
    target = np.asarray([[0.1, -0.2], [0.4, 0.3], [-0.1, 0.8]], dtype=np.float32)
    mean, sigma, nll = decoder_nll(trainer, latent, target, task)
    z = torch.as_tensor(latent)
    with torch.no_grad():
        adapted = trainer.auxiliary.input_adapters[task_index](z)
        identity = trainer.encoder.point.variable_identity.view(1, 2, -1).expand(3, -1, -1)
        task_embedding = trainer.auxiliary.task_identity[task_index].view(1, 1, -1).expand(3, 2, -1)
        position = torch.zeros((3, 2, 1))
        output = trainer.auxiliary.decoder(torch.cat((adapted, task_embedding, identity, position), dim=-1))
        expected_mean = output[..., 0].numpy()
        expected_sigma = (0.05 + torch.nn.functional.softplus(output[..., 1])).numpy()
        expected_nll = ((torch.as_tensor(target) - output[..., 0]).square() /
                        (2 * (0.05 + torch.nn.functional.softplus(output[..., 1])).square()) +
                        torch.log(0.05 + torch.nn.functional.softplus(output[..., 1])) +
                        0.5 * np.log(2 * np.pi)).numpy()
    np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=1e-7)
    np.testing.assert_allclose(sigma, expected_sigma, rtol=0, atol=1e-7)
    np.testing.assert_allclose(nll, expected_nll, rtol=0, atol=1e-7)
    chunked = [decoder_nll(trainer, latent[i:i + 1], target[i:i + 1], task)[2]
               for i in range(len(latent))]
    np.testing.assert_allclose(np.concatenate(chunked, axis=0), nll, rtol=0, atol=2e-7)


def test_layout_mismatch_rejected_before_diagnostic(tmp_path):
    import torch

    trainer, config = _trainer(tmp_path, evolution_dim=8)
    (tmp_path / "checkpoints").mkdir()
    checkpoint = tmp_path / "checkpoints" / "best_stage1.pt"
    torch.save({"encoder": trainer.encoder.state_dict(),
                "auxiliary": trainer.auxiliary.state_dict()}, checkpoint)
    bad_config = MNRFConfig(seq_len=32, pattern_len=32, batch_size=2,
                            evolution_dim=4, pattern_dim=8, relation_dim=8,
                            device="cpu", output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="config/layout mismatch"):
        load_decoder({"run": tmp_path, "config": bad_config, "variables": 2,
                      "layout": trainer.layout})


def test_endpoint_labels_are_taken_from_complete_axis_not_compressed_order():
    source = {
        "axes": {
            "all_global": np.asarray([10, 11, 15, 16]),
            "all_labels": np.asarray([0, 1, 0, 1], dtype=np.int8),
        },
        "endpoints": {"test": np.asarray([11, 16], dtype=np.int64)},
    }
    endpoints, labels = _endpoint_labels(source)
    np.testing.assert_array_equal(endpoints, [11, 16])
    np.testing.assert_array_equal(labels, [1, 1])


def test_cached_latent_targets_follow_endpoint_rows_and_chunk_boundaries(tmp_path):
    trainer, _ = _trainer(tmp_path)
    (tmp_path / "cache").mkdir()
    features = np.zeros((3, 2, 25), dtype=np.float32)
    features[:, :, 1:9] = np.arange(3 * 2 * 8, dtype=np.float32).reshape(3, 2, 8)
    np.save(tmp_path / "cache" / "test_features.npy", features)
    endpoints = np.asarray([2, 4, 5], dtype=np.int64)
    np.save(tmp_path / "cache" / "test_window_end_indices.npy", endpoints)
    normalized = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    np.save(tmp_path / "cache" / "normalized_values.npy", normalized)
    source = {
        "run": tmp_path,
        "variables": 2,
        "layout": trainer.layout,
        "normalized": np.load(tmp_path / "cache" / "normalized_values.npy", mmap_mode="r"),
        "endpoints": {"test": np.load(tmp_path / "cache" / "test_window_end_indices.npy", mmap_mode="r")},
    }
    result = _split_predictions(source, trainer, "test", "E", chunk_size=1)
    np.testing.assert_array_equal(result["target"], normalized[endpoints])
    assert result["nll"].shape == (3, 2)


def test_calibration_does_not_depend_on_labels_and_uses_strict_threshold():
    validation = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    calibration = np.asarray([[0.0], [1.0], [3.0]], dtype=np.float32)
    test = np.asarray([[0.0], [3.0]], dtype=np.float32)
    scores_a, threshold_a, _, _ = _score_from_matrix(test, validation, calibration)
    scores_b, threshold_b, _, _ = _score_from_matrix(test, validation, calibration)
    np.testing.assert_array_equal(scores_a, scores_b)
    assert threshold_a == threshold_b
    assert np.array_equal(scores_a > threshold_a, np.asarray([False, True]))
