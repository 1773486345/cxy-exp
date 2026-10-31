import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_mnrf_frozen_decoder_diagnostic import (
    _endpoint_labels,
    _boundary_metrics,
    _split_predictions,
    _score_from_matrix,
    decoder_nll,
    load_decoder,
    validate_source_contract,
)
from scripts import run_mnrf_frozen_decoder_diagnostic as diagnostic
from scripts.audit_mnrf_baseline_investigation import _effective_fields
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


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _synthetic_full_source(tmp_path):
    """Write a tiny, already-complete source without training or encoding."""
    import torch

    run = tmp_path / "synthetic-full"
    for name in ("checkpoints", "cache", "results"):
        (run / name).mkdir(parents=True)
    config = MNRFConfig(seed=2021, seq_len=32, pattern_len=32, batch_size=2,
                        stage1_epochs=1, stage2_epochs=1, patience=1,
                        device="cpu", output_dir=str(run), task_name="Genesis",
                        calibration_quantile=0.995, score_mad_floor=0.001)
    trainer = MNRFTrainer(config, variables=2, device=torch.device("cpu"))
    mean = np.asarray([0.25, -0.5], dtype=np.float32)
    scale = np.asarray([1.5, 0.75], dtype=np.float32)
    trainer.fixed_input.mean_ = mean
    trainer.fixed_input.scale_ = scale
    np.savez(run / "input_statistics.npz", mean=mean, scale=scale)

    rng = np.random.default_rng(2701)
    normalized = rng.normal(size=(64, 2)).astype(np.float32)
    np.save(run / "cache" / "normalized_values.npy", normalized)
    endpoints = {
        "train": np.arange(32, 40, dtype=np.int64),
        "validation": np.arange(40, 48, dtype=np.int64),
        "calibration": np.arange(48, 56, dtype=np.int64),
        "test": np.asarray([56, 57, 58, 59, 60, 62, 63], dtype=np.int64),
    }
    cache_hashes, endpoint_hashes, cache_shapes = {}, {}, {}
    for split, split_endpoints in endpoints.items():
        features = (rng.normal(size=(len(split_endpoints), 2, trainer.layout.total_dim))
                    .astype(np.float32) * 0.05)
        np.save(run / "cache" / f"{split}_features.npy", features)
        np.save(run / "cache" / f"{split}_window_end_indices.npy", split_endpoints)
        cache_hashes[split] = _sha(run / "cache" / f"{split}_features.npy")
        endpoint_hashes[split] = _sha(run / "cache" / f"{split}_window_end_indices.npy")
        cache_shapes[split] = list(features.shape)

    all_global = np.arange(64, dtype=np.int64)
    all_labels = np.zeros(64, dtype=np.int8)
    all_labels[[57, 58, 60, 61]] = 1
    all_times = all_global * np.int64(60_000_000_000)
    all_segments = np.zeros(64, dtype=np.int64)
    all_segments[61:] = 4
    test_global = np.arange(56, 64, dtype=np.int64)
    test_labels = all_labels[test_global]
    test_times = all_times[test_global]
    test_segments = all_segments[test_global]
    np.savez(run / "analysis_inputs.npz", all_global_indices=all_global,
             all_labels=all_labels, all_time_values_ns=all_times,
             all_segment_ids=all_segments, test_global_indices=test_global,
             test_labels=test_labels, test_time_values_ns=test_times,
             test_segment_ids=test_segments)

    full_scores = np.asarray([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    full_predictions = (full_scores > 0.5).astype(np.int8)
    np.savez(run / "results" / "aligned_scores.npz", scores=full_scores,
             predictions=full_predictions, actual=all_labels[endpoints["test"]],
             time_indices=endpoints["test"], local_indices=np.arange(len(full_scores)),
             valid_mask=np.ones(len(full_scores), dtype=bool), threshold=np.asarray(0.5))

    checkpoint = {"encoder": trainer.encoder.state_dict(),
                  "auxiliary": trainer.auxiliary.state_dict()}
    torch.save(checkpoint, run / "checkpoints" / "best_stage1.pt")
    torch.save(checkpoint, run / "checkpoints" / "best_encoder.pt")
    identity = {"task": "Genesis", "synthetic_fixture": "frozen-decoder-flow"}
    config_dict = config.to_dict()
    config_dict["resolved_layout"] = config.layout.to_dict()
    manifest = {
        "stage_status": "complete", "head_status": "complete", "head": "gmm",
        "head_type": "gmm", "artifact_type": "probe_head", "mode": "full_gmm",
        "artifact_policy": "full", "selected_views": ["S", "E", "P", "R"],
        "config": config_dict, "layout": config.layout.to_dict(),
        "variable_names": ["x", "y"], "source_identity": identity,
        "source_identity_sha256": diagnostic._json_hash(identity),
        "feature_fingerprint": trainer.fingerprint(),
        "normalized_cache_sha256": _sha(run / "cache" / "normalized_values.npy"),
        "input_statistics_sha256": _sha(run / "input_statistics.npz"),
        "checkpoint_sha256": {"best_encoder.pt": _sha(run / "checkpoints" / "best_encoder.pt")},
        "cache_sha256": cache_hashes, "endpoint_sha256": endpoint_hashes,
        "cache_shapes": cache_shapes,
    }
    manifest["aligned_scores_sha256"] = _sha(run / "results" / "aligned_scores.npz")
    (run / "probe_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def test_complete_cpu_run_task_writes_boundary_and_affiliation_evidence(tmp_path, monkeypatch):
    import csv

    source_path = _synthetic_full_source(tmp_path)
    output = tmp_path / "diagnostic-output"
    monkeypatch.setattr(diagnostic, "source_path",
                        lambda task: (source_path, {"synthetic_fixture": True}))

    validated = validate_source_contract(source_path)
    np.testing.assert_array_equal(validated["axes"]["test_segments"], [0, 0, 0, 0, 0, 4, 4, 4])
    assert validated["axes"]["test_times"].shape == (8,)
    decoder, _ = load_decoder(validated)
    assert decoder.auxiliary is not None

    diagnostic.run_task("Genesis", output, chunk_size=2)
    manifest = json.loads((output / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cpu_threads"]["requested"] == 1
    assert manifest["cpu_threads"]["torch_intraop"] == 1
    assert manifest["cpu_threads"]["torch_interop"] == 1
    assert all(pool["num_threads"] == 1 for pool in manifest["cpu_threads"]["native_threadpools"])
    assert (output / "E_squared_residual_test_scores.npz").is_file()
    assert (output / "E_nll_test_scores.npz").is_file()
    assert (output / "R_test_prediction_evidence.npz").is_file()
    assert (output / "per_variable_summary.json").is_file()
    assert (output / "README.md").is_file()

    with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    full = next(row for row in rows if row["view"] == "FULL")
    assert float(full["affiliation_f_valid_segment_macro"]) == pytest.approx(1.0)
    assert int(full["affiliation_defined_segments"]) == 1
    assert int(full["affiliation_total_segments"]) == 2
    assert int(full["anomaly_segments_total"]) == 3
    assert int(full["anomaly_segments_scored"]) == 2
    assert int(full["anomaly_segments_hit"]) == 2
    assert float(full["auc_roc"]) == pytest.approx(1.0)

    with np.load(output / "E_nll_test_scores.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["global_indices"], [56, 57, 58, 59, 60, 62, 63])
        assert saved["scores"].shape == (7,)
        assert np.isfinite(saved["threshold"])


def test_boundary_metrics_preserve_undefined_and_do_not_join_segments(tmp_path):
    source_path = _synthetic_full_source(tmp_path)
    source = validate_source_contract(source_path)
    axes = source["axes"]
    assert axes["test_segments"][4] != axes["test_segments"][5]
    # The event at global row 61 is unscored; segment accounting must retain it.
    result = _boundary_metrics(source, np.zeros(7, dtype=np.float32), 0.5)
    assert result["anomaly_segments_total"] == 3
    assert result["anomaly_segments_scored"] == 2
    assert result["anomaly_segments_hit"] == 0
    no_prediction = _boundary_metrics(source, np.zeros(7, dtype=np.float32), 0.5)
    assert no_prediction["affiliation_f_valid_segment_macro"] is None
    assert no_prediction["affiliation_defined_segments"] == 0
    assert any(item["reason"] == "non_finite_precision_or_recall"
               for item in no_prediction["affiliation_undefined_segments"])

    no_event = dict(source)
    no_event["axes"] = dict(axes)
    no_event["axes"]["test_labels"] = np.zeros_like(axes["test_labels"])
    undefined = _boundary_metrics(no_event, np.zeros(7, dtype=np.float32), 0.5)
    assert undefined["affiliation_f_valid_segment_macro"] is None
    assert undefined["affiliation_defined_segments"] == 0
    assert undefined["affiliation_total_segments"] == 2
    assert {item["reason"] for item in undefined["affiliation_undefined_segments"]} == {
        "no_ground_truth_event"
    }


def test_test_axis_segments_must_map_to_complete_axis(tmp_path):
    source_path = _synthetic_full_source(tmp_path)
    path = source_path / "analysis_inputs.npz"
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    arrays["test_segment_ids"] = arrays["test_segment_ids"].copy()
    arrays["test_segment_ids"][0] = 99
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="test_segment_ids differ"):
        validate_source_contract(source_path)


def test_tfad_effective_window_uses_n_window_default_and_override():
    default = _effective_fields("TFAD", "detect_score", "python TFAD.py")
    override = _effective_fields(
        "TFAD", "detect_score",
        "python TFAD.py --model-hyper-params '{\"n_window\": 256}'")
    assert default["effective_window"] == "192"
    assert override["effective_window"] == "256"
