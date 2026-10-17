import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ts_benchmark.baselines.mnrf_view_probe import MNRFViewProbe, ViewProbeConfig, load_probe
from ts_benchmark.baselines.mnrf_view_probe.probe import _select_stratified
from ts_benchmark.models.model_loader import get_models


def test_probe_is_loadable_through_benchmark_model_factory():
    factories = get_models({
        "models": [{"model_name": "mnrf_view_probe.MNRFViewProbe",
                    "model_hyper_params": {"device": "cpu"}}]
    })
    assert factories[0].model_name == "MNRFViewProbe"
    assert factories[0]().model_name == "MNRFViewProbe"


def test_view_selection_uses_canonical_layout_and_rejects_invalid_views():
    config = ViewProbeConfig(evolution_dim=3, pattern_dim=5, relation_dim=7, selected_views=("S", "P", "R"))
    assert config.layout.total_dim == 16
    assert config.selected_indices == [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert config.selected_dim == 13
    for views in ((), ("X",), ("S", "S"), ("P", "S")):
        try:
            ViewProbeConfig(selected_views=views)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid view selection accepted: {views}")


def test_stratified_sampling_handles_more_segments_than_samples():
    endpoints = np.arange(20, dtype=np.int64)
    segments = [(2 * index, 2 * index + 2) for index in range(10)]
    selected = _select_stratified(endpoints, segments, limit=3, seed=2021)
    assert selected.shape == (3,)
    assert len(np.unique(selected)) == 3
    assert np.all(np.isin(selected, endpoints))


def test_frozen_probe_gmm_random_and_selection_are_independent(tmp_path):
    rng = np.random.default_rng(2021)
    n, train_length = 520, 430
    frame = pd.DataFrame(
        rng.normal(size=(n, 2)).astype(np.float32), columns=("a", "b"),
        index=pd.date_range("2025-01-01", periods=n, freq="min"),
    )
    labels = np.zeros(n - train_length, dtype=np.int64)
    labels[::2] = 1
    frame["label"] = np.r_[np.zeros(train_length, dtype=np.int64), labels]

    source_dir = tmp_path / "source"
    source = MNRFViewProbe(
        seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1, patience=1,
        device="cpu", output_dir=str(source_dir), mode="full_gmm", gmm_fit_max_windows=32,
    )
    source.fit(frame, train_length, labels)
    assert json.loads((source_dir / "probe_manifest.json").read_text())["mode"] == "pretrained"
    assert (source_dir / "checkpoints" / "initial_encoder.pt").is_file()
    source_cache_hash = hashlib.sha256((source_dir / "cache" / "train_features.npy").read_bytes()).hexdigest()

    random_dir = tmp_path / "random"
    random_probe = MNRFViewProbe(
        seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1, patience=1,
        device="cpu", output_dir=str(random_dir), mode="random_gmm", gmm_fit_max_windows=32,
    )
    random_probe.fit(frame, train_length, labels)
    trained = source_dir / "checkpoints" / "best_encoder.pt"
    initial = random_dir / "checkpoints" / "initial_encoder.pt"
    assert trained.read_bytes() != initial.read_bytes()

    selected_dir = tmp_path / "drop-e"
    selected = MNRFViewProbe(
        seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1, patience=1,
        device="cpu", output_dir=str(selected_dir), mode="select_gmm", selected_views=("S", "P", "R"),
        source_dir=str(source_dir), gmm_fit_max_windows=32,
    )
    selected.fit(frame, train_length, labels)
    payload = json.loads((selected_dir / "results" / "metrics.json").read_text())
    assert payload["selected_dim"] == 17
    assert hashlib.sha256((source_dir / "cache" / "train_features.npy").read_bytes()).hexdigest() == source_cache_hash
    assert payload["auc_pr"] is not None
    status = json.loads((selected_dir / "heads" / "gmm_status.json").read_text())
    sample_path = selected_dir / "heads" / "gmm_fit_endpoints.npy"
    assert status["fit_endpoint_count"] == len(np.load(sample_path))
    assert status["fit_endpoint_sha256"] == hashlib.sha256(sample_path.read_bytes()).hexdigest()
    restored = load_probe(selected_dir, device="cpu")
    original_scores = np.load(selected_dir / "results" / "test_nll.npy")
    restored_scores = restored._gmm_scores(np.load(source_dir / "cache" / "test_features.npy", mmap_mode="r"))
    np.testing.assert_allclose(restored_scores, original_scores, rtol=0, atol=1e-6)

    stats_path = selected_dir / "heads" / "feature_statistics.npz"
    original_stats = stats_path.read_bytes()
    with np.load(stats_path) as stats:
        np.savez(stats_path, mean=stats["mean"], scale=np.zeros_like(stats["scale"]))
    with pytest.raises(ValueError, match="feature statistics hash|shape mismatch|scale"):
        load_probe(selected_dir, device="cpu")
    stats_path.write_bytes(original_stats)

    changed_frame = frame.copy()
    with pytest.raises(ValueError, match="identity|train split"):
        MNRFViewProbe(
            seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1, patience=1,
            device="cpu", output_dir=str(tmp_path / "bad-boundary"), mode="select_gmm",
            selected_views=("S", "P", "R"), source_dir=str(source_dir),
        ).fit(changed_frame, 480, np.zeros(len(changed_frame) - 480, dtype=np.int64))


def test_benchmark_fit_and_score_use_separate_train_test_frames(tmp_path):
    rng = np.random.default_rng(9)
    n, train_length = 520, 430
    frame = pd.DataFrame(
        rng.normal(size=(n, 2)).astype(np.float32), columns=("a", "b"),
        index=pd.date_range("2026-01-01", periods=n, freq="min"),
    )
    frame["label"] = np.r_[np.zeros(train_length, dtype=np.int64), np.ones(n - train_length, dtype=np.int64)]
    probe = MNRFViewProbe(
        seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1, patience=1,
        device="cpu", output_dir=str(tmp_path / "benchmark"), mode="full_gmm", gmm_fit_max_windows=16,
    )
    probe.detect_fit(frame.iloc[:train_length].copy(), frame.iloc[:train_length]["label"].to_numpy())
    scores, predictions = probe.detect_score(frame.iloc[train_length:].copy())
    assert len(scores) == len(predictions) > 0
    metadata = probe.score_with_metadata()
    assert metadata["time_indices"].min() >= train_length
    assert np.isfinite(scores).all()
