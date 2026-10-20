import copy
import hashlib
import json
import random
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from ts_benchmark.baselines.mnrf_view_probe import MNRFViewProbe, ViewProbeConfig, load_probe
from ts_benchmark.baselines.mnrf_view_probe.strategy import IndexedViewProbeStrategy
from ts_benchmark.baselines.mnrf_view_probe.probe import _json_hash, _select_stratified
from ts_benchmark.models.model_loader import get_models
from ts_benchmark.baselines.mnrf_lite_v2.training import MNRFTrainer


def test_probe_is_loadable_through_benchmark_model_factory():
    factories = get_models({
        "models": [{"model_name": "mnrf_view_probe.MNRFViewProbe",
                    "model_hyper_params": {"device": "cpu"}}]
    })
    assert factories[0].model_name == "MNRFViewProbe"
    assert factories[0]().model_name == "MNRFViewProbe"


def test_progress_events_follow_real_stage1_optimizer_steps(tmp_path):
    rng = np.random.default_rng(20261019)
    n, train_length = 520, 430
    frame = pd.DataFrame(rng.normal(size=(n, 2)).astype(np.float32), columns=("x", "y"),
                         index=pd.date_range("2026-01-01", periods=n, freq="min"))
    frame["label"] = np.r_[np.zeros(train_length, dtype=np.int64), np.arange(n - train_length) % 2]
    probe = MNRFViewProbe(seq_len=32, pattern_len=32, batch_size=32, stage1_epochs=1,
                          stage2_epochs=1, patience=1, device="cpu", output_dir=str(tmp_path / "run"),
                          mode="stage1", log_every_steps=2, log_every_seconds=30)
    events = []
    probe._log_event = lambda event, **fields: events.append((event, fields))
    probe.fit(frame, train_length)
    progress = [fields for event, fields in events if event == "stage1_batch"]
    assert [item["completed_steps"] for item in progress] == [1, 2, 4, 6, 8, 9]
    assert all(item["completed_steps"] == item["optimizer_steps"] for item in progress)
    checkpoint = torch.load(tmp_path / "run" / "checkpoints" / "stage1_latest.pt",
                            map_location="cpu", weights_only=False)
    assert checkpoint["completed_steps"] == 9
    assert progress[-1]["epoch_batch"] == progress[-1]["total_batches"] == 9


def test_auto_device_gmm_can_be_loaded_and_validated_on_cpu(tmp_path, monkeypatch):
    from scripts.validate_mnrf_view_probe import validate
    from scripts.summarize_mnrf_view_probe import collect

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(901)
    n, train_length = 520, 430
    frame = pd.DataFrame(rng.normal(size=(n, 2)).astype(np.float32), columns=("x", "y"),
                         index=pd.date_range("2026-03-01", periods=n, freq="min"))
    frame["label"] = np.r_[np.zeros(train_length, dtype=np.int64), np.arange(n - train_length) % 2]
    run = tmp_path / "auto-cpu"
    probe = MNRFViewProbe(seq_len=32, pattern_len=32, batch_size=32, stage1_epochs=1,
                          stage2_epochs=1, patience=1, device="auto", output_dir=str(run),
                          mode="full_gmm", gmm_fit_max_windows=32)
    probe.fit(frame, train_length, frame["label"].to_numpy()[train_length:])
    persisted = json.loads((run / "probe_manifest.json").read_text())
    status = json.loads((run / "heads" / "gmm_status.json").read_text())
    assert persisted["config"]["device"] == "auto"
    assert status["config"]["device"] == "auto"
    restored = load_probe(run, device="cpu")
    assert restored.device.type == "cpu"
    assert np.isfinite(restored.score_cached_test()["scores"]).all()
    assert validate(run)["status"] == "complete"
    summary = collect(tmp_path)
    assert summary.loc[summary["run"] == str(run), "validation_status"].iloc[0] == "valid"
    referenced = tmp_path / "auto-reference"
    selected = MNRFViewProbe(
        seq_len=32, pattern_len=32, batch_size=32, stage1_epochs=1, stage2_epochs=1,
        patience=1, device="auto", output_dir=str(referenced), mode="select_gmm",
        selected_views=("S", "P", "R"), source_dir=str(run), gmm_fit_max_windows=32,
    )
    selected.fit(frame, train_length, frame["label"].to_numpy()[train_length:])
    assert load_probe(referenced, device="cpu").device.type == "cpu"
    assert validate(referenced)["status"] == "complete"


def test_completed_stage1_resume_is_read_only_with_real_best_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(902)
    n, train_length = 520, 430
    frame = pd.DataFrame(rng.normal(size=(n, 2)).astype(np.float32), columns=("x", "y"),
                         index=pd.date_range("2026-04-01", periods=n, freq="min"))
    frame["label"] = np.r_[np.zeros(train_length, dtype=np.int64), np.zeros(n - train_length, dtype=np.int64)]
    run = tmp_path / "stage1"
    kwargs = dict(seq_len=32, pattern_len=32, batch_size=32, stage1_epochs=1,
                  stage2_epochs=1, patience=1, device="auto", output_dir=str(run), mode="stage1")
    MNRFViewProbe(**kwargs).fit(frame, train_length)
    names_before = sorted(path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file())
    hashes_before = {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in names_before}
    random.seed(4321)
    np.random.seed(4321)
    torch.manual_seed(4321)
    python_before = random.getstate()
    numpy_before = copy.deepcopy(np.random.get_state())
    torch_before = torch.get_rng_state().clone()
    MNRFViewProbe(**dict(kwargs, resume_from=str(run))).fit(frame, train_length)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())
    names_after = sorted(path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file())
    hashes_after = {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in names_after}
    assert names_after == names_before
    assert hashes_after == hashes_before


def test_flow_feature_statistics_shape_and_completed_stage2_resume_only_finishes(tmp_path, monkeypatch):
    rng = np.random.default_rng(904)
    n, train_length = 520, 430
    frame = pd.DataFrame(rng.normal(size=(n, 2)).astype(np.float32), columns=("x", "y"),
                         index=pd.date_range("2026-04-15", periods=n, freq="min"))
    frame["label"] = np.r_[np.zeros(train_length, dtype=np.int64), np.zeros(n - train_length, dtype=np.int64)]
    source_dir = tmp_path / "flow-source"
    MNRFViewProbe(seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1,
                  stage2_epochs=1, patience=1, device="cpu", output_dir=str(source_dir),
                  mode="stage1").fit(frame, train_length)
    initial = torch.load(source_dir / "checkpoints" / "initial_encoder.pt",
                         map_location="cpu", weights_only=False)["encoder"]
    best = torch.load(source_dir / "checkpoints" / "best_encoder.pt",
                      map_location="cpu", weights_only=False)["encoder"]
    assert any(not torch.equal(initial[name], best[name]) for name in initial)
    source_manifest = json.loads((source_dir / "probe_manifest.json").read_text())
    flow_dir = tmp_path / "flow"
    kwargs = dict(seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1,
                  stage2_epochs=1, patience=1, device="cpu", output_dir=str(flow_dir),
                  mode="flow", source_dir=str(source_dir))
    original_calibrate = MNRFTrainer.calibrate

    def fail_after_stage2(self, *args, **kwargs):
        raise RuntimeError("controlled post-stage2 interruption")

    monkeypatch.setattr(MNRFTrainer, "calibrate", fail_after_stage2)
    with pytest.raises(RuntimeError, match="controlled post-stage2"):
        flow_probe = MNRFViewProbe(**kwargs)
        flow_probe.fit(frame, train_length)
    assert flow_probe.trainer.fingerprint() == source_manifest["feature_fingerprint"]
    flow_manifest = json.loads((flow_dir / "probe_manifest.json").read_text())
    assert flow_manifest["stage2_identity"]
    assert flow_manifest["stage2_identity"] == _json_hash({
        "source_identity": flow_probe.source_identity,
        "feature_fingerprint": source_manifest["feature_fingerprint"],
        "layout": flow_probe.config.layout.to_dict(),
    })
    checkpoint_path = flow_dir / "checkpoints" / "stage2_latest.pt"
    checkpoint_before = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint_before["stage_status"] == "completed"
    with np.load(flow_dir / "heads" / "feature_statistics.npz") as stats:
        assert stats["mean"].shape == (2, 25)
        assert stats["scale"].shape == (2, 25)
    # Older Flow manifests stored the encoded-feature hash in the legacy
    # input_statistics field.  Keep that format loadable only with the
    # independent raw-input binding still present.
    manifest_path = flow_dir / "probe_manifest.json"
    legacy_manifest = json.loads(manifest_path.read_text())
    legacy_manifest["input_statistics_sha256"] = legacy_manifest["feature_statistics_sha256"]
    legacy_manifest.pop("feature_statistics_sha256")
    manifest_path.write_text(json.dumps(legacy_manifest))
    monkeypatch.setattr(MNRFTrainer, "calibrate", original_calibrate)
    restored = MNRFViewProbe(**dict(kwargs, resume_from=str(flow_dir)))
    restored.fit(frame, train_length)
    metrics = json.loads((flow_dir / "results" / "metrics.json").read_text())
    assert metrics["valid_points"] == 59
    assert metrics["excluded_points"] == 31
    checkpoint_after = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint_after["completed_steps"] == checkpoint_before["completed_steps"]
    assert checkpoint_after["next_epoch"] == checkpoint_before["next_epoch"]
    assert json.loads((flow_dir / "run_status.json").read_text())["status"] == "complete"


def test_flow_real_epoch_checkpoint_resumes_from_next_epoch(tmp_path, monkeypatch):
    rng = np.random.default_rng(905)
    n, train_length = 520, 430
    frame = pd.DataFrame(rng.normal(size=(n, 2)).astype(np.float32), columns=("x", "y"),
                         index=pd.date_range("2026-04-20", periods=n, freq="min"))
    frame["label"] = 0
    source_dir = tmp_path / "source"
    MNRFViewProbe(seq_len=32, pattern_len=32, evolution_dim=3, pattern_dim=5,
                  relation_dim=7, batch_size=64, stage1_epochs=1, stage2_epochs=1,
                  patience=1, device="cpu", output_dir=str(source_dir),
                  mode="stage1").fit(frame, train_length)
    flow_dir = tmp_path / "flow"
    kwargs = dict(seq_len=32, pattern_len=32, evolution_dim=3, pattern_dim=5,
                  relation_dim=7, batch_size=64, stage1_epochs=1, stage2_epochs=3,
                  patience=3, device="cpu", output_dir=str(flow_dir),
                  mode="flow", source_dir=str(source_dir))
    original_validation = MNRFTrainer._stage2_validation
    calls = {"count": 0}

    def interrupt_after_first_checkpoint(self, features):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("controlled epoch-boundary interruption")
        return original_validation(self, features)

    monkeypatch.setattr(MNRFTrainer, "_stage2_validation", interrupt_after_first_checkpoint)
    with pytest.raises(RuntimeError, match="epoch-boundary"):
        MNRFViewProbe(**kwargs).fit(frame, train_length)
    checkpoint_path = flow_dir / "checkpoints" / "stage2_latest.pt"
    interrupted = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert interrupted["stage_status"] == "in_progress"
    assert interrupted["next_epoch"] == 1
    monkeypatch.setattr(MNRFTrainer, "_stage2_validation", original_validation)
    MNRFViewProbe(**dict(kwargs, resume_from=str(flow_dir))).fit(frame, train_length)
    resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert resumed["completed_steps"] > interrupted["completed_steps"]
    assert resumed["next_epoch"] >= 2
    assert [item["epoch"] for item in resumed["history"]] == [1, 2, 3]
    assert resumed["stage_status"] == "completed"


def test_rejected_flow_resume_preserves_callers_rng_and_files(tmp_path):
    rng = np.random.default_rng(906)
    n, train_length = 520, 430
    frame = pd.DataFrame(rng.normal(size=(n, 2)).astype(np.float32), columns=("x", "y"),
                         index=pd.date_range("2026-04-25", periods=n, freq="min"))
    frame["label"] = np.r_[np.zeros(train_length, dtype=np.int64), np.arange(n - train_length) % 2]
    base = dict(seq_len=32, pattern_len=32, batch_size=32, stage1_epochs=1,
                stage2_epochs=1, patience=1, device="cpu", cpu_threads=1,
                gmm_fit_max_windows=32, task_name="SYN.csv", mode="stage1")
    source = tmp_path / "source"
    MNRFViewProbe(**dict(base, output_dir=str(source), artifact_policy="full")).fit(frame, train_length)
    flow_dir = tmp_path / "flow"
    flow_kwargs = dict(base, mode="flow", source_dir=str(source), output_dir=str(flow_dir),
                       artifact_policy="results")
    probe = MNRFViewProbe(**flow_kwargs)
    probe._finalize_results_artifact = lambda: (_ for _ in ()).throw(RuntimeError("controlled"))
    with pytest.raises(RuntimeError, match="controlled"):
        probe.fit(frame, train_length, frame["label"].to_numpy()[train_length:])
    stats_path = flow_dir / "input_statistics.npz"
    with np.load(stats_path, allow_pickle=False) as archive:
        stats = {key: archive[key].copy() for key in archive.files}
    stats["mean"].flat[0] += np.float32(0.1)
    np.savez(stats_path, **stats)

    def snapshot(path):
        return {str(item.relative_to(path)): (item.stat().st_size, item.stat().st_mtime_ns,
                hashlib.sha256(item.read_bytes()).hexdigest())
                for item in path.rglob("*") if item.is_file()}

    target_before = snapshot(flow_dir)
    source_before = snapshot(source)
    restored = MNRFViewProbe(**dict(flow_kwargs, resume_from=str(flow_dir)))

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    python_before = random.getstate()
    numpy_before = copy.deepcopy(np.random.get_state())
    torch_before = torch.get_rng_state().clone()

    def next_values(state):
        random.setstate(copy.deepcopy(state[0]))
        np.random.set_state(copy.deepcopy(state[1]))
        torch.set_rng_state(state[2].clone())
        return random.random(), np.random.rand(), torch.rand(3)

    expected = next_values((python_before, numpy_before, torch_before))
    random.setstate(python_before)
    np.random.set_state(numpy_before)
    torch.set_rng_state(torch_before)
    for _ in range(2):
        with pytest.raises(ValueError, match="raw input statistics binding"):
            restored.fit(frame, train_length, frame["label"].to_numpy()[train_length:])
        assert random.getstate() == python_before
        current_numpy = np.random.get_state()
        assert numpy_before[0] == current_numpy[0]
        assert np.array_equal(numpy_before[1], current_numpy[1])
        assert numpy_before[2:] == current_numpy[2:]
        assert torch.equal(torch_before, torch.get_rng_state())
        assert target_before == snapshot(flow_dir)
        assert source_before == snapshot(source)
        actual = next_values((random.getstate(), np.random.get_state(), torch.get_rng_state()))
        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        assert torch.equal(actual[2], expected[2])
        random.setstate(python_before)
        np.random.set_state(numpy_before)
        torch.set_rng_state(torch_before)


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

    aligned_path = selected_dir / "results" / "aligned_scores.npz"
    original_aligned = aligned_path.read_bytes()
    with np.load(aligned_path, allow_pickle=False) as result:
        altered = {name: np.asarray(result[name]) for name in result.files}
    altered["scores"] = altered["scores"].copy()
    altered["scores"][0] += 1.0
    np.savez_compressed(aligned_path, **altered)
    with pytest.raises(ValueError, match="result binding"):
        load_probe(selected_dir, device="cpu")
    aligned_path.write_bytes(original_aligned)

    metrics_path = selected_dir / "results" / "metrics.json"
    original_metrics = metrics_path.read_bytes()
    metrics = json.loads(original_metrics)
    metrics["auc_roc"] = 1.234
    metrics_path.write_text(json.dumps(metrics, sort_keys=True))
    with pytest.raises(ValueError, match="result binding"):
        load_probe(selected_dir, device="cpu")
    metrics_path.write_bytes(original_metrics)

    changed_frame = frame.copy()
    with pytest.raises(ValueError, match="identity|train split"):
        MNRFViewProbe(
            seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1, patience=1,
            device="cpu", output_dir=str(tmp_path / "bad-boundary"), mode="select_gmm",
            selected_views=("S", "P", "R"), source_dir=str(source_dir),
        ).fit(changed_frame, 480, np.zeros(len(changed_frame) - 480, dtype=np.int64))

    selected_status = selected_dir / "run_status.json"
    before_resume = selected_status.read_bytes()
    with pytest.raises(ValueError, match="identity|source identity|train split"):
        MNRFViewProbe(
            seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1, patience=1,
            device="cpu", output_dir=str(selected_dir), resume_from=str(selected_dir),
            mode="select_gmm", selected_views=("S", "P", "R"), source_dir=str(source_dir),
            gmm_fit_max_windows=32,
        ).fit(changed_frame, 480, np.zeros(len(changed_frame) - 480, dtype=np.int64))
    assert selected_status.read_bytes() == before_resume

    # A modified immutable cache must be rejected before resume writes any
    # status, manifest, progress, or checkpoint artifact.
    damaged_dir = tmp_path / "damaged-resume"
    shutil.copytree(source_dir, damaged_dir)
    damaged_status = json.loads((damaged_dir / "run_status.json").read_text())
    damaged_status["status"] = "gmm_fitting"
    (damaged_dir / "run_status.json").write_text(json.dumps(damaged_status, sort_keys=True))
    cache_path = damaged_dir / "cache" / "train_features.npy"
    cache = np.load(cache_path, mmap_mode="r+")
    cache[0, 0, 0] += 50.0
    cache.flush()
    before_files = sorted(path.relative_to(damaged_dir).as_posix() for path in damaged_dir.rglob("*") if path.is_file())
    before_hashes = {name: hashlib.sha256((damaged_dir / name).read_bytes()).hexdigest() for name in before_files}
    with pytest.raises(ValueError, match="cache hash differs"):
        MNRFViewProbe(
            seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1, patience=1,
            device="cpu", output_dir=str(damaged_dir), resume_from=str(damaged_dir),
            mode="full_gmm", gmm_fit_max_windows=32,
        ).fit(frame, train_length, labels)
    after_files = sorted(path.relative_to(damaged_dir).as_posix() for path in damaged_dir.rglob("*") if path.is_file())
    after_hashes = {name: hashlib.sha256((damaged_dir / name).read_bytes()).hexdigest() for name in after_files}
    assert after_files == before_files
    assert after_hashes == before_hashes


    # A completed variable cannot be resumed after its required statistics
    # binding is removed, even if the finite statistics file is edited too.
    missing_binding = tmp_path / "missing-stat-binding"
    shutil.copytree(source_dir, missing_binding)
    status_path = missing_binding / "run_status.json"
    status = json.loads(status_path.read_text())
    status["status"] = "gmm_fitting"
    status_path.write_text(json.dumps(status, sort_keys=True))
    manifest_path = missing_binding / "probe_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("feature_statistics_sha256", None)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    stats_path = missing_binding / "heads" / "feature_statistics.npz"
    with np.load(stats_path) as stats:
        np.savez(stats_path, mean=stats["mean"] + 25.0, scale=stats["scale"])
    before_files = sorted(path.relative_to(missing_binding).as_posix()
                          for path in missing_binding.rglob("*") if path.is_file())
    before_hashes = {name: hashlib.sha256((missing_binding / name).read_bytes()).hexdigest()
                     for name in before_files}
    with pytest.raises(ValueError, match="feature_statistics_sha256"):
        MNRFViewProbe(
            seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1, patience=1,
            device="cpu", output_dir=str(missing_binding), resume_from=str(missing_binding),
            mode="full_gmm", gmm_fit_max_windows=32,
        ).fit(frame, train_length, labels)
    after_files = sorted(path.relative_to(missing_binding).as_posix()
                         for path in missing_binding.rglob("*") if path.is_file())
    after_hashes = {name: hashlib.sha256((missing_binding / name).read_bytes()).hexdigest()
                    for name in after_files}
    assert after_files == before_files
    assert after_hashes == before_hashes


def test_flow_resume_missing_stage2_checkpoint_is_read_only_rejected(tmp_path):
    rng = np.random.default_rng(2026)
    n, train_length = 520, 430
    frame = pd.DataFrame(rng.normal(size=(n, 2)).astype(np.float32), columns=("a", "b"),
                         index=pd.date_range("2026-06-01", periods=n, freq="min"))
    source_dir = tmp_path / "flow-source"
    MNRFViewProbe(seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1,
                  patience=1, device="cpu", output_dir=str(source_dir), mode="stage1").fit(frame, train_length)
    flow_dir = tmp_path / "flow"
    flow_kwargs = dict(seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1,
                       stage2_epochs=1, patience=1, device="cpu", output_dir=str(flow_dir),
                       mode="flow", source_dir=str(source_dir), gmm_fit_max_windows=16)
    MNRFViewProbe(**flow_kwargs).fit(frame, train_length)
    damaged = tmp_path / "flow-damaged"
    shutil.copytree(flow_dir, damaged)
    (damaged / "checkpoints" / "stage2_latest.pt").unlink()
    status_path = damaged / "run_status.json"
    status = json.loads(status_path.read_text())
    status["status"] = "flow_training"
    status_path.write_text(json.dumps(status, sort_keys=True))
    before = {str(path.relative_to(damaged)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in damaged.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="stage2_latest.pt is missing"):
        MNRFViewProbe(**dict(flow_kwargs, output_dir=str(damaged), resume_from=str(damaged))).fit(frame, train_length)
    after = {str(path.relative_to(damaged)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in damaged.rglob("*") if path.is_file()}
    assert after == before


@pytest.fixture(scope="module")
def interrupted_flow_fixture(tmp_path_factory):
    """Create one real epoch-boundary interruption for all corruption cases."""
    rng = np.random.default_rng(20261018)
    n, train_length = 520, 430
    frame = pd.DataFrame(rng.normal(size=(n, 2)).astype(np.float32), columns=("a", "b"),
                         index=pd.date_range("2026-06-15", periods=n, freq="min"))
    frame["label"] = 0
    root = tmp_path_factory.mktemp("flow-corruption")
    source_dir = root / "source"
    MNRFViewProbe(seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1,
                  stage2_epochs=1, patience=1, device="cpu", output_dir=str(source_dir),
                  mode="stage1").fit(frame, train_length)
    flow_dir = root / "flow"
    kwargs = dict(seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1,
                  stage2_epochs=3, patience=3, device="cpu", output_dir=str(flow_dir),
                  mode="flow", source_dir=str(source_dir))
    original_validation = MNRFTrainer._stage2_validation
    calls = {"count": 0}

    def stop_after_checkpoint(self, features):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("controlled corruption fixture interruption")
        return original_validation(self, features)

    MNRFTrainer._stage2_validation = stop_after_checkpoint
    try:
        with pytest.raises(RuntimeError, match="controlled corruption fixture"):
            MNRFViewProbe(**kwargs).fit(frame, train_length)
    finally:
        MNRFTrainer._stage2_validation = original_validation
    checkpoint_path = flow_dir / "checkpoints" / "stage2_latest.pt"
    original = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert original["stage_status"] == "in_progress"
    assert "fusion_identity.E" in original["flow"]
    assert "variable_maps.0.weight" in original["flow"]
    optimizer_state_id = next(iter(original["optimizer"]["state"]))
    assert "exp_avg" in original["optimizer"]["state"][optimizer_state_id]
    original_step = original["optimizer"]["state"][optimizer_state_id]["step"]
    assert torch.is_tensor(original_step) and original_step.numel() == 1
    assert float(original_step.item()) > 0 and float(original_step.item()).is_integer()
    return {
        "root": root, "source_dir": source_dir, "flow_dir": flow_dir,
        "frame": frame, "train_length": train_length, "kwargs": kwargs,
        "optimizer_state_id": optimizer_state_id,
    }


@pytest.mark.parametrize(
    "corruption",
    (
        "missing-flow-key", "bad-flow-shape", "bad-global-rng", "bad-optimizer-moment",
        "missing-exp-avg", "missing-exp-avg-sq", "missing-step",
        "negative-tensor-step", "boolean-tensor-step", "fractional-tensor-step",
        "negative-python-step", "boolean-python-step", "fractional-python-step",
    ),
    ids=lambda value: value,
)
def test_flow_corrupt_checkpoint_payload_is_rejected_before_any_write(interrupted_flow_fixture, corruption):
    """Each real checkpoint corruption is independently visible in pytest output."""
    fixture = interrupted_flow_fixture
    source_dir = fixture["source_dir"]
    flow_dir = fixture["flow_dir"]
    root = fixture["root"]
    optimizer_state_id = fixture["optimizer_state_id"]
    damaged = root / corruption
    shutil.copytree(flow_dir, damaged)
    damaged_checkpoint = damaged / "checkpoints" / "stage2_latest.pt"
    payload = torch.load(damaged_checkpoint, map_location="cpu", weights_only=False)
    state = payload["optimizer"]["state"][optimizer_state_id]
    if corruption == "missing-flow-key":
        payload["flow"].pop("fusion_identity.E")
    elif corruption == "bad-flow-shape":
        payload["flow"]["variable_maps.0.weight"] = torch.ones(1)
    elif corruption == "bad-global-rng":
        payload["rng"]["torch"] = torch.zeros(1, dtype=torch.uint8)
    elif corruption == "bad-optimizer-moment":
        state["exp_avg"] = torch.ones(1)
    elif corruption == "missing-exp-avg":
        state.pop("exp_avg")
    elif corruption == "missing-exp-avg-sq":
        state.pop("exp_avg_sq")
    elif corruption == "missing-step":
        state.pop("step")
    elif corruption == "negative-tensor-step":
        state["step"] = torch.tensor(-1.0)
    elif corruption == "boolean-tensor-step":
        state["step"] = torch.tensor(True, dtype=torch.bool)
    elif corruption == "fractional-tensor-step":
        state["step"] = torch.tensor(0.5)
    elif corruption == "negative-python-step":
        state["step"] = -1.0
    elif corruption == "boolean-python-step":
        state["step"] = True
    elif corruption == "fractional-python-step":
        state["step"] = 0.5
    else:
        raise AssertionError(corruption)
    torch.save(payload, damaged_checkpoint)

    def snapshot(root):
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()
        }

    before_target = snapshot(damaged)
    before_source = snapshot(source_dir)
    process_python = random.getstate()
    process_numpy = np.random.get_state()
    process_torch = torch.get_rng_state().clone()
    optimizer_corruption = corruption not in {"missing-flow-key", "bad-flow-shape", "bad-global-rng"}
    expected_message = "resume rejected: Flow optimizer" if optimizer_corruption else "resume rejected"
    with pytest.raises(ValueError, match=expected_message):
        MNRFViewProbe(**dict(fixture["kwargs"], output_dir=str(damaged), resume_from=str(damaged))).fit(
            fixture["frame"], fixture["train_length"]
        )
    assert snapshot(damaged) == before_target, corruption
    assert snapshot(source_dir) == before_source, corruption
    assert random.getstate() == process_python
    after_numpy = np.random.get_state()
    assert after_numpy[0] == process_numpy[0]
    assert np.array_equal(after_numpy[1], process_numpy[1])
    assert after_numpy[2:] == process_numpy[2:]
    assert torch.equal(torch.get_rng_state(), process_torch)


def test_flow_resume_rejects_shared_wrong_source_identity_before_writes(interrupted_flow_fixture):
    fixture = interrupted_flow_fixture
    root = fixture["root"]
    damaged = root / "wrong-source-identity"
    shutil.copytree(fixture["flow_dir"], damaged)
    checkpoint_path = damaged / "checkpoints" / "stage2_latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["stage2_identity"] = "legacy-random-encoder-identity"
    torch.save(checkpoint, checkpoint_path)
    manifest_path = damaged / "probe_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stage2_identity"] = "legacy-random-encoder-identity"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    def snapshot(path):
        return {item.relative_to(path).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
                for item in path.rglob("*") if item.is_file()}

    before = snapshot(damaged)
    source_before = snapshot(fixture["source_dir"])
    with pytest.raises(ValueError, match="not bound to the validated source encoder"):
        MNRFViewProbe(**dict(fixture["kwargs"], output_dir=str(damaged),
                             resume_from=str(damaged))).fit(fixture["frame"], fixture["train_length"])
    assert snapshot(damaged) == before
    assert snapshot(fixture["source_dir"]) == source_before


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


def test_indexed_strategy_keeps_endpoint_coordinates_without_padding():
    class FakeProbe:
        output_dir = None

        def fit(self, frame, train_length, labels=None):
            self._metadata = {
                "scores": np.arange(59, dtype=np.float32),
                "predictions": np.zeros(59, dtype=np.int64),
                "time_indices": np.arange(121, 180, dtype=np.int64),
                "local_indices": np.arange(31, 90, dtype=np.int64),
            }
            return self

        def score_with_metadata(self):
            return self._metadata

    train = pd.DataFrame(np.zeros((90, 2), dtype=np.float32), columns=("a", "b"))
    test = pd.DataFrame(np.zeros((90, 2), dtype=np.float32), columns=("a", "b"))
    result = IndexedViewProbeStrategy(FakeProbe()).run(train, test, np.zeros(90, dtype=np.int64))
    assert np.array_equal(result["local_indices"], np.arange(31, 90))
    assert np.array_equal(result["time_indices"], np.arange(121, 180))


def test_indexed_strategy_preserves_timestamp_gap_and_rejects_cross_gap_scores():
    class GapProbe:
        output_dir = None

        def fit(self, frame, train_length, labels=None):
            self.seen_index = frame.index
            # The expected protocol endpoints for a 32-point window are the
            # 19 points before the gap and 9 points after it.
            self._metadata = {
                "scores": np.zeros(28, dtype=np.float32),
                "predictions": np.zeros(28, dtype=np.int64),
                "time_indices": np.r_[np.arange(461, 480), np.arange(511, 520)],
                "local_indices": np.r_[np.arange(31, 50), np.arange(81, 90)],
            }

        def score_with_metadata(self):
            return self._metadata

    train = pd.DataFrame(np.zeros((430, 2), dtype=np.float32), columns=("a", "b"),
                         index=pd.date_range("2026-01-01", periods=430, freq="min"))
    first = pd.date_range(train.index[-1] + pd.Timedelta(minutes=1), periods=50, freq="min")
    second = pd.date_range(first[-1] + pd.Timedelta(minutes=61), periods=40, freq="min")
    test = pd.DataFrame(np.zeros((90, 2), dtype=np.float32), columns=("a", "b"),
                        index=first.append(second))
    probe = GapProbe()
    result = IndexedViewProbeStrategy(probe).run(train, test, np.zeros(90, dtype=np.int64))
    assert isinstance(probe.seen_index, pd.DatetimeIndex)
    assert (probe.seen_index[480] - probe.seen_index[479]) == pd.Timedelta(minutes=61)
    assert len(result["time_indices"]) == 28
