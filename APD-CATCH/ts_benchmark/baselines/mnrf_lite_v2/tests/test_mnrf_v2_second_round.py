import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mnrf_lite_v2_preflight import preflight
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2, _hash_array
from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
from ts_benchmark.baselines.mnrf_lite_v2.training import MNRFTrainer
from ts_benchmark.baselines.mnrf_lite_v2.windows import (
    TimeIndex, continuous_segments, infer_expected_interval, normalize_time_index, window_endpoints,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _write_metadata(root, task, train_length, source):
    folder = root / "dataset" / ("external_validation" if source == "external_detect" else "anomaly_detect")
    (folder / "data").mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame([{"file_name": f"{task}.csv", "train_lens": train_length, "if_univariate": False}])
    metadata.to_csv(folder / ("EXTERNAL_DETECT_META.csv" if source == "external_detect" else "DETECT_META.csv"), index=False)
    return folder


@pytest.mark.parametrize("feature_count", [2, 4])
def test_preflight_uses_real_external_wide_loader(tmp_path, feature_count, capsys):
    task, n, train_n = "toy", 500, 420
    folder = _write_metadata(tmp_path, task, train_n, "external_detect")
    values = {f"v{i}": np.linspace(i, i + 1, n) for i in range(feature_count)}
    data = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=n, freq="s"), **values,
                         "label": np.r_[np.zeros(n - 1, dtype=int), 1]})
    data.to_csv(folder / "data" / f"{task}.csv", index=False)
    result = preflight(task, "external_detect", seq_len=32, root=tmp_path)
    assert len(result["feature_names"]) == feature_count
    assert result["train_length"] == train_n
    assert result["test_window_count"] > 0
    assert "training/cache not started" in capsys.readouterr().out


def test_preflight_uses_real_internal_long_loader(tmp_path):
    task, n, train_n = "toy", 500, 420
    folder = _write_metadata(tmp_path, task, train_n, "large_detect")
    timestamps = pd.date_range("2025-01-01", periods=n, freq="s")
    rows = []
    for column, values in (("a", np.arange(n)), ("b", np.arange(n) * 2), ("label", np.zeros(n))):
        rows.extend({"date": timestamps[i], "value": values[i], "cols": column} for i in range(n))
    pd.DataFrame(rows).to_csv(folder / "data" / f"{task}.csv", index=False)
    result = preflight(task, "large_detect", seq_len=32, root=tmp_path)
    assert result["feature_names"] == ("a", "b")
    assert result["train_length"] == train_n


def test_preflight_rejects_bad_external_schema_metadata_and_values(tmp_path):
    task = "toy"
    folder = _write_metadata(tmp_path, task, 420, "external_detect")
    data = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=500, freq="s"),
                         "a": np.arange(500), "b": np.arange(500), "label": np.zeros(500)})
    data.drop(columns="b").to_csv(folder / "data" / f"{task}.csv", index=False)
    with pytest.raises(ValueError, match="task=toy source=external_detect"):
        preflight(task, "external_detect", seq_len=32, root=tmp_path)
    data["b"] = np.arange(500, dtype=float)
    data.loc[20, "a"] = np.inf
    data.to_csv(folder / "data" / f"{task}.csv", index=False)
    with pytest.raises(ValueError, match="invalid prepared features|non-finite"):
        preflight(task, "external_detect", seq_len=32, root=tmp_path)
    data.loc[20, "a"] = 20
    data.to_csv(folder / "data" / f"{task}.csv", index=False)
    meta = pd.read_csv(folder / "EXTERNAL_DETECT_META.csv")
    pd.concat([meta, meta]).to_csv(folder / "EXTERNAL_DETECT_META.csv", index=False)
    with pytest.raises(ValueError, match="not unique"):
        preflight(task, "external_detect", seq_len=32, root=tmp_path)


def test_frozen_train_cadence_splits_doubled_test_rate_and_prefix_is_stable():
    train_times = pd.date_range("2025-01-01", periods=64, freq="s")
    expected = infer_expected_interval(normalize_time_index(train_times), len(train_times))
    test_times = pd.date_range("2025-02-01", periods=20, freq="2s")
    segments, audit = continuous_segments(normalize_time_index(test_times), len(test_times), expected_interval=expected)
    endpoints = window_endpoints(segments, 4)
    assert expected == 1_000_000_000
    assert segments == [(i, i + 1) for i in range(20)]
    assert len(endpoints) == 0
    prefix = pd.date_range("2025-02-01", periods=8, freq="s")
    prefix_segments, _ = continuous_segments(normalize_time_index(prefix), len(prefix), expected_interval=expected)
    appended = prefix.append(pd.date_range(prefix[-1] + pd.Timedelta(seconds=2), periods=4, freq="2s"))
    appended_segments, _ = continuous_segments(normalize_time_index(appended), len(appended), expected_interval=expected)
    assert prefix_segments == appended_segments[:len(prefix_segments)]
    assert window_endpoints(prefix_segments, 4).tolist() == window_endpoints(appended_segments, 4)[
        window_endpoints(appended_segments, 4) < len(prefix)
    ].tolist()
    assert audit["expected_interval"] == expected


def test_timezone_canonicalization_and_hash_are_offset_independent():
    local = pd.date_range("2025-01-01", periods=12, freq="min", tz="Asia/Shanghai")
    utc = local.tz_convert("UTC")
    local_index = normalize_time_index(local)
    utc_index = normalize_time_index(utc)
    assert np.array_equal(local_index.values_ns, utc_index.values_ns)
    assert _hash_array(local_index) == _hash_array(utc_index)
    local_segments = continuous_segments(local_index, len(local), expected_interval=60_000_000_000)[0]
    utc_segments = continuous_segments(utc_index, len(utc), expected_interval=60_000_000_000)[0]
    assert local_segments == utc_segments == [(0, 12)]
    gap = local.copy()
    gap_values = gap.asi8.copy()
    gap_values[7:] += 86_400_000_000_000
    gap_index = pd.DatetimeIndex(gap_values, tz="UTC")
    gap_segments, _ = continuous_segments(normalize_time_index(gap_index), len(gap_index),
                                          expected_interval=60_000_000_000)
    assert gap_segments == [(0, 7), (7, 12)]
    assert _hash_array(normalize_time_index(gap_index)) == _hash_array(normalize_time_index(gap_index.tz_convert("Asia/Shanghai")))


def test_datetime_coordinates_are_canonical_nanoseconds_across_resolutions():
    base = pd.date_range("2025-01-01", periods=12, freq="s")
    base_ns = base.to_numpy()
    canonical = None
    for unit in ("ns", "us", "ms", "s"):
        represented = base_ns.astype(f"datetime64[{unit}]")
        normalized = normalize_time_index(represented)
        if canonical is None:
            canonical = normalized
        assert np.array_equal(normalized.values_ns, canonical.values_ns)
        assert infer_expected_interval(normalized) == 1_000_000_000
        assert _hash_array(normalized) == _hash_array(canonical)
        assert _hash_array(represented) == _hash_array(base_ns)
    aware_utc = base.tz_localize("UTC")
    aware_local = aware_utc.tz_convert("Asia/Shanghai")
    assert np.array_equal(normalize_time_index(aware_utc).values_ns,
                          normalize_time_index(aware_local).values_ns)
    naive_ns = normalize_time_index(base_ns.astype("datetime64[ns]"))
    naive_seconds = normalize_time_index(base_ns.astype("datetime64[s]"))
    assert np.array_equal(naive_ns.values_ns, naive_seconds.values_ns)
    numpy_seconds = base.to_numpy().astype("datetime64[s]")
    assert np.array_equal(normalize_time_index(numpy_seconds).values_ns, naive_ns.values_ns)


def test_gap_and_multiple_entity_boundaries_have_same_indices_across_units():
    base = pd.date_range("2025-01-01", periods=80, freq="s")
    raw = base.to_numpy()
    raw[40:] += np.timedelta64(30, "m")
    index = pd.DatetimeIndex(raw)
    normalized_ns = normalize_time_index(raw.astype("datetime64[ns]"))
    normalized_seconds = normalize_time_index(raw.astype("datetime64[s]"))
    boundaries = np.array(["a"] * 20 + ["b"] * 40 + ["c"] * 20)
    segments_ns, _ = continuous_segments(normalized_ns, 80, expected_interval=1_000_000_000,
                                         boundary_ids=boundaries)
    segments_seconds, _ = continuous_segments(normalized_seconds, 80, expected_interval=1_000_000_000,
                                              boundary_ids=boundaries)
    endpoints_ns = window_endpoints(segments_ns, 8)
    endpoints_seconds = window_endpoints(segments_seconds, 8)
    assert segments_ns == segments_seconds
    assert np.array_equal(endpoints_ns, endpoints_seconds)
    assert 46 not in endpoints_ns
    assert 47 in endpoints_ns


def test_datetime_column_and_index_share_canonical_coordinates_and_bad_times_fail():
    column = MNRFLiteV2._time_index(pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=5, freq="s")}))
    index = MNRFLiteV2._time_index(pd.DataFrame(index=pd.date_range("2025-01-01", periods=5, freq="s")))
    assert np.array_equal(column.values_ns, index.values_ns)
    with pytest.raises(ValueError, match="NaT"):
        normalize_time_index(pd.DatetimeIndex(["2025-01-01", "NaT"]))
    valid = pd.date_range("2025-01-01", periods=5, freq="s")
    with pytest.raises(ValueError, match="duplicate or reverse"):
        continuous_segments(valid.insert(3, valid[2]), 6, expected_interval=1_000_000_000)
    with pytest.raises(ValueError, match="duplicate or reverse"):
        continuous_segments(valid[::-1], 5, expected_interval=1_000_000_000)
    with pytest.raises(ValueError, match="cannot be parsed"):
        normalize_time_index(np.array(["2025-01-01", "not-a-time"], dtype=object), explicit_time=True)


def test_stage1_completed_early_stop_resume_does_not_train_again(tmp_path, monkeypatch):
    config = MNRFConfig(seq_len=32, batch_size=8, stage1_epochs=5, stage2_epochs=1,
                        patience=1, device="cpu")
    values = np.random.default_rng(1).normal(size=(80, 2)).astype(np.float32)
    trainer = MNRFTrainer(config, 2, torch.device("cpu"))
    validation_values = iter((1.0, 2.0))
    monkeypatch.setattr(trainer, "_validation_stage1", lambda *args, **kwargs: next(validation_values))
    batch_count = [0]
    original = trainer._pretrain_batch
    def counted(*args, **kwargs):
        batch_count[0] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(trainer, "_pretrain_batch", counted)
    trainer.stage1(values, (0, 48), (48, 80), tmp_path, resume_id="same")
    assert torch.load(tmp_path / "stage1_latest.pt", weights_only=False)["completion_reason"] == "early_stopping"
    completed_batches = batch_count[0]
    resume_config = MNRFConfig(**{**config.to_dict(), "resume_from": str(tmp_path)})
    resumed = MNRFTrainer(resume_config, 2, torch.device("cpu"))
    monkeypatch.setattr(resumed, "_validation_stage1", lambda *args, **kwargs: pytest.fail("completed stage1 revalidated"))
    monkeypatch.setattr(resumed, "_pretrain_batch", counted)
    resumed.stage1(values, (0, 48), (48, 80), tmp_path, resume_id="same")
    assert batch_count[0] == completed_batches


def test_stage1_epoch_boundary_resume_matches_uninterrupted_best_state(tmp_path, monkeypatch):
    from ts_benchmark.baselines.mnrf_lite_v2 import training as training_module
    from ts_benchmark.baselines.mnrf_lite_v2.training import seed_all

    config_values = dict(seq_len=32, batch_size=8, stage1_epochs=2, stage2_epochs=1,
                         patience=5, device="cpu")
    values = np.random.default_rng(77).normal(size=(80, 2)).astype(np.float32)
    uninterrupted_dir = tmp_path / "uninterrupted"
    interrupted_dir = tmp_path / "interrupted"
    uninterrupted_dir.mkdir()
    interrupted_dir.mkdir()

    seed_all(1234)
    uninterrupted = MNRFTrainer(MNRFConfig(**config_values), 2, torch.device("cpu"))
    monkeypatch.setattr(uninterrupted, "_validation_stage1", lambda *args, **kwargs: 1.0)
    uninterrupted.stage1(values, (0, 40), (40, 72), uninterrupted_dir, resume_id="same")
    expected = torch.load(uninterrupted_dir / "best_encoder.pt", weights_only=False)

    seed_all(1234)
    interrupted = MNRFTrainer(MNRFConfig(**config_values), 2, torch.device("cpu"))
    monkeypatch.setattr(interrupted, "_validation_stage1", lambda *args, **kwargs: 1.0)
    original_append = training_module._append_log
    raised = [False]
    def durable_log_then_interrupt(path, record):
        original_append(path, record)
        if record.get("stage") == "stage1" and record.get("epoch") == 1 and not raised[0]:
            raised[0] = True
            raise RuntimeError("simulated interruption after durable epoch checkpoint")
    monkeypatch.setattr(training_module, "_append_log", durable_log_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted.stage1(values, (0, 40), (40, 72), interrupted_dir,
                           log_path=interrupted_dir / "logs" / "training.jsonl", resume_id="same")
    monkeypatch.setattr(training_module, "_append_log", original_append)

    seed_all(9999)
    resumed = MNRFTrainer(MNRFConfig(**{**config_values, "resume_from": str(interrupted_dir)}), 2,
                          torch.device("cpu"))
    monkeypatch.setattr(resumed, "_validation_stage1", lambda *args, **kwargs: 1.0)
    resumed.stage1(values, (0, 40), (40, 72), interrupted_dir,
                   log_path=interrupted_dir / "logs" / "training.jsonl", resume_id="same")
    actual = torch.load(interrupted_dir / "best_encoder.pt", weights_only=False)
    assert expected.keys() == actual.keys()
    for key in expected["encoder"]:
        assert torch.allclose(expected["encoder"][key], actual["encoder"][key], atol=1e-7, rtol=1e-7)
    latest = torch.load(interrupted_dir / "stage1_latest.pt", weights_only=False)
    assert latest["next_epoch"] == 2 and latest["completion_reason"] == "epoch_limit"
    records = [json.loads(line) for line in (interrupted_dir / "logs" / "training.jsonl").read_text().splitlines()]
    assert [record["epoch"] for record in records if record.get("stage") == "stage1" and "epoch" in record] == [1, 2]


def test_stage2_early_stop_resume_skips_optimizer_batches_and_checks_binding(tmp_path, monkeypatch):
    config = MNRFConfig(seq_len=8, pattern_len=8, batch_size=4, stage1_epochs=1, stage2_epochs=5,
                        patience=1, device="cpu")
    train = np.random.default_rng(2).normal(size=(4, 2, 25)).astype(np.float32)
    validation = train.copy()
    trainer = MNRFTrainer(config, 2, torch.device("cpu"))
    trainer.fixed_feature.fit(train)
    sequence = iter((0.2, 0.3))
    monkeypatch.setattr(trainer, "_stage2_validation", lambda _: next(sequence))
    binding = {"encoder_fingerprint": "fixed"}
    trainer.stage2(train, validation, tmp_path, resume_id="same", stage2_identity=binding)
    checkpoint = torch.load(tmp_path / "stage2_latest.pt", weights_only=False)
    assert checkpoint["completion_reason"] == "early_stopping"
    steps = checkpoint["completed_steps"]
    resumed = MNRFTrainer(MNRFConfig(**{**config.to_dict(), "resume_from": str(tmp_path)}), 2, torch.device("cpu"))
    resumed.fixed_feature.fit(train)
    monkeypatch.setattr(resumed, "_stage2_validation", lambda _: pytest.fail("completed stage2 revalidated"))
    resumed.stage2(train, validation, tmp_path, resume_id="same", stage2_identity=binding)
    assert torch.load(tmp_path / "stage2_latest.pt", weights_only=False)["completed_steps"] == steps
    with pytest.raises(ValueError, match="frozen encoder/statistics/cache/time identity"):
        MNRFTrainer(MNRFConfig(**{**config.to_dict(), "resume_from": str(tmp_path)}), 2,
                    torch.device("cpu")).stage2(train, validation, tmp_path, resume_id="same",
                                                  stage2_identity={"encoder_fingerprint": "changed"})


def test_stage2_validation_disables_grad_only_for_validation_and_preserves_training(tmp_path):
    config = MNRFConfig(batch_size=2, device="cpu")
    trainer = MNRFTrainer(config, 2, torch.device("cpu"))
    features = np.random.default_rng(4).normal(size=(3, 2, 25)).astype(np.float32)
    trainer.fixed_feature.fit(features)
    first_parameter = next(trainer.flow.parameters())
    first_parameter.grad = torch.full_like(first_parameter, 0.25)
    prior_grad = first_parameter.grad.clone()
    grad_states, grad_fns = [], []
    def record_forward(module, inputs, output):
        grad_states.append(torch.is_grad_enabled())
        grad_fns.append(output["nll"].grad_fn)
    handle = trainer.flow.register_forward_hook(record_forward)
    loss = trainer._stage2_validation(features)
    handle.remove()
    assert np.isfinite(loss)
    assert grad_states and not any(grad_states)
    assert all(fn is None for fn in grad_fns)
    assert torch.equal(first_parameter.grad, prior_grad)
    trainer.flow.train()
    batch = torch.as_tensor(trainer._normalized_feature_batch(features, slice(0, 2)))
    train_loss = trainer.flow(batch, trainer.encoder.point.variable_identity)["nll"].mean()
    train_loss.backward()
    assert first_parameter.grad is not None and torch.isfinite(first_parameter.grad).all()
