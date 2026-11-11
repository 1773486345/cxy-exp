"""Formal MSL mapping/target wiring, without fitting or model forward."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import mnrf_msl_sequence_boundaries as boundaries
from scripts.analyze_mnrf_psm_tail_fusion import normal_window_groups, tail_fusion, validation_tail
from scripts.preflight_mnrf_lite_v2 import preflight
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
from ts_benchmark.baselines.mnrf_lite_v2.readout import flow_scores, score_target_indices
from ts_benchmark.baselines.mnrf_lite_v2.training import MNRFTrainer, purged_ranges
from ts_benchmark.baselines.mnrf_lite_v2.windows import continuous_segments, intersect_segments, window_endpoints
from ts_benchmark.evaluation.strategy import mnrf_valid_detect as strategy_module


@pytest.fixture
def mapped_source(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    channels = [f"C-{index:02}" for index in range(27)]
    n_train, n_test = 27*80, 27*45
    frame = pd.DataFrame(np.random.default_rng(2021).normal(size=(n_train+n_test, 55)).astype(np.float32),
                         columns=boundaries.FEATURE_COLUMNS,
                         index=pd.date_range("2025", periods=n_train+n_test, freq="min"))
    frame["label"] = 0
    frame.iloc[n_train+34:n_train+38, -1] = 1
    root = tmp_path / "data_root"
    (root / "data").mkdir(parents=True)
    path = root / "data/MSL.csv"
    frame.rename_axis("date").reset_index().melt(id_vars="date", var_name="cols", value_name="data")[
        ["date", "data", "cols"]].to_csv(path, index=False)
    pd.DataFrame([dict(file_name="MSL.csv", train_lens=n_train)]).to_csv(root / "DETECT_META.csv", index=False)
    # Bind the synthetic values through the actual loader, not a fake source validator.
    from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource
    loader = LocalAnomalyDetectDataSource(str(root))
    loader.load_series_list(["MSL.csv"])
    loaded = loader.dataset.get_series("MSL.csv")
    digest = boundaries._frame_feature_hash(loaded, boundaries.FEATURE_COLUMNS)
    monkeypatch.setattr(boundaries, "EXPECTED_FEATURE_SHA256", digest)
    mapping = dict(schema="msl-raw-mapping-v1", channels=channels,
                   channel_row_counts={name: dict(train=80, test=45) for name in channels},
                   train_rows=n_train, test_rows=n_test, local_feature_names=boundaries.FEATURE_COLUMNS,
                   columns=55, expected_feature_sha256=digest, local_csv_sha256=boundaries.sha256_file(path))
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping))
    return root, loaded, n_train, mapping_path


def test_real_strategy_annotates_full_axis_before_split_and_binds_contract(mapped_source, tmp_path, monkeypatch):
    root, frame, n_train, mapping_path = mapped_source
    pool = SimpleNamespace(get_series=lambda name: frame,
                           get_series_meta_info=lambda name: pd.Series(dict(train_lens=n_train)))
    monkeypatch.setattr(strategy_module, "DataPool", lambda: SimpleNamespace(get_pool=lambda: pool))
    strategy = strategy_module.MNRFValidDetect(dict(strategy_name="mnrf_valid_detect",
        msl_sequence_mapping=str(mapping_path), readout_data_root=str(root)), None)
    train, _, test, _ = strategy.split_data("MSL.csv")
    assert "entity_id" not in frame  # the shared source is never changed
    assert train.entity_id.iloc[-1] == "train:C-26"
    assert test.entity_id.iloc[0] == "test:C-00"
    assert list(MNRFLiteV2._features_from_frame(train)[1]) == boundaries.FEATURE_COLUMNS
    assert train.attrs == test.attrs == strategy._full_input_frame.attrs
    model = MNRFLiteV2(task_name="MSL.csv", score_target="col_0", seq_len=32,
                       device="cpu", output_dir=str(tmp_path / "run"))
    model.bind_readout_data(strategy._full_input_frame, n_train, "MSL.csv", "MSL-sequence-bounded-col0", str(root))
    contract = model._readout_data_contract
    assert contract["score_target"] == "col_0" and contract["full_entity_ids_sha256"]
    assert contract["sequence_mapping"]["mapping_sha256"] == boundaries.sha256_file(mapping_path)
    ranges = purged_ranges(len(train), 32)
    segments, _ = continuous_segments(model._time_index(train), len(train),
                                      boundary_ids=model._boundary_ids(train))
    for name in ("train", "validation", "calibration"):
        endpoints = window_endpoints(intersect_segments(segments, *ranges[name]), 32)
        for end in endpoints:
            assert train.entity_id.iloc[end-31:end+1].nunique() == 1
    test_segments, _ = continuous_segments(model._time_index(test), len(test),
                                           boundary_ids=model._boundary_ids(test))
    test_endpoints = window_endpoints(test_segments, 32)
    assert len(test_endpoints) == 27*(45-31)
    assert 45 not in test_endpoints and 76 in test_endpoints  # no cross-channel warmup
    params = dict(task_name="MSL.csv", score_target="col_0", seq_len=32, score_readout="flow", device="cpu")
    result = preflight("MSL", root, params, msl_sequence_mapping=mapping_path)
    assert result["test_endpoints"] == len(test_endpoints)
    assert all(r["cross_boundary_windows_remaining"] == 0 for r in result["boundary_window_checks"].values())
    assert result["boundary_window_checks"]["test"]["excluded_cross_boundary_windows"] == 26*31


def test_mapping_source_and_order_mismatch_rejected(mapped_source):
    root, frame, train, mapping_path = mapped_source
    swapped = frame.copy()
    swapped.columns = ["col_1", "col_0"] + list(swapped.columns[2:])
    with pytest.raises(ValueError, match="feature order"):
        boundaries.prepare_formal_frame(swapped, train, root / "data/MSL.csv", mapping_path)
    changed = frame.copy()
    changed.iloc[0, 0] += 1
    with pytest.raises(ValueError, match="feature values/order"):
        boundaries.prepare_formal_frame(changed, train, root / "data/MSL.csv", mapping_path)
    with (root / "data/MSL.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="CSV digest"):
        boundaries.prepare_formal_frame(frame, train, root / "data/MSL.csv", mapping_path)


def test_mapping_and_target_persist_in_actual_run_identity_and_deployment(mapped_source, tmp_path, monkeypatch):
    root, frame, train_length, path = mapped_source
    annotated = boundaries.prepare_formal_frame(frame, train_length, root / "data/MSL.csv", path)
    run = tmp_path / "bound-run"
    model = MNRFLiteV2(seq_len=32, task_name="MSL.csv", score_target="col_0", output_dir=str(run), device="cpu")
    model.bind_readout_data(annotated, train_length, "MSL.csv", "MSL-sequence-bounded-col0", str(root))
    def stop_before_training(*args, **kwargs):
        raise RuntimeError("synthetic stop before training")
    monkeypatch.setattr(MNRFTrainer, "stage1", stop_before_training)
    with pytest.raises(RuntimeError, match="synthetic stop"):
        model.detect_fit(annotated.iloc[:train_length].drop(columns="label"))
    identity = json.loads((run / "run_identity.json").read_text())
    contract = json.loads((run / "readout_data_contract.json").read_text())
    assert identity["config"]["score_target"] == contract["score_target"] == "col_0"
    assert identity["temporal_rule"]["sequence_mapping"] == contract["sequence_mapping"]
    assert identity["temporal_rule"]["entity_boundary_columns"] == ["entity_id"]
    model.trainer.fixed_feature.mean_ = np.zeros((55, 25), np.float32)
    model.trainer.fixed_feature.scale_ = np.ones((55, 25), np.float32)
    model.trainer.calibration_center = np.zeros(55, np.float32)
    model.trainer.calibration_scale = np.ones(55, np.float32)
    model.trainer.threshold = 3.5
    model.trainer.noise_vectors = torch.zeros((4, 25))
    model._save_artifacts()
    restored = MNRFLiteV2(device="cpu").load_checkpoint(run / "deployment.pt")
    assert restored.config.score_target == "col_0" and restored.trainer.threshold == 3.5
    assert restored.temporal_rule == identity["temporal_rule"]
    old = torch.load(run / "deployment.pt", weights_only=False)
    old["config"].pop("score_target")
    old["temporal_rule"].pop("sequence_mapping")
    old["temporal_rule"].pop("score_target")
    torch.save(old, tmp_path / "old-all-variable.pt")
    requested = MNRFLiteV2(task_name="MSL.csv", score_target="col_0", device="cpu")
    with pytest.raises(ValueError, match="score target differs"):
        requested.load_checkpoint(tmp_path / "old-all-variable.pt")


def test_col0_calibration_and_real_scoring_path_ignore_other_nll(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = MNRFConfig(seq_len=32, batch_size=64, task_name="MSL.csv", score_target="col_0")
    trainer = MNRFTrainer(config, 55, torch.device("cpu"))
    names = boundaries.FEATURE_COLUMNS
    indices = score_target_indices(names, config.score_target)
    validation = np.zeros((5, 55, 25), np.float32)
    calibration = np.zeros((7, 55, 25), np.float32)
    validation[:, 0, 0] = np.arange(5)
    calibration[:, 0, 0] = np.arange(7)
    validation[:, 1:, 0] = 1000*np.arange(5)[:, None]
    calibration[:, 1:, 0] = 1e9
    def saved_nll(features, output_path=None):
        value = np.array(features[..., 0], copy=True)
        if output_path is not None:
            np.save(output_path, value)
        return value
    monkeypatch.setattr(trainer, "per_variable_scores", saved_nll)
    trainer.calibrate(validation, calibration, tmp_path / "cal", score_indices=indices)
    expected = flow_scores(calibration[..., 0], trainer.calibration_center, trainer.calibration_scale, indices)
    assert trainer.threshold == float(np.quantile(expected, .995))
    assert trainer.threshold < 10
    assert np.max(flow_scores(calibration[..., 0], trainer.calibration_center, trainer.calibration_scale)) > 1e5
    model = MNRFLiteV2(seq_len=32, batch_size=64, task_name="MSL.csv", score_target="col_0", device="cpu")
    model.trainer, model.variable_names, model._fitted = trainer, tuple(names), True
    mapping = {"mapping_sha256": "synthetic-boundary"}
    model.temporal_rule = dict(time_kind="row_order", expected_interval_ns=None, tolerance=.05,
                               entity_boundary_columns=["entity_id"], sequence_mapping=mapping, score_target="col_0")
    trainer.fixed_input.mean_ = np.zeros(55, np.float32)
    trainer.fixed_input.scale_ = np.ones(55, np.float32)
    # Isolate scoring/aggregation: no learned encoder or Flow forward is invoked.
    def encode(windows):
        raw = torch.zeros((len(windows), 55, 25))
        raw[:, :, 0] = windows[:, -1, :]
        return {"features": raw}
    monkeypatch.setattr(trainer.encoder, "encode_views", encode)
    frame = pd.DataFrame(np.zeros((80, 55), np.float32), columns=names)
    frame["col_0"] = np.arange(80) % 7
    frame["entity_id"] = ["test:A"]*40 + ["test:B"]*40
    frame.attrs["mnrf_sequence_mapping"] = mapping
    first = model.score_with_metadata(frame, index_offset=2160)
    changed = frame.copy()
    changed.loc[:, names[1:]] = 1e12
    second = model.score_with_metadata(changed, index_offset=2160)
    for key in ("scores", "predictions", "time_indices", "valid_mask"):
        np.testing.assert_array_equal(first[key], second[key])
    assert len(first["scores"]) == 18
    np.testing.assert_array_equal(first["local_time_indices"], np.r_[31:40, 71:80])
    expected_test = (frame.col_0.to_numpy(dtype=np.float32)[first["local_time_indices"]] - trainer.calibration_center[0]) / trainer.calibration_scale[0]
    np.testing.assert_array_equal(first["scores"], expected_test)
    np.testing.assert_array_equal(first["predictions"], expected_test > trainer.threshold)
    bad = frame.copy()
    bad.attrs.clear()
    with pytest.raises(ValueError, match="mapping differs"):
        model.score_with_metadata(bad)


@pytest.mark.parametrize("old_target,old_mapping", [("", False), ("", True), ("col_0", False)])
def test_old_all_variable_or_unbounded_identity_cannot_resume(mapped_source, tmp_path, old_target, old_mapping):
    root, frame, n_train, path = mapped_source
    annotated = boundaries.prepare_formal_frame(frame, n_train, root / "data/MSL.csv", path)
    run = tmp_path / "old-run"
    run.mkdir()
    identity = dict(resume_id="old-id", config=dict(score_target=old_target),
                    temporal_rule=dict(sequence_mapping=annotated.attrs) if old_mapping else {})
    (run / "run_identity.json").write_text(json.dumps(identity))
    before = (run / "run_identity.json").read_bytes()
    model = MNRFLiteV2(seq_len=32, task_name="MSL.csv", score_target="col_0", output_dir=str(run),
                       resume_from=str(run), device="cpu")
    with pytest.raises(ValueError, match="boundary mapping or score target differs"):
        model.detect_fit(annotated.iloc[:n_train].drop(columns="label"))
    assert (run / "run_identity.json").read_bytes() == before
    assert sorted(p.name for p in run.iterdir()) == ["run_identity.json"]


def test_msl_wrapper_captures_real_parameters_without_starting(tmp_path):
    root = Path(__file__).resolve().parents[4]
    project = tmp_path / "project"
    wrapper = project / "scripts/multivariate_detection/detect_score/MSL_script/MNRFLiteV2.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_bytes((root / "scripts/multivariate_detection/detect_score/MSL_script/MNRFLiteV2.sh").read_bytes())
    capture = project / "scripts/run_mnrf_lite_v2_task.sh"
    capture.write_text('python - "$@" <<\'PY\'\nimport json, os, sys\nprint(json.dumps({"args":sys.argv[1:], "target":os.environ["MNRF_SCORE_TARGET"], "mapping":os.environ["MNRF_MSL_SEQUENCE_MAPPING"], "policy":os.environ["MNRF_ARTIFACT_POLICY"], "cuda":os.environ.get("CUDA_VISIBLE_DEVICES")}))\nPY\n')
    env = dict(os.environ, PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ["PATH"], CUDA_VISIBLE_DEVICES="3,5")
    env.pop("MNRF_ARTIFACT_POLICY", None)
    result = subprocess.run(["bash", str(wrapper), "--preflight"], env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["args"] == ["MSL", "large_detect", "--preflight"]
    assert value["target"] == "col_0" and value["policy"] == "readout" and value["cuda"] == "3,5"
    assert value["mapping"].endswith("msl-raw-mapping-20261031.json")


def test_tail_rule_inclusive_ties_saturation_and_budget():
    from scripts.analyze_mnrf_r_residual_readout import budget_point
    np.testing.assert_array_equal(validation_tail([1, 2, 2, 4], [0, 2, 3, 4, 5]), [1, .8, .4, .4, .2])
    values = tail_fusion([1, 2, 3], [1, 2, 3], [4, 5, 0], [0, 0, 4])
    np.testing.assert_array_equal(values, np.repeat(np.log(4), 3))
    threshold, prediction = budget_point(values, np.array([1, 0, 1]), 0)
    assert not prediction.any() and threshold == values.max()
    with pytest.raises(ValueError, match="nonfinite"):
        validation_tail([1, np.nan], [1])


def test_entire_window_normal_partition_keeps_endpoint_labels():
    labels = np.array([0, 1, 0, 0, 0, 0, 1, 0])
    endpoints = np.array([2, 3, 4, 5, 6, 7])
    groups = normal_window_groups(labels, endpoints, 3)
    np.testing.assert_array_equal(groups["entire_window_normal"], [False, False, True, True, False, False])
    np.testing.assert_array_equal(groups["normal_endpoint_anomalous_context"], [True, True, False, False, False, True])
