import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.mnrf_common_data import load_task_frame
from scripts.evaluate_mnrf_view_probe_common import _same_task_identity
from scripts import run_mnrf_common_baselines as common
from scripts.run_mnrf_common_baselines import _existing_catch_run, _fit_light_method, _fit_residual_pca, _full_axis_metrics
from ts_benchmark.baselines.mnrf_view_probe.evaluation import calibrated_quantile


def _save_indices(path, values):
    np.save(path, np.asarray(values, dtype=np.int64))
    return {"path": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "count": int(len(values))}


def test_benchmark_long_csv_loader_preserves_variable_order_time_and_binary_labels(tmp_path):
    times = pd.date_range("2026-01-01", periods=5, freq="min")
    rows = []
    columns = {"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50], "label": [0, 0, 1, 0, 1]}
    for name, values in columns.items():
        rows.extend({"date": stamp.isoformat(), "data": value, "cols": name}
                    for stamp, value in zip(times, values))
    path = tmp_path / "long.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    frame, label_column = load_task_frame(path)
    assert list(frame.columns) == ["x", "y", "label"]
    assert label_column == "label"
    assert frame.index.equals(pd.DatetimeIndex(times))
    assert frame["label"].dtype == np.int8
    np.testing.assert_array_equal(frame["label"].to_numpy(), [0, 0, 1, 0, 1])


def test_light_baselines_use_fixed_fit_rows_and_calibration_coordinates(tmp_path):
    rng = np.random.default_rng(2021)
    values = rng.normal(size=(160, 3)).astype(np.float32)
    labels = np.zeros(160, dtype=np.int64)
    labels[120:124] = 1
    labels[145:148] = 1
    times = np.arange(160, dtype=np.int64) * 60_000_000_000
    train_rows = np.arange(0, 80, dtype=np.int64)
    calibration = np.arange(100, 110, dtype=np.int64)
    test = np.arange(120, 160, dtype=np.int64)
    train_record = _save_indices(tmp_path / "train_rows.npy", train_rows)
    calibration_record = _save_indices(tmp_path / "calibration.npy", calibration)
    test_record = _save_indices(tmp_path / "test.npy", test)
    contract_file = tmp_path / "common_contract.json"
    contract_file.write_text("{}", encoding="utf-8")
    contract = {
        "task_name": "SYN", "common_contract_path": str(contract_file),
        "probe_run": str(tmp_path / "probe"), "probe_manifest_sha256": "source-manifest",
        "data_csv_path": str(tmp_path / "source.csv"), "data_csv_sha256": "source-bytes",
        "data_identity": "data", "time_identity": "time", "variable_names": ["a", "b", "c"],
        "boundary_identity": None, "train_rows": train_record,
        "catch_splits": {"calibration": calibration_record}, "catch_test": test_record,
        "probe_source_identity": {"test_segments": [[120, 160]]},
    }
    for method in ("IsolationForest", "PCA"):
        score_path, evidence = _fit_light_method(method, tmp_path / method, values, labels, times, contract)
        with np.load(score_path, allow_pickle=False) as payload:
            np.testing.assert_array_equal(payload["time_indices"], test)
            np.testing.assert_array_equal(payload["time_values_ns"], times[test])
            np.testing.assert_array_equal(payload["actual"], labels[test])
            assert np.array_equal(payload["predictions"], payload["scores"] > payload["threshold"])
            threshold = float(payload["threshold"])
        with np.load(tmp_path / method / "results" / "calibration_scores.npz", allow_pickle=False) as calibration_payload:
            assert np.array_equal(calibration_payload["time_indices"], calibration)
            assert threshold == calibrated_quantile(calibration_payload["scores"], 0.995)
        manifest = json.loads((tmp_path / method / "evidence" / "baseline.json").read_text())
        assert manifest["training_protocol"].endswith("no internal split")
        assert manifest["seed"] == 2021
        assert evidence["parameters"]


def test_metrics_keep_coordinate_gaps_and_report_segment_coverage():
    labels = np.zeros(20, dtype=np.int64)
    labels[4:7] = 1
    labels[14:17] = 1
    indices = np.asarray([3, 4, 6, 7, 12, 14, 15, 17], dtype=np.int64)
    scores = np.asarray([0.1, 0.9, 0.2, 0.1, 0.1, 0.8, 0.2, 0.1])
    metrics = _full_axis_metrics(labels, indices, scores, 0.5, [[0, 10], [10, 20]], None)
    assert (metrics["tp"], metrics["fn"]) == (2, 2)
    assert metrics["total_anomaly_segments"] == 2
    assert metrics["scored_anomaly_segments"] == 2
    assert metrics["hit_anomaly_segments"] == 2
    assert metrics["affiliation"]["total_segment_count"] == 5


def test_strict_task_identity_accepts_csv_and_dataset_stem_only_when_equal():
    assert _same_task_identity("PSM.csv", "PSM")
    assert _same_task_identity("/datasets/PSM.csv", "PSM")
    assert not _same_task_identity("MSL.csv", "PSM")
    assert not _same_task_identity(None, "PSM")


def test_complete_catch_output_can_be_reused_but_partial_output_is_not_overwritten(tmp_path):
    output = tmp_path / "catch"
    files = (
        "checkpoints/catch_common.pt", "normalization.npz", "evidence/training.json",
        "evidence/calibration.json", "results/aligned_scores.npz",
        "results/calibration_scores.npz", "results/metrics.json",
    )
    for relative in files:
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    (output / "results/metrics.json").write_text(json.dumps({
        "status": "complete", "task_name": "PSM", "training_time_seconds": 12,
        "calibration_and_inference_time_seconds": 3,
    }), encoding="utf-8")
    assert _existing_catch_run(output, "PSM") == 15
    with pytest.raises(ValueError, match="not a completed run"):
        _existing_catch_run(output, "MSL")
    (output / "results/aligned_scores.npz").unlink()
    with pytest.raises(FileExistsError, match="incomplete"):
        _existing_catch_run(output, "PSM")


def test_catch_common_cli_has_no_duplicate_options():
    result = subprocess.run([sys.executable, "scripts/run_mnrf_catch_common.py", "--help"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("\n  --auxi-lambda ") == 1
    assert result.stdout.count("\n  --inference-patch-size ") == 1


def test_common_baselines_cli_exposes_resume_batch():
    result = subprocess.run([sys.executable, "scripts/run_mnrf_common_baselines.py", "--help"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, result.stderr
    assert "--resume-batch" in result.stdout


def test_pca_residual_rule_applies_to_all_dimensions_uniformly():
    rng = np.random.RandomState(17)
    for dimension in (2, 3, 9, 25, 55, 72):
        train = rng.normal(size=(240, dimension))
        model, params = _fit_residual_pca(train, dimension)
        assert model.n_components_ < dimension
        assert params["residual_dimensions"] >= 1
        assert params["selection_rule"] == "min(components_for_95_percent_variance, D-1)"


def test_resume_batch_reuses_complete_batch_and_preserves_published_files_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "TASKS", ("PSM",))
    archive_summary = tmp_path / "physical_task_metrics.csv"
    archive_summary.write_text("fixture\n", encoding="utf-8")
    configs = {"PSM": {"resolved_common_run_model_config": {"seq_len": 32}}}
    archive_rows = [{"dataset": "PSM", "reference": "fixture"}]
    monkeypatch.setattr(common, "_archive_references", lambda: (configs, archive_rows, archive_summary))
    calls = {"count": 0, "fail": False}

    def task_pipeline(task, batch, analysis_root, score_root, catch_configs, method_rows, task_records):
        calls["count"] += 1
        if calls["fail"]:
            raise RuntimeError("injected partial resume failure")
        method_rows.append({"dataset": task, "method": "MNRF Full", "roc_auc": 0.5})
        task_records.append({"dataset": task, "status": "complete", "methods": {}})

    monkeypatch.setattr(common, "_task_pipeline", task_pipeline)
    monkeypatch.setattr(common, "_write_readme",
                        lambda path, *args: Path(path).write_text("complete report\n", encoding="utf-8"))
    args = SimpleNamespace(batch_name="resume-fixture", resume_batch=False)
    out = common.run_all(args)
    metrics = out / "common_protocol_metrics.csv"
    status = out / "task_status.json"
    readme = out / "README.md"
    first = (metrics.read_bytes(), status.read_bytes(), readme.read_bytes())

    args.resume_batch = True
    common.run_all(args)
    assert calls["count"] == 2
    second = (metrics.read_bytes(), status.read_bytes(), readme.read_bytes())
    assert json.loads(status.read_text())["completed_tasks"] == 1

    calls["fail"] = True
    with pytest.raises(RuntimeError, match="previous complete batch report was preserved"):
        common.run_all(args)
    assert (metrics.read_bytes(), status.read_bytes(), readme.read_bytes()) == second
    assert list(out.glob("resume-attempt-*.json"))
    assert first[0] == second[0]


