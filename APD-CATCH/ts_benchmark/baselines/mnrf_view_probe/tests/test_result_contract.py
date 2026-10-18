import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ts_benchmark.baselines.mnrf_view_probe import probe as probe_module


def _bundle(tmp_path):
    run = tmp_path / "result"
    (run / "results").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    (run / "cache").mkdir()
    (run / "heads").mkdir()
    local = np.r_[np.arange(31, 50), np.arange(112, 120)].astype(np.int64)
    endpoints = 430 + local
    valid = np.zeros(120, dtype=bool)
    valid[local] = True
    scores = np.linspace(0.1, 2.0, len(local), dtype=np.float32)
    predictions = (scores > 1.0).astype(np.int64)
    actual = (np.arange(len(local)) % 2).astype(np.int64)
    aligned_scores = np.full(120, np.nan, dtype=np.float32)
    aligned_scores[local] = scores
    aligned_predictions = np.full(120, -1, dtype=np.int64)
    aligned_predictions[local] = predictions
    aligned_actual = np.full(120, -1, dtype=np.int64)
    aligned_actual[local] = actual
    time_values = np.arange(len(local), dtype=np.int64) + 1_000
    payload = {
        "scores": scores, "predictions": predictions, "time_indices": endpoints,
        "time_values_ns": time_values, "local_indices": local, "valid_mask": valid,
        "aligned_scores": aligned_scores, "aligned_predictions": aligned_predictions,
        "actual": actual, "aligned_actual": aligned_actual, "threshold": np.asarray(1.0),
    }
    np.savez_compressed(run / "results" / "aligned_scores.npz", **payload)
    metrics = {
        "status": "complete", "head": "gmm", "threshold": 1.0,
        "valid_points": len(scores), "excluded_points": int((~valid).sum()),
        "coverage": len(scores) / 120.0,
        "auc_pr": float(average_precision_score(actual, scores)),
        "auc_roc": float(roc_auc_score(actual, scores)),
    }
    (run / "results" / "metrics.json").write_text(json.dumps(metrics))
    source_identity = {
        "train_length": 430, "input_length": 550, "seq_len": 32,
        "test_segments": [[430, 480], [511, 550]],
        "test_endpoints_sha256": probe_module._chunked_array_hash(endpoints),
        "test_endpoints_count": len(endpoints),
    }
    manifest = {
        "schema": probe_module.PROBE_SCHEMA, "artifact_type": probe_module.ARTIFACT_RESULTS,
        "artifact_policy": "results", "result_bundle_schema": probe_module.RESULT_SCHEMA,
        "result_cleanup_status": "in_progress", "source_identity": source_identity,
        "metrics_sha256": probe_module._sha256_file(run / "results" / "metrics.json"),
        "aligned_scores_sha256": probe_module._sha256_file(run / "results" / "aligned_scores.npz"),
        "test_endpoint_sha256": probe_module._chunked_array_hash(endpoints),
        "test_time_values_sha256": probe_module._chunked_array_hash(time_values),
    }
    (run / "probe_manifest.json").write_text(json.dumps(manifest))
    (run / "run_status.json").write_text(json.dumps({"schema": probe_module.PROBE_SCHEMA,
                                                        "status": "organizing_results"}))
    return run


def test_result_coordinates_and_labels_are_semantically_bound(tmp_path):
    run = _bundle(tmp_path)
    assert probe_module._validate_result_bundle(run, allow_in_progress=True)["status"] == "valid"
    with np.load(run / "results" / "aligned_scores.npz") as source:
        payload = {key: source[key] for key in source.files}
    payload["time_indices"] = payload["time_indices"].copy()
    payload["time_indices"][0] += 1
    np.savez_compressed(run / "results" / "aligned_scores.npz", **payload)
    with np.testing.assert_raises(ValueError):
        probe_module._validate_result_bundle(run, allow_in_progress=True)


def test_cleanup_failure_preserves_valid_metrics(tmp_path, monkeypatch):
    run = _bundle(tmp_path)
    original = probe_module.shutil.rmtree

    def fail_checkpoints(path, *args, **kwargs):
        if str(path).endswith("checkpoints"):
            raise OSError("injected cleanup interruption")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(probe_module.shutil, "rmtree", fail_checkpoints)
    with np.testing.assert_raises(OSError):
        probe_module._organize_results_transaction(run)
    manifest = json.loads((run / "probe_manifest.json").read_text())
    assert manifest["result_valid"] is True
    assert manifest["result_cleanup_status"] == "failed"
    assert probe_module._validate_result_bundle(run, allow_in_progress=True)["metrics"]["auc_pr"] is not None


def test_gmm_identity_projection_excludes_logging_and_storage_fields():
    config = {"gmm_components": 4, "gmm_reg_covar": 0.001, "log_every_steps": 1,
              "log_every_seconds": 2.0, "gmm_log_every_iters": 3, "device": "cpu",
              "output_dir": "/tmp/run", "batch_size": 4}
    changed = dict(config, log_every_steps=999, log_every_seconds=999.0,
                   gmm_log_every_iters=999, device="cuda:0", output_dir="/other")
    assert probe_module._gmm_numeric_config(config) == probe_module._gmm_numeric_config(changed)


def _refresh_result_hash(run):
    manifest_path = run / "probe_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["aligned_scores_sha256"] = probe_module._sha256_file(run / "results" / "aligned_scores.npz")
    manifest["metrics_sha256"] = probe_module._sha256_file(run / "results" / "metrics.json")
    manifest_path.write_text(json.dumps(manifest))


def test_result_validator_rejects_fractional_predictions_and_counts_after_rebinding(tmp_path):
    run = _bundle(tmp_path)
    with np.load(run / "results" / "aligned_scores.npz", allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload["predictions"] = payload["predictions"].astype(np.float32)
    payload["predictions"][0] = 0.5
    payload["aligned_predictions"] = payload["aligned_predictions"].astype(np.float32)
    payload["aligned_predictions"][payload["local_indices"][0]] = 0.5
    np.savez_compressed(run / "results" / "aligned_scores.npz", **payload)
    _refresh_result_hash(run)
    with np.testing.assert_raises(ValueError):
        probe_module._validate_result_bundle(run, allow_in_progress=True)


def test_summary_isolates_corrupt_gmm_status_from_valid_runs(tmp_path, monkeypatch):
    from scripts import summarize_mnrf_view_probe as summary

    for name, corrupt in (("valid", False), ("broken", True)):
        run = tmp_path / name
        (run / "results").mkdir(parents=True)
        (run / "heads").mkdir()
        (run / "probe_manifest.json").write_text(json.dumps({
            "artifact_type": "probe_head", "artifact_policy": "full", "mode": "full_gmm",
            "config": {"task_name": name, "seed": 2021}, "head_status": "complete",
        }))
        (run / "run_status.json").write_text(json.dumps({"status": "complete"}))
        (run / "results" / "metrics.json").write_text(json.dumps({"status": "complete", "auc_roc": 0.5}))
        (run / "heads" / "gmm_status.json").write_text("{ broken" if corrupt else "{}")
    monkeypatch.setattr(summary, "load_probe", lambda *args, **kwargs: object())
    frame = summary.collect(tmp_path)
    assert set(frame["task"]) == {"valid", "broken"}
    assert frame.loc[frame["task"] == "valid", "validation_status"].iloc[0] == "valid"
    broken_reason = frame.loc[frame["task"] == "broken", "validation_reason"].iloc[0]
    assert "gmm_status.json" in broken_reason

    run = _bundle(tmp_path / "counts")
    metrics_path = run / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["valid_points"] = 59.9
    metrics_path.write_text(json.dumps(metrics))
    _refresh_result_hash(run)
    with np.testing.assert_raises(ValueError):
        probe_module._validate_result_bundle(run, allow_in_progress=True)
