import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import export_mnrf_view_probe_analysis as analysis
from scripts.export_mnrf_view_probe_analysis import export
from scripts import supplement_mnrf_view_probe_analysis_inputs as supplement_tool
from scripts.run_mnrf_view_probe import _write_run_supporting_evidence, _write_run_supporting_evidence_unchecked
from scripts.validate_mnrf_view_probe import validate
from ts_benchmark.baselines.mnrf_view_probe import MNRFViewProbe
from ts_benchmark.baselines.mnrf_view_probe.probe import _chunked_array_hash


def _snapshot(root):
    snapshot = {}
    for path in [root] + list(root.rglob("*")):
        relative = "." if path == root else str(path.relative_to(root))
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path))
        elif path.is_dir():
            snapshot[relative] = ("directory",)
        elif path.is_file():
            snapshot[relative] = ("file", hashlib.sha256(path.read_bytes()).hexdigest())
    return snapshot


def _complete_source(tmp_path):
    rng = np.random.default_rng(20261013)
    n, train = 520, 430
    frame = pd.DataFrame(rng.normal(size=(n, 2)).astype(np.float32), columns=("x", "y"),
                         index=pd.date_range("2026-01-01", periods=n, freq="min"))
    labels = np.zeros(n, dtype=np.int64)
    labels[train:train + 5] = 1
    labels[train + 48:train + 52] = 1
    frame["label"] = labels
    run = tmp_path / "full"
    MNRFViewProbe(seq_len=32, pattern_len=32, batch_size=32, stage1_epochs=1,
                  stage2_epochs=1, patience=1, device="cpu", output_dir=str(run),
                  mode="full_gmm", gmm_components=1, gmm_fit_max_windows=8,
                  artifact_policy="full", task_name="SYN.csv").fit(
                      frame, train, labels[train:])
    data_path = tmp_path / "SYN.csv"
    metadata_path = tmp_path / "DETECT_META.csv"
    data_path.write_text("synthetic input identity only\n")
    metadata_path.write_text("file_name,train_lens\nSYN.csv,430\n")
    manifest = json.loads((run / "probe_manifest.json").read_text())
    times = frame.index.view("int64")
    provenance = {
        "captured_before_training": True,
        "git_head": "synthetic-test",
        "git_dirty": True,
        "code_files": [],
        "startup_script": {"path": "synthetic.sh",
                           "sha256": hashlib.sha256(b"#!/usr/bin/env bash\n# synthetic fixture\n").hexdigest(),
                           "text": "#!/usr/bin/env bash\n# synthetic fixture\n"},
        "training_environment": {"python": "test", "torch": "test"},
        "source_data": {"task": "SYN", "data_source": manifest["config"]["data_source"],
                        "rows": n, "train_length": train,
                        "variable_names": manifest["source_identity"]["variable_names"],
                        "time_kind": manifest["source_identity"]["time_index"]["kind"],
                        "time_values_sha256": _chunked_array_hash(times),
                        "feature_data_sha256": manifest["source_identity"]["feature_data_sha256"],
                        "boundary_columns": manifest["source_identity"].get("boundary_columns", []),
                        "boundary_values_sha256": manifest["source_identity"].get("boundary_values_sha256"),
                        "selected_metadata_row": {"file_name": "SYN.csv", "train_lens": "430"},
                        "data_file": str(data_path), "data_file_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                        "metadata_file": str(metadata_path), "metadata_file_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()},
    }
    _write_run_supporting_evidence(run, frame, train, labels[train:], data_path,
                                   metadata_path, provenance)
    assert validate(run)["status"] == "complete"
    return run, frame, labels, train


@pytest.fixture(scope="module")
def complete_source(tmp_path_factory):
    return _complete_source(tmp_path_factory.mktemp("analysis-source"))


def test_full_source_analysis_export_is_read_only_and_reports_unscored_anomaly(tmp_path, monkeypatch, complete_source):
    run, frame, _, train = complete_source
    before = _snapshot(run)
    analysis_dir = tmp_path / "analysis"
    calls = []
    real_validate = analysis.validate

    def counted_validate(path):
        calls.append(Path(path))
        return real_validate(path)

    monkeypatch.setattr(analysis, "validate", counted_validate)
    result = export(run, analysis_dir)
    assert result["status"] == "complete"
    assert calls == [run]
    assert _snapshot(run) == before
    summary = json.loads((analysis_dir / "anomaly_segments.json").read_text())
    assert summary["segment_summary"]["raw_anomaly_segments"] == 2
    first, second = summary["segments"]
    assert (first["start_original"], first["end_original"], first["point_count"]) == (430, 434, 5)
    assert first["valid_points"] == 0 and first["hit_points"] == 0
    assert first["no_valid_scores"] is True and first["hit"] is False
    assert first["start_time_ns"] == int(frame.index.view("int64")[430])
    assert first["end_time_ns"] == int(frame.index.view("int64")[434])
    assert (second["start_original"], second["end_original"], second["point_count"]) == (478, 481, 4)
    diagnostics = json.loads((analysis_dir / "variable_diagnostics.json").read_text())
    assert diagnostics["aggregation_matches_scores"]
    assert diagnostics["aggregation_matches_predictions"]
    manifest = json.loads((analysis_dir / "analysis_manifest.json").read_text())
    assert manifest["source_directory_logical_bytes"] == sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
    assert manifest["training_environment"] == {"python": "test", "torch": "test"}
    assert manifest["source_cache_inventory"]
    with pytest.raises(FileExistsError):
        export(run, analysis_dir)


@pytest.mark.parametrize("target_kind", ["equal", "parent", "inside", "alias", "parent_alias"])
def test_rejects_overlapping_or_aliased_output_before_mutation(tmp_path, target_kind, complete_source):
    run, _, _, _ = complete_source
    source_before = _snapshot(run)
    other = tmp_path / "other.txt"
    other.write_text("untouched")
    if target_kind == "equal":
        target = run
    elif target_kind == "parent":
        target = run.parent
    elif target_kind == "inside":
        target = run / "cache" / "analysis"
    elif target_kind == "alias":
        target = tmp_path / "source-link"
        target.symlink_to(run, target_is_directory=True)
    else:
        target = tmp_path / "parent-link"
        target.symlink_to(run.parent, target_is_directory=True)
        target = target / run.name / "new-analysis"
    with pytest.raises(ValueError, match="outside the source tree|symlink"):
        export(run, target, overwrite=True)
    assert _snapshot(run) == source_before
    assert other.read_text() == "untouched"


def test_unknown_overwrite_targets_are_preserved(tmp_path, complete_source):
    run, _, _, _ = complete_source
    for target in (tmp_path / "plain-file", tmp_path / "unknown-dir", tmp_path / "unknown-link"):
        if target.name == "plain-file":
            target.write_text("keep")
        elif target.name == "unknown-dir":
            target.mkdir(); (target / "sentinel").write_text("keep")
        else:
            referent = tmp_path / "referent"; referent.mkdir()
            (referent / "sentinel").write_text("keep")
            target.symlink_to(referent, target_is_directory=True)
        before = _snapshot(target.parent)
        with pytest.raises(ValueError):
            export(run, target, overwrite=True)
        assert _snapshot(target.parent) == before


def test_same_source_overwrite_is_transactional_on_export_failure(tmp_path, monkeypatch, complete_source):
    run, _, _, _ = complete_source
    output = tmp_path / "analysis"
    export(run, output)
    old = _snapshot(output)
    source = _snapshot(run)
    original = analysis._feature_overview

    def fail(*args, **kwargs):
        raise RuntimeError("injected diagnostic failure")

    monkeypatch.setattr(analysis, "_feature_overview", fail)
    with pytest.raises(RuntimeError, match="injected"):
        export(run, output, overwrite=True)
    assert _snapshot(output) == old
    assert _snapshot(run) == source
    monkeypatch.setattr(analysis, "_feature_overview", original)
    result = export(run, output, overwrite=True)
    assert result["status"] == "complete"
    assert json.loads((output / "analysis_complete.json").read_text())["status"] == "complete"
    assert _snapshot(run) == source


def test_same_source_overwrite_rejects_damaged_existing_package(tmp_path, complete_source):
    run, _, _, _ = complete_source
    output = tmp_path / "analysis"
    export(run, output)
    evidence = output / "evidence" / "results" / "test_nll.npy"
    evidence.write_bytes(evidence.read_bytes() + b"tamper")
    damaged_package = _snapshot(output)
    source = _snapshot(run)
    with pytest.raises(ValueError, match="damaged or missing evidence"):
        export(run, output, overwrite=True)
    assert _snapshot(output) == damaged_package
    assert _snapshot(run) == source


def test_rejects_missing_or_tampered_bound_analysis_inputs_without_writes(tmp_path, complete_source):
    source, _, _, _ = complete_source
    run = tmp_path / "full-copy"
    shutil.copytree(source, run)
    before = _snapshot(run)
    inputs = run / "analysis_inputs.npz"
    with np.load(inputs, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    arrays["test_labels"][0] = 1 - arrays["test_labels"][0]
    np.savez(inputs, **arrays)
    damaged = _snapshot(run)
    with pytest.raises(ValueError, match="SHA-256|bound"):
        export(run, tmp_path / "bad-analysis")
    assert _snapshot(run) == damaged
    assert not (tmp_path / "bad-analysis").exists()
    assert before != damaged


def test_legacy_skip_flag_is_rejected_by_python_api(tmp_path, complete_source):
    run, _, _, _ = complete_source
    with pytest.raises(ValueError, match="already_validated is forbidden"):
        export(run, tmp_path / "analysis", already_validated=True)
    assert not (tmp_path / "analysis").exists()


def test_supplemental_inputs_can_finish_without_training(tmp_path, monkeypatch, complete_source):
    run, frame, labels, train = complete_source
    manifest_path = run / "probe_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    provenance_path = run / "training_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance.pop("analysis_inputs", None)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    for name in ("analysis_inputs.npz", "analysis_inputs.json"):
        (run / name).unlink()
    artifacts = manifest.get("analysis_artifacts", {})
    for key in ("analysis_inputs_sha256", "test_time_values_sha256", "full_axis_sha256", "training_provenance_sha256"):
        artifacts.pop(key, None)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    data_path = Path(provenance["source_data"]["data_file"])
    metadata_path = Path(provenance["source_data"]["metadata_file"])
    monkeypatch.setattr(supplement_tool, "_read_task", lambda task, source, data_root=None: (
        frame, train, labels[train:], data_path, metadata_path,
        {"file_name": "SYN.csv", "train_lens": "430"}))
    result = supplement_tool.supplement(run, tmp_path / "analysis")
    assert result["status"] == "complete"
    assert (run / "analysis_inputs.npz").is_file()
    assert json.loads((tmp_path / "analysis" / "analysis_complete.json").read_text())["status"] == "complete"


def test_supporting_evidence_failure_rolls_back_auxiliary_files(tmp_path, monkeypatch, complete_source):
    run, frame, labels, train = complete_source
    before = _snapshot(run)
    data_path = run / "synthetic.csv"
    metadata_path = run / "synthetic-meta.csv"
    data_path.write_text("synthetic\n")
    metadata_path.write_text("file_name,train_lens\nSYN.csv,430\n")
    provenance = json.loads((run / "training_provenance.json").read_text())
    # Refresh fixture-only source file references, which are not used by this
    # injected failure but must remain structurally valid for the writer.
    provenance["source_data"]["data_file"] = str(data_path)
    provenance["source_data"]["metadata_file"] = str(metadata_path)
    provenance["source_data"]["data_file_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    provenance["source_data"]["metadata_file_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    (run / "existing-recovery.tmp").write_bytes(b"keep recovery evidence")
    (run / ".analysis_inputs.tmp-98765.npz").write_bytes(b"keep old npz temporary")
    (run / "preexisting.json.tmp").write_bytes(b"keep old json temporary")
    (run / "training_provenance.json.rollback.tmp").write_bytes(b"keep rollback evidence")
    old_temp_dir = run / ".analysis_inputs.tmp-old-dir"
    old_temp_dir.mkdir()
    (old_temp_dir / "payload").write_bytes(b"keep directory")
    external = tmp_path / "external-target"
    external.mkdir()
    (external / "payload").write_bytes(b"outside remains unchanged")
    link = run / "analysis_inputs.tmp-link"
    link.symlink_to(external, target_is_directory=True)
    before = _snapshot(run)

    def partial_then_fail(*args, **kwargs):
        staging = Path(args[0])
        path = staging / "probe_manifest.json"
        path.write_text("partial mutation")
        (staging / "training_provenance.json").write_text("partial mutation")
        (staging / "unpublished.partial.tmp").write_text("owned by transaction")
        raise RuntimeError("injected support-write interruption")

    monkeypatch.setattr("scripts.run_mnrf_view_probe._write_run_supporting_evidence_unchecked", partial_then_fail)
    with pytest.raises(RuntimeError, match="injected"):
        _write_run_supporting_evidence(run, frame, train, labels[train:], data_path,
                                       metadata_path, provenance)
    assert _snapshot(run) == before
    assert (external / "payload").read_bytes() == b"outside remains unchanged"
    assert json.loads((run / "run_status.json").read_text())["status"] == "complete"


def test_invalid_label_preflight_preserves_all_preexisting_temp_evidence(tmp_path, complete_source):
    run, frame, labels, train = complete_source
    data_path = tmp_path / "data.csv"
    metadata_path = tmp_path / "meta.csv"
    data_path.write_text("data\n")
    metadata_path.write_text("file_name,train_lens\nSYN.csv,430\n")
    sentinels = {
        "existing-recovery.tmp": b"recovery",
        ".analysis_inputs.tmp-98765.npz": b"npz temp",
        "preexisting.json.tmp": b"json temp",
        "training_provenance.json.rollback.tmp": b"rollback temp",
    }
    for name, payload in sentinels.items():
        (run / name).write_bytes(payload)
    before = _snapshot(run)
    invalid = labels[train:].astype(np.float64)
    invalid[0] = 0.5
    with pytest.raises(ValueError, match="exact binary"):
        _write_run_supporting_evidence(run, frame, train, invalid, data_path,
                                       metadata_path, json.loads((run / "training_provenance.json").read_text()))
    assert _snapshot(run) == before


def test_commit_failure_rolls_back_and_retry_preserves_old_temp_files(tmp_path, monkeypatch, complete_source):
    run, frame, labels, train = complete_source
    data_path = tmp_path / "data.csv"
    metadata_path = tmp_path / "meta.csv"
    data_path.write_text("data\n")
    metadata_path.write_text("file_name,train_lens\nSYN.csv,430\n")
    (run / "existing-recovery.tmp").write_bytes(b"preserve")
    (run / ".analysis_inputs.tmp-98765.npz").write_bytes(b"preserve")
    provenance = json.loads((run / "training_provenance.json").read_text())
    provenance["source_data"]["data_file"] = str(data_path)
    provenance["source_data"]["metadata_file"] = str(metadata_path)
    provenance["source_data"]["data_file_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    provenance["source_data"]["metadata_file_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    before = _snapshot(run)
    module = __import__("scripts.run_mnrf_view_probe", fromlist=["os"])
    real_replace = os.replace
    failed = {"once": False}
    target = run / "analysis_inputs.json"

    def fail_once_on_final_metadata(source, destination):
        if Path(destination) == target and not failed["once"]:
            failed["once"] = True
            raise OSError("injected final metadata publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_once_on_final_metadata)
    with pytest.raises(OSError, match="injected final metadata"):
        _write_run_supporting_evidence(run, frame, train, labels[train:], data_path,
                                       metadata_path, provenance)
    assert _snapshot(run) == before
    assert not list(run.glob(".viewprobe-aux-txn-*"))
    monkeypatch.setattr(module.os, "replace", real_replace)
    _write_run_supporting_evidence(run, frame, train, labels[train:], data_path,
                                   metadata_path, provenance)
    assert (run / "analysis_inputs.npz").is_file()
    assert (run / "analysis_inputs.json").is_file()
    assert (run / "existing-recovery.tmp").read_bytes() == b"preserve"
    assert (run / ".analysis_inputs.tmp-98765.npz").read_bytes() == b"preserve"
    assert not list(run.glob(".viewprobe-aux-txn-*"))


def test_rollback_failure_reports_and_retains_private_recovery_backup(tmp_path, monkeypatch, complete_source):
    run, frame, labels, train = complete_source
    data_path = tmp_path / "data.csv"
    metadata_path = tmp_path / "meta.csv"
    data_path.write_text("data\n")
    metadata_path.write_text("file_name,train_lens\nSYN.csv,430\n")
    provenance = json.loads((run / "training_provenance.json").read_text())
    provenance["source_data"]["data_file"] = str(data_path)
    provenance["source_data"]["metadata_file"] = str(metadata_path)
    provenance["source_data"]["data_file_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    provenance["source_data"]["metadata_file_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    provenance["rollback_test_marker"] = "force-provenance-replacement"
    before = _snapshot(run)
    module = __import__("scripts.run_mnrf_view_probe", fromlist=["os"])
    real_replace = os.replace
    failures = {"analysis": False, "provenance_rollback": False}

    def fail_publish_and_one_rollback(source, destination):
        destination = Path(destination)
        if destination == run / "analysis_inputs.json" and not failures["analysis"]:
            failures["analysis"] = True
            (run / "training_provenance.json").write_bytes(b"concurrent external update")
            failures["provenance_rollback"] = True
            raise OSError("injected publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_publish_and_one_rollback)
    with pytest.raises(RuntimeError, match="rollback failed.*recovery snapshots retained"):
        _write_run_supporting_evidence(run, frame, train, labels[train:], data_path,
                                       metadata_path, provenance)
    assert failures == {"analysis": True, "provenance_rollback": True}
    transaction_dirs = list(run.glob(".viewprobe-aux-txn-*"))
    assert len(transaction_dirs) == 1
    backup = transaction_dirs[0] / "backup"
    assert hashlib.sha256((backup / "probe_manifest.json").read_bytes()).hexdigest() == before["probe_manifest.json"][1]
    assert hashlib.sha256((backup / "training_provenance.json").read_bytes()).hexdigest() == before["training_provenance.json"][1]
    assert (run / "training_provenance.json").read_bytes() == b"concurrent external update"
    assert json.loads((run / "run_status.json").read_text())["status"] == "complete"


def test_feature_variance_is_over_time_per_variable_and_branch(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "cache").mkdir(parents=True)
    output.mkdir()
    values = np.zeros((4, 2, 25), dtype=np.float32)
    values[:, 1, :] = 100.0
    np.save(source / "cache" / "train_features.npy", values)
    manifest = {"layout": {"F": 25, "slices": {"S": [0, 1], "E": [1, 9], "P": [9, 17], "R": [17, 25]}}}
    result = analysis._feature_overview(source, output, manifest, np.random.default_rng(5))
    branches = result["splits"]["train"]["branches"]
    assert branches["S"]["low_variance_dimensions"] == 2
    assert branches["E"]["low_variance_dimensions"] == 16
    assert branches["P"]["low_variance_dimensions"] == 16
    assert branches["R"]["low_variance_dimensions"] == 16
    assert sum(item["low_variance_dimensions"] for item in branches.values()) == 50
    assert branches["E"]["low_variance_unit"] == "variable x feature dimension pair"

    values[:, 1, :] = np.arange(4, dtype=np.float32)[:, None]
    np.save(source / "cache" / "train_features.npy", values)
    mixed = analysis._feature_overview(source, output, manifest, np.random.default_rng(5))
    assert mixed["splits"]["train"]["branches"]["R"]["low_variance_by_variable"] == [8, 0]


def test_rare_test_categories_are_stratified_and_repeatable(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "cache").mkdir(parents=True)
    (source / "results").mkdir()
    (source / "heads").mkdir()
    output.mkdir()
    (output / "samples").mkdir()
    endpoints = np.arange(100, 120, dtype=np.int64)
    features = np.arange(20 * 2 * 25, dtype=np.float32).reshape(20, 2, 25)
    np.save(source / "cache" / "test_features.npy", features)
    np.save(source / "cache" / "test_window_end_indices.npy", endpoints)
    labels = np.zeros(250, dtype=np.int64)
    labels[105] = 1
    labels[111] = 1
    times = np.arange(250, dtype=np.int64) + 1_000
    global_indices = np.arange(100, 250, dtype=np.int64)
    segments = np.zeros(150, dtype=np.int32)
    scores = np.full(20, 0.1, dtype=np.float32)
    scores[2] = 0.9  # false positive
    scores[5] = 0.9  # true positive
    scores[11] = 0.1  # false negative
    predictions = (scores > 0.5).astype(np.int64)
    np.savez(source / "results" / "aligned_scores.npz", time_indices=endpoints,
             scores=scores, predictions=predictions, valid_mask=np.ones(20, dtype=bool))
    np.savez(source / "heads" / "calibration.npz", threshold=np.asarray(0.5))
    analysis_inputs = (labels[100:], times[100:], global_indices[100:], {}, segments[100:],
                       labels, times, np.arange(250, dtype=np.int64), np.zeros(250, dtype=np.int32))
    manifest = {"layout": {"F": 25, "slices": {"S": [0, 1], "E": [1, 9], "P": [9, 17], "R": [17, 25]}},
                "source_identity": {"train_length": 100}}
    first = analysis._samples(source, output, manifest, np.random.default_rng(20261013), analysis_inputs)
    saved_path = output / "samples" / "test.npz"
    with np.load(saved_path, allow_pickle=False) as data:
        first_indices = np.asarray(data["sample_indices"]).copy()
        reasons = np.asarray(data["sample_reason"]).astype(str)
        actual = np.asarray(data["actual"])
        sample_times = np.asarray(data["time_values_ns"])
    assert len(first_indices) <= 512 and len(np.unique(first_indices)) == len(first_indices)
    assert {"tp", "fp", "fn", "tn"}.issubset(set(reasons))
    assert np.sum(actual == 1) == 2
    assert np.array_equal(sample_times, endpoints[first_indices] + 1_000)
    (output / "samples" / "test.npz").unlink()
    analysis._samples(source, output, manifest, np.random.default_rng(20261013), analysis_inputs)
    with np.load(saved_path, allow_pickle=False) as data:
        assert np.array_equal(first_indices, data["sample_indices"])
