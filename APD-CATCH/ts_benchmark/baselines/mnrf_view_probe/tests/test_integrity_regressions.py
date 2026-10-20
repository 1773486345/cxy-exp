import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import export_mnrf_view_probe_analysis as exporter
from scripts import supplement_mnrf_view_probe_analysis_inputs as supplementer
from scripts.run_mnrf_view_probe import _read_task, _write_run_supporting_evidence
from ts_benchmark.baselines.mnrf_view_probe.integrity import (
    canonical_segment_ids, normalize_binary_labels, validate_normalized_values_cache,
)
from ts_benchmark.baselines.mnrf_view_probe import MNRFViewProbe
from scripts.validate_mnrf_view_probe import validate


def _core_snapshot(root):
    excluded = {"probe_manifest.json", "training_provenance.json", "analysis_inputs.npz", "analysis_inputs.json"}
    return {str(path.relative_to(root)): exporter._sha256(path)
            for path in root.rglob("*") if path.is_file()
            and path.name not in excluded and "legacy-analysis-inputs-" not in str(path)}


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.float32, np.float64, np.bool_])
def test_binary_label_normalizer_accepts_exact_binary_numeric_types(dtype):
    labels = normalize_binary_labels(np.asarray([0, 1], dtype=dtype), 2)
    assert labels.dtype == np.int8
    assert np.array_equal(labels, np.asarray([0, 1], dtype=np.int8))


@pytest.mark.parametrize("values", [
    [0.5], [0.999999999999], [-1], [2], [np.nan], [np.inf],
    np.asarray(["0", "1"], dtype=object), np.asarray([1 + 0j]),
    np.asarray([[0, 1]]), [0, 1, 0],
])
def test_binary_label_normalizer_rejects_ambiguous_or_invalid_inputs(values):
    with pytest.raises(ValueError):
        normalize_binary_labels(values, 1)


def test_canonical_test_segment_ids_are_a_slice_of_full_ids():
    identity = {"input_length": 12, "train_length": 8,
                "segments": [[0, 4], [4, 12]],
                "test_segments": [[8, 12]]}
    full = canonical_segment_ids(identity)
    assert np.array_equal(full, np.asarray([0] * 4 + [1] * 8, dtype=np.int32))
    assert np.array_equal(full[8:], np.asarray([1] * 4, dtype=np.int32))
    with pytest.raises(ValueError, match="intersections"):
        canonical_segment_ids({**identity, "test_segments": [[8, 10], [10, 12]]})


def _integrity_fixture(root, values=None, declared=True):
    import hashlib
    cache = root / "cache" / "normalized_values.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    stats = root / "input_statistics.npz"
    np.savez(stats, mean=np.zeros(2, dtype=np.float32), scale=np.ones(2, dtype=np.float32))
    if values is not None:
        np.save(cache, values)
    digest = hashlib.sha256(cache.read_bytes()).hexdigest() if cache.is_file() else None
    return {"source_identity": {"input_length": 4, "variable_names": ["x", "y"]},
            "variable_names": ["x", "y"],
            "normalized_cache_sha256": digest if declared else None,
            "raw_input_statistics_sha256": hashlib.sha256(stats.read_bytes()).hexdigest()}


def test_normalized_cache_requires_exact_shape_dtype_finiteness_and_authoritative_hash(tmp_path):
    root = tmp_path / "source"
    values = np.arange(8, dtype=np.float32).reshape(4, 2)
    manifest = _integrity_fixture(root, values)
    assert validate_normalized_values_cache(root, manifest)["sha256"] == manifest["normalized_cache_sha256"]
    variants = [
        (np.arange(6, dtype=np.float32).reshape(3, 2), "shape/dtype mismatch"),
        (values.astype(np.float64), "shape/dtype mismatch"),
        (np.asarray([[np.nan, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32), "non-finite"),
    ]
    for changed, reason in variants:
        np.save(root / "cache" / "normalized_values.npy", changed)
        stale = dict(manifest)
        with pytest.raises(ValueError, match=reason):
            validate_normalized_values_cache(root, stale)
    np.save(root / "cache" / "normalized_values.npy", values)
    stale = dict(manifest)
    stale["normalized_cache_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_normalized_values_cache(root, stale)
    del stale["normalized_cache_sha256"]
    with pytest.raises(ValueError, match="no authoritative"):
        validate_normalized_values_cache(root, stale)
    (root / "cache" / "normalized_values.npy").unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_normalized_values_cache(root, manifest)


@pytest.mark.parametrize("task, floating, gap_at", [
    ("FLOAT_AUDIT", True, None),
    ("INT_AUDIT", False, None),
    ("GAP_AUDIT", False, 150),
])
def test_loader_and_real_cli_accept_labels_and_export_full_axis(tmp_path, task, floating, gap_at):
    dataset_root = tmp_path / "anomaly_detect"
    data_dir = dataset_root / "data"
    data_dir.mkdir(parents=True)
    total, train = 520, 430
    times = pd.date_range("2026-01-01", periods=total, freq="min")
    if gap_at is not None:
        times = times.where(np.arange(total) < gap_at, times + pd.Timedelta(minutes=10))
    if floating:
        x = np.sin(np.arange(total) / 11.0)
        y = np.cos(np.arange(total) / 17.0)
        label_dtype = np.float64
    else:
        x = (np.arange(total) % 17).astype(np.int64)
        y = ((np.arange(total) * 3) % 23).astype(np.int64)
        label_dtype = np.int64
    labels = np.zeros(total, dtype=label_dtype)
    labels[430:435] = 1.0
    labels[478:482] = 1.0
    rows = []
    for name, values in (("x", x), ("y", y), ("label", labels)):
        rows.extend({"date": stamp, "data": value, "cols": name}
                    for stamp, value in zip(times, values))
    csv_path = data_dir / f"{task}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    metadata_path = dataset_root / "DETECT_META.csv"
    pd.DataFrame([{"file_name": f"{task}.csv", "train_lens": train}]).to_csv(metadata_path, index=False)

    loaded, loaded_train, test_labels, *_ = _read_task(task, "large_detect", dataset_root)
    assert loaded_train == train and loaded["label"].dtype.kind == ("f" if floating else "i")
    assert test_labels.dtype == np.int8 and test_labels.sum() == 9

    run_dir = tmp_path / "run"
    analysis_dir = tmp_path / "analysis"
    command = [sys.executable, str(Path(exporter.ROOT) / "scripts/run_mnrf_view_probe.py"),
               "--task", task, "--data-root", str(dataset_root),
               "--mode", "full_gmm", "--artifact-policy", "full",
               "--output-dir", str(run_dir), "--seq-len", "32", "--pattern-len", "32",
               "--batch-size", "32", "--stage1-epochs", "1", "--patience", "1",
               "--device", "cpu", "--gmm-components", "1", "--gmm-fit-max-windows", "8",
               "--startup-script", str(Path(exporter.ROOT) / "scripts/multivariate_detection/detect_score/PSM_script/MNRFViewProbe.sh")]
    environment = os.environ.copy()
    completed = subprocess.run(command, cwd=exporter.ROOT, env=environment,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert (run_dir / "training_provenance.json").is_file()
    provenance = json.loads((run_dir / "training_provenance.json").read_text())
    assert provenance["captured_before_training"] is True
    assert provenance["startup_script"]["snapshot"] == "provenance/startup_script.sh"
    durable_log = (run_dir / "run.log").read_text()
    assert durable_log.index("training_receipt_saved_before_fit") < durable_log.index("stage1_start")
    assert validate(run_dir)["status"] == "complete"

    before = exporter._tree_snapshot(run_dir)
    exported = exporter.export(run_dir, analysis_dir)
    assert exported["status"] == "complete"
    assert exporter._tree_snapshot(run_dir) == before
    with np.load(run_dir / "analysis_inputs.npz", allow_pickle=False) as payload:
        assert payload["all_labels"].dtype == np.int8
        assert np.array_equal(payload["test_segment_ids"], payload["all_segment_ids"][train:])
        if gap_at is not None:
            assert int(payload["test_segment_ids"][0]) == int(payload["all_segment_ids"][train]) == 1
    summary = json.loads((analysis_dir / "anomaly_segments.json").read_text())
    assert [(item["start_original"], item["end_original"], item["valid_points"])
            for item in summary["segments"]] == [(430, 434, 0), (478, 481, 4)]

    if floating:
        invalid_root = tmp_path / "invalid-data"
        (invalid_root / "data").mkdir(parents=True)
        raw = pd.read_csv(csv_path)
        label_rows = np.flatnonzero(raw["cols"].to_numpy() == "label")
        raw.loc[label_rows[10], "data"] = 0.5
        raw.to_csv(invalid_root / "data" / "INVALID.csv", index=False)
        pd.DataFrame([{"file_name": "INVALID.csv", "train_lens": train}]).to_csv(
            invalid_root / "DETECT_META.csv", index=False)
        invalid_run = tmp_path / "invalid-run"
        invalid_command = list(command)
        invalid_command[invalid_command.index("FLOAT_AUDIT")] = "INVALID"
        invalid_command[invalid_command.index(str(dataset_root))] = str(invalid_root)
        invalid_command[invalid_command.index(str(run_dir))] = str(invalid_run)
        rejected = subprocess.run(invalid_command, cwd=exporter.ROOT, env=environment,
                                  capture_output=True, text=True, timeout=30)
        assert rejected.returncode != 0
        assert "stage1_start" not in rejected.stdout
        assert not invalid_run.exists()

        # Emulate the historical completed-core/late-label-write failure. The
        # recovery command uses the same real long-table loader and does not
        # rerun the model or infer training-time environment facts.
        legacy_run = tmp_path / "legacy-run"
        shutil.copytree(run_dir, legacy_run)
        (legacy_run / "analysis_inputs.npz").unlink()
        (legacy_run / "analysis_inputs.json").unlink()
        (legacy_run / "training_provenance.json").unlink()
        historical_core = _core_snapshot(legacy_run)
        legacy_analysis = tmp_path / "legacy-analysis"
        recovered = subprocess.run([
            sys.executable, str(Path(exporter.ROOT) / "scripts/supplement_mnrf_view_probe_analysis_inputs.py"),
            "--run-dir", str(legacy_run), "--output-dir", str(legacy_analysis),
            "--recover-legacy", "--data-root", str(dataset_root),
        ], cwd=exporter.ROOT, capture_output=True, text=True, timeout=120)
        assert recovered.returncode == 0, recovered.stdout + "\n" + recovered.stderr
        recovered_manifest = json.loads((legacy_analysis / "analysis_manifest.json").read_text())
        assert recovered_manifest["training_provenance_status"] == "partial_historical_recovery"
        assert recovered_manifest["training_environment"] is None
        assert _core_snapshot(legacy_run) == historical_core
    if gap_at is not None:
        legacy_run = tmp_path / "legacy-gap-run"
        shutil.copytree(run_dir, legacy_run)
        manifest_path = legacy_run / "probe_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        provenance_path = legacy_run / "training_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        inputs_path = legacy_run / "analysis_inputs.npz"
        with np.load(inputs_path, allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        local_ids = np.full(len(arrays["test_segment_ids"]), -1, dtype=np.int32)
        for local_id, (left, right) in enumerate(manifest["source_identity"]["test_segments"]):
            a, b = max(train, left) - train, min(total, right) - train
            local_ids[a:b] = local_id
        arrays["test_segment_ids"] = local_ids
        temp_npz = inputs_path.with_suffix(".tmp.npz")
        np.savez(temp_npz, **arrays)
        os.replace(temp_npz, inputs_path)
        from scripts.run_mnrf_view_probe import _sha256_file
        provenance["analysis_inputs"] = {"npz": str(inputs_path), "sha256": _sha256_file(inputs_path)}
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        manifest.setdefault("analysis_artifacts", {})["analysis_inputs_sha256"] = _sha256_file(inputs_path)
        manifest["analysis_artifacts"]["training_provenance_sha256"] = _sha256_file(provenance_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        metadata_path = legacy_run / "analysis_inputs.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["labels_sha256"] = _sha256_file(inputs_path)
        metadata["source_manifest_sha256"] = _sha256_file(manifest_path)
        metadata["provenance_sha256"] = _sha256_file(provenance_path)
        metadata["test_segment_identity"] = [
            {"segment_id": index, "boundary_value": None}
            for index, _ in enumerate(manifest["source_identity"]["test_segments"])
        ]
        metadata["boundary_identity"]["test_segment_identity"] = metadata["test_segment_identity"]
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        assert supplementer._known_old_segment_numbering(legacy_run, manifest)
        migration_core = _core_snapshot(legacy_run)
        migration_output = tmp_path / "legacy-gap-analysis"
        migrated = subprocess.run([
            sys.executable, str(Path(exporter.ROOT) / "scripts/supplement_mnrf_view_probe_analysis_inputs.py"),
            "--run-dir", str(legacy_run), "--output-dir", str(migration_output),
            "--migrate-segment-ids", "--data-root", str(dataset_root),
        ], cwd=exporter.ROOT, capture_output=True, text=True, timeout=120)
        assert migrated.returncode == 0, migrated.stdout + "\n" + migrated.stderr
        with np.load(legacy_run / "analysis_inputs.npz", allow_pickle=False) as payload:
            assert np.array_equal(payload["test_segment_ids"], payload["all_segment_ids"][train:])
        migrated_provenance = json.loads((legacy_run / "training_provenance.json").read_text())
        assert migrated_provenance["migrations"][-1]["operation"] == "canonicalize_full_axis_segment_ids"
        assert _core_snapshot(legacy_run) == migration_core


def test_validator_rejects_finite_normalized_cache_tampering(tmp_path):
    # Build a small actual full source through the in-process API, then alter a
    # finite value without changing shape/dtype or the frozen file digest.
    total, train = 520, 430
    frame = pd.DataFrame(np.random.default_rng(8).normal(size=(total, 2)).astype(np.float32),
                         columns=("x", "y"), index=pd.date_range("2026-01-01", periods=total, freq="min"))
    labels = np.zeros(total, dtype=np.int8)
    frame["label"] = labels
    run = tmp_path / "small-run"
    MNRFViewProbe(seq_len=32, pattern_len=32, batch_size=32, stage1_epochs=1,
                  patience=1, device="cpu", output_dir=str(run), mode="full_gmm",
                  gmm_components=1, gmm_fit_max_windows=4, artifact_policy="full",
                  task_name="TAMPER.csv").fit(frame, train, labels[train:])
    normalized = np.load(run / "cache/normalized_values.npy", mmap_mode="r+")
    normalized[0, 0] += np.float32(100)
    normalized.flush()
    with pytest.raises(ValueError, match="normalized_values.npy sha256 mismatch"):
        validate(run)
    with pytest.raises(ValueError, match="normalized_values.npy sha256 mismatch"):
        exporter.export(run, tmp_path / "analysis")
    assert not (tmp_path / "analysis").exists()
