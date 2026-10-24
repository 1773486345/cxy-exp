"""Bounded tests for the frozen-cache recovery entry point."""

import hashlib
import os
from pathlib import Path
import tarfile
import io
import zipfile

import numpy as np
import torch

import scripts.recover_mnrf_view_probe_cache as recovery
from scripts.recover_mnrf_view_probe_cache import (
    _find_exact_backup, _install_exact, _required_artifacts, _test_forward,
)
from ts_benchmark.baselines.mnrf_lite_v2.config import MNRFConfig
from ts_benchmark.baselines.mnrf_lite_v2.training import MNRFTrainer


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_exact_install_is_idempotent_and_never_overwrites(tmp_path):
    payload = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    generated = tmp_path / "generated.npy"
    target = tmp_path / "target.npy"
    np.save(generated, payload)
    digest = _sha(generated)
    status, installed = _install_exact(generated, target, digest)
    assert status == "installed"
    assert installed == digest
    original = target.read_bytes()
    replacement = tmp_path / "replacement.npy"
    np.save(replacement, payload)
    status, installed = _install_exact(replacement, target, digest)
    assert status == "race_reused"
    assert installed == digest
    assert target.read_bytes() == original
    assert replacement.exists()
    np.save(replacement, payload + 1)
    wrong = _sha(replacement)
    try:
        _install_exact(replacement, target, digest)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("a changed file must not be accepted")
    assert wrong != digest


def test_frozen_forward_writes_float32_bounded_cache(tmp_path):
    config = MNRFConfig(
        seq_len=32, pattern_len=32, hidden_dim=8, view_dim=8,
        batch_size=2, encoding_batch_size=2, device="cpu",
        output_dir=str(tmp_path / "trainer"),
    )
    trainer = MNRFTrainer(config, variables=2, device=torch.device("cpu"))
    trainer.encoder.eval()
    values = np.random.default_rng(7).normal(size=(64, 2)).astype(np.float32)
    endpoints = np.arange(31, 64, dtype=np.int64)
    output = tmp_path / "features.npy"
    temporary, elapsed = _test_forward(
        trainer, values, endpoints, seq_len=32, batch_size=2,
        device=torch.device("cpu"), out_path=output,
    )
    assert elapsed >= 0
    array = np.load(temporary, mmap_mode="r", allow_pickle=False)
    assert array.shape == (len(endpoints), 2, 25)
    assert array.dtype == np.dtype("float32")
    assert np.isfinite(array).all()
    status, digest = _install_exact(temporary, output, _sha(temporary))
    assert status == "installed"
    assert digest == _sha(output)


def test_hash_mismatch_rejects_without_install(tmp_path):
    generated = tmp_path / "generated.npy"
    target = tmp_path / "target.npy"
    np.save(generated, np.zeros((2, 2, 3), dtype=np.float32))
    before = set(tmp_path.iterdir())
    try:
        _install_exact(generated, target, "0" * 64)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("hash mismatch must be rejected")
    assert not target.exists()
    assert set(tmp_path.iterdir()) == before
    assert generated.exists()


def test_exact_backup_lookup_is_project_scoped(tmp_path, monkeypatch):
    project = tmp_path / "project"
    source = project / "result" / "score" / "task" / "run"
    backup = project / "result" / "backup" / "train_features.npy"
    source.mkdir(parents=True)
    backup.parent.mkdir(parents=True)
    np.save(backup, np.zeros((2, 2, 3), dtype=np.float32))
    monkeypatch.setattr(recovery, "ROOT", project)
    found = _find_exact_backup(source, "train", _sha(backup), report_dir=tmp_path / "scratch")
    assert found["kind"] == "file"
    assert Path(found["path"]) == backup
    assert _find_exact_backup(source, "train", "0" * 64,
                              report_dir=tmp_path / "scratch")["kind"] is None


def test_install_race_created_equal_target_is_reused(tmp_path, monkeypatch):
    staged = tmp_path / "staged.npy"
    target = tmp_path / "target.npy"
    np.save(staged, np.arange(12, dtype=np.float32))
    digest = _sha(staged)
    link = os.link
    injected = {"done": False}

    def race_link(source, destination, *args, **kwargs):
        if Path(destination) == target and not injected["done"]:
            injected["done"] = True
            target.write_bytes(staged.read_bytes())
        return link(source, destination, *args, **kwargs)

    monkeypatch.setattr(recovery.os, "link", race_link)
    status, actual = _install_exact(staged, target, digest)
    assert injected["done"]
    assert status == "race_reused"
    assert actual == digest
    assert _sha(target) == digest
    assert staged.exists()


def test_install_race_conflict_is_not_overwritten(tmp_path, monkeypatch):
    staged = tmp_path / "staged.npy"
    target = tmp_path / "target.npy"
    np.save(staged, np.arange(12, dtype=np.float32))
    digest = _sha(staged)
    conflict = b"concurrent writer owns these bytes"
    link = os.link
    injected = {"done": False}

    def race_link(source, destination, *args, **kwargs):
        if Path(destination) == target and not injected["done"]:
            injected["done"] = True
            target.write_bytes(conflict)
        return link(source, destination, *args, **kwargs)

    monkeypatch.setattr(recovery.os, "link", race_link)
    try:
        _install_exact(staged, target, digest)
    except FileExistsError as error:
        assert "conflict" in str(error)
    else:
        raise AssertionError("concurrent conflicting target must be rejected")
    assert target.read_bytes() == conflict
    assert staged.exists()


def test_install_refuses_dangling_symlink(tmp_path):
    staged = tmp_path / "staged.npy"
    target = tmp_path / "target.npy"
    np.save(staged, np.arange(4, dtype=np.float32))
    target.symlink_to(tmp_path / "missing-target")
    link_before = os.readlink(str(target))
    try:
        _install_exact(staged, target, _sha(staged))
    except FileExistsError as error:
        assert "non-regular" in str(error)
    else:
        raise AssertionError("dangling symlink must not be considered absent")
    assert target.is_symlink()
    assert os.readlink(str(target)) == link_before
    assert not (tmp_path / "missing-target").exists()


def test_install_exdev_uses_no_replace_on_target_filesystem(tmp_path, monkeypatch):
    staged = tmp_path / "staged.npy"
    target = tmp_path / "target.npy"
    np.save(staged, np.arange(4, dtype=np.float32))
    expected = _sha(staged)
    link = os.link
    calls = {"count": 0}
    conflict = b"target created during EXDEV copy"

    def exdev_then_race(source, destination, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError(recovery.errno.EXDEV, "cross-device link")
        target.write_bytes(conflict)
        return link(source, destination, *args, **kwargs)

    monkeypatch.setattr(recovery.os, "link", exdev_then_race)
    try:
        _install_exact(staged, target, expected)
    except FileExistsError as error:
        assert "conflict" in str(error)
    else:
        raise AssertionError("EXDEV fallback must still not replace a racing target")
    assert target.read_bytes() == conflict
    assert not list(tmp_path.glob(".target.npy.recover-*.tmp"))
    assert staged.exists()


def test_archive_search_streams_exact_zip_and_tar_members(tmp_path, monkeypatch):
    project = tmp_path / "project"
    source = project / "result" / "score" / "task" / "run"
    archive_dir = project / "result" / "backup"
    source.mkdir(parents=True)
    archive_dir.mkdir(parents=True)
    data = np.arange(60, dtype=np.float32).reshape(10, 2, 3)
    cache = tmp_path / "train_features.npy"
    np.save(cache, data)
    digest = _sha(cache)
    zip_path = archive_dir / "cache.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(cache, "backup/old/train_features.npy")
    tar_path = archive_dir / "cache.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(cache, arcname="run/cache/train_features.npy")
    monkeypatch.setattr(recovery, "ROOT", project)
    found_zip = _find_exact_backup(source, "train", digest, report_dir=tmp_path / "zip-stage")
    assert found_zip["kind"] == "archive"
    assert found_zip["member"].endswith("train_features.npy")
    assert _sha(found_zip["staged_path"]) == digest
    found_tar = _find_exact_backup(source, "train", digest, report_dir=tmp_path / "tar-stage")
    # The fixed archive ordering is lexical filesystem discovery; either exact
    # archive member is acceptable and must retain its archive provenance.
    assert found_tar["kind"] == "archive"
    assert found_tar["path"].endswith(("cache.zip", "cache.tar.gz"))
    assert _sha(found_tar["staged_path"]) == digest


def test_full_source_inventory_requires_all_four_splits_and_core_artifacts(tmp_path):
    manifest = {
        "cache_shapes": {name: [2, 1, 25] for name in
                         ("train", "validation", "calibration", "test")},
        "checkpoint_sha256": {"best_encoder.pt": "b", "initial_encoder.pt": "i"},
        "input_statistics_sha256": "s", "normalized_cache_sha256": "n",
        "feature_statistics_sha256": "f", "gmm_status_sha256": "g",
        "gmm_fit_endpoints_sha256": "e",
    }
    required, missing = _required_artifacts(tmp_path, manifest)
    assert len(required) >= 25
    assert "cache/validation_features.npy" in missing
    assert "cache/calibration_window_end_indices.npy" in missing
    for name in ("train", "validation", "calibration", "test"):
        (tmp_path / "cache").mkdir(exist_ok=True)
        (tmp_path / "cache" / f"{name}_features.npy").touch()
        (tmp_path / "cache" / f"{name}_window_end_indices.npy").touch()
    for relative in required:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=True)
    _, missing_after = _required_artifacts(tmp_path, manifest)
    assert missing_after == []


def test_train_and_test_alone_never_count_as_a_complete_source(tmp_path, monkeypatch):
    project = tmp_path / "project"
    source = project / "result" / "score" / "MNRF-ViewProbe" / "PSM" / "run"
    source.mkdir(parents=True)
    (source / "cache").mkdir()
    manifest = {
        "artifact_policy": "full", "head_type": "gmm",
        "cache_shapes": {name: [2, 1, 25] for name in
                         ("train", "validation", "calibration", "test")},
        "cache_sha256": {}, "endpoint_sha256": {},
    }
    for name in ("train", "test"):
        cache = source / "cache" / f"{name}_features.npy"
        endpoint = source / "cache" / f"{name}_window_end_indices.npy"
        np.save(cache, np.zeros((2, 1, 25), dtype=np.float32))
        np.save(endpoint, np.asarray([31, 32], dtype=np.int64))
        manifest["cache_sha256"][name] = _sha(cache)
        manifest["endpoint_sha256"][name] = _sha(endpoint)
    for name in ("validation", "calibration"):
        endpoint = source / "cache" / f"{name}_window_end_indices.npy"
        np.save(endpoint, np.asarray([31, 32], dtype=np.int64))
        manifest["cache_sha256"][name] = "a" * 64
        manifest["endpoint_sha256"][name] = _sha(endpoint)
    (source / "probe_manifest.json").write_text(__import__("json").dumps(manifest))
    monkeypatch.setattr(recovery, "ROOT", project)
    monkeypatch.setattr(recovery, "_source_path", lambda task: source)
    result = recovery.recover_task("PSM", tmp_path / "report", "cpu",
                                   requested_splits=["train", "test"])
    assert result["status"] in {"blocked", "partial_unrequested_split"}
    assert result["status"] != "already_complete"
    assert set(result.get("missing", [])) & {
        "cache/validation_features.npy", "cache/calibration_features.npy",
        "validation", "calibration",
    }
