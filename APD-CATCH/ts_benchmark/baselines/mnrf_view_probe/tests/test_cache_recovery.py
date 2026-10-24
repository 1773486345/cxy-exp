"""Bounded tests for the frozen-cache recovery entry point."""

import hashlib
from pathlib import Path

import numpy as np
import torch

import scripts.recover_mnrf_view_probe_cache as recovery
from scripts.recover_mnrf_view_probe_cache import _find_exact_backup, _install_exact, _test_forward
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
    assert not replacement.exists()
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
    assert set(tmp_path.iterdir()) == before - {generated}


def test_exact_backup_lookup_is_project_scoped(tmp_path, monkeypatch):
    project = tmp_path / "project"
    source = project / "result" / "score" / "task" / "run"
    backup = project / "result" / "review" / "backup" / "train_features.npy"
    source.mkdir(parents=True)
    backup.parent.mkdir(parents=True)
    np.save(backup, np.zeros((2, 2, 3), dtype=np.float32))
    monkeypatch.setattr(recovery, "ROOT", project)
    assert _find_exact_backup(source, "train", _sha(backup)) == backup
    assert _find_exact_backup(source, "train", "0" * 64) is None
