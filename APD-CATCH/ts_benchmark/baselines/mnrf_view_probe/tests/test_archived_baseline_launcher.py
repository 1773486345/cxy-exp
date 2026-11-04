import json
import subprocess
import sys
from argparse import Namespace
from collections import namedtuple
from pathlib import Path

import numpy as np

from scripts.evaluate_added_baselines import evaluate
from scripts import run_archived_baseline
from scripts.run_archived_baseline import (
    _effective_config,
    _estimate_point_artifact_bytes,
    _global_artifact_indices,
    _storage_preflight,
    _unique_run_path,
)


ROOT = Path(__file__).resolve().parents[4]


def test_archived_launcher_help_is_static():
    result = subprocess.run(
        [sys.executable, "scripts/run_archived_baseline.py", "--help"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert result.returncode == 0
    assert "provenance-rich archive" in result.stdout


def test_evaluation_rejects_coordinate_mismatch(tmp_path):
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    np.savez(left, task=np.asarray(["PSM"]), global_indices=np.asarray([0, 2, 4]), labels=np.asarray([0, 1, 0]), scores=np.asarray([.1, .9, .2]))
    np.savez(right, task=np.asarray(["PSM"]), global_indices=np.asarray([0, 1, 4]), labels=np.asarray([0, 1, 0]), scores=np.asarray([.2, .8, .3]))
    rows = evaluate("PSM", [("left", left), ("right", right)], tmp_path / "out.csv")
    assert {row["common_points"] for row in rows} == {2}
    assert (tmp_path / "out.csv").is_file()


def test_unfixed_artifact_offset_restores_global_coordinates():
    actual = np.asarray([0, 1, 2], dtype=np.int8)
    payload = {"coordinate_offset": 7}
    np.testing.assert_array_equal(_global_artifact_indices(actual, payload), [7, 8, 9])


def test_evaluation_rejects_label_mismatch_on_common_coordinates(tmp_path):
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    np.savez(left, task=np.asarray(["PSM"]), global_indices=np.asarray([0, 1]),
             labels=np.asarray([0, 1]), scores=np.asarray([.1, .9]))
    np.savez(right, task=np.asarray(["PSM"]), global_indices=np.asarray([0, 1]),
             labels=np.asarray([1, 0]), scores=np.asarray([.2, .8]))
    import pytest
    with pytest.raises(ValueError, match="labels disagree"):
        evaluate("PSM", [("left", left), ("right", right)], tmp_path / "out.csv")


def test_evaluation_requires_bound_direction_and_identity(tmp_path):
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    common = dict(task=np.asarray(["PSM"]), global_indices=np.asarray([0, 1]),
                  labels=np.asarray([0, 1]), scores=np.asarray([.1, .9]),
                  score_direction=np.asarray(["higher_is_anomalous"]),
                  source_data_sha256=np.asarray(["raw-psm-v1"]))
    np.savez(left, **common)
    np.savez(right, **common)
    rows = evaluate("PSM", [("left", left), ("right", right)], tmp_path / "strict.csv",
                     require_identity=True)
    assert {row["identity_status"] for row in rows} == {"bound"}
    bad = tmp_path / "bad.npz"
    np.savez(bad, task=np.asarray(["PSM"]), global_indices=np.asarray([0, 1]),
             labels=np.asarray([0, 1]), scores=np.asarray([.1, .9]))
    import pytest
    with pytest.raises(ValueError, match="source identity"):
        evaluate("PSM", [("bad", bad), ("left", left)], tmp_path / "bad.csv",
                 require_identity=True)


def test_evaluation_accepts_compact_scores_with_aligned_mask(tmp_path):
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    fields = dict(task=np.asarray(["PSM"]), global_indices=np.asarray([1, 2]),
                  labels=np.asarray([0, 1]), scores=np.asarray([.1, .9]),
                  valid_mask=np.asarray([False, True, True]),
                  aligned_scores=np.asarray([np.nan, .1, .9]),
                  score_direction=np.asarray(["higher_is_anomalous"]),
                  source_data_sha256=np.asarray(["raw-psm-v1"]))
    np.savez(left, **fields)
    np.savez(right, **fields)
    rows = evaluate("PSM", [("left", left), ("right", right)], tmp_path / "aligned.csv",
                     require_identity=True)
    assert rows[0]["common_points"] == 2


def test_point_artifact_size_uses_official_test_length(tmp_path):
    metadata = tmp_path / "DETECT_META.csv"
    metadata.write_text("file_name,test_lens\nPSM.csv,87841\n", encoding="utf-8")
    expected = 87841 * (8 + 64 + 1 + 1 + 8) + 1024 * 1024
    assert _estimate_point_artifact_bytes("PSM", "score", metadata) == expected


def test_storage_preflight_accounts_for_artifact_and_reserve(tmp_path):
    Usage = namedtuple("Usage", "total used free")
    enough = _storage_preflight(tmp_path, 100, 50, lambda _: Usage(1000, 800, 200))
    low = _storage_preflight(tmp_path, 151, 50, lambda _: Usage(1000, 800, 200))
    assert enough["passed"] is True
    assert enough["required_before_run_bytes"] == 150
    assert low["passed"] is False
    assert low["required_before_run_bytes"] == 201


def test_config_can_skip_duplicate_point_payload_for_native_only(tmp_path):
    config = Path("config/unfixed_detect_score_multi_config.json").resolve()
    args = Namespace(
        config_path=str(config), model_hyper_params="{}", seed=2021,
        model_name="merlion.DAGMM", task="PSM", gpus=0, num_workers=1,
        num_cpus=1, timeout=10, entry="score", point_artifact="native-only",
    )
    _, effective = _effective_config(args, "score/DAGMM/PSM/run-test")
    strategy = effective["evaluation_config"]["strategy_args"]
    assert strategy["save_anomaly_artifacts"] is False
    assert strategy["artifact_kind"] == "score"


def test_external_storage_root_is_supported_and_confined(tmp_path, monkeypatch):
    root = tmp_path / "external-results"
    root.mkdir()
    monkeypatch.setenv("BASELINE_STORAGE_ROOT", str(root))
    target, save_path = _unique_run_path("score", "DAGMM", "PSM")
    assert target == root / "score" / "DAGMM" / "PSM" / target.name
    assert save_path == str(target)
    monkeypatch.setenv("BASELINE_STORAGE_ROOT", str(root))
    import pytest
    with pytest.raises(ValueError, match="escapes"):
        _unique_run_path("score", "../../outside", "PSM")


def test_disk_preflight_blocks_before_benchmark_and_keeps_small_receipt(tmp_path, monkeypatch):
    import pytest

    script = tmp_path / "launch.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    archive = tmp_path / "archive"
    config = Path("config/unfixed_detect_score_multi_config.json").resolve()
    effective = {
        "data_config": {}, "model_config": {"models": []},
        "evaluation_config": {"metrics": "all", "strategy_args": {}},
        "report_config": {"save_path": "pending"},
        "resources": {},
    }
    monkeypatch.setattr(run_archived_baseline, "_unique_run_path",
                        lambda *args: (archive, str(archive)))
    monkeypatch.setattr(run_archived_baseline, "_effective_config",
                        lambda *args: (config, json.loads(json.dumps(effective))))
    monkeypatch.setattr(run_archived_baseline, "_data_evidence", lambda _: {})
    monkeypatch.setattr(run_archived_baseline, "_package_versions", lambda: {})
    monkeypatch.setattr(run_archived_baseline, "_git_value", lambda *_: "test")
    monkeypatch.setattr(run_archived_baseline, "_estimate_point_artifact_bytes", lambda *_: 100)
    monkeypatch.setattr(run_archived_baseline, "_storage_preflight", lambda *_: {
        "passed": False, "filesystem_free_bytes": 50, "required_before_run_bytes": 101,
    })
    def should_not_launch(*_):
        pytest.fail("benchmark must not start after a disk preflight failure")
    monkeypatch.setattr(run_archived_baseline, "_run_and_tee", should_not_launch)
    args = Namespace(
        task="PSM", entry="score", method_path="DAGMM", model_name="merlion.DAGMM",
        model_hyper_params="{}", config_path=str(config), startup_script=str(script),
        upstream="test", seed=2021, gpus=0, num_workers=1, num_cpus=1,
        timeout=1, min_free_gib=0, point_artifact="compact",
    )
    assert run_archived_baseline.run(args) == 1
    manifest = json.loads((archive / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked_disk_space"
    assert manifest["benchmark_returncode"] is None
    assert manifest["storage_preflight"]["passed"] is False
    assert not (archive / "stdout.log").exists()
    assert not (archive / "stderr.log").exists()
    assert (archive / "startup_script.sh").read_text(encoding="utf-8") == "#!/bin/sh\n"

