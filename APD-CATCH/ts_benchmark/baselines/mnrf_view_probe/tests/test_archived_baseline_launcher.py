import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.evaluate_added_baselines import evaluate
from scripts.run_archived_baseline import _global_artifact_indices


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

